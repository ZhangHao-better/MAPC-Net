import os
import os.path as osp
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils import import_modules_from_strings
from mmseg.utils import register_all_modules


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def sanitize_decode_head_for_tsne(cfg):
    """适配当前代码库里的 seg-only ProgressiveContrastiveHead。
    当前生效的 head 不接受 contrastive_loss_cfg 等 kwargs，
    但我们画 t-SNE 只需要 forward 时内部产生的 z1/z2 特征，不需要真的构建 contrastive 分支。
    """
    dh = cfg.model.decode_head

    drop_keys = [
        "contrastive_loss_cfg",
        "contrastive_weight",
        "contrastive_warmup_iters",
        "contrastive_levels",
        "contrastive_proj_dim",
        "contrastive_samples_per_img",
        "contrastive_boundary_ratio",
        "contrastive_gather",
        "ignore_index",
    ]

    for k in drop_keys:
        if k in dh:
            dh.pop(k)

    # 强制走 seg-only 版 head 的构造路径
    dh.use_contrastive = False
    return cfg

def build_runner_and_loader(cfg_path, ckpt_path):
    # 先注册 mmseg / 自定义模块，和 train.py 保持一致
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(cfg_path)

    if "custom_imports" in cfg:
        import_modules_from_strings(**cfg["custom_imports"])

    # ===== 关键修复 2：把 contrastive 配置裁掉，适配当前 seg-only head =====
    cfg = sanitize_decode_head_for_tsne(cfg)

    # ===== 关键修复 1：补 work_dir =====
    if cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs",
            osp.splitext(osp.basename(cfg_path))[0]
        )

    # 可选：避免脚本里误触 resume
    cfg.resume = False

    runner = Runner.from_cfg(cfg)
    model = unwrap_model(runner.model)

    # w/ contrast 的 checkpoint 里可能含有额外 projection / contrastive 分支参数
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)
    model.cuda()
    model.eval()

    # val dataloader
    if hasattr(runner, "val_loop") and runner.val_loop is not None:
        data_loader = runner.val_loop.dataloader
    elif hasattr(runner, "val_dataloader"):
        data_loader = runner.val_dataloader
    else:
        raise RuntimeError("Cannot find val dataloader from runner.")

    return model, data_loader


def get_hook_module(model, feature_name="z1"):
    # 你当前 head 里 z1/z2/z3/z4 分别对应 getmask1/2/3/4
    mapping = {
        "z1": model.decode_head.getmask1,
        "z2": model.decode_head.getmask2,
        "z3": model.decode_head.getmask3,
        "z4": model.decode_head.getmask4,
    }
    return mapping[feature_name]


def stack_gt_from_samples(data_samples):
    # mmseg 的 gt_sem_seg 一般在 data_sample.gt_sem_seg.data
    gts = []
    for ds in data_samples:
        gt = ds.gt_sem_seg.data
        if gt.dim() == 3:
            gt = gt.squeeze(0)
        gts.append(gt)
    return torch.stack(gts, dim=0)  # [B, H, W]


def sample_pixel_features(feat_map, gt_map, n_pos=32, n_neg=32, ignore_index=255):
    """
    feat_map: [B, C, H, W]
    gt_map:   [B, H, W]  0/1/255
    """
    B, C, H, W = feat_map.shape
    feat_map = feat_map.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]

    feats = []
    labels = []

    for b in range(B):
        gt = gt_map[b]
        feat = feat_map[b]

        pos = torch.nonzero(gt == 1, as_tuple=False)
        neg = torch.nonzero(gt == 0, as_tuple=False)

        if len(pos) < 2 or len(neg) < 2:
            continue

        pos_num = min(n_pos, len(pos))
        neg_num = min(n_neg, len(neg))

        pos_idx = pos[torch.randperm(len(pos), device=pos.device)[:pos_num]]
        neg_idx = neg[torch.randperm(len(neg), device=neg.device)[:neg_num]]

        pos_feat = feat[pos_idx[:, 0], pos_idx[:, 1]]   # [Np, C]
        neg_feat = feat[neg_idx[:, 0], neg_idx[:, 1]]   # [Nn, C]

        feats.append(pos_feat)
        feats.append(neg_feat)

        labels.append(torch.ones(pos_feat.size(0), device=feat.device, dtype=torch.long))
        labels.append(torch.zeros(neg_feat.size(0), device=feat.device, dtype=torch.long))

    if len(feats) == 0:
        return None, None

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


@torch.no_grad()
def extract_features_for_tsne(
    cfg_path,
    ckpt_path,
    feature_name="z1",
    max_images=200,
    n_pos_per_img=32,
    n_neg_per_img=32,
):
    model, data_loader = build_runner_and_loader(cfg_path, ckpt_path)

    cache = {}

    def hook_fn(module, inp, out):
        # NonLocalMask 返回 (mask, z)
        cache["feat"] = out[1].detach()

    handle = get_hook_module(model, feature_name).register_forward_hook(hook_fn)

    all_feats = []
    all_labels = []

    seen_imgs = 0

    for data in data_loader:
        # 预处理
        batch = model.data_preprocessor(data, training=False)

        inputs = batch["inputs"].cuda()
        extras = batch.get("extras", None)
        data_samples = batch["data_samples"]

        if extras is not None:
            for k, v in extras.items():
                if torch.is_tensor(v):
                    extras[k] = v.cuda()

        # 编码 + 解码，hook 会自动抓到 z1/z2/z3/z4
        feats = model.forward_encoder(inputs, extras)
        _ = model.decode_head.forward(feats)

        feat_map = cache["feat"]  # [B, C, H, W]
        gt = stack_gt_from_samples(data_samples).to(feat_map.device)  # [B,H0,W0]

        # resize GT 到对应特征分辨率
        gt_rs = F.interpolate(
            gt.unsqueeze(1).float(),
            size=feat_map.shape[-2:],
            mode="nearest"
        ).squeeze(1).long()

        feats_sampled, labels_sampled = sample_pixel_features(
            feat_map,
            gt_rs,
            n_pos=n_pos_per_img,
            n_neg=n_neg_per_img,
            ignore_index=255,
        )

        if feats_sampled is not None:
            all_feats.append(feats_sampled.cpu())
            all_labels.append(labels_sampled.cpu())

        seen_imgs += inputs.size(0)
        if seen_imgs >= max_images:
            break

    handle.remove()

    if len(all_feats) == 0:
        raise RuntimeError("No valid sampled features found.")

    X = torch.cat(all_feats, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()
    return X, y


def run_tsne(X, random_state=42, pca_dim=50, perplexity=30):
    # 可选：先 PCA 降一轮，t-SNE 更稳更快
    if X.shape[1] > pca_dim and X.shape[0] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)

    perplexity = min(perplexity, max(5, (len(X) - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        n_iter=2000,
    )
    Z = tsne.fit_transform(X)
    return Z


def plot_two_panel_tsne(
    Z1, y1, title1,
    Z2, y2, title2,
    save_path="tsne_compare.png"
):
    plt.figure(figsize=(10, 4.5), dpi=180)

    # 左图
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(Z1[y1 == 1, 0], Z1[y1 == 1, 1], s=6, c="red", label="Tampered", alpha=0.8)
    ax1.scatter(Z1[y1 == 0, 0], Z1[y1 == 0, 1], s=6, c="blue", label="Authentic", alpha=0.8)
    ax1.set_title(title1)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 右图
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(Z2[y2 == 1, 0], Z2[y2 == 1, 1], s=6, c="red", label="Tampered", alpha=0.8)
    ax2.scatter(Z2[y2 == 0, 0], Z2[y2 == 0, 1], s=6, c="blue", label="Authentic", alpha=0.8)
    ax2.set_title(title2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    set_seed(42)

    # ===== 你改这里 =====
    cfg_wo = "/home/zhlinux/RTM-progressive/ASCFormer/configs/ascformer/ascformer_rtm_progressive_21.32.py"
    ckpt_wo = "/home/zhlinux/RTM-progressive/ASCFormer/work_dirs/ascformer_rtm_progressive_acc3_crop0.7/iter_240000.pth"

    cfg_w = "/home/zhlinux/RTM-progressive/ASCFormer/configs/ascformer/ascformer_rtm_progressive_22.33.py"
    ckpt_w = "/home/zhlinux/RTM-progressive/ASCFormer/work_dirs/ascformer_rtm_progressive_contrastive_exp20z1/iter_200000.pth"

    feature_name = "z1"   # 推荐先画 z1
    max_images = 200
    n_pos_per_img = 32
    n_neg_per_img = 32
    print("1")
    # 1) 提取 w/o contrast 特征
    X_wo, y_wo = extract_features_for_tsne(
        cfg_wo, ckpt_wo,
        feature_name=feature_name,
        max_images=max_images,
        n_pos_per_img=n_pos_per_img,
        n_neg_per_img=n_neg_per_img,
    )
    print("2")
    # 2) 提取 w/ contrast 特征
    X_w, y_w = extract_features_for_tsne(
        cfg_w, ckpt_w,
        feature_name=feature_name,
        max_images=max_images,
        n_pos_per_img=n_pos_per_img,
        n_neg_per_img=n_neg_per_img,
    )
    print("3")
    # 可选：L2 normalize
    X_wo = X_wo / (np.linalg.norm(X_wo, axis=1, keepdims=True) + 1e-12)
    X_w = X_w / (np.linalg.norm(X_w, axis=1, keepdims=True) + 1e-12)
    print("4")
    # 3) t-SNE
    Z_wo = run_tsne(X_wo, random_state=42, pca_dim=50, perplexity=30)
    Z_w = run_tsne(X_w, random_state=42, pca_dim=50, perplexity=30)
    print("5")
    # 4) 画图
    plot_two_panel_tsne(
    Z1=Z_wo,
    y1=y_wo,
    title1="(a) w/o Contrast",
    Z2=Z_w,
    y2=y_w,
    title2="(b) w/ Contrast",
    save_path=f"tsne_{feature_name}_compare.png"
)