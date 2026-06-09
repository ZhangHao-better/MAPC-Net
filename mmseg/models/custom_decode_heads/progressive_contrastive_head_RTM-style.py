# Copyright (c) OpenMMLab. All rights reserved.
"""
Progressive Contrastive Head with PSCC-Net Architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import FocalLoss, BinaryDiceLoss
from typing import List, Tuple, Dict, Optional

class MemoryBank(object):
    """RTM-style memory bank for contrastive learning.

    Store a short queue of positive/negative features sampled from recent iterations.
    Each memory item is a tensor of shape [N, 1, C] (L2-normalized).
    """

    def __init__(self, max_steps: int = 1, max_sample: int = 256):
        self.max_steps = int(max_steps)
        self.max_sample = int(max_sample)
        self.reset()

    def reset(self):
        self.pos_bank = []
        self.neg_bank = []

    def _trim(self):
        if len(self.pos_bank) > self.max_steps:
            self.pos_bank = self.pos_bank[-self.max_steps:]
        if len(self.neg_bank) > self.max_steps:
            self.neg_bank = self.neg_bank[-self.max_steps:]

    @torch.no_grad()
    def update(self,
               feats_flat: torch.Tensor,
               pos_idx_all: torch.Tensor, pos_idx_unused: torch.Tensor,
               neg_idx_all: torch.Tensor, neg_idx_unused: torch.Tensor):
        """Update memory bank.

        Args:
            feats_flat: [N, 1, C] (detached)
            pos_idx_all/neg_idx_all: indices of all pos/neg pixels (flattened)
            pos_idx_unused/neg_idx_unused: indices of unused pixels (prefer to store)
        """
        self.pos_bank.append(self._exclusive_sampling(feats_flat, pos_idx_all, pos_idx_unused))
        self.neg_bank.append(self._exclusive_sampling(feats_flat, neg_idx_all, neg_idx_unused))
        self._trim()

    @torch.no_grad()
    def _exclusive_sampling(self, feats_flat: torch.Tensor,
                            idx_all: torch.Tensor, idx_unused: torch.Tensor):
        if feats_flat is None:
            return None
        C = feats_flat.size(-1)
        if idx_all is None or idx_all.numel() == 0:
            return feats_flat.new_zeros((0, 1, C))

        idx_all = idx_all.view(-1)
        idx_unused = idx_unused.view(-1) if idx_unused is not None else idx_all.new_zeros((0,), dtype=torch.long)

        # 1) take from unused first
        samples = feats_flat.new_zeros((0, 1, C))
        if idx_unused.numel() > 0:
            num_take = min(idx_unused.numel(), self.max_sample)
            take = idx_unused[torch.randperm(idx_unused.numel(), device=idx_unused.device)[:num_take]]
            samples = feats_flat[take]

        # 2) supplement from all if still not enough
        if samples.size(0) < self.max_sample:
            need = self.max_sample - samples.size(0)
            num_take = min(idx_all.numel(), need)
            take = idx_all[torch.randperm(idx_all.numel(), device=idx_all.device)[:num_take]]
            samples2 = feats_flat[take]
            samples = torch.cat([samples, samples2], dim=0)

        return F.normalize(samples, dim=2)

    @torch.no_grad()
    def sample(self, num_sample: Optional[int] = None, pos: bool = True):
        bank = self.pos_bank if pos else self.neg_bank
        if len(bank) == 0:
            return None
        mem = torch.cat([m for m in bank if m is not None], dim=0)
        if mem.numel() == 0:
            return None
        if num_sample is None or num_sample >= mem.size(0):
            return mem
        return mem[-num_sample:]

class NonLocalMask(nn.Module):
    """NonLocal Mask模块"""
    
    def __init__(self, in_channels: int, reduce_scale: int = 1):
        super(NonLocalMask, self).__init__()
        self.r = reduce_scale
        self.ic = in_channels * self.r * self.r
        self.mc = self.ic
        
        self.g = nn.Conv2d(self.ic, self.ic, kernel_size=1)
        self.theta = nn.Conv2d(self.ic, self.mc, kernel_size=1)
        self.phi = nn.Conv2d(self.ic, self.mc, kernel_size=1)
        self.W_s = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_c = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma_s = nn.Parameter(torch.ones(1))
        self.gamma_c = nn.Parameter(torch.ones(1))
        
        self.getmask = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
            # 移除Sigmoid，输出logits用于Focal Loss
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x1 = x.reshape(b, self.ic, h // self.r, w // self.r) if self.r > 1 else x
        
        g_x = self.g(x1).view(b, self.ic, -1).permute(0, 2, 1)
        theta_x = self.theta(x1).view(b, self.mc, -1)
        phi_x = self.phi(x1).view(b, self.mc, -1)
        
        # Spatial attention
        f_s = torch.matmul(theta_x.permute(0, 2, 1), phi_x)
        f_s_div = F.softmax(f_s, dim=-1)
        
        # Channel attention
        f_c = torch.matmul(theta_x, phi_x.permute(0, 2, 1))
        f_c_div = F.softmax(f_c, dim=-1)
        
        y_s = torch.matmul(f_s_div, g_x).permute(0, 2, 1).contiguous().view(b, c, h, w)
        y_c = torch.matmul(g_x, f_c_div).view(b, c, h, w)
        
        z = x + self.gamma_s * self.W_s(y_s) + self.gamma_c * self.W_c(y_c)
        mask = self.getmask(z)
        
        return mask, z


@MODELS.register_module()
class ProgressiveContrastiveHead(BaseDecodeHead):
    """渐进式对比学习解码头"""
    
    def __init__(self, crop_size=(512, 512), reduce_scales=[4, 2, 2, 1], 
                 use_contrastive=False,
                 contrastive_loss_cfg=None,
                 contrastive_weight=0.02,
                 contrastive_warmup_iters=20000,
                 contrastive_levels=('z1', 'z2', 'z3'),
                 contrastive_proj_dim=128,
                 contrastive_samples_per_img=256,   # pixel-style: 每张图：fg=n, bg=n
                 contrastive_boundary_ratio=0.5,    # 0~1；不足自动回退
                 ignore_index=255,
                 contrastive_gather=True,
                 # --- Contrastive style ---
                 contrastive_style='pixel',         # 'pixel'(exp20) | 'rtm'(RTM-style)
                 # --- RTM-style sampling/memory ---
                 rtm_cl_sampler='ori',              # 'ori' | 'balance' | 'max' | 'hard' | 'edge'
                 rtm_batch_cl=True,                 # sample across batch (like RTM)
                 rtm_use_memory=False,
                 rtm_max_points=1024,               # per class (pos/neg)
                 rtm_max_memory_step=1,
                 rtm_max_memory_size=256,
                 **kwargs):
        super(ProgressiveContrastiveHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        
        self.crop_size = crop_size
        self.reduce_scales = reduce_scales
        self.use_contrastive = use_contrastive
        self.ignore_index = ignore_index

        # ===== Contrastive branch =====
        if self.use_contrastive:
            assert contrastive_loss_cfg is not None, "need contrastive_loss_cfg when use_contrastive=True"
            # 建议 cfg 里 type='SupConLoss'
            # 让 contrastive_gather 真正生效：覆盖 cfg 里的 gather
            cfg = copy.deepcopy(contrastive_loss_cfg)
            cfg['gather'] = bool(contrastive_gather)
            # 注意：RTM 风格采样/记忆库会导致不同显卡上采样点数 M 不一致。
            # 如果启用跨卡聚合（gather=True），SupConLoss 的聚合实现必须支持“变长采样”。
            self.contrastive_loss = MODELS.build(cfg)
            self.contrastive_gather = bool(contrastive_gather)
            self.contrastive_weight = float(contrastive_weight)
            self.contrastive_warmup_iters = int(contrastive_warmup_iters)
            self.contrastive_levels = tuple(contrastive_levels)
            self.contrastive_proj_dim = int(contrastive_proj_dim)
            self.contrastive_samples_per_img = int(contrastive_samples_per_img)
            self.contrastive_boundary_ratio = float(contrastive_boundary_ratio)

            # contrastive style switch
            self.contrastive_style = str(contrastive_style)
            assert self.contrastive_style in ['pixel', 'rtm'], \
                f"invalid contrastive_style={self.contrastive_style}"

            # --- RTM-style params ---
            self.rtm_cl_sampler = str(rtm_cl_sampler)
            self.rtm_batch_cl = bool(rtm_batch_cl)
            self.rtm_use_memory = bool(rtm_use_memory) and (self.contrastive_style == 'rtm')
            self.rtm_max_points = int(rtm_max_points)
            self.rtm_all_points = int(2 * self.rtm_max_points)
            self.rtm_max_memory_step = int(rtm_max_memory_step)
            self.rtm_max_memory_size = int(rtm_max_memory_size)
            self.rtm_memory_bank = None
            if self.rtm_use_memory:
                self.rtm_memory_bank = MemoryBank(
                    max_steps=self.rtm_max_memory_step,
                    max_sample=self.rtm_max_memory_size
                )

            # projection head (per level)
            self.proj = nn.ModuleDict()
            ch_map = {'z1': self.in_channels[0], 'z2': self.in_channels[1],
                      'z3': self.in_channels[2], 'z4': self.in_channels[3]}
            for k in self.contrastive_levels:
                in_ch = ch_map[k]
                if self.contrastive_style == 'rtm':
                    # same as RTM: 3x3 -> (dim*4) + BN + act, then 1x1 -> dim (no BN/act)
                    self.proj[k] = nn.Sequential(
                        ConvModule(
                            in_channels=in_ch,
                            out_channels=self.contrastive_proj_dim * 4,
                            kernel_size=3,
                            padding=1,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg),
                        ConvModule(
                            in_channels=self.contrastive_proj_dim * 4,
                            out_channels=self.contrastive_proj_dim,
                            kernel_size=1,
                            norm_cfg=None,
                            act_cfg=None),
                    )
                else:
                    # exp20 style: simple 1x1 projection
                    self.proj[k] = nn.Conv2d(in_ch, self.contrastive_proj_dim, kernel_size=1, bias=False)


        self.getmask4 = NonLocalMask(self.in_channels[3], reduce_scales[3])
        self.getmask3 = NonLocalMask(self.in_channels[2], reduce_scales[2])
        self.getmask2 = NonLocalMask(self.in_channels[1], reduce_scales[1])
        self.getmask1 = NonLocalMask(self.in_channels[0], reduce_scales[0])
        
        # 初始化Focal Loss
        self.focal_loss = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0
        )
        
        # 初始化Binary Dice Loss (仅用于mask1)
        self.dice_loss = BinaryDiceLoss(
            smooth=1,
            exponent=2,
            reduction='mean',
            loss_weight=1.0  # Dice自身的内部权重
        )
        self.dice_loss_weight = 0.01  # 在总损失中给Dice的系数
        self.boundary_width = 2         # 边界宽度：1~3都行，先用2（对应5x5核）

        # 1) 定义 total_loss_scale（先设为2.0，保证训练强度对比loss2时匹配）
        self.total_loss_scale = 2.0  # 调整为合适的 scale
    
    def forward(self, inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = self._transform_inputs(inputs)
        s1, s2, s3, s4 = inputs
        
        target_1 = self.crop_size
        target_2 = tuple([i // 2 for i in self.crop_size])
        target_3 = tuple([i // 4 for i in self.crop_size])
        target_4 = tuple([i // 8 for i in self.crop_size])
        
        if s1.shape[2:] != target_1:
            s1 = F.interpolate(s1, size=target_1, mode='bilinear', align_corners=False)
            s2 = F.interpolate(s2, size=target_2, mode='bilinear', align_corners=False)
            s3 = F.interpolate(s3, size=target_3, mode='bilinear', align_corners=False)
            s4 = F.interpolate(s4, size=target_4, mode='bilinear', align_corners=False)
        
        # 残差门控: s_refined = s + (s * sigmoid(mask_up))
        mask4, z4 = self.getmask4(s4)
        mask4_up = F.interpolate(mask4, size=s3.size()[2:], mode='bilinear', align_corners=False)
        gate4 = torch.sigmoid(mask4_up)
        s3_refined = s3 + (s3 * gate4)
        mask3, z3 = self.getmask3(s3_refined)
        
        mask3_up = F.interpolate(mask3, size=s2.size()[2:], mode='bilinear', align_corners=False)
        gate3 = torch.sigmoid(mask3_up)
        s2_refined = s2 + (s2 * gate3)
        mask2, z2 = self.getmask2(s2_refined)
        
        mask2_up = F.interpolate(mask2, size=s1.size()[2:], mode='bilinear', align_corners=False)
        gate2 = torch.sigmoid(mask2_up)
        s1_refined = s1 + (s1 * gate2)
        mask1, z1 = self.getmask1(s1_refined)
        
        if self.training:
            out = {'mask1': mask1, 'mask2': mask2, 'mask3': mask3, 'mask4': mask4}
            if self.use_contrastive:
                # 只返回 config 里指定的对比层，避免“没用到的 z 也被传出去”
                z_map = {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4}
                for k in self.contrastive_levels:
                    if k in z_map:
                        out[k] = z_map[k]
            return out
        return {'mask1': mask1}
    
    def _get_iter(self):
        # 尽量从 MessageHub 取（支持 resume），取不到就退化成本地计数
        if not hasattr(self, "_local_iter"):
            self._local_iter = 0
        self._local_iter += 1
        it = self._local_iter
        try:
            from mmengine.logging import MessageHub
            hub = MessageHub.get_current_instance()
            it2 = hub.get_info('iter')
            if it2 is not None:
                it = int(it2)
        except Exception:
            pass
        return it

    def _warmup_weight(self):
        if self.contrastive_warmup_iters <= 0:
            return self.contrastive_weight
        t = self._get_iter()
        return self.contrastive_weight * min(1.0, max(0.0, t / float(self.contrastive_warmup_iters)))

    def _make_boundary(self, tgt01, width):
        # tgt01: [B,1,H,W] in {0,1}
        w = int(max(1, width))
        k = 2 * w + 1
        dilate = F.max_pool2d(tgt01, kernel_size=k, stride=1, padding=w)
        erode  = 1.0 - F.max_pool2d(1.0 - tgt01, kernel_size=k, stride=1, padding=w)
        return (dilate - erode).clamp(0, 1)

    def _sample_points(self, z_proj, gt_hw, boundary=None):
        # z_proj: [B,C,H,W] (requires grad)
        # gt_hw:  [B,H,W] long (0/1/255)
        B, C, H, W = z_proj.shape
        n = self.contrastive_samples_per_img
        feats = []
        labs = []
        device = z_proj.device

        z_flat = z_proj.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B,HW,C]
        gt_flat = gt_hw.reshape(B, H*W)

        if boundary is not None:
            bd_flat = boundary.reshape(B, H*W) > 0.5
        else:
            bd_flat = None

        # 只对“索引/采样” no_grad：先算 idx / label
        # 注意：特征 gather 必须在 no_grad 外做，否则会 detach，contrastive 不反传
        chosen = []
        # ---- stats (python number; no_grad) ----
        used_imgs = 0
        pos_avail_sum = 0
        neg_avail_sum = 0
        pos_bd_avail_sum = 0
        neg_bd_avail_sum = 0
        pos_unique_sum = 0
        neg_unique_sum = 0
        pos_need_repl_sum = 0
        neg_need_repl_sum = 0
        pos_bd_sel_sum = 0
        neg_bd_sel_sum = 0
        with torch.no_grad():
            for b in range(B):
                gt_b = gt_flat[b]
                valid = (gt_b != self.ignore_index)
                pos = valid & (gt_b == 1)
                neg = valid & (gt_b == 0)

                pos_idx = torch.nonzero(pos, as_tuple=False).squeeze(1)
                neg_idx = torch.nonzero(neg, as_tuple=False).squeeze(1)

                if pos_idx.numel() < 2 or neg_idx.numel() < 2:
                    continue

                def pick(idx_all, idx_bd, want):
                    if idx_bd is None or idx_bd.numel() == 0 or self.contrastive_boundary_ratio <= 0:
                        # 全局采样
                        if idx_all.numel() >= want:
                            sel = idx_all[torch.randperm(idx_all.numel(), device=device)[:want]]
                        else:
                            sel = idx_all[torch.randint(0, idx_all.numel(), (want,), device=device)]
                        return sel

                    want_bd = int(round(want * self.contrastive_boundary_ratio))
                    want_bd = max(0, min(want, want_bd))
                    want_glb = want - want_bd

                    # 先取边界
                    if idx_bd.numel() >= want_bd:
                        sel_bd = idx_bd[torch.randperm(idx_bd.numel(), device=device)[:want_bd]]
                    else:
                        sel_bd = idx_bd[torch.randint(0, idx_bd.numel(), (want_bd,), device=device)] if want_bd > 0 else idx_bd[:0]
                    # 再补全局
                    if idx_all.numel() >= want_glb:
                        sel_glb = idx_all[torch.randperm(idx_all.numel(), device=device)[:want_glb]]
                    else:
                        sel_glb = idx_all[torch.randint(0, idx_all.numel(), (want_glb,), device=device)] if want_glb > 0 else idx_all[:0]
                    return torch.cat([sel_bd, sel_glb], dim=0)

                if bd_flat is not None:
                    pos_bd = pos_idx[bd_flat[b][pos_idx]]
                    neg_bd = neg_idx[bd_flat[b][neg_idx]]
                else:
                    pos_bd, neg_bd = None, None

                sel_pos = pick(pos_idx, pos_bd, n)
                sel_neg = pick(neg_idx, neg_bd, n)

                sel = torch.cat([sel_pos, sel_neg], dim=0)  # [2n]
                lab = torch.cat([
                    torch.ones(n, device=device, dtype=torch.long),
                    torch.zeros(n, device=device, dtype=torch.long)
                ], dim=0)
                chosen.append((b, sel, lab))

                # ---- stats update ----
                used_imgs += 1
                pos_avail_sum += int(pos_idx.numel())
                neg_avail_sum += int(neg_idx.numel())
                pos_unique_sum += int(sel_pos.unique().numel())
                neg_unique_sum += int(sel_neg.unique().numel())
                pos_need_repl_sum += int(pos_idx.numel() < n)
                neg_need_repl_sum += int(neg_idx.numel() < n)
                if bd_flat is not None:
                    pos_bd_avail_sum += int(pos_bd.numel()) if pos_bd is not None else 0
                    neg_bd_avail_sum += int(neg_bd.numel()) if neg_bd is not None else 0
                    pos_bd_sel_sum += int(bd_flat[b][sel_pos].sum().item())
                    neg_bd_sel_sum += int(bd_flat[b][sel_neg].sum().item())
        
        # gather 特征要保留梯度：不能在 no_grad 里做
        for b, sel, lab in chosen:
            feats.append(z_flat[b, sel])  # [2n,C] keep grad
            labs.append(lab)

        if len(feats) == 0:
            # 返回空：不能 reshape 原 tensor，直接创建空 tensor
            feat = z_proj.new_zeros((0, C))
            lab = z_proj.new_zeros((0,), dtype=torch.long)
        else:
            feat = torch.cat(feats, dim=0)  # [M,C]
            lab = torch.cat(labs, dim=0)    # [M]

        feat = F.normalize(feat, dim=1)
        feat = feat.unsqueeze(1)  # [M,1,C] -> SupConLoss 需要 [bsz,n_views,C]
        # ---- pack stats (python numbers) ----
        if used_imgs > 0:
            stats = dict(
                used_imgs=used_imgs,
                skip_imgs=int(B - used_imgs),
                M=int(used_imgs * 2 * n),
                pos_avail_mean=float(pos_avail_sum / used_imgs),
                neg_avail_mean=float(neg_avail_sum / used_imgs),
                pos_unique_ratio=float(pos_unique_sum / (used_imgs * n)),
                neg_unique_ratio=float(neg_unique_sum / (used_imgs * n)),
                pos_need_repl_rate=float(pos_need_repl_sum / used_imgs),
                neg_need_repl_rate=float(neg_need_repl_sum / used_imgs),
                pos_bd_avail_mean=float(pos_bd_avail_sum / used_imgs),
                neg_bd_avail_mean=float(neg_bd_avail_sum / used_imgs),
                pos_bd_sel_ratio=float(pos_bd_sel_sum / (used_imgs * n)),
                neg_bd_sel_ratio=float(neg_bd_sel_sum / (used_imgs * n)),
            )
        else:
            stats = dict(
                used_imgs=0, skip_imgs=int(B), M=0,
                pos_avail_mean=0.0, neg_avail_mean=0.0,
                pos_unique_ratio=0.0, neg_unique_ratio=0.0,
                pos_need_repl_rate=0.0, neg_need_repl_rate=0.0,
                pos_bd_avail_mean=0.0, neg_bd_avail_mean=0.0,
                pos_bd_sel_ratio=0.0, neg_bd_sel_ratio=0.0,
            )
        return feat, lab, stats
    
    def _sample_points_rtm(self, z_proj, gt_hw, boundary=None, prob=None):
        """RTM 风格的点采样 +（可选）记忆库。

        目的：对齐“全批次扁平化采样”逻辑，替代 exp20 的“逐图固定数量采样”。

        采样特点：
        - 先把单卡进程内的一个批次全部展开到同一维度（rtm_batch_cl=True）
        - 采样器：ori / balance / max（可选 hard / edge）
        - 可选记忆库：rtm_use_memory=True 时，会把历史正/负样本拼接进来

        返回：
        - feat:  [M, 1, C]（已做 L2 归一化）
        - lab:   [M]（1=篡改前景，0=背景）
        - stats: dict（仅用于日志打印，不参与反传）
        """
        # z_proj: [B,C,H,W]
        # gt_hw:  [B,H,W] long (0/1/ignore)
        B, C, H, W = z_proj.shape
        device = z_proj.device

        if boundary is not None and boundary.dim() == 4:
            boundary = boundary.squeeze(1)  # [B,H,W]
        if prob is not None and prob.dim() == 4:
            prob = prob.squeeze(1)          # [B,H,W]

        # flatten GT
        gt_flat = gt_hw.reshape(B, H * W)

        # ---------- availability stats (per image) ----------
        with torch.no_grad():
            valid = (gt_flat != self.ignore_index)
            pos_avail_each = (valid & (gt_flat == 1)).sum(dim=1)  # [B]
            neg_avail_each = (valid & (gt_flat == 0)).sum(dim=1)
            used_mask = (pos_avail_each > 0) & (neg_avail_each > 0)
            used_imgs = int(used_mask.sum().item())
            pos_avail_mean = float(pos_avail_each[used_mask].float().mean().item()) if used_imgs > 0 else 0.0
            neg_avail_mean = float(neg_avail_each[used_mask].float().mean().item()) if used_imgs > 0 else 0.0

        if boundary is not None:
            bd_flat = (boundary.reshape(B, H * W) > 0.5)
            with torch.no_grad():
                pos_bd_avail_each = (bd_flat & (gt_flat == 1)).sum(dim=1)
                neg_bd_avail_each = (bd_flat & (gt_flat == 0)).sum(dim=1)
                pos_bd_avail_mean = float(pos_bd_avail_each[used_mask].float().mean().item()) if used_imgs > 0 else 0.0
                neg_bd_avail_mean = float(neg_bd_avail_each[used_mask].float().mean().item()) if used_imgs > 0 else 0.0
        else:
            bd_flat = None
            pos_bd_avail_mean = 0.0
            neg_bd_avail_mean = 0.0

        prob_flat = prob.reshape(B, H * W) if prob is not None else None

        # ---------- flatten features (keep grad) ----------
        feats = z_proj.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,HW,C]
        feats_flat = feats.reshape(-1, 1, C)  # [N,1,C]
        gt_all = gt_flat.reshape(-1)          # [N]
        bd_all = bd_flat.reshape(-1) if bd_flat is not None else None
        prob_all = prob_flat.reshape(-1) if prob_flat is not None else None

        # ---------- choose indices (no grad) ----------
        with torch.no_grad():
            pos_idx = torch.nonzero(gt_all == 1, as_tuple=False).squeeze(1)  # [Np]
            neg_idx = torch.nonzero(gt_all == 0, as_tuple=False).squeeze(1)  # [Nn]

            # 样本不足：直接返回空（本层对比学习会被跳过）
            if pos_idx.numel() < 2 or neg_idx.numel() < 2:
                feat = z_proj.new_zeros((0, 1, C))
                lab = z_proj.new_zeros((0,), dtype=torch.long)
                stats = dict(
                    used_imgs=0, skip_imgs=int(B), M=0,
                    pos_avail_mean=0.0, neg_avail_mean=0.0,
                    pos_unique_ratio=0.0, neg_unique_ratio=0.0,
                    pos_need_repl_rate=0.0, neg_need_repl_rate=0.0,
                    pos_bd_avail_mean=0.0, neg_bd_avail_mean=0.0,
                    pos_bd_sel_ratio=0.0, neg_bd_sel_ratio=0.0,
                )
                return feat, lab, stats

            sampler = str(self.rtm_cl_sampler).lower()
            max_p = int(self.rtm_max_points)
            all_p = int(self.rtm_all_points)

            # hard 采样需要 prob（预测概率图）
            if sampler == 'hard' and prob_all is None:
                raise ValueError('rtm_cl_sampler=hard requires prob map (sigmoid(logits))')

            # 对非 hard 的采样器先打乱索引，保证随机性
            if sampler not in ['hard']:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=device)]
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=device)]

            # 根据采样器类型确定本次想要采多少正/负样本
            if sampler == 'balance':
                k = min(pos_idx.numel(), neg_idx.numel(), max_p)
                want_pos = want_neg = int(k)
            elif sampler == 'ori':
                if pos_idx.numel() < max_p:
                    want_pos = int(pos_idx.numel())
                    want_neg = int(min(neg_idx.numel(), all_p - want_pos))
                elif neg_idx.numel() < max_p:
                    want_neg = int(neg_idx.numel())
                    want_pos = int(min(pos_idx.numel(), all_p - want_neg))
                else:
                    want_pos = want_neg = int(max_p)
            elif sampler == 'max':
                want_pos = int(min(pos_idx.numel(), max_p))
                want_neg = int(min(neg_idx.numel(), max_p))
            elif sampler == 'edge':
                # 仅在边界上采样；如果没有 boundary，则退化为全局采样
                if bd_all is None:
                    want_pos = int(min(pos_idx.numel(), max_p))
                    want_neg = int(min(neg_idx.numel(), max_p))
                else:
                    pos_bd = pos_idx[bd_all[pos_idx]]
                    neg_bd = neg_idx[bd_all[neg_idx]]
                    pos_idx = pos_bd
                    neg_idx = neg_bd
                    want_pos = int(min(pos_idx.numel(), max_p))
                    want_neg = int(min(neg_idx.numel(), max_p))
            elif sampler == 'hard':
                want_pos = int(min(pos_idx.numel(), max_p))
                want_neg = int(min(neg_idx.numel(), max_p))
            else:
                raise ValueError(f'Unsupported rtm_cl_sampler={sampler}')

            # 工具函数：按（可选）边界优先或 hard 分数选点
            def pick(idx_all_1d, want, bd_mask_1d=None, score_1d=None, largest=True):
                if want <= 0 or idx_all_1d.numel() == 0:
                    return idx_all_1d[:0]
                if score_1d is not None:
                    score = score_1d[idx_all_1d]
                    k = min(int(want), int(score.numel()))
                    top = torch.topk(score, k=k, largest=largest).indices
                    return idx_all_1d[top]

                if bd_mask_1d is not None and self.contrastive_boundary_ratio > 0:
                    is_bd = bd_mask_1d[idx_all_1d]
                    idx_bd = idx_all_1d[is_bd]
                    idx_glb = idx_all_1d[~is_bd]
                    want_bd = int(round(want * self.contrastive_boundary_ratio))
                    want_bd = max(0, min(want, want_bd))
                    sel_bd = idx_bd[:min(idx_bd.numel(), want_bd)]
                    sel_glb = idx_glb[:min(idx_glb.numel(), want - sel_bd.numel())]
                    return torch.cat([sel_bd, sel_glb], dim=0)

                # 朴素策略：直接取前缀（前面已打乱过）
                return idx_all_1d[:min(idx_all_1d.numel(), want)]

            if sampler == 'hard':
                sel_pos = pick(pos_idx, want_pos, bd_mask_1d=None, score_1d=(1.0 - prob_all), largest=True)
                sel_neg = pick(neg_idx, want_neg, bd_mask_1d=None, score_1d=prob_all, largest=True)
            else:
                sel_pos = pick(pos_idx, want_pos, bd_mask_1d=bd_all, score_1d=None)
                sel_neg = pick(neg_idx, want_neg, bd_mask_1d=bd_all, score_1d=None)

            # 记忆库用的“未被本次选中”的索引：all - selected
            sel_mask = torch.zeros_like(gt_all, dtype=torch.bool)
            sel_mask[torch.cat([sel_pos, sel_neg], dim=0)] = True
            pos_unused = pos_idx[~sel_mask[pos_idx]]
            neg_unused = neg_idx[~sel_mask[neg_idx]]

            # boundary selection ratio stats
            if bd_all is not None:
                pos_bd_sel = int(bd_all[sel_pos].sum().item()) if sel_pos.numel() > 0 else 0
                neg_bd_sel = int(bd_all[sel_neg].sum().item()) if sel_neg.numel() > 0 else 0
            else:
                pos_bd_sel = 0
                neg_bd_sel = 0

        # ---------- 取特征（保留梯度） ----------
        pos_feats = feats_flat[sel_pos]  # [Np,1,C]
        neg_feats = feats_flat[sel_neg]  # [Nn,1,C]
        pos_feats = F.normalize(pos_feats, dim=2)
        neg_feats = F.normalize(neg_feats, dim=2)

        # ---------- 拼接记忆库样本（作为额外对比样本） ----------
        if self.rtm_memory_bank is not None:
            pos_mem = self.rtm_memory_bank.sample(pos=True)
            neg_mem = self.rtm_memory_bank.sample(pos=False)
            if pos_mem is not None and pos_mem.numel() > 0:
                pos_feats = torch.cat([pos_feats, pos_mem], dim=0)
            if neg_mem is not None and neg_mem.numel() > 0:
                neg_feats = torch.cat([neg_feats, neg_mem], dim=0)

            # 采样完成后再更新记忆库（保存当前迭代的特征）
            self.rtm_memory_bank.update(
                feats_flat.detach(),
                pos_idx_all=pos_idx, pos_idx_unused=pos_unused,
                neg_idx_all=neg_idx, neg_idx_unused=neg_unused
            )

        feat = torch.cat([pos_feats, neg_feats], dim=0)  # [M,1,C]
        lab = torch.cat([
            torch.ones(pos_feats.size(0), device=device, dtype=torch.long),
            torch.zeros(neg_feats.size(0), device=device, dtype=torch.long),
        ], dim=0)

        # ---------- 统计量 ----------
        M = int(feat.size(0))
        pos_unique_ratio = float(sel_pos.unique().numel() / max(1, sel_pos.numel()))
        neg_unique_ratio = float(sel_neg.unique().numel() / max(1, sel_neg.numel()))
        # 注意：RTM 风格不做“放回采样”，这里仅表示“是否样本不足”
        pos_need_repl_rate = float(pos_idx.numel() < max_p)
        neg_need_repl_rate = float(neg_idx.numel() < max_p)
        pos_bd_sel_ratio = float(pos_bd_sel / max(1, sel_pos.numel()))
        neg_bd_sel_ratio = float(neg_bd_sel / max(1, sel_neg.numel()))

        stats = dict(
            used_imgs=used_imgs,
            skip_imgs=int(B - used_imgs),
            M=M,
            pos_avail_mean=pos_avail_mean,
            neg_avail_mean=neg_avail_mean,
            pos_unique_ratio=pos_unique_ratio,
            neg_unique_ratio=neg_unique_ratio,
            pos_need_repl_rate=pos_need_repl_rate,
            neg_need_repl_rate=neg_need_repl_rate,
            pos_bd_avail_mean=pos_bd_avail_mean,
            neg_bd_avail_mean=neg_bd_avail_mean,
            pos_bd_sel_ratio=pos_bd_sel_ratio,
            neg_bd_sel_ratio=neg_bd_sel_ratio,
        )
        return feat, lab, stats
    
    def loss_by_feat(self, seg_logits: Dict[str, torch.Tensor], 
                     batch_data_samples) -> Dict[str, torch.Tensor]:
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        B, H, W = seg_label.shape
        
        # Downsample ground truth masks to match prediction sizes
        # 转为long类型（FocalLoss要求类别索引）
        gt_mask1 = seg_label.long()
        gt_mask2 = F.interpolate(seg_label.unsqueeze(1).float(), 
                                 size=(H//2, W//2), mode='nearest').squeeze(1).long()
        gt_mask3 = F.interpolate(seg_label.unsqueeze(1).float(), 
                                 size=(H//4, W//4), mode='nearest').squeeze(1).long()
        gt_mask4 = F.interpolate(seg_label.unsqueeze(1).float(), 
                                 size=(H//8, W//8), mode='nearest').squeeze(1).long()
        
        # 使用Focal Loss + 渐进式权重 [0.6, 0.8, 1.0, 1.0]
        # mask1: Focal Loss + Binary Dice Loss
        logit1 = seg_logits['mask1']  # [B, 1, H, W]
        
        # 1) Focal Loss (和之前一样)
        loss1_focal = self.focal_loss(logit1, gt_mask1)
        
        # # 2) Binary Dice: 先sigmoid得到概率，再和0/1 GT计算(之前的整体Dice，先注释掉，改成边界Dice)
        # prob1 = torch.sigmoid(logit1)  # [B, 1, H, W], ∈[0,1]
        # dice_target = gt_mask1.float().unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        # loss1_dice = self.dice_loss(prob1, dice_target)

        # 2) Boundary Dice (只在边界带算 Dice)
        prob1 = torch.sigmoid(logit1)  # [B,1,H,W]
        tgt1  = (gt_mask1 == 1).float().unsqueeze(1)  # [B,1,H,W]，确保是0/1

        # --- 用形态学梯度生成边界带 ---
        w = self.boundary_width
        k = 2 * w + 1

        # dilation / erosion (纯 torch，无需opencv)
        dilate = F.max_pool2d(tgt1, kernel_size=k, stride=1, padding=w)
        erode  = 1.0 - F.max_pool2d(1.0 - tgt1, kernel_size=k, stride=1, padding=w)

        boundary = (dilate - erode).clamp(0, 1)  # 边界带（大约2w像素厚）

        # 如果未来真的出现 ignore=255，也可以加这行更稳（现在可选）
        # valid = (gt_mask1 != 255).float().unsqueeze(1)
        # boundary = boundary * valid

        prob_b = prob1 * boundary
        tgt_b  = tgt1  * boundary

        # 边界带为空（例如该 crop 没有篡改区域）时，bdice 置 0，避免出现无意义/不稳定的统计
        if boundary.sum() < 1:
            loss1_bdice = prob1.sum() * 0.0   # 保证是 tensor、在同设备上
        else:
            loss1_bdice = self.dice_loss(prob_b, tgt_b)

        
        # 额外统计（detach，避免参与反传）
        boundary_ratio = boundary.mean().detach()
        bdice_raw = loss1_bdice.detach()
        bdice_coef = (1.0 - bdice_raw)
        
        losses = {
            'loss_mask1_focal': loss1_focal * 0.6,
            'loss_mask1_bdice': loss1_bdice * self.dice_loss_weight,

            # === 仅用于日志观察，不参与 loss 求和 ===
            'mask1_boundary_ratio': boundary_ratio,
            'mask1_bdice_raw': bdice_raw,
            'mask1_bdice_coef': bdice_coef,
        }
        
        if 'mask2' in seg_logits:
            loss2 = self.focal_loss(seg_logits['mask2'], gt_mask2)
            losses['loss_mask2'] = loss2 * 0.8
        
        if 'mask3' in seg_logits:
            loss3 = self.focal_loss(seg_logits['mask3'], gt_mask3)
            losses['loss_mask3'] = loss3 * 1.0
        
        if 'mask4' in seg_logits:
            loss4 = self.focal_loss(seg_logits['mask4'], gt_mask4)
            losses['loss_mask4'] = loss4 * 1.0
        
        # ===== Contrastive loss =====
        if self.use_contrastive and any(k in seg_logits for k in self.contrastive_levels):
            w_con = self._warmup_weight()
            con_sum = None
            con_cnt = 0

            # 给不同尺度一个简单权重（可在 cfg 里扩展）
            lvl_w = {'z1': 1.0, 'z2': 1.0, 'z3': 1.0, 'z4': 1.0}
            gt_map = {'z1': gt_mask1, 'z2': gt_mask2, 'z3': gt_mask3, 'z4': gt_mask4}

            for k in self.contrastive_levels:
                z = seg_logits[k]                # [B,C,H,W]
                gt = gt_map[k]                   # [B,H,W]
                z_proj = self.proj[k](z)         # [B,proj_dim,H,W]

                # boundary（尺度越粗，width 越小）
                if self.contrastive_boundary_ratio > 0:
                    tgt01 = (gt == 1).float().unsqueeze(1)
                    width = 2 if k == 'z1' else 1
                    bd = self._make_boundary(tgt01, width)
                else:
                    bd = None

                if getattr(self, 'contrastive_style', 'pixel') == 'rtm':
                    prob = None
                    if str(self.rtm_cl_sampler).lower() == 'hard':
                        # use corresponding mask logits as hardness source
                        mask_key = {'z1': 'mask1', 'z2': 'mask2', 'z3': 'mask3', 'z4': 'mask4'}.get(k, 'mask1')
                        if mask_key in seg_logits:
                            prob = torch.sigmoid(seg_logits[mask_key]).detach()
                            if prob.shape[2:] != z_proj.shape[2:]:
                                prob = F.interpolate(prob, size=z_proj.shape[2:], mode='bilinear', align_corners=False)
                    feat, lab, stat = self._sample_points_rtm(z_proj, gt, boundary=bd, prob=prob)
                else:
                    feat, lab, stat = self._sample_points(z_proj, gt, bd)

                
                loss_k = self.contrastive_loss(feat, lab)
                # DDP 稳定性修复：
                # 即使当前 rank 采样为空，或 contrastive loss 主要由其它 rank 的常量特征构成，
                # 也强制把“本地当前这条投影分支”的真实前向路径挂到计算图上。
                # 这样比直接对参数 p.sum()*0 更稳，因为它保证使用的是当前迭代的本地 z_proj 计算图。
                local_dummy = z_proj.mean() * 0.0

                if loss_k is None:
                    loss_k = local_dummy
                else:
                    loss_k = loss_k + local_dummy

                if con_sum is None:
                    con_sum = lvl_w.get(k, 1.0) * loss_k
                else:
                    con_sum = con_sum + lvl_w.get(k, 1.0) * loss_k
                con_cnt += 1

            # ---- per-level logs (no grad) ----
                dev = gt_mask1.device
                def _f(x): return torch.tensor(float(x), device=dev).detach()
                def _i(x): return torch.tensor(int(x), device=dev, dtype=torch.float32).detach()
                losses[f'contrast_{k}_used_imgs'] = _i(stat['used_imgs'])
                losses[f'contrast_{k}_skip_imgs'] = _i(stat['skip_imgs'])
                losses[f'contrast_{k}_M'] = _i(stat['M'])
                losses[f'contrast_{k}_pos_avail_mean'] = _f(stat['pos_avail_mean'])
                losses[f'contrast_{k}_neg_avail_mean'] = _f(stat['neg_avail_mean'])
                losses[f'contrast_{k}_pos_unique_ratio'] = _f(stat['pos_unique_ratio'])
                losses[f'contrast_{k}_neg_unique_ratio'] = _f(stat['neg_unique_ratio'])
                losses[f'contrast_{k}_pos_need_repl_rate'] = _f(stat['pos_need_repl_rate'])
                losses[f'contrast_{k}_neg_need_repl_rate'] = _f(stat['neg_need_repl_rate'])
                losses[f'contrast_{k}_pos_bd_avail_mean'] = _f(stat['pos_bd_avail_mean'])
                losses[f'contrast_{k}_neg_bd_avail_mean'] = _f(stat['neg_bd_avail_mean'])
                losses[f'contrast_{k}_pos_bd_sel_ratio'] = _f(stat['pos_bd_sel_ratio'])
                losses[f'contrast_{k}_neg_bd_sel_ratio'] = _f(stat['neg_bd_sel_ratio'])
                # 注意：mmengine parse_losses 会把 key 里包含 "loss" 的项计入总 loss
                # 这里是纯日志项，必须避开 "loss" 子串
                losses[f'contrast_{k}_raw'] = loss_k.detach()

            # ---- final (weighted) loss + raw log ----
            if con_cnt > 0 and con_sum is not None:
                con_raw = con_sum / con_cnt
            else:
                con_raw = gt_mask1.sum() * 0.0
            losses['loss_contrastive'] = con_raw * w_con
            # 纯日志项：避开 "loss" 子串
            losses['contrastive_raw'] = con_raw.detach()

        # 2) 对每个 loss 逐项乘以 total_loss_scale
        if self.total_loss_scale != 1.0:
            for k in list(losses.keys()):
                # if k.startswith('loss_'):
                if k.startswith('loss_') and (k != 'loss_contrastive'):
                    losses[k] = losses[k] * self.total_loss_scale

        
        return losses
    
    def predict(self, inputs: List[torch.Tensor], batch_img_metas: List[dict], 
                test_cfg: dict) -> torch.Tensor:
        mask_logits = self.forward(inputs)['mask1']
        # 测试时需要sigmoid转换为概率
        return torch.sigmoid(mask_logits)
