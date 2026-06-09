import argparse
import json
import random
from types import MethodType

import numpy as np

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings


def _quantiles(arr, qs=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    if len(arr) == 0:
        return {str(q): None for q in qs}
    a = np.asarray(arr, dtype=np.float64)
    return {str(q): float(np.percentile(a, q)) for q in qs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to your mmseg config, e.g. configs/ascformer_rtm_progressive.py")
    parser.add_argument("--num", type=int, default=2000, help="how many samples to run (default: 2000)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--out", type=str, default="fg_crop_stats_2k.json", help="output json file")
    args = parser.parse_args()

    # -------------------------
    # Reproducibility
    # -------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------
    # Load config + custom imports
    # -------------------------
    cfg = Config.fromfile(args.config)
    if "custom_imports" in cfg and cfg.custom_imports is not None:
        import_modules_from_strings(**cfg.custom_imports)

    # Register mmseg modules (IMPORTANT: init default scope to mmseg)
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    # Force default scope to mmseg, so 'LoadAnnotations' resolves to mmseg version not mmcv
    from mmengine.registry import DefaultScope
    DefaultScope.get_instance('fg_stats', scope_name='mmseg')

    # Build dataset (train split)
    from mmseg.registry import DATASETS
    train_dataset_cfg = cfg.train_dataloader.dataset
    dataset = DATASETS.build(train_dataset_cfg)

    # -------------------------
    # Find RandomCropWithExtra in pipeline
    # -------------------------
    pipe = getattr(dataset, "pipeline", None)
    if pipe is None or not hasattr(pipe, "transforms"):
        raise RuntimeError("dataset.pipeline not found or has no .transforms; cannot locate RandomCropWithExtra.")

    crop_t = None
    for t in pipe.transforms:
        if t.__class__.__name__ == "RandomCropWithExtra":
            crop_t = t
            break

    if crop_t is None:
        # 有些项目是嵌套 Compose
        for t in pipe.transforms:
            if hasattr(t, "transforms"):
                for tt in t.transforms:
                    if tt.__class__.__name__ == "RandomCropWithExtra":
                        crop_t = tt
                        break
            if crop_t is not None:
                break

    if crop_t is None:
        raise RuntimeError("Cannot find RandomCropWithExtra in dataset pipeline. "
                           "请确认 train pipeline 里确实用了 RandomCropWithExtra，并且脚本跑的是 train_dataloader.dataset。")

    stats = {
        "records": [],
        "errors": 0,
    }

    # -------------------------
    # Monkeypatch crop_bbox: 复刻你 transforms.py 里逻辑 + 统计
    # -------------------------
    def instrumented_crop_bbox(self, results: dict) -> tuple:
        img = results["img"]

        def generate_crop_bbox(img_: np.ndarray) -> tuple:
            margin_h = max(img_.shape[0] - self.crop_size[0], 0)
            margin_w = max(img_.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)

            if self.stride is not None:
                offset_h = int(offset_h / self.stride) * self.stride
                offset_w = int(offset_w / self.stride) * self.stride

            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
            return crop_y1, crop_y2, crop_x1, crop_x2

        crop_bbox = generate_crop_bbox(img)

        seg_full = results.get("gt_seg_map", None)
        rec = {
            "fg_full_pixels": None,
            "fg_crop_pixels": None,
            "ensure_triggered": False,
            "ensure_success": None,
            "ensure_retry": 0,
            "min_required": 0,
            "best_fg": None,
            "last_fg": None,
            "catmax_applied": bool(getattr(self, "cat_max_ratio", 1.0) < 1.0),
        }

        # gt 全 0 短路
        if seg_full is None:
            stats["records"].append(rec)
            return crop_bbox

        valid_full = (seg_full != self.ignore_index)
        fg_full_pixels = int(((seg_full > 0) & valid_full).sum())
        rec["fg_full_pixels"] = fg_full_pixels

        if fg_full_pixels == 0:
            # 直接随机 crop（无前景可采）
            seg_tmp = self.crop(results["gt_seg_map"], crop_bbox)
            valid = (seg_tmp != self.ignore_index)
            rec["fg_crop_pixels"] = int(((seg_tmp > 0) & valid).sum())
            stats["records"].append(rec)
            return crop_bbox

        # cat_max_ratio（如果你设成 1.0，这段不会生效）
        if self.cat_max_ratio < 1.0:
            for _ in range(10):
                seg_temp = self.crop(results["gt_seg_map"], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and (np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        # ensure_fg：放到最后
        if self.ensure_fg_prob > 0 and np.random.rand() < self.ensure_fg_prob:
            rec["ensure_triggered"] = True

            min_required = int(self.ensure_fg_min_pixels)
            if getattr(self, "ensure_fg_min_ratio", 0.0) > 0:
                min_required = min(
                    min_required,
                    max(1, int(fg_full_pixels * float(self.ensure_fg_min_ratio)))
                )
            min_required = max(1, int(min_required))
            rec["min_required"] = int(min_required)

            success = False
            best_bbox = crop_bbox
            best_fg = -1
            last_fg = 0
            retry = 0

            for retry in range(1, int(self.ensure_fg_max_retry) + 1):
                seg_temp = self.crop(results["gt_seg_map"], crop_bbox)
                valid = (seg_temp != self.ignore_index)
                fg = (seg_temp > 0) & valid
                last_fg = int(fg.sum())

                if last_fg >= min_required:
                    success = True
                    # 如果不开 select_best，命中就立刻停
                    if not getattr(self, "ensure_fg_select_best", False):
                        break

                if last_fg > best_fg:
                    best_fg = last_fg
                    best_bbox = crop_bbox

                crop_bbox = generate_crop_bbox(img)

            rec["ensure_success"] = bool(success)
            rec["ensure_retry"] = int(retry)
            rec["best_fg"] = int(best_fg if best_fg >= 0 else last_fg)
            rec["last_fg"] = int(last_fg)

            # 不成功 或 select_best=True 时，用 best_bbox
            if (not success) or getattr(self, "ensure_fg_select_best", False):
                crop_bbox = best_bbox

        # 最终 crop 的 fg 像素
        seg_tmp = self.crop(results["gt_seg_map"], crop_bbox)
        valid = (seg_tmp != self.ignore_index)
        rec["fg_crop_pixels"] = int(((seg_tmp > 0) & valid).sum())

        stats["records"].append(rec)
        return crop_bbox

    crop_t.crop_bbox = MethodType(instrumented_crop_bbox, crop_t)

    # -------------------------
    # Run N samples
    # -------------------------
    N = args.num
    for i in range(N):
        try:
            _ = dataset[i % len(dataset)]  # 触发 pipeline
        except Exception as e:
            stats["errors"] += 1
            # 不中断，继续跑
            continue

    recs = stats["records"]
    fg_crop = [r["fg_crop_pixels"] for r in recs if r["fg_crop_pixels"] is not None]
    fg_full = [r["fg_full_pixels"] for r in recs if r["fg_full_pixels"] is not None]

    trig = [r for r in recs if r["ensure_triggered"]]
    trig_succ = [r for r in trig if r["ensure_success"] is True]
    trig_fail = [r for r in trig if r["ensure_success"] is False]

    retry = [r["ensure_retry"] for r in trig]
    minreq = [r["min_required"] for r in trig]
    last_fg = [r["last_fg"] for r in trig if r["last_fg"] is not None]

    summary = {
        "num_requested": N,
        "num_records": len(recs),
        "errors": stats["errors"],

        "fg_full_zero_rate": float(np.mean([1 if (r["fg_full_pixels"] == 0) else 0 for r in recs if r["fg_full_pixels"] is not None])) if fg_full else None,
        "fg_crop_zero_rate": float(np.mean([1 if (x == 0) else 0 for x in fg_crop])) if fg_crop else None,

        "fg_crop_pixels_quantiles": _quantiles(fg_crop),
        "fg_full_pixels_quantiles": _quantiles(fg_full),

        "ensure_trigger_rate": (len(trig) / len(recs)) if len(recs) else None,
        "ensure_success_rate_given_trigger": (len(trig_succ) / len(trig)) if len(trig) else None,
        "ensure_avg_retry_given_trigger": float(np.mean(retry)) if retry else None,
        "ensure_retry_quantiles": _quantiles(retry),
        "ensure_min_required_quantiles": _quantiles(minreq),
        "ensure_last_fg_quantiles": _quantiles(last_fg),

        "counts": {
            "triggered": len(trig),
            "triggered_success": len(trig_succ),
            "triggered_fail": len(trig_fail),
        }
    }

    out_obj = {"summary": summary, "records": recs[:200]}  # records 太大了，先截前 200 条方便你看
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print("\n======== FG CROP STATS (SUMMARY) ========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {args.out} (contains summary + first 200 records)")


if __name__ == "__main__":
    main()
