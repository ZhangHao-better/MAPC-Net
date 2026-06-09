import argparse
import json
import random
import traceback
from collections import Counter
from types import MethodType

import numpy as np

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings


def _quantiles(arr, qs=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    if len(arr) == 0:
        return {str(q): None for q in qs}
    a = np.asarray(arr, dtype=np.float64)
    return {str(q): float(np.percentile(a, q)) for q in qs}


def _mean(arr):
    return float(np.mean(arr)) if len(arr) else None


def _bucket_id(fg_full: int) -> str:
    if fg_full == 0:
        return "fg_full=0"
    if fg_full <= 64:
        return "1-64"
    if fg_full <= 256:
        return "65-256"
    if fg_full <= 1024:
        return "257-1024"
    return "1025+"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to your mmseg config, e.g. configs/ascformer_rtm_progressive.py")
    parser.add_argument("--num", type=int, default=2000, help="how many samples to run (default: 2000)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--out", type=str, default="fg_crop_stats_v4.json", help="output json file")
    parser.add_argument("--record_limit", type=int, default=200, help="how many per-sample records to dump")

    # 可选：不改 config 直接覆盖关键超参，方便对照
    parser.add_argument("--prob", type=float, default=None, help="override ensure_fg_prob")
    parser.add_argument("--min_pixels", type=int, default=None, help="override ensure_fg_min_pixels")
    parser.add_argument("--max_retry", type=int, default=None, help="override ensure_fg_max_retry")
    parser.add_argument("--min_ratio", type=float, default=None, help="override ensure_fg_min_ratio")
    parser.add_argument("--select_best", type=int, default=None, help="override ensure_fg_select_best (0/1)")
    parser.add_argument("--cat_max_ratio", type=float, default=None, help="override cat_max_ratio")

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

    # Force default scope to mmseg
    from mmengine.registry import DefaultScope
    DefaultScope.get_instance("fg_stats", scope_name="mmseg")

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
        # nested Compose
        for t in pipe.transforms:
            if hasattr(t, "transforms"):
                for tt in t.transforms:
                    if tt.__class__.__name__ == "RandomCropWithExtra":
                        crop_t = tt
                        break
            if crop_t is not None:
                break

    if crop_t is None:
        raise RuntimeError(
            "Cannot find RandomCropWithExtra in dataset pipeline. "
            "请确认 train pipeline 里确实用了 RandomCropWithExtra，并且脚本跑的是 train_dataloader.dataset。"
        )

    # -------------------------
    # Optional overrides (对照实验不用改 config)
    # -------------------------
    if args.prob is not None:
        crop_t.ensure_fg_prob = float(args.prob)
    if args.min_pixels is not None:
        crop_t.ensure_fg_min_pixels = int(args.min_pixels)
    if args.max_retry is not None:
        crop_t.ensure_fg_max_retry = int(args.max_retry)
    if args.min_ratio is not None:
        crop_t.ensure_fg_min_ratio = float(args.min_ratio)
    if args.select_best is not None:
        crop_t.ensure_fg_select_best = bool(int(args.select_best))
    if args.cat_max_ratio is not None:
        crop_t.cat_max_ratio = float(args.cat_max_ratio)

    stats = {
        "records": [],
        "errors": 0,
        "error_types": {},
        "first_error_traceback": None,
    }

    # -------------------------
    # Monkeypatch crop_bbox: 基于你原版逻辑 + 新增诊断统计
    # -------------------------
    def instrumented_crop_bbox(self, results: dict) -> tuple:
        img = results["img"]

        # 用你原版的本地 generate_crop_bbox（不依赖类里是否有同名函数）
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
            "fg_cover_ratio": None,  # fg_crop / fg_full

            # 新增：初始随机 crop 的 fg（用于衡量 ensure_fg 的“提升量”）
            "fg_init": None,

            # 新增：catmax 后、ensure 前的 fg
            "fg_after_catmax": None,

            # ensure 统计
            "ensure_triggered": False,
            "ensure_success": None,
            "ensure_retry": 0,

            # 新增：阈值拆解（定位 min_ratio 是否把门槛压低）
            "min_required": 0,
            "min_required_pixels": int(getattr(self, "ensure_fg_min_pixels", 0)),
            "min_required_ratio": None,
            "min_required_was_ratio_active": None,  # min_required < min_pixels ?

            # 新增：达到门槛那一刻的 fg（证明是否“很早停”）
            "stop_fg": None,
            "stop_reason": None,  # fg_full_zero / not_triggered / ensure_hit / ensure_max_retry

            "best_fg": None,
            "last_fg": None,

            # 新增：最终 bbox 是否来自 best
            "final_bbox_is_best": None,

            "catmax_applied": bool(getattr(self, "cat_max_ratio", 1.0) < 1.0),
        }

        if seg_full is None:
            # pipeline 里如果 crop 发生在 LoadAnnotations 前，seg_full 可能 None
            rec["stop_reason"] = "seg_full_none"
            if len(stats["records"]) < args.record_limit:
                stats["records"].append(rec)
            return crop_bbox

        valid_full = (seg_full != self.ignore_index)
        fg_full_pixels = int(((seg_full > 0) & valid_full).sum())
        rec["fg_full_pixels"] = fg_full_pixels

        # 初始随机 crop 的 fg
        seg_tmp0 = self.crop(seg_full, crop_bbox)
        valid0 = (seg_tmp0 != self.ignore_index)
        rec["fg_init"] = int(((seg_tmp0 > 0) & valid0).sum())

        # gt 全 0 短路
        if fg_full_pixels == 0:
            rec["stop_reason"] = "fg_full_zero"
            rec["fg_crop_pixels"] = rec["fg_init"]
            rec["fg_cover_ratio"] = None
            if len(stats["records"]) < args.record_limit:
                stats["records"].append(rec)
            return crop_bbox

        # cat_max_ratio（如果你设成 1.0，这段不会生效）
        if self.cat_max_ratio < 1.0:
            for _ in range(10):
                seg_temp = self.crop(seg_full, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and (np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        seg_tmp1 = self.crop(seg_full, crop_bbox)
        valid1 = (seg_tmp1 != self.ignore_index)
        rec["fg_after_catmax"] = int(((seg_tmp1 > 0) & valid1).sum())

        # ensure_fg：放到最后
        if self.ensure_fg_prob > 0 and np.random.rand() < self.ensure_fg_prob:
            rec["ensure_triggered"] = True

            # 对齐你当前 transforms.py 的 min_required 语义（安全版：再 clip 到 fg_full_pixels）
            min_required = int(self.ensure_fg_min_pixels)
            ratio = float(getattr(self, "ensure_fg_min_ratio", 0.0) or 0.0)
            if ratio > 0:
                rec["min_required_ratio"] = int(max(1, fg_full_pixels * ratio))
                min_required = min(min_required, max(1, int(fg_full_pixels * ratio)))
            min_required = min(min_required, int(fg_full_pixels))
            min_required = max(1, int(min_required))

            rec["min_required"] = int(min_required)
            rec["min_required_was_ratio_active"] = bool(min_required < int(self.ensure_fg_min_pixels))

            success = False
            best_bbox = crop_bbox
            best_fg = -1
            last_fg = 0
            stop_fg = None
            stop_reason = None
            retry = 0

            for retry in range(1, int(self.ensure_fg_max_retry) + 1):
                seg_temp = self.crop(seg_full, crop_bbox)
                valid = (seg_temp != self.ignore_index)
                fg_pixels = int(((seg_temp > 0) & valid).sum())
                last_fg = fg_pixels

                if fg_pixels > best_fg:
                    best_fg = fg_pixels
                    best_bbox = crop_bbox

                if fg_pixels >= min_required:
                    success = True
                    if stop_fg is None:
                        stop_fg = fg_pixels
                    stop_reason = "ensure_hit"
                    if not getattr(self, "ensure_fg_select_best", False):
                        break

                crop_bbox = generate_crop_bbox(img)

            rec["ensure_success"] = bool(success)
            rec["ensure_retry"] = int(retry)
            rec["best_fg"] = int(best_fg if best_fg >= 0 else last_fg)
            rec["last_fg"] = int(last_fg)
            rec["stop_fg"] = int(stop_fg if stop_fg is not None else last_fg)
            rec["stop_reason"] = stop_reason if stop_reason is not None else "ensure_max_retry"

            use_best = (not success) or getattr(self, "ensure_fg_select_best", False)
            rec["final_bbox_is_best"] = bool(use_best)
            if use_best:
                crop_bbox = best_bbox
        else:
            rec["stop_reason"] = "not_triggered"
            rec["final_bbox_is_best"] = None  # 没走 ensure 时无意义

        # 最终 crop 的 fg 像素
        seg_tmp = self.crop(seg_full, crop_bbox)
        valid = (seg_tmp != self.ignore_index)
        fg_crop_pixels = int(((seg_tmp > 0) & valid).sum())
        rec["fg_crop_pixels"] = fg_crop_pixels
        rec["fg_cover_ratio"] = float(fg_crop_pixels / fg_full_pixels) if fg_full_pixels > 0 else None

        if len(stats["records"]) < args.record_limit:
            stats["records"].append(rec)
        return crop_bbox

    crop_t.crop_bbox = MethodType(instrumented_crop_bbox, crop_t)

    # -------------------------
    # Run N samples
    # -------------------------
    N = args.num
    err_counter = Counter()
    first_tb = None

    # 注意：records 只保留 record_limit 条，但我们会另外累积 summary 所需的数组
    all_recs = []

    for i in range(N):
        try:
            _ = dataset[i % len(dataset)]  # 触发 pipeline
            # 由于 instrumented_crop_bbox 内部把 rec append 到 stats["records"]（截断），
            # 我们在这里额外同步抓取“最后一条”来做完整统计：不太可靠（多线程 pipeline 会乱）
            # 所以：更稳的方式是 instrumented_crop_bbox 里再 append 一个 all_recs。
        except Exception as e:
            stats["errors"] += 1
            err_counter[type(e).__name__] += 1
            if first_tb is None:
                first_tb = traceback.format_exc()
            continue

    # 这里用 stats["records"] 做示例展示；真正 summary 用下面再“重新跑一遍”收集全量更稳：
    # 但你原脚本也是只用 records 统计（它并没有截断 records），所以这里我们把 records 改成全量保存。
    # ——为了不改变你原脚本行为，我们这里直接用 recs_full: instrumented_crop_bbox 已经只保存 record_limit，
    # 所以我们需要“全量保存”用于 summary：最简单是把 instrumented_crop_bbox 里的 append 改成无条件 append。
    # 下面我直接实现：重新绑定一个不截断版本来收全量（只跑一次更好，但为了让你直接可用，我用“二次跑”）。
    # 如果你嫌慢，把 args.num=2000 这点二次跑也就几秒。

    # 重新跑一次：全量统计（不影响 pipeline），但保证 summary 有 2000 条
    stats_full = {"records": [], "errors": 0}
    def instrumented_crop_bbox_full(self, results: dict) -> tuple:
        bbox = instrumented_crop_bbox(self, results)
        # instrumented_crop_bbox 里 append 了 record_limit；这里我们自己从 rec 重新构建更稳？
        # 简化：直接复制 instrumented_crop_bbox 的逻辑太冗余。
        # 所以我们采取：instrumented_crop_bbox 里最后 append 的那条就是当前 rec（在单进程 dataset[i] 下稳定）。
        # 取 stats["records"] 的最后一条复制出来（注意 record_limit 够大时才可靠，所以这里临时设大）
        return bbox

    # 临时扩大 record_limit 来避免截断影响（只用于 full run）
    old_limit = args.record_limit
    args.record_limit = N  # 全量保存
    stats["records"] = []  # 清空
    crop_t.crop_bbox = MethodType(instrumented_crop_bbox, crop_t)

    err_counter2 = Counter()
    first_tb2 = None
    for i in range(N):
        try:
            _ = dataset[i % len(dataset)]
        except Exception as e:
            stats_full["errors"] += 1
            err_counter2[type(e).__name__] += 1
            if first_tb2 is None:
                first_tb2 = traceback.format_exc()
            continue

    args.record_limit = old_limit  # 还原

    # 全量 recs
    recs = stats["records"]

    # error report
    stats["error_types"] = dict(err_counter2)
    stats["first_error_traceback"] = first_tb2

    fg_crop = [r["fg_crop_pixels"] for r in recs if r.get("fg_crop_pixels") is not None]
    fg_full = [r["fg_full_pixels"] for r in recs if r.get("fg_full_pixels") is not None]
    cover = [r["fg_cover_ratio"] for r in recs if r.get("fg_cover_ratio") is not None]

    fg_init = [r["fg_init"] for r in recs if r.get("fg_init") is not None]
    fg_after_catmax = [r["fg_after_catmax"] for r in recs if r.get("fg_after_catmax") is not None]

    trig = [r for r in recs if r.get("ensure_triggered")]
    trig_succ = [r for r in trig if r.get("ensure_success") is True]
    trig_fail = [r for r in trig if r.get("ensure_success") is False]

    retry = [r["ensure_retry"] for r in trig if r.get("ensure_retry") is not None]
    minreq = [r["min_required"] for r in trig if r.get("min_required") is not None]
    stop_fg = [r["stop_fg"] for r in trig if r.get("stop_fg") is not None]
    best_fg = [r["best_fg"] for r in trig if r.get("best_fg") is not None]
    final_fg = [r["fg_crop_pixels"] for r in trig if r.get("fg_crop_pixels") is not None]

    ratio_active = [r for r in trig if r.get("min_required_was_ratio_active")]

    # bucket stats by fg_full
    buckets = {}
    for r in recs:
        b = _bucket_id(int(r.get("fg_full_pixels") or 0))
        buckets.setdefault(b, []).append(r)

    bucket_summary = {}
    for b, rr in buckets.items():
        bb_fg_crop = [x["fg_crop_pixels"] for x in rr if x.get("fg_crop_pixels") is not None]
        bb_cover = [x["fg_cover_ratio"] for x in rr if x.get("fg_cover_ratio") is not None]
        bb_trig = [x for x in rr if x.get("ensure_triggered")]
        bb_ratio = [x for x in bb_trig if x.get("min_required_was_ratio_active")]

        bucket_summary[b] = {
            "count": len(rr),
            "fg_crop_zero_rate": float(np.mean([1 if (x.get("fg_crop_pixels") == 0) else 0 for x in rr])) if rr else None,
            "ensure_trigger_rate": (len(bb_trig) / len(rr)) if rr else None,
            "ratio_active_rate_given_trigger": (len(bb_ratio) / len(bb_trig)) if bb_trig else None,
            "fg_crop_pixels_quantiles": _quantiles(bb_fg_crop),
            "fg_cover_ratio_quantiles": _quantiles(bb_cover),
        }

    summary = {
        "num_requested": N,
        "num_records": len(recs),
        "errors": stats_full["errors"],
        "error_types": stats["error_types"],
        "first_error_traceback": stats["first_error_traceback"],

        "fg_full_zero_rate": float(np.mean([1 if (r.get("fg_full_pixels") == 0) else 0 for r in recs if r.get("fg_full_pixels") is not None])) if fg_full else None,
        "fg_crop_zero_rate": float(np.mean([1 if (x == 0) else 0 for x in fg_crop])) if fg_crop else None,

        "fg_full_pixels_quantiles": _quantiles(fg_full),
        "fg_crop_pixels_quantiles": _quantiles(fg_crop),
        "fg_cover_ratio_quantiles": _quantiles(cover),

        "fg_init_quantiles": _quantiles(fg_init),
        "fg_after_catmax_quantiles": _quantiles(fg_after_catmax),

        "ensure_trigger_rate": (len(trig) / len(recs)) if len(recs) else None,
        "ensure_success_rate_given_trigger": (len(trig_succ) / len(trig)) if len(trig) else None,
        "ensure_avg_retry_given_trigger": _mean(retry),
        "ensure_retry_quantiles": _quantiles(retry),
        "ensure_min_required_quantiles": _quantiles(minreq),

        # 新增关键诊断
        "ratio_active_rate_given_trigger": (len(ratio_active) / len(trig)) if len(trig) else None,
        "ensure_stop_fg_quantiles": _quantiles(stop_fg),
        "ensure_best_fg_quantiles": _quantiles(best_fg),
        "ensure_final_fg_quantiles": _quantiles(final_fg),

        "bucket_summary_by_fg_full": bucket_summary,

        "counts": {
            "triggered": len(trig),
            "triggered_success": len(trig_succ),
            "triggered_fail": len(trig_fail),
        }
    }

    # 输出 records（截前 record_limit 条）
    out_obj = {"summary": summary, "records": recs[:args.record_limit]}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print("\n======== FG CROP STATS (SUMMARY) ========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["errors"]:
        print("\n[ErrorTypeCounts]", summary["error_types"])
        if summary["first_error_traceback"] is not None:
            print("\n[FirstErrorTraceback]\n", summary["first_error_traceback"])
    print(f"\nSaved: {args.out} (contains summary + first {args.record_limit} records)")


if __name__ == "__main__":
    main()
