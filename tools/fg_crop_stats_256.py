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


def _rate(preds):
    preds = list(preds)
    return float(np.mean(preds)) if len(preds) else None


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

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    if "custom_imports" in cfg and cfg.custom_imports is not None:
        import_modules_from_strings(**cfg.custom_imports)

    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    from mmengine.registry import DefaultScope
    DefaultScope.get_instance("fg_stats", scope_name="mmseg")

    from mmseg.registry import DATASETS
    dataset = DATASETS.build(cfg.train_dataloader.dataset)

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
    # Optional overrides
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

    # 保存最终生效的超参（写入 summary）
    effective = {
        "crop_size": tuple(getattr(crop_t, "crop_size", (None, None))),
        "stride": getattr(crop_t, "stride", None),
        "cat_max_ratio": float(getattr(crop_t, "cat_max_ratio", 1.0)),
        "ensure_fg_prob": float(getattr(crop_t, "ensure_fg_prob", 0.0)),
        "ensure_fg_min_pixels": int(getattr(crop_t, "ensure_fg_min_pixels", 0)),
        "ensure_fg_max_retry": int(getattr(crop_t, "ensure_fg_max_retry", 0)),
        "ensure_fg_min_ratio": float(getattr(crop_t, "ensure_fg_min_ratio", 0.0) or 0.0),
        "ensure_fg_select_best": bool(getattr(crop_t, "ensure_fg_select_best", False)),
    }

    # 全量记录 + 预览记录
    full_records = []
    preview_records = []

    errors = 0
    err_counter = Counter()
    first_tb = None

    # -------------------------
    # Monkeypatch crop_bbox
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

        def count_fg(seg_map, bbox) -> int:
            seg_tmp = self.crop(seg_map, bbox)
            valid = (seg_tmp != self.ignore_index)
            return int(((seg_tmp > 0) & valid).sum())

        crop_bbox = generate_crop_bbox(img)
        seg_full = results.get("gt_seg_map", None)

        rec = {
            "fg_full_pixels": None,
            "fg_crop_pixels": None,
            "fg_cover_ratio": None,

            "fg_init": None,
            "fg_after_catmax": None,

            "ensure_triggered": False,
            "ensure_success": None,
            "ensure_retry": 0,

            # 阈值拆解
            "min_required": 0,
            "min_required_pixels": int(getattr(self, "ensure_fg_min_pixels", 0)),
            "min_required_ratio": None,
            "ratio_active": False,      # ratio_req < min_pixels 导致 min_required 变小
            "clip_active": False,       # min_required 被 fg_full clip 变小（fg_full < raw）

            "stop_fg": None,
            "stop_reason": None,        # seg_full_none / fg_full_zero / not_triggered / ensure_hit / ensure_max_retry

            "best_fg": None,
            "last_fg": None,
            "final_bbox_is_best": None,

            "catmax_applied": bool(getattr(self, "cat_max_ratio", 1.0) < 1.0),
        }

        if seg_full is None:
            rec["stop_reason"] = "seg_full_none"
            full_records.append(rec)
            if len(preview_records) < args.record_limit:
                preview_records.append(rec)
            return crop_bbox

        fg_full_pixels = int(((seg_full > 0) & (seg_full != self.ignore_index)).sum())
        rec["fg_full_pixels"] = fg_full_pixels

        rec["fg_init"] = count_fg(seg_full, crop_bbox)

        # gt 全 0 短路
        if fg_full_pixels == 0:
            rec["stop_reason"] = "fg_full_zero"
            rec["fg_crop_pixels"] = rec["fg_init"]
            rec["fg_cover_ratio"] = None
            full_records.append(rec)
            if len(preview_records) < args.record_limit:
                preview_records.append(rec)
            return crop_bbox

        # cat_max_ratio（你设 1.0 时不生效）
        if self.cat_max_ratio < 1.0:
            for _ in range(10):
                seg_temp = self.crop(seg_full, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and (np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        rec["fg_after_catmax"] = count_fg(seg_full, crop_bbox)

        # ensure_fg：放到最后
        if self.ensure_fg_prob > 0 and np.random.rand() < self.ensure_fg_prob:
            rec["ensure_triggered"] = True

            min_pixels = int(self.ensure_fg_min_pixels)
            ratio = float(getattr(self, "ensure_fg_min_ratio", 0.0) or 0.0)

            raw = max(1, min_pixels)
            ratio_req = None
            if ratio > 0:
                ratio_req = max(1, int(fg_full_pixels * ratio))
                rec["min_required_ratio"] = int(ratio_req)
                if ratio_req < raw:
                    rec["ratio_active"] = True
                raw = min(raw, ratio_req)

            # clip（和你当前训练代码一致的安全版）
            used = raw
            if fg_full_pixels < used:
                rec["clip_active"] = True
            used = min(used, fg_full_pixels)
            used = max(1, int(used))
            rec["min_required"] = int(used)

            success = False
            best_bbox = crop_bbox
            best_fg = -1
            last_fg = 0
            stop_fg = None
            stop_reason = None
            retry = 0

            for retry in range(1, int(self.ensure_fg_max_retry) + 1):
                fg_now = count_fg(seg_full, crop_bbox)
                last_fg = fg_now

                if fg_now > best_fg:
                    best_fg = fg_now
                    best_bbox = crop_bbox

                if fg_now >= used:
                    success = True
                    if stop_fg is None:
                        stop_fg = fg_now
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

        fg_crop_pixels = count_fg(seg_full, crop_bbox)
        rec["fg_crop_pixels"] = int(fg_crop_pixels)
        rec["fg_cover_ratio"] = float(fg_crop_pixels / fg_full_pixels) if fg_full_pixels > 0 else None

        full_records.append(rec)
        if len(preview_records) < args.record_limit:
            preview_records.append(rec)
        return crop_bbox

    crop_t.crop_bbox = MethodType(instrumented_crop_bbox, crop_t)

    # -------------------------
    # Run N samples
    # -------------------------
    N = int(args.num)
    for i in range(N):
        try:
            _ = dataset[i % len(dataset)]  # trigger pipeline
        except Exception as e:
            errors += 1
            err_counter[type(e).__name__] += 1
            if first_tb is None:
                first_tb = traceback.format_exc(limit=80)

    recs = full_records

    # -------------------------
    # Aggregate
    # -------------------------
    def pick(key, cond=None):
        out = []
        for r in recs:
            if cond is not None and not cond(r):
                continue
            v = r.get(key, None)
            if v is None:
                continue
            out.append(v)
        return out

    fg_full = pick("fg_full_pixels")
    fg_crop = pick("fg_crop_pixels")
    cover = pick("fg_cover_ratio")
    fg_init = pick("fg_init")
    fg_after_catmax = pick("fg_after_catmax")

    trig = [r for r in recs if r.get("ensure_triggered")]
    trig_succ = [r for r in trig if r.get("ensure_success") is True]
    trig_fail = [r for r in trig if r.get("ensure_success") is False]

    retry = pick("ensure_retry", cond=lambda r: r.get("ensure_triggered"))
    minreq = pick("min_required", cond=lambda r: r.get("ensure_triggered"))
    stop_fg = pick("stop_fg", cond=lambda r: r.get("ensure_triggered"))
    best_fg = pick("best_fg", cond=lambda r: r.get("ensure_triggered"))
    final_fg_trig = pick("fg_crop_pixels", cond=lambda r: r.get("ensure_triggered"))

    ratio_active_rate = _rate([1 if r.get("ratio_active") else 0 for r in trig]) if len(trig) else None
    clip_active_rate = _rate([1 if r.get("clip_active") else 0 for r in trig]) if len(trig) else None

    # <128 / <256 统计（全体）
    p_lt_128 = _rate([1 if x < 128 else 0 for x in fg_crop]) if len(fg_crop) else None
    p_lt_256 = _rate([1 if x < 256 else 0 for x in fg_crop]) if len(fg_crop) else None

    # 逻辑自检：final 是否 < min_required（触发 ensure 的样本上）
    p_final_lt_minreq = None
    if len(trig):
        bad = 0
        total = 0
        for r in trig:
            if r.get("fg_crop_pixels") is None or r.get("min_required") is None:
                continue
            total += 1
            if int(r["fg_crop_pixels"]) < int(r["min_required"]):
                bad += 1
        p_final_lt_minreq = (bad / total) if total else None

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
        bb_ratio = [x for x in bb_trig if x.get("ratio_active")]
        bb_clip = [x for x in bb_trig if x.get("clip_active")]

        bucket_summary[b] = {
            "count": len(rr),
            "fg_crop_zero_rate": _rate([1 if (x.get("fg_crop_pixels") == 0) else 0 for x in rr]) if rr else None,
            "ensure_trigger_rate": (len(bb_trig) / len(rr)) if rr else None,
            "ratio_active_rate_given_trigger": (len(bb_ratio) / len(bb_trig)) if bb_trig else None,
            "clip_active_rate_given_trigger": (len(bb_clip) / len(bb_trig)) if bb_trig else None,
            "fg_crop_pixels_quantiles": _quantiles(bb_fg_crop),
            "fg_cover_ratio_quantiles": _quantiles(bb_cover),
        }

    summary = {
        "effective_params": effective,

        "num_requested": N,
        "num_records": len(recs),

        "errors": int(errors),
        "error_types": dict(err_counter),
        "first_error_traceback": first_tb,

        "fg_full_zero_rate": _rate([1 if x == 0 else 0 for x in fg_full]) if fg_full else None,
        "fg_crop_zero_rate": _rate([1 if x == 0 else 0 for x in fg_crop]) if fg_crop else None,

        "p_fg_crop_lt_128": p_lt_128,
        "p_fg_crop_lt_256": p_lt_256,
        "p_fg_crop_lt_min_required_given_trigger": p_final_lt_minreq,

        "fg_full_pixels_quantiles": _quantiles(fg_full),
        "fg_init_quantiles": _quantiles(fg_init),
        "fg_after_catmax_quantiles": _quantiles(fg_after_catmax),
        "fg_crop_pixels_quantiles": _quantiles(fg_crop),
        "fg_cover_ratio_quantiles": _quantiles(cover),

        "ensure_trigger_rate": (len(trig) / len(recs)) if len(recs) else None,
        "ensure_success_rate_given_trigger": (len(trig_succ) / len(trig)) if len(trig) else None,
        "ensure_avg_retry_given_trigger": _mean(retry),
        "ensure_retry_quantiles": _quantiles(retry),
        "ensure_min_required_quantiles": _quantiles(minreq),

        "ratio_active_rate_given_trigger": ratio_active_rate,
        "clip_active_rate_given_trigger": clip_active_rate,
        "ensure_stop_fg_quantiles": _quantiles(stop_fg),
        "ensure_best_fg_quantiles": _quantiles(best_fg),
        "ensure_final_fg_quantiles": _quantiles(final_fg_trig),

        "bucket_summary_by_fg_full": bucket_summary,

        "counts": {
            "triggered": len(trig),
            "triggered_success": len(trig_succ),
            "triggered_fail": len(trig_fail),
        },
    }

    out_obj = {"summary": summary, "records": preview_records}
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
