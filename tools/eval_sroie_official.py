"""Evaluate T-SROIE predictions with box-level Precision / Recall / F1.

This follows the paper's synthetic-dataset protocol:
  1) convert tampered text boxes to binary masks for training;
  2) at inference, binarize predicted masks;
  3) extract connected components;
  4) convert each component to its maximum circumscribed rectangle;
  5) optionally filter tiny boxes;
  6) evaluate one-to-one matches with IoU >= 0.5.

Expected prediction directory:
  pred_dir/
    X51005711444.png
    X51006556865.png
    ...

Ground truth:
  sroie_test_1011.json (COCO-style, categories include text / text_temp)
"""

import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 (exclusive)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate T-SROIE with box-level Precision / Recall / F1')
    parser.add_argument('--pred_dir', required=True, help='Directory of predicted masks (.png).')
    parser.add_argument('--gt_json', required=True, help='Path to sroie_test_1011.json.')
    parser.add_argument('--tampered_category', default='text_temp', help='GT category name treated as tampered.')
    parser.add_argument('--threshold', type=float, default=127.0,
                        help='Binarization threshold for predicted masks. If image range is [0,1], use 0.5.')
    parser.add_argument('--iou_thr', type=float, default=0.5, help='IoU threshold for one-to-one matching.')
    parser.add_argument('--min_box_area', type=float, default=0.0,
                        help='Filter predicted boxes with area < min_box_area.')
    parser.add_argument('--min_box_side', type=float, default=0.0,
                        help='Filter predicted boxes with min(w, h) < min_box_side.')
    parser.add_argument('--save_json', default=None, help='Optional path to save detailed results as json.')
    return parser.parse_args()


def bbox_xywh_to_xyxy(bbox) -> Box:
    x, y, w, h = bbox
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = int(np.ceil(x + w))
    y2 = int(np.ceil(y + h))
    return x1, y1, x2, y2


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / max(union, 1e-12)


def load_gt_boxes(gt_json: str, tampered_category: str) -> Dict[str, List[Box]]:
    with open(gt_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cat_name_to_id = {c['name']: c['id'] for c in data['categories']}
    if tampered_category not in cat_name_to_id:
        raise KeyError(f'Category {tampered_category!r} not found. Available: {list(cat_name_to_id.keys())}')
    tamp_id = cat_name_to_id[tampered_category]

    imgid_to_name = {img['id']: osp.splitext(img['file_name'])[0] for img in data['images']}
    gt_boxes = defaultdict(list)
    for ann in data['annotations']:
        if ann.get('category_id') != tamp_id:
            continue
        name = imgid_to_name[ann['image_id']]
        gt_boxes[name].append(bbox_xywh_to_xyxy(ann['bbox']))
    return dict(gt_boxes)


def mask_to_boxes(mask: np.ndarray,
                  threshold: float = 127.0,
                  min_box_area: float = 0.0,
                  min_box_side: float = 0.0) -> List[Box]:
    mask = mask.astype(np.float32)
    thr = threshold
    if mask.max() <= 1.0:
        thr = 0.5 if threshold > 1.0 else threshold
    binary = (mask > thr).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: List[Box] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_box_area:
            continue
        if min(w, h) < min_box_side:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def greedy_match(pred_boxes: List[Box], gt_boxes: List[Box], iou_thr: float) -> Tuple[int, int, int]:
    if not pred_boxes and not gt_boxes:
        return 0, 0, 0
    candidates = []
    for pi, pbox in enumerate(pred_boxes):
        for gi, gbox in enumerate(gt_boxes):
            iou = box_iou(pbox, gbox)
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True)

    matched_pred = set()
    matched_gt = set()
    tp = 0
    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def main():
    args = parse_args()
    gt_map = load_gt_boxes(args.gt_json, args.tampered_category)

    pred_files = [f for f in os.listdir(args.pred_dir) if f.lower().endswith('.png')]
    pred_files.sort()

    total_tp = total_fp = total_fn = 0
    per_image = {}

    all_names = set(gt_map.keys()) | {osp.splitext(f)[0] for f in pred_files}
    for name in sorted(all_names):
        pred_path = osp.join(args.pred_dir, f'{name}.png')
        if osp.exists(pred_path):
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if pred is None:
                raise FileNotFoundError(f'Failed to read predicted mask: {pred_path}')
            pred_boxes = mask_to_boxes(
                pred,
                threshold=args.threshold,
                min_box_area=args.min_box_area,
                min_box_side=args.min_box_side)
        else:
            pred_boxes = []

        gt_boxes = gt_map.get(name, [])
        tp, fp, fn = greedy_match(pred_boxes, gt_boxes, args.iou_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_image[name] = dict(
            num_pred=len(pred_boxes),
            num_gt=len(gt_boxes),
            tp=tp,
            fp=fp,
            fn=fn,
        )

    precision = total_tp / (total_tp + total_fp + 1e-12)
    recall = total_tp / (total_tp + total_fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    summary = dict(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        pred_dir=args.pred_dir,
        gt_json=args.gt_json,
        threshold=args.threshold,
        iou_thr=args.iou_thr,
        min_box_area=args.min_box_area,
        min_box_side=args.min_box_side,
    )

    print(json.dumps(summary, indent=2))

    if args.save_json is not None:
        payload = dict(summary=summary, per_image=per_image)
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'Saved detailed results to: {args.save_json}')


if __name__ == '__main__':
    main()