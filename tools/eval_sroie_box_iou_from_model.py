import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import torch

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint
from mmseg.apis import init_model
from mmseg.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate T-SROIE with custom box-level IoU diagnostics from a model checkpoint.')
    parser.add_argument('config', help='Inference config file.')
    parser.add_argument('checkpoint', help='Model checkpoint.')
    parser.add_argument('--gt-json', required=True, help='Path to sroie_test_1011.json')
    parser.add_argument('--out-dir', required=True, help='Output directory.')
    parser.add_argument('--device', default='cuda:0', help='Inference device.')
    parser.add_argument('--tampered-category', default='text_temp',
                        help='Tampered category name in JSON.')
    parser.add_argument('--positive-class', type=int, default=1,
                        help='Positive class id in pred_sem_seg. '
                             'For official ASC-Former with 2-class CE, use 1.')
    parser.add_argument('--min-box-area', type=float, default=0.0,
                        help='Filter predicted boxes whose connected-component area is smaller than this.')
    parser.add_argument('--min-box-side', type=float, default=0.0,
                        help='Filter predicted boxes whose min(width, height) is smaller than this.')
    parser.add_argument('--match-iou-thr', type=float, default=0.5,
                        help='IoU threshold used for greedy matched IoU statistics.')
    parser.add_argument('--save-mask', action='store_true',
                        help='Whether to save predicted binary masks.')
    parser.add_argument('--save-box-json', action='store_true',
                        help='Whether to save per-image predicted boxes as JSON.')
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def bbox_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = int(np.ceil(x + w))
    y2 = int(np.ceil(y + h))
    return x1, y1, x2, y2


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a, b):
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


def rasterize_boxes(boxes, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1 = int(np.clip(x1, 0, w))
        x2 = int(np.clip(x2, 0, w))
        y1 = int(np.clip(y1, 0, h))
        y2 = int(np.clip(y2, 0, h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def mask_iou(a, b):
    inter = int(((a == 1) & (b == 1)).sum())
    union = int(((a == 1) | (b == 1)).sum())
    if union == 0:
        return 1.0
    return inter / union


def pairwise_iou_matrix(gt_boxes, pred_boxes):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    mat = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            mat[i, j] = box_iou(g, p)
    return mat


def greedy_match_ious(iou_mat, thr=0.5):
    """Greedy one-to-one matching by descending IoU."""
    if iou_mat.size == 0:
        return []
    pairs = []
    rows, cols = iou_mat.shape
    for i in range(rows):
        for j in range(cols):
            iou = float(iou_mat[i, j])
            if iou >= thr:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_r = set()
    used_c = set()
    matched_ious = []
    matched_pairs = []
    for iou, r, c in pairs:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        matched_ious.append(iou)
        matched_pairs.append((r, c, iou))
    return matched_pairs


def best_ious_over_gt(iou_mat):
    if iou_mat.size == 0:
        return np.zeros((iou_mat.shape[0],), dtype=np.float32)
    return iou_mat.max(axis=1)


def best_ious_over_pred(iou_mat):
    if iou_mat.size == 0:
        return np.zeros((iou_mat.shape[1],), dtype=np.float32)
    return iou_mat.max(axis=0)


def load_sroie_json(json_path, tampered_category):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = data.get('categories', [])
    cat_name_to_id = {c['name']: c['id'] for c in categories}
    if tampered_category not in cat_name_to_id:
        raise KeyError(
            f'Tampered category "{tampered_category}" not found. '
            f'Available categories: {list(cat_name_to_id.keys())}')
    tampered_cat_id = cat_name_to_id[tampered_category]

    images = data.get('images', [])
    anns = data.get('annotations', [])
    anns_by_image = defaultdict(list)
    for ann in anns:
        anns_by_image[ann['image_id']].append(ann)

    gt_by_stem = {}
    image_meta_by_stem = {}
    for img in images:
        stem = osp.splitext(img['file_name'])[0]
        image_meta_by_stem[stem] = dict(
            id=img['id'],
            file_name=img['file_name'],
            width=img['width'],
            height=img['height'],
        )
        gt_boxes = []
        for ann in anns_by_image.get(img['id'], []):
            if ann.get('category_id') != tampered_cat_id:
                continue
            gt_boxes.append(bbox_xywh_to_xyxy(ann['bbox']))
        gt_by_stem[stem] = gt_boxes

    return gt_by_stem, image_meta_by_stem


def mask_to_boxes(binary_mask, min_box_area=0.0, min_box_side=0.0):
    binary_mask = binary_mask.astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    boxes = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_box_area:
            continue
        if min(w, h) < min_box_side:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def pred_sample_to_binary_mask(result, positive_class=1):
    pred = result.pred_sem_seg.data.squeeze().detach().cpu().numpy()
    if pred.ndim != 2:
        raise ValueError(f'Unexpected pred shape: {pred.shape}')
    binary = (pred == positive_class).astype(np.uint8)
    return binary


def build_model_without_checkpoint_meta(cfg, checkpoint_path, device):
    model = init_model(cfg, checkpoint=None, device=device)
    ckpt = load_checkpoint(
        model,
        checkpoint_path,
        map_location='cpu',
        revise_keys=[(r'^module\.', '')])

    model.cfg = cfg

    dataset_meta = None
    if isinstance(ckpt, dict):
        meta = ckpt.get('meta', None)
        if isinstance(meta, dict):
            dataset_meta = meta.get('dataset_meta', None)
            if dataset_meta is None and 'CLASSES' in meta:
                classes = meta['CLASSES']
                palette = meta.get('PALETTE', None)
                dataset_meta = dict(classes=classes, palette=palette)

    if dataset_meta is None:
        dataset_meta = dict(
            classes=('background', 'tampered_text'),
            palette=[[0, 0, 0], [255, 255, 255]])

    model.dataset_meta = dataset_meta
    model.eval()
    return model


def build_test_dataset_from_cfg(cfg):
    if 'test_dataloader' not in cfg or 'dataset' not in cfg.test_dataloader:
        raise KeyError(
            'Config must contain test_dataloader.dataset so that this script '
            'can reuse the exact same pipeline as tools/test.py.')
    dataset_cfg = cfg.test_dataloader.dataset.copy()
    dataset = DATASETS.build(dataset_cfg)
    return dataset


def main():
    args = parse_args()

    ensure_dir(args.out_dir)
    if args.save_mask:
        ensure_dir(osp.join(args.out_dir, 'pred_mask'))
    if args.save_box_json:
        ensure_dir(osp.join(args.out_dir, 'pred_boxes_json'))

    gt_by_stem, image_meta_by_stem = load_sroie_json(args.gt_json, args.tampered_category)

    cfg = Config.fromfile(args.config)
    model = build_model_without_checkpoint_meta(cfg, args.checkpoint, args.device)
    dataset = build_test_dataset_from_cfg(cfg)

    # dataset-level accumulators
    total_gt_boxes = 0
    total_pred_boxes = 0
    total_matched = 0

    sum_best_iou_over_gt = 0.0
    sum_best_iou_over_pred = 0.0
    sum_matched_iou = 0.0

    global_rect_intersection = 0
    global_rect_union = 0

    per_image = {}

    for idx in range(len(dataset)):
        data_batch = pseudo_collate([dataset[idx]])
        with torch.no_grad():
            result = model.test_step(data_batch)[0]

        img_meta = result.metainfo
        img_path = img_meta.get('img_path', None)
        if img_path is None:
            raise KeyError('img_path not found in prediction metainfo.')

        file_name = osp.basename(img_path)
        stem = osp.splitext(file_name)[0]

        if stem not in gt_by_stem:
            raise KeyError(f'Image stem "{stem}" not found in GT JSON.')

        gt_boxes = gt_by_stem[stem]
        image_meta = image_meta_by_stem[stem]
        h, w = image_meta['height'], image_meta['width']

        binary_mask = pred_sample_to_binary_mask(result, positive_class=args.positive_class)
        pred_boxes = mask_to_boxes(
            binary_mask,
            min_box_area=args.min_box_area,
            min_box_side=args.min_box_side)

        if args.save_mask:
            cv2.imwrite(
                osp.join(args.out_dir, 'pred_mask', f'{stem}.png'),
                binary_mask * 255)

        if args.save_box_json:
            with open(osp.join(args.out_dir, 'pred_boxes_json', f'{stem}.json'),
                      'w', encoding='utf-8') as f:
                json.dump(
                    dict(
                        file_name=file_name,
                        gt_boxes=gt_boxes,
                        pred_boxes=pred_boxes,
                    ),
                    f,
                    ensure_ascii=False,
                    indent=2)

        iou_mat = pairwise_iou_matrix(gt_boxes, pred_boxes)
        gt_best = best_ious_over_gt(iou_mat)
        pred_best = best_ious_over_pred(iou_mat)
        matched_pairs = greedy_match_ious(iou_mat, thr=args.match_iou_thr)
        matched_ious = [p[2] for p in matched_pairs]

        gt_rect_mask = rasterize_boxes(gt_boxes, h, w)
        pred_rect_mask = rasterize_boxes(pred_boxes, h, w)
        rect_iou = mask_iou(gt_rect_mask, pred_rect_mask)

        inter = int(((gt_rect_mask == 1) & (pred_rect_mask == 1)).sum())
        union = int(((gt_rect_mask == 1) | (pred_rect_mask == 1)).sum())
        global_rect_intersection += inter
        global_rect_union += union

        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(pred_boxes)
        total_matched += len(matched_ious)

        sum_best_iou_over_gt += float(gt_best.sum()) if len(gt_best) > 0 else 0.0
        sum_best_iou_over_pred += float(pred_best.sum()) if len(pred_best) > 0 else 0.0
        sum_matched_iou += float(sum(matched_ious))

        per_image[stem] = dict(
            file_name=file_name,
            num_gt=len(gt_boxes),
            num_pred=len(pred_boxes),
            num_matched=len(matched_ious),
            mean_best_iou_over_gt=float(gt_best.mean()) if len(gt_best) > 0 else 0.0,
            mean_best_iou_over_pred=float(pred_best.mean()) if len(pred_best) > 0 else 0.0,
            mean_greedy_matched_iou=float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0,
            rect_mask_iou=float(rect_iou),
            matched_pairs=matched_pairs,
        )

        if (idx + 1) % 20 == 0 or (idx + 1) == len(dataset):
            print(f'[{idx + 1}/{len(dataset)}] processed')

    summary = dict(
        num_images=len(dataset),
        num_gt_boxes=total_gt_boxes,
        num_pred_boxes=total_pred_boxes,
        num_matched_boxes_at_thr=total_matched,
        match_iou_thr=args.match_iou_thr,
        mean_best_iou_over_gt=(sum_best_iou_over_gt / total_gt_boxes) if total_gt_boxes > 0 else 0.0,
        mean_best_iou_over_pred=(sum_best_iou_over_pred / total_pred_boxes) if total_pred_boxes > 0 else 0.0,
        mean_greedy_matched_iou=(sum_matched_iou / total_matched) if total_matched > 0 else 0.0,
        global_rect_mask_iou=(global_rect_intersection / global_rect_union) if global_rect_union > 0 else 1.0,
        total_iou=(global_rect_intersection / global_rect_union) if global_rect_union > 0 else 1.0,
        total_iou_percent=100.0 * ((global_rect_intersection / global_rect_union) if global_rect_union > 0 else 1.0),
        note='This is a custom box-level IoU diagnostic, not the Table-10 mask IoU.'
    )

    save_path = osp.join(args.out_dir, 'box_iou_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dict(summary=summary, per_image=per_image), f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved results to: {save_path}')


if __name__ == '__main__':
    main()