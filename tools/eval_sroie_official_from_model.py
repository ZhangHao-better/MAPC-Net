import argparse
import importlib.util
import json
import os
import os.path as osp
import sys
import zipfile
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
        description='Run official T-SROIE/Tampered-IC13 style evaluation from a model checkpoint.')
    parser.add_argument('config', help='Inference/export config file.')
    parser.add_argument('checkpoint', help='Model checkpoint.')
    parser.add_argument('--gt-json', required=True, help='Path to sroie_test_1011.json')
    parser.add_argument('--official-script', required=True,
                        help='Path to official script.py. '
                             'Its helper file rrc_evaluation_funcs_1_1.py should be in the same directory.')
    parser.add_argument('--out-dir', required=True, help='Output directory.')
    parser.add_argument('--device', default='cuda:0', help='Inference device.')
    parser.add_argument('--tampered-category', default='text_temp', help='Tampered category name in JSON.')
    parser.add_argument('--min-box-area', type=float, default=0.0,
                        help='Filter predicted boxes whose connected-component area is smaller than this.')
    parser.add_argument('--min-box-side', type=float, default=0.0,
                        help='Filter predicted boxes whose min(width, height) is smaller than this.')
    parser.add_argument('--save-mask', action='store_true', help='Whether to save binary masks.')
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


def zip_dir(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for file in sorted(files):
                full_path = osp.join(root, file)
                rel_path = osp.relpath(full_path, src_dir)
                zf.write(full_path, rel_path)


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

    return data, images, anns_by_image, tampered_cat_id


def build_gt_txts(images, anns_by_image, tampered_cat_id, gt_txt_dir):
    ensure_dir(gt_txt_dir)
    for img in images:
        img_id = img['id']
        txt_name = f'img_{img_id}.txt'
        txt_path = osp.join(gt_txt_dir, txt_name)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for ann in anns_by_image.get(img_id, []):
                if ann.get('category_id') != tampered_cat_id:
                    continue
                x1, y1, x2, y2 = bbox_xywh_to_xyxy(ann['bbox'])
                # official script with LTRB=True expects:
                # left,top,right,bottom,transcription
                f.write(f'{x1},{y1},{x2},{y2},text\n')


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


def pred_sample_to_binary_mask(result):
    pred = result.pred_sem_seg.data.squeeze().detach().cpu().numpy()
    if pred.ndim != 2:
        raise ValueError(f'Unexpected pred shape: {pred.shape}')
    # official ASC-Former uses 2-class CE, class 1 is tampered
    binary = (pred == 1).astype(np.uint8)
    return binary


def load_official_script(script_path):
    script_dir = osp.dirname(osp.abspath(script_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec = importlib.util.spec_from_file_location('official_eval_script', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    gt_txt_dir = osp.join(args.out_dir, 'gt_txt')
    det_txt_dir = osp.join(args.out_dir, 'det_txt')
    pred_mask_dir = osp.join(args.out_dir, 'pred_mask')
    ensure_dir(gt_txt_dir)
    ensure_dir(det_txt_dir)
    if args.save_mask:
        ensure_dir(pred_mask_dir)

    data, images, anns_by_image, tampered_cat_id = load_sroie_json(
        args.gt_json, args.tampered_category)

    # Build GT txt files
    build_gt_txts(images, anns_by_image, tampered_cat_id, gt_txt_dir)

    # Init model
    cfg = Config.fromfile(args.config)
    model = build_model_without_checkpoint_meta(cfg, args.checkpoint, args.device)

    # Build dataset using the same pipeline as tools/test.py
    dataset = build_test_dataset_from_cfg(cfg)

    # Inference by iterating dataset items, not inference_model(...)
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

        # Map file stem -> image_id from gt json
        matched = [img for img in images if osp.splitext(img['file_name'])[0] == stem]
        if len(matched) != 1:
            raise ValueError(f'Cannot uniquely map predicted file "{stem}" to image_id in GT json.')
        img_id = matched[0]['id']

        binary_mask = pred_sample_to_binary_mask(result)

        if args.save_mask:
            cv2.imwrite(osp.join(pred_mask_dir, f'{stem}.png'), binary_mask * 255)

        boxes = mask_to_boxes(
            binary_mask,
            min_box_area=args.min_box_area,
            min_box_side=args.min_box_side)

        det_txt_path = osp.join(det_txt_dir, f'img_{img_id}.txt')
        with open(det_txt_path, 'w', encoding='utf-8') as f:
            for x1, y1, x2, y2 in boxes:
                # official script with LTRB=True and CONFIDENCES=True
                f.write(f'{x1},{y1},{x2},{y2},1.0\n')

        if (idx + 1) % 20 == 0 or (idx + 1) == len(dataset):
            print(f'[{idx + 1}/{len(dataset)}] processed')

    # Zip GT and detections for official script
    gt_zip = osp.join(args.out_dir, 'gt.zip')
    det_zip = osp.join(args.out_dir, 'det.zip')
    zip_dir(gt_txt_dir, gt_zip)
    zip_dir(det_txt_dir, det_zip)

    # Official evaluation
    official = load_official_script(args.official_script)
    params = official.default_evaluation_params()
    params['GT_SAMPLE_NAME_2_ID'] = r'img_([0-9]+)\.txt'
    params['DET_SAMPLE_NAME_2_ID'] = r'img_([0-9]+)\.txt'
    params['LTRB'] = True
    params['CONFIDENCES'] = True

    official.validate_data(gt_zip, det_zip, params)
    results = official.evaluate_method(gt_zip, det_zip, params)

    summary = {
        'method': results.get('method', {}),
    }
    if 'samples' in results:
        summary['samples'] = results['samples']

    save_json = osp.join(args.out_dir, 'official_eval_results.json')
    with open(save_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary.get('method', {}), ensure_ascii=False, indent=2))
    print(f'Saved results to: {save_json}')
    print(f'GT zip: {gt_zip}')
    print(f'DET zip: {det_zip}')


if __name__ == '__main__':
    main()