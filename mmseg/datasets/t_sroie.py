import json
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class TSROIEDataset(BaseSegDataset):
    """T-SROIE dataset adapter.

    The original annotations are stored in a COCO-style JSON file with two
    categories:
      - text: authentic text
      - text_temp: tampered text

    This dataset keeps the original image path and attaches all annotations to
    ``instances``. A custom loading transform then rasterizes the tampered
    polygons / boxes into a binary mask for segmentation training.
    """

    METAINFO = dict(
        classes=('background', 'tampered_text'),
        palette=[[0, 0, 0], [255, 255, 255]],
    )

    def __init__(self,
                 ann_file: str,
                 img_suffix: str = '.jpg',
                 tampered_category: str = 'text_temp',
                 use_bbox_fallback: bool = True,
                 **kwargs) -> None:
        self.tampered_category = tampered_category
        self.use_bbox_fallback = use_bbox_fallback
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix='.png',  # unused, kept for BaseSegDataset API
            reduce_zero_label=False,
            **kwargs)

    def _resolve_path(self, path: str) -> str:
        if path is None or path == '':
            return path
        # mmengine/BaseDataset may already have joined data_root.
        # If the path already exists, keep it as is.
        if osp.isabs(path) or osp.exists(path):
            return path
        if self.data_root is None:
            return path
        norm_root = osp.normpath(self.data_root)
        norm_path = osp.normpath(path)
        if norm_path == norm_root or norm_path.startswith(norm_root + osp.sep):
            return path
        return osp.join(self.data_root, path)

    def load_data_list(self) -> List[Dict[str, Any]]:
        # self.ann_file may already be expanded by BaseDataset
        ann_path = self._resolve_path(self.ann_file)
        with open(ann_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        categories = raw.get('categories', [])
        cat_name_to_id = {c['name']: c['id'] for c in categories}
        if self.tampered_category not in cat_name_to_id:
            raise KeyError(
                f'Tampered category "{self.tampered_category}" not found in '
                f'{ann_path}. Available categories: {list(cat_name_to_id.keys())}')
        tampered_cat_id = cat_name_to_id[self.tampered_category]

        anns_by_image = defaultdict(list)
        for ann in raw.get('annotations', []):
            anns_by_image[ann['image_id']].append(ann)

        img_root = self.data_prefix.get('img_path', '') if isinstance(self.data_prefix, dict) else ''
        img_root = self._resolve_path(img_root) if img_root else (self.data_root or '')

        data_list: List[Dict[str, Any]] = []
        for img_info in raw.get('images', []):
            img_path = osp.join(img_root, img_info['file_name']) if img_root else img_info['file_name']
            instances: List[Dict[str, Any]] = []
            for ann in anns_by_image.get(img_info['id'], []):
                seg = ann.get('segmentation', [])
                if isinstance(seg, list) and seg and isinstance(seg[0], (int, float)):
                    seg = [seg]
                instances.append(dict(
                    id=ann.get('id', -1),
                    category_id=ann.get('category_id', -1),
                    is_tampered=(ann.get('category_id', -1) == tampered_cat_id),
                    segmentation=seg,
                    bbox=ann.get('bbox', None),
                    area=ann.get('area', None),
                    iscrowd=ann.get('iscrowd', 0),
                ))

            data_info = dict(
                img_path=img_path,
                img_id=img_info['id'],
                file_name=img_info['file_name'],
                height=img_info['height'],
                width=img_info['width'],
                ori_shape=(img_info['height'], img_info['width']),
                img_shape=(img_info['height'], img_info['width']),
                reduce_zero_label=False,
                seg_fields=[],
                instances=instances,
                use_bbox_fallback=self.use_bbox_fallback,
            )
            data_list.append(data_info)

        return data_list


@TRANSFORMS.register_module()
class LoadSROIEAnnotations(BaseTransform):
    """Rasterize tampered T-SROIE annotations into a binary mask.

    Expected input keys:
      - img
      - instances

    Added keys:
      - gt_seg_map
      - seg_fields
    """

    def __init__(self,
                 binary: bool = True,
                 ignore_index: int = 255,
                 use_bbox_fallback: Optional[bool] = None) -> None:
        self.binary = binary
        self.ignore_index = ignore_index
        self.use_bbox_fallback = use_bbox_fallback

    @staticmethod
    def _clip_bbox(x: float, y: float, w: float, h: float, img_w: int, img_h: int):
        x1 = max(int(np.floor(x)), 0)
        y1 = max(int(np.floor(y)), 0)
        x2 = min(int(np.ceil(x + w)), img_w)
        y2 = min(int(np.ceil(y + h)), img_h)
        return x1, y1, x2, y2

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'img' not in results:
            raise KeyError('LoadSROIEAnnotations requires "img" to be loaded first.')

        img_h, img_w = results['img'].shape[:2]
        gt = np.zeros((img_h, img_w), dtype=np.uint8)
        use_bbox_fallback = results.get('use_bbox_fallback', True) if self.use_bbox_fallback is None else self.use_bbox_fallback

        for inst in results.get('instances', []):
            if not inst.get('is_tampered', False):
                continue

            drawn = False
            polygons = inst.get('segmentation', [])
            for poly in polygons:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] < 3:
                    continue
                pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(gt, [pts], 1)
                drawn = True

            if (not drawn) and use_bbox_fallback and inst.get('bbox', None) is not None:
                x, y, w, h = inst['bbox']
                x1, y1, x2, y2 = self._clip_bbox(x, y, w, h, img_w, img_h)
                if x2 > x1 and y2 > y1:
                    gt[y1:y2, x1:x2] = 1

        if not self.binary:
            gt = gt.astype(np.int64)

        results['gt_seg_map'] = gt
        results.setdefault('seg_fields', [])
        if 'gt_seg_map' not in results['seg_fields']:
            results['seg_fields'].append('gt_seg_map')
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(binary={self.binary}, '
                f'ignore_index={self.ignore_index}, '
                f'use_bbox_fallback={self.use_bbox_fallback})')

@TRANSFORMS.register_module()
class SyncOriShapeWithImgShape(BaseTransform):
    """Force ori_shape to be the current resized image shape.

    This is useful for online validation/testing when GT masks are evaluated
    at the resized resolution (e.g. after Resize in pipeline), while the model
    postprocess would otherwise restore predictions back to the original image
    size, causing shape mismatch in pixel-level evaluators.
    """

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'img' not in results:
            raise KeyError('SyncOriShapeWithImgShape requires "img" in results.')
        h, w = results['img'].shape[:2]
        results['ori_shape'] = (h, w)
        results['img_shape'] = (h, w)
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'