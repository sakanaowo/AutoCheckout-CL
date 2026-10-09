"""Class-incremental detection metrics (V2, plan section 6.5).

Two protocols, both computed from a single ``pred_<split>.npz`` per stage plus the full
split annotation file (``val_full.json`` / ``test_full.json``, see docs/data_preprocessing/formats.md):

- **M1** (paper protocol): mAP@C/P/A, reproduced from the original PDP code
  (``pdp/main.py`` builds a separate COCO ground-truth file per task group and runs
  ``pycocotools.cocoeval.COCOeval`` on it with default ``catIds``/``imgIds``, i.e. all
  categories and all images of that file). Restricting ``COCOeval.params.catIds`` to a
  group's labels and ``params.imgIds`` to the images that contain an object of that group,
  on the *same* full ground-truth file, is equivalent: COCOeval only visits the requested
  categories and images, so results for other classes/images never affect the group's AP.
- **M2** (store protocol): AP over the whole split, with predictions that land on a
  not-yet-learned SKU (IoU >= 0.5 with its GT box) removed first, because the store has
  not started selling that SKU at stage t.

Label space: everywhere here ``category_id`` (ground truth) and ``label`` (predictions)
are both *model labels* (0..num_slots-1) -- ``tasks/<name>/*_full.json`` stores the model
label directly in ``category_id`` (docs/data_preprocessing/formats.md section 4), so no RPC-id mapping is
needed for these metrics.
"""

from __future__ import annotations

import contextlib
import copy
import io
import re
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from autocheckout.predictions import Predictions
from autocheckout.taskcfg import TaskConfig

_STAGE_DIR_RE = re.compile(r"^task_(\d+)$")


@contextlib.contextmanager
def _quiet():
    """pycocotools prints a handful of lines per call; silence them, we call it a lot."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_coco(ann: dict[str, Any]) -> COCO:
    """Build a ``pycocotools`` index over an already-parsed COCO annotation dict."""
    coco = COCO()
    coco.dataset = ann
    with _quiet():
        coco.createIndex()
    return coco


def images_with_labels(coco: COCO, labels: Iterable[int]) -> list[int]:
    """Images containing >=1 GT annotation whose category is in ``labels`` (union, not
    intersection -- ``COCO.getImgIds(catIds=...)`` intersects, which is not what M1 wants).
    """
    label_set = set(labels)
    image_ids = {
        ann["image_id"] for ann in coco.dataset.get("annotations", []) if ann["category_id"] in label_set
    }
    return sorted(image_ids)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``[x1, y1, x2, y2]`` boxes: shape (N, M)."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    boxes1 = boxes1.astype(np.float64)
    boxes2 = boxes2.astype(np.float64)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _load_dt(coco_gt: COCO, results: list[dict[str, Any]]) -> COCO:
    """Like ``coco_gt.loadRes(results)``, but tolerates an empty result list (which
    ``loadRes`` cannot: it indexes ``results[0]`` unconditionally).
    """
    if not results:
        res = COCO()
        res.dataset = {
            "images": coco_gt.dataset.get("images", []),
            "categories": coco_gt.dataset.get("categories", []),
            "annotations": [],
        }
        with _quiet():
            res.createIndex()
        return res
    with _quiet():
        return coco_gt.loadRes(results)


_NAN_STATS = {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}


class EvaluatedPairs:
    """``COCOeval.evaluate()`` run once; AP of any subset of categories and images is then
    accumulated from the stored per-(category, image) results.

    Exact: pycocotools evaluates every (category, image) pair independently (matching, maxDets
    truncation), so a subset's AP equals running COCOeval on that subset alone. Only the area range
    'all' is evaluated (4x fewer pairs), which is the only range reported. ``COCOeval.accumulate(p)``
    cannot be used for subsets: it indexes the stored results with positions in ``p``.
    """

    def __init__(self, coco_gt: COCO, coco_dt: COCO, cat_ids: Sequence[int], img_ids: Sequence[int]) -> None:
        self.cat_ids, self.img_ids = list(cat_ids), list(img_ids)
        self.coco_eval = None
        if not self.cat_ids or not self.img_ids:
            return
        with _quiet():
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.areaRng = coco_eval.params.areaRng[:1]
            coco_eval.params.areaRngLbl = coco_eval.params.areaRngLbl[:1]
            coco_eval.evaluate()
        self.coco_eval = coco_eval
        # evaluate() stores its own (sorted) id lists; index the results by them
        self._cat_pos = {c: i for i, c in enumerate(coco_eval._paramsEval.catIds)}
        self._img_pos = {m: i for i, m in enumerate(coco_eval._paramsEval.imgIds)}

    def stats(self, cat_ids: Iterable[int], img_ids: Iterable[int]) -> dict[str, float]:
        """AP (0.50:0.95), AP50, AP75 on a subset; NaN when there is nothing to evaluate."""
        cat_ids = [c for c in cat_ids if c in getattr(self, "_cat_pos", {})]
        img_ids = [m for m in img_ids if m in getattr(self, "_img_pos", {})]
        if self.coco_eval is None or not cat_ids or not img_ids:
            return {**_NAN_STATS, "n_images": len(img_ids)}
        full = self.coco_eval
        n_img = len(full._paramsEval.imgIds)
        subset = copy.copy(full)
        subset.evalImgs = [full.evalImgs[self._cat_pos[c] * n_img + self._img_pos[m]]
                           for c in cat_ids for m in img_ids]
        subset._paramsEval = copy.deepcopy(full._paramsEval)
        subset._paramsEval.catIds, subset._paramsEval.imgIds = cat_ids, img_ids
        subset.params = copy.deepcopy(full.params)
        subset.params.catIds, subset.params.imgIds = cat_ids, img_ids
        with _quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # empty small/medium/large ranges
            subset.accumulate()
            subset.summarize()
        stats = subset.stats
        return {"AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]), "n_images": len(img_ids)}


def ap_stats(
    coco_gt: COCO, coco_dt: COCO, cat_ids: Sequence[int], img_ids: Sequence[int]
) -> dict[str, float]:
    """AP (0.50:0.95), AP50, AP75 restricted to ``cat_ids``/``img_ids``. NaN when there is
    nothing to evaluate (empty group), instead of relying on pycocotools' edge-case behaviour.
    """
    return EvaluatedPairs(coco_gt, coco_dt, cat_ids, img_ids).stats(cat_ids, img_ids)


def group_ap(coco_gt: COCO, preds: Predictions, labels: Sequence[int]) -> dict[str, float]:
    """AP stats for one label group, on the images that contain >=1 GT object of that group
    (M1's per-group protocol)."""
    img_ids = images_with_labels(coco_gt, labels)
    coco_dt = _load_dt(coco_gt, preds.to_coco_results())
    return ap_stats(coco_gt, coco_dt, labels, img_ids)


def m1_stage(coco_gt: COCO, preds: Predictions, cfg: TaskConfig, stage: int) -> dict[str, Any]:
    """mAP@C/P/A and the per-task-group AP50 row of the accuracy matrix, at ``stage``."""
    coco_dt = _load_dt(coco_gt, preds.to_coco_results())
    all_labels = list(range(cfg.seen_classes(stage)))
    evaluated = EvaluatedPairs(coco_gt, coco_dt, all_labels, images_with_labels(coco_gt, all_labels))

    def stats(labels: Sequence[int]) -> dict[str, float]:
        return evaluated.stats(list(labels), images_with_labels(coco_gt, labels))

    map_p = stats(range(cfg.seen_classes(stage - 1))) if stage >= 2 else None
    matrix_row = {g: stats(cfg.task(g).labels)["AP50"] for g in range(1, stage + 1)}
    return {"mAP_C": stats(cfg.task(stage).labels), "mAP_P": map_p, "mAP_A": stats(all_labels),
            "matrix_row": matrix_row}


def m2_stage(coco_gt: COCO, preds: Predictions, seen_classes: int) -> dict[str, Any]:
    """Store protocol at ``stage``: AP over the whole split, with predictions overlapping
    not-yet-learned GT removed first. ``overall`` uses every image of the split; ``by_level``
    breaks it down by ``images[].level``."""
    learned = list(range(seen_classes))
    filtered = filter_predictions_overlapping_unlearned(preds, coco_gt, seen_classes)
    coco_dt = _load_dt(coco_gt, filtered.to_coco_results())
    all_images = coco_gt.getImgIds()
    evaluated = EvaluatedPairs(coco_gt, coco_dt, learned, all_images)

    overall = evaluated.stats(learned, all_images)
    by_level: dict[str, dict[str, float]] = {}
    for level in ("easy", "medium", "hard"):
        level_images = [img_id for img_id in all_images if coco_gt.imgs[img_id].get("level") == level]
        by_level[level] = evaluated.stats(learned, level_images)
    return {"overall": overall, "by_level": by_level}


def _xywh_to_xyxy(bbox: Sequence[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def filter_predictions_overlapping_unlearned(
    preds: Predictions, coco_gt: COCO, seen_classes: int, iou_threshold: float = 0.5
) -> Predictions:
    """Drop prediction rows whose box overlaps (IoU >= ``iou_threshold``) a GT box of a class
    that is not learned yet: at stage t the store has not started selling those SKUs, so a
    detection on one of them is not a real false positive (plan section 6.5, M2)."""
    if len(preds) == 0:
        return preds
    order = np.argsort(preds.image_id, kind="stable")
    sorted_ids = preds.image_id[order]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    groups = np.split(order, boundaries)

    keep = np.ones(len(preds), dtype=bool)
    for idxs in groups:
        image_id = int(preds.image_id[idxs[0]])
        unlearned = [ann for ann in coco_gt.imgToAnns.get(image_id, []) if ann["category_id"] >= seen_classes]
        if not unlearned:
            continue
        gt_boxes = np.array([_xywh_to_xyxy(ann["bbox"]) for ann in unlearned], dtype=np.float32)
        overlap = box_iou(preds.boxes[idxs], gt_boxes).max(axis=1) >= iou_threshold
        keep[idxs[overlap]] = False
    return preds.subset(keep)


def matrix_forgetting(matrix: dict[int, dict[int, float]]) -> dict[str, Any]:
    """Forgetting per group g<T = max over t<T of acc[t][g] minus acc[T][g], averaged over
    g<T; plus the average of the last available row. ``T`` is the last stage present in
    ``matrix``. Returns ``None`` averages when there is nothing to compare (e.g. a
    joint-training run that only has its single final stage)."""
    if not matrix:
        return {"per_group": {}, "average": None, "avg_last_row": None}
    last_stage = max(matrix)
    last_row = matrix.get(last_stage, {})

    per_group: dict[int, float] = {}
    for g in range(1, last_stage):
        earlier = [matrix[t][g] for t in matrix if t < last_stage and g in matrix[t]]
        if earlier and g in last_row:
            per_group[g] = max(earlier) - last_row[g]

    average = float(np.mean(list(per_group.values()))) if per_group else None
    avg_last_row = float(np.mean(list(last_row.values()))) if last_row else None
    return {"per_group": per_group, "average": average, "avg_last_row": avg_last_row}


def discover_stage_predictions(run_dir: str | Path, split: str) -> dict[int, Path]:
    """Map stage (task id) -> ``pred_<split>.npz`` path, for every ``task_<t>/`` subdirectory
    that has one. Stages need not be contiguous: a joint-training upper bound run (E0) may
    only have its last stage."""
    stages: dict[int, Path] = {}
    run_dir = Path(run_dir)
    for child in sorted(run_dir.iterdir()):
        match = _STAGE_DIR_RE.match(child.name)
        if not match:
            continue
        pred_path = child / f"pred_{split}.npz"
        if pred_path.is_file():
            stages[int(match.group(1))] = pred_path
    return stages


def check_ann_md5(meta: dict[str, Any], expected_md5: str, *, context: str) -> None:
    """Refuse to evaluate a prediction file against the wrong annotation file."""
    actual = meta.get("ann_md5")
    if actual is not None and actual != expected_md5:
        raise ValueError(
            f"{context}: prediction meta.ann_md5={actual!r} does not match the md5 of the "
            f"given annotation file ({expected_md5!r}); predictions were produced from a "
            "different annotation file."
        )


def check_stage_meta(meta: dict[str, Any], cfg: TaskConfig, stage: int) -> int:
    """Cross-check a prediction file's meta against the task config for this stage, and
    return the number of learned classes (``seen_classes``) to use."""
    expected_seen = cfg.seen_classes(stage)
    task_id = meta.get("task_id")
    if task_id is not None and int(task_id) != stage:
        raise ValueError(f"task_{stage}: meta.task_id={task_id} does not match the directory name")
    actual_seen = meta.get("seen_classes", expected_seen)
    if int(actual_seen) != expected_seen:
        raise ValueError(
            f"task_{stage}: meta.seen_classes={actual_seen} but task config {cfg.name!r} says {expected_seen}"
        )
    return expected_seen
