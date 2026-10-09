"""Product-counting metrics (V3, plan section 6.5), matching the RPC leaderboard tool
(``rpctool``, https://github.com/RPC-Dataset/RPC-Leaderboard, ``evaluate_v1/evaluate.py``).

``rpctool`` is not installed here (it depends on ``boxx``, which we were told not to
install); the score formulas below are a direct transcription of
``rpctool/evaluate_v1/evaluate.py``'s ``score1``/``score2``/``score4``/``score5``, cross
checked in ``tests/test_counting.py`` against a second transcription kept in the test file.

Per image and class, the predicted count is the number of ``top1_per_query()`` rows with
score >= threshold (one label per physical query/object, per docs/data_preprocessing/formats.md section 5);
the GT count comes from the annotation. Only learned classes (label < seen_classes) are
counted on both sides -- unlearned GT objects are ignored entirely, and mid-run cAcc is
therefore always computed on the SKUs learned so far, never on the full 200.

Deviation from rpctool: rpctool always evaluates all K=200 RPC classes, each guaranteed to
have GT somewhere in the (fixed, full) split it runs on, so ``score4``/``score5``'s
per-class average over K never divides by zero. Here K is the number of *learned* classes,
which can be small (e.g. stage 1 of a 5-task config) and is not guaranteed to have every
class present with a positive count in every image subset we evaluate (e.g. a single
level). Classes with zero total GT in the evaluated subset are therefore excluded from the
``mCCD``/``mCIoU`` class averages (averaged over the classes that do have GT, ``K_eff``)
instead of producing a division by zero; ``cAcc`` and ``ACD`` are unaffected by this since
they do not divide per class. This choice, and both ``K`` and ``K_eff``, are recorded in
every score dict so a reader can see when it applies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pycocotools.coco import COCO

from autocheckout.cl_metrics import box_iou
from autocheckout.predictions import Predictions

#: Deterministic grid, plan section 6.5: 0.05..0.95 step 0.01 (91 points).
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 + 0.01 * i, 2) for i in range(91))

TIE_BREAK_NOTE = "ties broken by the smallest threshold that reaches the maximum cAcc"

ZERO_GT_NOTE = (
    "classes with zero GT in the evaluated image subset are excluded from the mCCD/mCIoU "
    "class averages (divided by K_eff, not K); cAcc and ACD are unaffected"
)


def dedup_detections(preds: Predictions, iou_threshold: float) -> Predictions:
    """One detection per physical object: class-agnostic greedy NMS per image on the top-1-per-query rows,
    highest score first (counting option, 30/09). DETR models are meant to need no NMS, but a model trained
    on duplicate pseudo-labels reports some objects twice (F14). Rows below the lowest threshold of the grid
    are dropped first: they never count and cannot suppress a higher-scoring row."""
    top1 = preds.top1_per_query()
    top1 = top1.subset(top1.score >= THRESHOLD_GRID[0])
    if len(top1) == 0:
        return top1
    keep = np.zeros(len(top1), dtype=bool)
    order = np.lexsort((-top1.score, top1.image_id))
    image_ids = top1.image_id[order]
    starts = np.flatnonzero(np.r_[True, image_ids[1:] != image_ids[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(order)], strict=True):
        rows = order[start:end]
        ious = box_iou(top1.boxes[rows], top1.boxes[rows])
        suppressed = np.zeros(len(rows), dtype=bool)
        for i in range(len(rows)):
            if not suppressed[i]:
                keep[rows[i]] = True
                suppressed |= ious[i] >= iou_threshold
    return top1.subset(keep)


def gt_counts(coco_gt: COCO, image_ids: Sequence[int], labels: Sequence[int]) -> np.ndarray:
    """``counts[i, c]`` = number of GT objects of ``labels[c]`` in ``image_ids[i]``."""
    label_index = {label: c for c, label in enumerate(labels)}
    img_index = {image_id: i for i, image_id in enumerate(image_ids)}
    counts = np.zeros((len(image_ids), len(labels)), dtype=np.int64)
    for ann in coco_gt.dataset.get("annotations", []):
        c = label_index.get(ann["category_id"])
        i = img_index.get(ann["image_id"])
        if c is not None and i is not None:
            counts[i, c] += 1
    return counts


def pred_counts(
    preds: Predictions, image_ids: Sequence[int], labels: Sequence[int], threshold: float
) -> np.ndarray:
    """``counts[i, c]`` = number of top1-per-query detections of ``labels[c]`` in
    ``image_ids[i]`` with score >= ``threshold``."""
    return count_matrix(preds.top1_per_query(), image_ids, labels, threshold)


def count_matrix(
    top1: Predictions, image_ids: Sequence[int], labels: Sequence[int], threshold: float
) -> np.ndarray:
    """Like ``pred_counts`` for rows that are already one per query (``top1_per_query()``), so
    threshold sweeps compute the top-1 rows once."""
    counts = np.zeros((len(image_ids), len(labels)), dtype=np.int64)
    if len(top1) == 0 or len(image_ids) == 0 or len(labels) == 0:
        return counts
    image_ids_arr = np.asarray(image_ids, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    image_order = np.argsort(image_ids_arr)
    label_order = np.argsort(labels_arr)
    keep = top1.score >= threshold
    rows_img, rows_label = top1.image_id[keep], top1.label[keep].astype(np.int64)
    # position of each row's image / label in the requested lists; rows not in the lists are dropped
    i = np.clip(np.searchsorted(image_ids_arr, rows_img, sorter=image_order), 0, len(image_ids_arr) - 1)
    c = np.clip(np.searchsorted(labels_arr, rows_label, sorter=label_order), 0, len(labels_arr) - 1)
    found = (image_ids_arr[image_order[i]] == rows_img) & (labels_arr[label_order[c]] == rows_label)
    np.add.at(counts, (image_order[i[found]], label_order[c[found]]), 1)
    return counts


def counting_scores(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """cAcc, ACD, mCCD, mCIoU, transcribed from rpctool (see module docstring). ``pred`` and
    ``gt`` are (n_images, n_classes) count matrices, aligned by row and column."""
    n_images, k = gt.shape
    if n_images == 0:
        return {"cAcc": float("nan"), "ACD": float("nan"), "mCCD": float("nan"), "mCIoU": float("nan"),
                "K": k, "K_eff": 0}

    # score1 (cAcc): fraction of images whose full count vector matches exactly.
    cacc = float(np.mean(np.all(pred == gt, axis=1)))
    # score2 (ACD): mean over images of the total absolute count difference.
    acd = float(np.sum(np.abs(pred - gt)) / n_images)

    class_gt = gt.sum(axis=0)
    has_gt = class_gt > 0
    k_eff = int(has_gt.sum())
    if k_eff:
        # score4 (mCCD): per class, sum|pred-gt| / sum(gt), averaged over classes with GT.
        class_cd = np.abs(pred - gt).sum(axis=0)
        mccd = float(np.sum(class_cd[has_gt] / class_gt[has_gt]) / k_eff)
        # score5 (mCIoU): per class, sum(min(pred,gt)) / sum(max(pred,gt)), averaged.
        class_min = np.minimum(pred, gt).sum(axis=0)
        class_max = np.maximum(pred, gt).sum(axis=0)
        mciou = float(np.sum(class_min[has_gt] / class_max[has_gt]) / k_eff)
    else:
        mccd = float("nan")
        mciou = float("nan")

    return {"cAcc": cacc, "ACD": acd, "mCCD": mccd, "mCIoU": mciou, "K": k, "K_eff": k_eff}


def class_group_scores(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """mCCD and mCCS over the classes (columns) given, as IncreACO (WACV 2021, eq. 11-12) reports them
    separately for old and new classes. mCCS = per class sum(pred) / sum(gt), averaged: 1 means as many
    objects counted as there are; below 1, the model counts too few. Classes with zero GT are excluded,
    as in ``counting_scores``."""
    class_gt = gt.sum(axis=0)
    has_gt = class_gt > 0
    k_eff = int(has_gt.sum())
    if not k_eff:
        return {"mCCD": float("nan"), "mCCS": float("nan"), "K": gt.shape[1], "K_eff": 0}
    mccd = float(np.mean(np.abs(pred - gt).sum(axis=0)[has_gt] / class_gt[has_gt]))
    mccs = float(np.mean(pred.sum(axis=0)[has_gt] / class_gt[has_gt]))
    return {"mCCD": mccd, "mCCS": mccs, "K": gt.shape[1], "K_eff": k_eff}


def select_threshold(
    coco_gt: COCO, preds: Predictions, labels: Sequence[int], thresholds: Sequence[float] = THRESHOLD_GRID
) -> tuple[float, dict[str, float]]:
    """Grid search maximising cAcc (plan section 6.5, V3). Deterministic tie-break: the
    smallest threshold that reaches the maximum cAcc (thresholds are scanned ascending and
    only a strictly larger cAcc replaces the current best)."""
    image_ids = sorted(coco_gt.getImgIds())
    gt = gt_counts(coco_gt, image_ids, labels)
    top1 = preds.top1_per_query()
    best_threshold, best_scores = None, None
    for threshold in thresholds:
        scores = counting_scores(count_matrix(top1, image_ids, labels, threshold), gt)
        if best_scores is None or scores["cAcc"] > best_scores["cAcc"]:
            best_threshold, best_scores = float(threshold), scores
    return best_threshold, best_scores


def scores_by_level(
    coco_gt: COCO, preds: Predictions, labels: Sequence[int], threshold: float
) -> dict[str, dict[str, float]]:
    """Counting scores at a fixed threshold, overall and broken down by ``images[].level``."""
    all_images = coco_gt.getImgIds()
    top1 = preds.top1_per_query()
    result = {"overall": counting_scores(count_matrix(top1, all_images, labels, threshold),
                                          gt_counts(coco_gt, all_images, labels))}
    for level in ("easy", "medium", "hard"):
        level_images = [i for i in all_images if coco_gt.imgs[i].get("level") == level]
        result[level] = counting_scores(count_matrix(top1, level_images, labels, threshold),
                                         gt_counts(coco_gt, level_images, labels))
    return result


def oracle_by_level(coco_gt: COCO, preds: Predictions, labels: Sequence[int]) -> dict[str, dict[str, Any]]:
    """rpctool-style oracle: threshold searched on the evaluated subset itself ("looking at
    the answer"), reported per level plus overall -- for leaderboard comparison only."""
    all_images = coco_gt.getImgIds()
    subsets = {"overall": all_images}
    for level in ("easy", "medium", "hard"):
        subsets[level] = [i for i in all_images if coco_gt.imgs[i].get("level") == level]

    top1 = preds.top1_per_query()
    result: dict[str, dict[str, Any]] = {}
    for name, image_ids in subsets.items():
        gt = gt_counts(coco_gt, image_ids, labels)
        best_threshold, best_scores = None, None
        for threshold in THRESHOLD_GRID:
            scores = counting_scores(count_matrix(top1, image_ids, labels, threshold), gt)
            if best_scores is None or scores["cAcc"] > best_scores["cAcc"]:
                best_threshold, best_scores = float(threshold), scores
        result[name] = {"threshold": best_threshold, **best_scores}
    return result
