"""CLI for V3: product-counting metrics (cAcc, ACD, mCCD, mCIoU), rpctool-compatible.

Usage::

    python -m tools.eval_count --run-dir RUN --val-ann path/to/val_full.json \\
        --test-ann path/to/test_full.json --task-config configs/tasks_<name>.json

For every stage that has both ``task_<t>/pred_val.npz`` and ``task_<t>/pred_test.npz``: pick
the score threshold on val (grid search maximising cAcc, see
``autocheckout.counting.select_threshold``), apply it to test, and also report the
rpctool-style oracle threshold (searched on test itself, "looking at the answer"; for
leaderboard comparison only, plan section 6.5). Writes ``RUN/metrics_count_test.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO

from autocheckout.cl_metrics import check_ann_md5, check_stage_meta, discover_stage_predictions, load_coco
from autocheckout.counting import (
    THRESHOLD_GRID,
    TIE_BREAK_NOTE,
    ZERO_GT_NOTE,
    class_group_scores,
    dedup_detections,
    gt_counts,
    oracle_by_level,
    pred_counts,
    scores_by_level,
    select_threshold,
)
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import Predictions, load_predictions
from autocheckout.taskcfg import TaskConfig


def evaluate_run(run_dir: Path, val_ann_path: Path, test_ann_path: Path, cfg: TaskConfig,
                 nms_iou: float = 0.0) -> dict[str, Any]:
    val_md5 = md5_file(val_ann_path)
    test_md5 = md5_file(test_ann_path)
    coco_val = load_coco(load_json(val_ann_path))
    coco_test = load_coco(load_json(test_ann_path))

    val_stages = discover_stage_predictions(run_dir, "val")
    test_stages = discover_stage_predictions(run_dir, "test")
    common = sorted(set(val_stages) & set(test_stages))
    if not common:
        raise ValueError(f"no stage under {run_dir} has both pred_val.npz and pred_test.npz")

    stages: dict[int, dict[str, Any]] = {}
    for stage in common:
        val_preds = load_predictions(val_stages[stage])
        test_preds = load_predictions(test_stages[stage])
        check_ann_md5(val_preds.meta, val_md5, context=f"task_{stage}/pred_val.npz")
        check_ann_md5(test_preds.meta, test_md5, context=f"task_{stage}/pred_test.npz")
        seen_classes = check_stage_meta(val_preds.meta, cfg, stage)
        check_stage_meta(test_preds.meta, cfg, stage)
        labels = list(range(seen_classes))
        if nms_iou > 0:
            val_preds, test_preds = dedup_detections(val_preds, nms_iou), dedup_detections(test_preds, nms_iou)

        threshold, val_scores = select_threshold(coco_val, val_preds, labels)
        stages[stage] = {
            "task_id": stage,
            "seen_classes": seen_classes,
            "threshold": {"value": threshold, "cAcc_val": val_scores["cAcc"]},
            "test": scores_by_level(coco_test, test_preds, labels, threshold),
            "oracle": oracle_by_level(coco_test, test_preds, labels),
            "test_by_task": scores_by_task_group(coco_test, test_preds, cfg, stage, threshold),
        }

    return {
        "run_dir": str(run_dir),
        "task_config": cfg.name,
        "val_ann": str(val_ann_path),
        "val_ann_md5": val_md5,
        "test_ann": str(test_ann_path),
        "test_ann_md5": test_md5,
        "threshold_grid": list(THRESHOLD_GRID),
        "tie_break": TIE_BREAK_NOTE,
        "zero_gt_class_handling": ZERO_GT_NOTE,
        "nms_iou": nms_iou,
        "stages": stages,
    }


def scores_by_task_group(coco_gt: COCO, preds: Predictions, cfg: TaskConfig, stage: int,
                         threshold: float) -> dict[str, dict[str, float]]:
    """mCCD/mCCS on all test images for the classes of each task learned so far, plus "new" (the classes of
    this stage's task) and "old" (all classes learned before it; absent at stage 1), as in IncreACO's Table 3."""
    image_ids = coco_gt.getImgIds()
    labels = list(range(cfg.seen_classes(stage)))
    pred = pred_counts(preds, image_ids, labels, threshold)
    gt = gt_counts(coco_gt, image_ids, labels)
    groups = {f"task_{t}": list(cfg.task(t).labels) for t in range(1, stage + 1)}
    groups["new"] = groups[f"task_{stage}"]
    if stage > 1:
        groups["old"] = list(range(cfg.seen_classes(stage - 1)))
    return {name: class_group_scores(pred[:, cols], gt[:, cols]) for name, cols in groups.items()}


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def to_markdown(result: dict[str, Any]) -> str:
    nms = f", class-agnostic NMS at IoU {result['nms_iou']}" if result.get("nms_iou") else ""
    lines = [f"# Counting metrics: {result['run_dir']} (test, threshold picked on val{nms})", ""]
    lines.append("| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | "
                  "oracle cAcc | oracle threshold |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for stage in sorted(result["stages"]):
        s = result["stages"][stage]
        t, o = s["test"]["overall"], s["oracle"]["overall"]
        lines.append(
            f"| {stage} | {_fmt(s['threshold']['value'])} | {_fmt(s['threshold']['cAcc_val'])} | "
            f"{_fmt(t['cAcc'])} | {_fmt(t['ACD'])} | {_fmt(t['mCCD'])} | {_fmt(t['mCIoU'])} | "
            f"{_fmt(o['cAcc'])} | {_fmt(o['threshold'])} |"
        )
    lines += ["", "Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):", "",
              "| stage | mCCD old | mCCD new | mCCS old | mCCS new |", "|---|---|---|---|---|"]
    for stage in sorted(result["stages"]):
        groups = result["stages"][stage]["test_by_task"]
        old = groups.get("old", {"mCCD": float("nan"), "mCCS": float("nan")})
        new = groups["new"]
        lines.append(f"| {stage} | {_fmt(old['mCCD'])} | {_fmt(new['mCCD'])} | "
                     f"{_fmt(old['mCCS'])} | {_fmt(new['mCCS'])} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="V3: product-counting metrics")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--val-ann", required=True, type=Path)
    parser.add_argument("--test-ann", required=True, type=Path)
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--nms-iou", type=float, default=0.0,
                        help="one detection per object: class-agnostic NMS at this IoU before counting (0 = off); "
                             "writes metrics_count_test_nms<iou>.json")
    args = parser.parse_args()

    cfg = TaskConfig.load(args.task_config)
    result = evaluate_run(args.run_dir, args.val_ann, args.test_ann, cfg, nms_iou=args.nms_iou)

    suffix = f"_nms{args.nms_iou:g}" if args.nms_iou > 0 else ""
    out_path = args.run_dir / f"metrics_count_test{suffix}.json"
    save_json(out_path, result, indent=1)
    print(to_markdown(result))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
