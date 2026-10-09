"""CLI for V3: product-counting metrics (cAcc, ACD, mCCD, mCIoU), rpctool-compatible.

Usage::

    python -m tools.eval_count --run-dir RUN --val-ann path/to/val_full.json \\
        --test-ann path/to/test_full.json --task-config configs/tasks_<name>.json

Use --calibrate-only to write a val-only policy, then --policy-in to evaluate test without
reading validation inputs or retuning. Oracle is opt-in via --oracle and labelled separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO

from autocheckout.cl_metrics import check_ann_md5, check_stage_meta, discover_stage_predictions, load_coco
from autocheckout.counting import (
    THRESHOLD_GRID,
    ZERO_GT_NOTE,
    class_group_scores,
    dedup_detections,
    gt_counts,
    oracle_by_level,
    pred_counts,
    scores_by_level,
    select_threshold,
    validated_grid,
)
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.model_acceptance import now
from autocheckout.predictions import Predictions, load_predictions
from autocheckout.taskcfg import TaskConfig

CALIBRATION_GRID = tuple(round(0.10 + 0.01 * i, 2) for i in range(81))
POLICY_TIE_BREAK = "maximum val cAcc, then smallest score threshold, then smallest NMS IoU (0 = off)"


def _config_digest(cfg: TaskConfig) -> str:
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()


def _checkpoint_digest(run_dir: Path, stage: int) -> str | None:
    checkpoint = run_dir / f"task_{stage}" / "task_final.pth"
    return md5_file(checkpoint) if checkpoint.is_file() else None


def _validated_predictions(path: Path, coco: COCO, cfg: TaskConfig, stage: int,
                           split: str, ann_md5: str) -> Predictions:
    preds = load_predictions(path)
    if preds.meta.get("split") != split:
        raise ValueError(f"{path}: prediction split must be {split}")
    if not preds.meta.get("ann_md5"):
        raise ValueError(f"{path}: prediction ann_md5 is required")
    check_ann_md5(preds.meta, ann_md5, context=str(path))
    seen = check_stage_meta(preds.meta, cfg, stage)
    if preds.meta.get("coordinates", "native_pixels") != "native_pixels":
        raise ValueError(f"{path}: coordinates must be native_pixels")
    if not np.isin(preds.image_id, coco.getImgIds()).all():
        raise ValueError(f"{path}: image IDs outside the annotation split")
    if ((preds.label < 0) | (preds.label >= seen)).any():
        raise ValueError(f"{path}: label outside learned classes")
    if not np.isfinite(preds.score).all() or ((preds.score < 0) | (preds.score > 1)).any():
        raise ValueError(f"{path}: score must be finite and in [0, 1]")
    if not np.isfinite(preds.boxes).all() or (preds.boxes[:, 2:] < preds.boxes[:, :2]).any():
        raise ValueError(f"{path}: boxes must be finite xyxy in annotation pixels")
    return preds


def calibrate_run(run_dir: Path, val_ann_path: Path, cfg: TaskConfig, *,
                  thresholds: Sequence[float] = CALIBRATION_GRID,
                  nms_ious: Sequence[float] = (0.45,), stages: Sequence[int] | None = None) -> dict[str, Any]:
    """Choose one counting policy per checkpoint using validation inputs exclusively."""
    thresholds = validated_grid(thresholds, "threshold grid")
    nms_ious = validated_grid(nms_ious, "NMS grid")
    val_md5 = md5_file(val_ann_path)
    coco = load_coco(load_json(val_ann_path))
    available = discover_stage_predictions(run_dir, "val")
    selected_stages = sorted(available if stages is None else stages)
    if not selected_stages:
        raise ValueError(f"no pred_val.npz under {run_dir}")
    policies = {}
    for stage in selected_stages:
        preds = _validated_predictions(available[stage], coco, cfg, stage, "val", val_md5)
        labels = list(range(cfg.seen_classes(stage)))
        candidates = []
        for nms_iou in nms_ious:
            processed = dedup_detections(preds, nms_iou) if nms_iou else preds.top1_per_query()
            threshold, scores = select_threshold(coco, processed, labels, thresholds)
            candidates.append(dict(score_threshold=threshold, nms_iou=nms_iou, val_scores=scores))
        best = min(candidates, key=lambda p: (-p["val_scores"]["cAcc"], p["score_threshold"], p["nms_iou"]))
        policies[str(stage)] = {
            **best, "task_id": stage, "seen_classes": len(labels), "candidates": candidates,
            "checkpoint_md5": _checkpoint_digest(run_dir, stage),
            "val_prediction_md5": md5_file(available[stage]),
        }
    return {
        "schema_version": 1, "created_at": now(), "selection_split": "val", "objective": "cAcc",
        "task_config": cfg.name, "task_config_sha256": _config_digest(cfg),
        "val_ann": str(val_ann_path), "val_ann_md5": val_md5,
        "threshold_grid": thresholds, "nms_grid": nms_ious, "tie_break": POLICY_TIE_BREAK,
        "zero_gt_class_handling": ZERO_GT_NOTE, "stages": policies,
    }


def evaluate_policy(run_dir: Path, test_ann_path: Path, cfg: TaskConfig, policy: dict[str, Any], *,
                    oracle: bool = False) -> dict[str, Any]:
    """Apply a saved policy without accessing validation files or searching on test."""
    if policy.get("schema_version") != 1 or policy.get("selection_split") != "val":
        raise ValueError("counting policy must be version 1 and selected on val")
    if policy.get("task_config_sha256") != _config_digest(cfg):
        raise ValueError("counting policy task config does not match")
    paths = discover_stage_predictions(run_dir, "test")
    if set(paths) != {int(stage) for stage in policy["stages"]}:
        raise ValueError("test stages do not match the calibrated policy stages")
    if not paths:
        raise ValueError("counting policy has no stages")
    test_md5 = md5_file(test_ann_path)
    coco = load_coco(load_json(test_ann_path))
    results = {}
    for stage, path in sorted(paths.items()):
        selected = policy["stages"][str(stage)]
        if selected["task_id"] != stage or selected["seen_classes"] != cfg.seen_classes(stage):
            raise ValueError("policy task/seen_classes does not match task config")
        if selected["checkpoint_md5"] != _checkpoint_digest(run_dir, stage):
            raise ValueError(f"task_{stage}: checkpoint differs from the calibrated policy")
        threshold, nms_iou = selected["score_threshold"], selected["nms_iou"]
        validated_grid([threshold], "policy threshold grid")
        validated_grid([nms_iou], "policy NMS grid")
        preds = _validated_predictions(path, coco, cfg, stage, "test", test_md5)
        processed = dedup_detections(preds, nms_iou) if nms_iou else preds.top1_per_query()
        labels = list(range(cfg.seen_classes(stage)))
        results[stage] = {
            "task_id": stage, "seen_classes": len(labels), "nms_iou": nms_iou,
            "threshold": {"value": threshold, "cAcc_val": selected["val_scores"]["cAcc"]},
            "test": scores_by_level(coco, processed, labels, threshold),
            "test_by_task": scores_by_task_group(coco, processed, cfg, stage, threshold),
        }
        if oracle:
            results[stage]["oracle"] = oracle_by_level(coco, processed, labels)
    nms_values = {stage["nms_iou"] for stage in results.values()}
    return {
        "run_dir": str(run_dir), "task_config": cfg.name,
        "val_ann": policy["val_ann"], "val_ann_md5": policy["val_ann_md5"],
        "test_ann": str(test_ann_path), "test_ann_md5": test_md5,
        "threshold_grid": policy["threshold_grid"], "tie_break": policy["tie_break"],
        "zero_gt_class_handling": ZERO_GT_NOTE, "nms_iou": next(iter(nms_values)) if len(nms_values) == 1 else None,
        "oracle_enabled": oracle, "calibration_policy": policy, "stages": results,
    }


def evaluate_run(run_dir: Path, val_ann_path: Path, test_ann_path: Path, cfg: TaskConfig,
                 nms_iou: float = 0.0, *, oracle: bool = False) -> dict[str, Any]:
    # Compatibility for callers evaluating a subset of stages with both predictions present.
    common = sorted(set(discover_stage_predictions(run_dir, "val")) & set(discover_stage_predictions(run_dir, "test")))
    policy = calibrate_run(run_dir, val_ann_path, cfg, thresholds=THRESHOLD_GRID, nms_ious=[nms_iou], stages=common)
    return evaluate_policy(run_dir, test_ann_path, cfg, policy, oracle=oracle)


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
    show_oracle = result.get("oracle_enabled", False)
    lines = [f"# Counting metrics: {result['run_dir']} (test, policy selected on val)", ""]
    header = "| stage | NMS IoU | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU |"
    separator = "|---|---|---|---|---|---|---|---|"
    if show_oracle:
        lines.append("Oracle enabled: diagnostic comparison selected on test; excluded from research policy.")
        lines.append("")
        header += " oracle cAcc | oracle threshold |"
        separator += "---|---|"
    lines.extend([header, separator])
    for stage in sorted(result["stages"]):
        s = result["stages"][stage]
        t = s["test"]["overall"]
        row = (f"| {stage} | {_fmt(s['nms_iou'])} | {_fmt(s['threshold']['value'])} | "
               f"{_fmt(s['threshold']['cAcc_val'])} | {_fmt(t['cAcc'])} | {_fmt(t['ACD'])} | "
               f"{_fmt(t['mCCD'])} | {_fmt(t['mCIoU'])} |")
        if show_oracle:
            o = s["oracle"]["overall"]
            row += f" {_fmt(o['cAcc'])} | {_fmt(o['threshold'])} |"
        lines.append(row)
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
    parser.add_argument("--val-ann", type=Path)
    parser.add_argument("--test-ann", type=Path)
    parser.add_argument("--task-config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate-only", action="store_true", help="read validation only, save policy and exit")
    mode.add_argument("--policy-in", type=Path, help="locked policy; evaluate test without reading validation")
    parser.add_argument("--policy-out", type=Path, help="default RUN/calibration_count.json")
    parser.add_argument("--threshold-grid", help="comma-separated score thresholds; default 0.10..0.90 step 0.01")
    nms = parser.add_mutually_exclusive_group()
    nms.add_argument("--nms-iou", type=float, help="fixed class-agnostic NMS IoU (0 = off); default 0.45")
    nms.add_argument("--nms-grid", help="comma-separated IoUs selected on val only; 0 disables NMS")
    parser.add_argument("--oracle", action="store_true",
                        help="explicit test-selected diagnostic; never used for policy")
    args = parser.parse_args()

    cfg = TaskConfig.load(args.task_config)
    if args.policy_in:
        if args.threshold_grid is not None or args.nms_grid is not None or args.nms_iou is not None or args.policy_out:
            parser.error("--policy-in cannot be combined with calibration settings")
        policy = load_json(args.policy_in)
        policy_path = args.policy_in
    else:
        if args.val_ann is None:
            parser.error("--val-ann is required for calibration")
        if args.calibrate_only and args.oracle:
            parser.error("--oracle is only available during explicit test evaluation")
        thresholds = [float(x) for x in args.threshold_grid.split(",")] if args.threshold_grid else CALIBRATION_GRID
        nms_ious = [float(x) for x in args.nms_grid.split(",")] if args.nms_grid else [
            0.45 if args.nms_iou is None else args.nms_iou]
        policy = calibrate_run(args.run_dir, args.val_ann, cfg, thresholds=thresholds, nms_ious=nms_ious)
        policy_path = args.policy_out or args.run_dir / "calibration_count.json"
        save_json(policy_path, policy, indent=2)
        print(f"Wrote validation policy {policy_path}")
        if args.calibrate_only:
            return
    if args.test_ann is None:
        parser.error("--test-ann is required for test evaluation")
    result = evaluate_policy(args.run_dir, args.test_ann, cfg, policy, oracle=args.oracle)
    result["calibration_policy_path"] = str(policy_path)
    result["calibration_policy_md5"] = md5_file(policy_path)

    suffix = f"_nms{args.nms_iou:g}" if args.nms_iou is not None and args.nms_iou > 0 else ""
    if args.oracle:
        suffix += "_oracle"
    out_path = args.run_dir / f"metrics_count_test{suffix}.json"
    save_json(out_path, result, indent=1)
    print(to_markdown(result))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
