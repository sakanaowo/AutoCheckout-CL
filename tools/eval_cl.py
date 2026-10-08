"""CLI for V2: class-incremental detection metrics (mAP@C/P/A, M2, accuracy matrix, forgetting).

Usage::

    python -m tools.eval_cl --run-dir RUN --split test --ann path/to/test_full.json \\
        --task-config configs/tasks_<name>.json

Reads ``RUN/task_<t>/pred_<split>.npz`` for every stage present (stages need not be
contiguous: a joint-training upper bound run may only have its last stage, see
``autocheckout.cl_metrics.discover_stage_predictions``). Writes
``RUN/metrics_cl_<split>.json`` (docs/data_preprocessing/formats.md section 6) and prints a markdown summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from autocheckout.cl_metrics import (
    check_ann_md5,
    check_stage_meta,
    discover_stage_predictions,
    load_coco,
    m1_stage,
    m2_stage,
    matrix_forgetting,
)
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import load_predictions
from autocheckout.taskcfg import TaskConfig


def evaluate_run(run_dir: Path, split: str, ann_path: Path, cfg: TaskConfig) -> dict[str, Any]:
    ann = load_json(ann_path)
    ann_md5 = md5_file(ann_path)
    coco_gt = load_coco(ann)

    stage_paths = discover_stage_predictions(run_dir, split)
    if not stage_paths:
        raise ValueError(f"no task_<t>/pred_{split}.npz found under {run_dir}")

    stages: dict[int, dict[str, Any]] = {}
    matrix: dict[int, dict[int, float]] = {}
    for stage, path in sorted(stage_paths.items()):
        preds = load_predictions(path)
        check_ann_md5(preds.meta, ann_md5, context=f"task_{stage}/pred_{split}.npz")
        seen_classes = check_stage_meta(preds.meta, cfg, stage)

        m1 = m1_stage(coco_gt, preds, cfg, stage)
        m2 = m2_stage(coco_gt, preds, seen_classes)
        stages[stage] = {"task_id": stage, "seen_classes": seen_classes, "m1": m1, "m2": m2}
        matrix[stage] = m1["matrix_row"]

    forgetting = matrix_forgetting(matrix)

    return {
        "run_dir": str(run_dir),
        "split": split,
        "ann": str(ann_path),
        "ann_md5": ann_md5,
        "task_config": cfg.name,
        "stages": stages,
        "matrix": matrix,
        "forgetting": forgetting,
    }


def _fmt(x: float | None) -> str:
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.4f}"


def to_markdown(result: dict[str, Any]) -> str:
    lines = [f"# Class-incremental metrics: {result['run_dir']} ({result['split']})", ""]
    lines.append("| stage | mAP@C AP50 | mAP@C AP | mAP@P AP50 | mAP@P AP | "
                  "mAP@A AP50 | mAP@A AP | M2 AP50 | M2 AP |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for stage in sorted(result["stages"]):
        s = result["stages"][stage]
        c, p, a, m2 = s["m1"]["mAP_C"], s["m1"]["mAP_P"], s["m1"]["mAP_A"], s["m2"]["overall"]
        lines.append(
            f"| {stage} | {_fmt(c['AP50'])} | {_fmt(c['AP'])} | "
            f"{_fmt(p['AP50'] if p else None)} | {_fmt(p['AP'] if p else None)} | "
            f"{_fmt(a['AP50'])} | {_fmt(a['AP'])} | {_fmt(m2['AP50'])} | {_fmt(m2['AP'])} |"
        )
    lines.append("")

    last_stage = max(result["stages"])
    lines.append(f"## Accuracy matrix (AP50, task group as column, after stage as row; last = {last_stage})")
    groups = sorted(result["matrix"][last_stage])
    lines.append("| stage \\ group | " + " | ".join(str(g) for g in groups) + " |")
    lines.append("|" + "---|" * (len(groups) + 1))
    for stage in sorted(result["matrix"]):
        row = result["matrix"][stage]
        cells = [_fmt(row.get(g)) for g in groups]
        lines.append(f"| {stage} | " + " | ".join(cells) + " |")
    lines.append("")

    forgetting = result["forgetting"]
    lines.append(f"Forgetting (average over groups < {last_stage}): {_fmt(forgetting['average'])}")
    lines.append(f"Average of last row: {_fmt(forgetting['avg_last_row'])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="V2: class-incremental detection metrics")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument("--task-config", required=True, type=Path)
    args = parser.parse_args()

    cfg = TaskConfig.load(args.task_config)
    result = evaluate_run(args.run_dir, args.split, args.ann, cfg)

    out_path = args.run_dir / f"metrics_cl_{args.split}.json"
    save_json(out_path, result, indent=1)
    print(to_markdown(result))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
