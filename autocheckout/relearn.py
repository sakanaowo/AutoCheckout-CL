"""Relearn: tasks 6-7 relearn 50 SKUs of tasks 1-5 under alias classes
(docs/project/task6-relearn-plan-2026-10-30.md).

An alias class has its own label but the RPC id of the class it relearns. Before scoring, predictions are
resolved back to the original labels, so a relearn run is scored like a stage-5 model of the seed config:
``merge`` keeps the old slot and the alias of a relearned SKU (one object counted once), ``alias_only``
keeps only the alias.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from autocheckout.predictions import Predictions, load_predictions, save_predictions
from autocheckout.taskcfg import TaskConfig

MODES = ("merge", "alias_only")
GAIN = {6: 0.0165, 7: 0.03}  # cAcc-M2 gain over the seed required after task 6 and task 7 (Q10)
OTHER_DROP = 0.01  # largest fall of the AP50 of the 150 SKUs not relearned


def alias_val_file(val_full: dict[str, Any], cfg: TaskConfig, task_id: int) -> dict[str, Any]:
    """Val images with an object of the SKUs alias task ``task_id`` relearns, only those objects, under the
    alias labels (the val_task file PDP reads for that task)."""
    classes = cfg.task(task_id).classes
    alias = {c.alias_of: c.label for c in classes}
    annotations = [{**a, "category_id": alias[a["category_id"]]} for a in val_full["annotations"]
                   if a["category_id"] in alias]
    keep = {a["image_id"] for a in annotations}
    return {**val_full, "images": [img for img in val_full["images"] if img["id"] in keep],
            "annotations": annotations,
            "categories": [{"id": c.label, "name": c.name, "supercategory": c.supercategory} for c in classes]}


def resolve_predictions(preds: Predictions, cfg: TaskConfig, *, mode: str, as_stage: int) -> Predictions:
    """Predictions of a relearn stage with the aliases mapped back to the original labels, as stage ``as_stage``.

    ``alias_only`` first drops the rows of the old slot of every SKU relearned so far. Rows that become the same
    (image, query, label) keep the highest score.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    task_id = int(preds.meta["task_id"])
    seen = cfg.seen_classes(task_id)
    if ((preds.label < 0) | (preds.label >= seen)).any():
        raise ValueError(f"labels outside the {seen} classes learned by task {task_id}")
    aliases = {c.label: c.alias_of for t in cfg.tasks if t.task_id <= task_id for c in t.classes
               if c.alias_of is not None}
    out = preds.subset(~np.isin(preds.label, list(aliases.values()))) if mode == "alias_only" else preds
    original = np.arange(seen)
    original[list(aliases)] = list(aliases.values())
    out = Predictions(out.image_id, out.query, original[out.label], out.score, out.boxes)
    order = np.lexsort((-out.score, out.label, out.query, out.image_id))
    key = np.stack([out.image_id[order], out.query[order], out.label[order]])
    first = np.ones(len(order), dtype=bool)
    first[1:] = (key[:, 1:] != key[:, :-1]).any(axis=0)
    out = out.subset(order[first])
    out.meta = {**preds.meta, "task_id": as_stage, "seen_classes": cfg.seen_classes(as_stage),
                "resolved_from": {"mode": mode, "task_id": task_id}}
    return out


def resolve_run(run: Path, stage: int, cfg: TaskConfig, target: Path, *, mode: str, as_stage: int,
                splits: Sequence[str] = ("val",)) -> Path:
    """``target/task_<as_stage>/``: the resolved predictions of ``run/task_<stage>`` and links to its checkpoint
    and run_info. The target holds that stage only: evaluate_policy wants the stages of the policy, each with
    the md5 of its checkpoint."""
    run, target = Path(run), Path(target)
    others = sorted(p.name for p in target.glob("task_*") if p.name != f"task_{as_stage}")
    if others:
        raise ValueError(f"{target} already holds {others}; the evaluators need task_{as_stage} only")
    source, out = run / f"task_{stage}", target / f"task_{as_stage}"
    out.mkdir(parents=True, exist_ok=True)
    for split in splits:
        preds = load_predictions(source / f"pred_{split}.npz")
        save_predictions(out / f"pred_{split}.npz", resolve_predictions(preds, cfg, mode=mode, as_stage=as_stage))
    for name in ("task_final.pth", "run_info.json"):
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink((source / name).resolve(), link)
    return target


def relearn_verdict(report: dict[str, Any], seed: dict[str, Any], stage: int) -> tuple[str, str]:
    """Gate after relearn task ``stage``: cAcc-M2 must rise by GAIN[stage] over the seed and the AP50 of the SKUs
    not relearned fall by at most OTHER_DROP. Returns (go | stop, reason naming the failed gates)."""
    gain = round(report["cAcc_m2"] - seed["cAcc_m2"], 6)  # compared after rounding, as pilot_verdict
    drop = round(seed["AP50_other"] - report["AP50_other"], 6)
    scores = (f"cAcc-M2 {seed['cAcc_m2']:.4f} -> {report['cAcc_m2']:.4f} ({gain:+.4f}, needs {GAIN[stage]:+.4f}); "
              f"AP50 of the other SKUs {seed['AP50_other']:.4f} -> {report['AP50_other']:.4f} "
              f"({-drop:+.4f}, may fall {OTHER_DROP:.4f})")
    failed = [name for name, bad in (("cAcc-M2 gain", gain < GAIN[stage]),
                                     ("AP50 of the other SKUs", drop > OTHER_DROP)) if bad]
    if failed:
        return "stop", f"{scores}: failed {' and '.join(failed)}"
    return "go", scores
