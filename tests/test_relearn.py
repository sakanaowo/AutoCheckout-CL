"""Relearn helpers: alias val files, alias resolution of predictions, verdict (task6-relearn plan, Task 7)."""

import json

import numpy as np
import pytest

from autocheckout.cl_metrics import discover_stage_predictions, load_coco
from autocheckout.io import md5_file
from autocheckout.predictions import Predictions, load_predictions, save_predictions
from autocheckout.relearn import alias_val_file, relearn_verdict, resolve_predictions, resolve_run
from autocheckout.taskcfg import TaskConfig
from pdp_helpers import task_config_dict
from tools.eval_count import _validated_predictions
from tools.make_relearn_config import relearn_config

BASE = TaskConfig.from_dict(task_config_dict(sizes=(2, 1), reserved=0))  # labels 0, 1 | 2
CFG = relearn_config(BASE, [[0], [2]], reserved=1, name="toy+relearn")    # task 3: alias 3 of 0; task 4: alias 4 of 2


def preds(task_id, rows, **meta):
    """Predictions of image 1 from (query, label, score) rows."""
    query, label, score = zip(*rows)
    boxes = [[10, 10, 20 + q, 20 + q] for q in query]
    return Predictions(image_id=[1] * len(rows), query=query, label=label, score=score, boxes=boxes,
                       meta={"task_id": task_id, "seen_classes": CFG.seen_classes(task_id), "split": "val", **meta})


def rows(p):
    return sorted(zip(p.query.tolist(), p.label.tolist(), [round(s, 2) for s in p.score.tolist()]))


TASK4 = [(0, 0, .6), (0, 3, .9), (1, 4, .5), (3, 0, .7)]


def val_full():
    return {"info": {"name": "toy"}, "images": [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100}
                                                for i in (1, 2, 3)],
            "annotations": [{"id": k, "image_id": i, "category_id": c, "bbox": [1, 1, 5, 5], "area": 25, "iscrowd": 0}
                            for k, (i, c) in enumerate([(1, 0), (1, 1), (2, 1), (3, 2), (3, 0)], start=1)],
            "categories": [{"id": c.label, "name": c.name} for c in BASE.tasks[0].classes + BASE.tasks[1].classes]}


def test_alias_val_file_keeps_the_relearned_objects_under_their_alias_labels():
    out = alias_val_file(val_full(), CFG, 3)
    assert [i["id"] for i in out["images"]] == [1, 3] and out["info"] == {"name": "toy"}
    assert [(a["id"], a["image_id"], a["category_id"]) for a in out["annotations"]] == [(1, 1, 3), (5, 3, 3)]
    assert out["categories"] == [{"id": 3, "name": "sku_1@t3", "supercategory": "drink"}]
    assert [a["category_id"] for a in alias_val_file(val_full(), CFG, 4)["annotations"]] == [4]


def test_merge_counts_an_object_seen_by_the_old_slot_and_the_alias_once():
    out = resolve_predictions(preds(4, TASK4, ann_md5="x"), CFG, mode="merge", as_stage=2)
    assert rows(out) == [(0, 0, .9), (1, 2, .5), (3, 0, .7)]
    assert out.boxes[out.query == 0].tolist() == [[10, 10, 20, 20]]
    assert out.meta == {"task_id": 2, "seen_classes": CFG.seen_classes(2), "split": "val", "ann_md5": "x",
                        "resolved_from": {"mode": "merge", "task_id": 4}}


def test_alias_only_drops_the_old_slot_of_relearned_skus():
    assert rows(resolve_predictions(preds(4, TASK4), CFG, mode="alias_only", as_stage=2)) == [(0, 0, .9), (1, 2, .5)]


def test_after_task_3_the_old_slot_of_a_task_4_sku_is_kept():
    out = resolve_predictions(preds(3, [(0, 2, .8), (1, 0, .4), (1, 3, .6)]), CFG, mode="alias_only", as_stage=2)
    assert rows(out) == [(0, 2, .8), (1, 0, .6)]


@pytest.mark.parametrize("mode", ["merge", "alias_only"])
def test_resolve_refuses_labels_not_yet_seen(mode):
    with pytest.raises(ValueError, match="learned"):
        resolve_predictions(preds(3, [(0, 4, .8)]), CFG, mode=mode, as_stage=2)


def test_resolve_refuses_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        resolve_predictions(preds(4, TASK4), CFG, mode="both", as_stage=2)


def test_resolve_run_lays_out_a_stage_the_evaluators_read(tmp_path):
    ann = tmp_path / "val_full.json"
    ann.write_text(json.dumps(val_full()))
    source = tmp_path / "run/task_4"
    source.mkdir(parents=True)
    (source / "task_final.pth").write_bytes(b"weights")
    (source / "run_info.json").write_text("{}")
    save_predictions(source / "pred_val.npz", preds(4, TASK4, ann_md5=md5_file(ann)))

    target = resolve_run(tmp_path / "run", 4, CFG, tmp_path / "resolved", mode="merge", as_stage=2)
    assert target == tmp_path / "resolved" and sorted(p.name for p in target.iterdir()) == ["task_2"]
    assert discover_stage_predictions(target, "val") == {2: target / "task_2/pred_val.npz"}
    checked = _validated_predictions(target / "task_2/pred_val.npz", load_coco(val_full()), BASE, 2, "val",
                                     md5_file(ann))
    assert rows(checked) == [(0, 0, .9), (1, 2, .5), (3, 0, .7)]
    for name in ("task_final.pth", "run_info.json"):
        link = target / "task_2" / name
        assert link.is_symlink() and link.resolve() == (source / name).resolve()

    resolve_run(tmp_path / "run", 4, CFG, target, mode="alias_only", as_stage=2)  # rerun replaces it
    assert rows(load_predictions(target / "task_2/pred_val.npz")) == [(0, 0, .9), (1, 2, .5)]
    with pytest.raises(ValueError, match="task_2"):  # the evaluators need exactly one stage in the target
        resolve_run(tmp_path / "run", 4, CFG, target, mode="merge", as_stage=1)


@pytest.mark.parametrize("stage, cacc, ap_other, decision", [
    (6, 0.4533, 0.94, "go"), (6, 0.4532, 0.95, "stop"), (6, 0.4533, 0.9399, "stop"),
    (7, 0.4668, 0.94, "go"), (7, 0.4667, 0.95, "stop"), (7, 0.4668, 0.9399, "stop"), (7, 0.50, 0.96, "go")])
def test_relearn_verdict_needs_the_gain_of_its_stage_without_hurting_the_other_skus(stage, cacc, ap_other, decision):
    seed = {"cAcc_m2": 0.4368, "AP50_other": 0.95}
    verdict, reason = relearn_verdict({"cAcc_m2": cacc, "AP50_other": ap_other}, seed, stage)
    assert verdict == decision
    if decision == "stop":
        assert ("cAcc-M2" in reason.split(":")[-1]) == (round(cacc - 0.4368, 6) < (0.0165 if stage == 6 else 0.03))
