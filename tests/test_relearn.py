"""Relearn helpers: alias val files, alias resolution of predictions, verdict (task6-relearn plan, Task 7)."""

import json
from pathlib import Path

import numpy as np
import pytest

from autocheckout.cl_metrics import discover_stage_predictions, load_coco
from autocheckout.io import md5_file
from autocheckout.predictions import Predictions, load_predictions, save_predictions
from autocheckout.relearn import alias_val_file, per_sku_accuracy, relearn_verdict, resolve_predictions, resolve_run
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


def test_per_sku_accuracy_counts_right_wrong_missed_and_extra_boxes_as_the_selection_did():
    # image 1: A (label 0) and B (label 1); image 2: C (label 1), no box; one box under its threshold is not counted
    coco = load_coco({"images": [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100} for i in (1, 2)],
                      "annotations": [{"id": k, "image_id": i, "category_id": c, "bbox": b, "area": 100, "iscrowd": 0}
                                      for k, (i, c, b) in enumerate([(1, 0, [0, 0, 10, 10]), (1, 1, [50, 50, 10, 10]),
                                                                     (2, 1, [0, 0, 10, 10])], start=1)],
                      "categories": [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}]})
    counted = Predictions(image_id=[1, 1, 1, 1], query=[0, 1, 2, 3], label=[0, 0, 1, 1], score=[.9, .8, .9, .2],
                          boxes=[[0, 0, 10, 10], [50, 50, 60, 60], [80, 80, 95, 95], [0, 0, 10, 10]], meta={})
    acc = per_sku_accuracy(coco, counted, np.array([.5, .5]), seen=2)
    assert acc[0] == {"n_val": 1, "recall": 1.0, "precision": 0.5, "wrong": 0, "missed": 0, "others_called_this": 1,
                      "extra_boxes": 0}
    assert acc[1] == {"n_val": 2, "recall": 0.0, "precision": 0.0, "wrong": 1, "missed": 1, "others_called_this": 0,
                      "extra_boxes": 1}


def test_a_config_whose_learned_tasks_differ_from_the_seed_is_refused():
    from autocheckout.relearn import relearn_tasks
    seed = TaskConfig.from_dict(task_config_dict(sizes=(2, 1, 1, 1, 1), reserved=0))
    assert relearn_tasks(relearn_config(seed, [[1], [3]], reserved=1, name="x"), seed) == [6, 7]
    other = TaskConfig.from_dict(task_config_dict(sizes=(1, 2, 1, 1, 1), reserved=0))
    with pytest.raises(ValueError, match="tasks 1-5"):
        relearn_tasks(relearn_config(other, [[1], [3]], reserved=1, name="x"), seed)
    with pytest.raises(ValueError, match="alias tasks"):
        relearn_tasks(relearn_config(seed, [[1]], reserved=1, name="x"), seed)


def test_prepare_refuses_to_mix_a_dry_run_and_a_real_seed(tmp_path):
    from autocheckout import relearn
    (tmp_path / "seed").mkdir()
    (tmp_path / "seed/plan.json").write_text(json.dumps({"dry_run": False, "campaign_id": "full-syn-r1"}))
    with pytest.raises(ValueError, match="dry run"):
        relearn.prepare({"campaign_id": "rl", "round": 1, "artifact_root": str(tmp_path), "dry_run": True,
                         "seed_campaign": str(tmp_path / "seed")})


def test_task_7_is_refused_when_the_task_6_gate_said_stop(tmp_path):
    from autocheckout import relearn
    from autocheckout.full_syn import GateStop
    plan = {"root": str(tmp_path), "dry_run": False, "relearn_tasks": [6, 7]}
    (tmp_path / "stages").mkdir()
    gate = {"name": "relearn task 6", "decision": "stop", "reason": "no gain", "enforced": True}
    (tmp_path / "stages/task6.json").write_text(json.dumps({"stage": 6, "gate": gate}))
    with pytest.raises(GateStop, match="no gain"):
        relearn.stage_task(plan, 7)
    assert not (tmp_path / "attempts").exists()  # nothing was started


@pytest.mark.slow
def test_relearn_dry_run_expands_trains_two_tasks_and_scores_both_modes(tmp_path):
    import torch

    from autocheckout import full_syn, relearn
    base = dict(round=1, artifact_root=str(tmp_path), dry_run=True, batch_size=1, effective_batch=2,
                resolution=640, fsa_epochs=8, epochs=16, audit_images=4)
    seed = full_syn.prepare({**base, "campaign_id": "seed"})
    for stage in (full_syn.stage_fsa, full_syn.stage_task1, full_syn.stage_tasks):
        stage(seed)
    settings = dict(campaign_id="rl", round=1, artifact_root=str(tmp_path), dry_run=True, audit_images=4,
                    seed_campaign=str(tmp_path / "dry_run/seed"), task_config="configs/relearn/task_config_relearn.json")
    plan = relearn.prepare(settings)
    assert (plan["n_classes"], plan["relearn_tasks"], plan["relearned"]) == (10, [6, 7], [1, 3])
    assert sorted(p.name for p in Path(plan["task_dir"]).iterdir()) == [
        "test_full.json", "train_task_6.json", "train_task_7.json", "val_full.json", "val_task_6.json", "val_task_7.json"]
    run = tmp_path / "dry_run/rl/runs" / relearn.RELEARN

    relearn.stage_expand(plan)
    r6, r7 = relearn.stage_task(plan, 6), relearn.stage_task(plan, 7)
    assert torch.load(run / "task_5/task_final.pth", map_location="cpu")["model"]["class_embed.0.weight"].shape[0] == 10
    assert load_predictions(run / "task_7/pred_val.npz").label.max() < 8
    for r in (r6, r7):
        assert set(r["modes"]) == {"merge", "alias_only"} and r["mode"] in r["modes"]
        assert {"cAcc_m2", "AP50_other"} <= set(r["modes"]["merge"]) and set(r["seed"]) == {"cAcc_m2", "AP50_other"}
        assert r["gate"]["decision"] in {"go", "stop"} and r["gate"]["enforced"] is False
    mtime = (run / "task_7/task_final.pth").stat().st_mtime
    assert relearn.stage_task(plan, 7)["finished_at"] == r7["finished_at"]
    acc = relearn.sku_accuracy(plan, 7)
    assert sorted(acc) == list(range(6)) and all(0 <= a["precision"] <= 1 for a in acc.values())
    assert (run / "task_7/task_final.pth").stat().st_mtime == mtime

    with pytest.raises(RuntimeError, match="confirm"):
        relearn.stage_final_test(plan)
    final = relearn.stage_final_test(plan, confirm=True)
    assert final["mode"] == r7["mode"] and set(final["test2019"]["counting"]) == {
        "calibration_count_groups.json", "calibration_count.json"}
    assert relearn.stage_final_test(plan, confirm=True)["finished_at"] == final["finished_at"]
    assert relearn.prepare(settings) == plan
