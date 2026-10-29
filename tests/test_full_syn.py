"""FULL-SYN package (docs/project/full-syn-train-plan-2026-10-22.md): inputs, gates and stages."""

import hashlib
import json

import pytest
from conftest import REPO_ROOT

from autocheckout.full_syn import N_TASKS, link_tasks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toy_releases(root):
    """Synthetic release and evaluation set with the file names and manifests of the real ones."""
    release, eval_root = root / "synth", root / "eval"
    (release / "tasks/synth_v2").mkdir(parents=True)
    eval_root.mkdir()
    tasks = {}
    for t in range(1, N_TASKS + 1):
        files = {}
        for name in (f"train_task_{t}.json", f"train_task_{t}_gt_full.json"):
            path = release / "tasks/synth_v2" / name
            path.write_text(json.dumps({"name": name}))
            files[name] = sha(path)
        tasks[str(t)] = {"files": files}
    (release / "manifest.json").write_text(json.dumps({"tasks": tasks}))
    files = {}
    for name in [f"val_task_{t}.json" for t in range(1, N_TASKS + 1)] + ["val_full.json", "test_full.json"]:
        (eval_root / name).write_text(json.dumps({"name": name}))
        files[name] = {"sha256": sha(eval_root / name)}
    (eval_root / "manifest.json").write_text(json.dumps({"files": files}))
    return release, eval_root


def test_links_point_to_synthetic_train_and_real_eval_files_without_gt_full(tmp_path):
    release, eval_root = toy_releases(tmp_path)
    hashes = link_tasks(release, eval_root, tmp_path / "links")
    names = sorted(p.name for p in (tmp_path / "links").iterdir())
    assert len(names) == 2 * N_TASKS + 2 and not any("gt_full" in n for n in names)
    assert (tmp_path / "links/train_task_3.json").resolve() == (release / "tasks/synth_v2/train_task_3.json").resolve()
    assert (tmp_path / "links/val_task_3.json").resolve() == (eval_root / "val_task_3.json").resolve()
    assert hashes["test_full.json"] == sha(eval_root / "test_full.json")
    assert link_tasks(release, eval_root, tmp_path / "links") == hashes  # idempotent


def test_links_refuse_changed_files_and_foreign_files(tmp_path):
    release, eval_root = toy_releases(tmp_path)
    (release / "tasks/synth_v2/train_task_2.json").write_text("{}")
    with pytest.raises(ValueError, match="does not match its manifest"):
        link_tasks(release, eval_root, tmp_path / "links")

    release, eval_root = toy_releases(tmp_path / "again")
    (tmp_path / "again/links").mkdir()
    audit_file = release / "tasks/synth_v2/train_task_1_gt_full.json"
    (tmp_path / "again/links/train_task_1_gt_full.json").symlink_to(audit_file)
    with pytest.raises(ValueError, match="not FULL-SYN inputs"):
        link_tasks(release, eval_root, tmp_path / "again/links")


RELEASE = REPO_ROOT / "data/processed/rpc_synth_v2_v1"
EVAL = REPO_ROOT / "data/processed/rpc_eval_official_v1"


@pytest.mark.skipif(not (RELEASE / "manifest.json").is_file() or not (EVAL / "manifest.json").is_file(),
                    reason="synthetic v2 release or evaluation set not on this machine")
def test_released_files_match_their_manifests(tmp_path):
    hashes = link_tasks(RELEASE, EVAL, tmp_path / "links")
    assert len(hashes) == 2 * N_TASKS + 2


def test_subset_run_scores_the_old_test_images_with_the_val_policy(tmp_path):
    """F0.8: predictions on the full test, filtered to the B1/B2 test images, evaluate like a run made on them."""
    from autocheckout.full_syn import subset_run
    from autocheckout.io import md5_file, save_json
    from autocheckout.predictions import load_predictions, save_predictions
    from tests.coco_helpers import make_ann, make_gt, make_image, make_preds, three_task_config
    from tools import eval_count

    cfg = three_task_config()
    full = make_gt(images=[make_image(i) for i in (1, 2, 3)],
                   annotations=[make_ann(i, i, 0, [0, 0, 10, 10]) for i in (1, 2, 3)], num_labels=6)
    old = {**full, "images": full["images"][:2], "annotations": full["annotations"][:2]}
    for name, gt in (("val_full.json", full), ("test_full.json", full), ("old_test.json", old)):
        save_json(tmp_path / name, gt)
    run = tmp_path / "run"
    (run / "task_1").mkdir(parents=True)
    (run / "task_1/task_final.pth").write_bytes(b"weights")
    # image 3: wrong SKU
    rows = [(1, 0, 0, 0.9, [0, 0, 10, 10]), (2, 0, 0, 0.9, [0, 0, 10, 10]), (3, 0, 1, 0.9, [0, 0, 10, 10])]
    for split in ("val", "test"):
        meta = {"task_id": 1, "seen_classes": 2, "split": split, "ann_md5": md5_file(tmp_path / f"{split}_full.json")}
        save_predictions(run / f"task_1/pred_{split}.npz", make_preds(rows, meta))
    policy = eval_count.calibrate_run(run, tmp_path / "val_full.json", cfg)

    subset = subset_run(run, tmp_path / "old_test.json", tmp_path / "compare")
    assert sorted(set(load_predictions(subset / "task_1/pred_test.npz").image_id.tolist())) == [1, 2]
    full_result = eval_count.evaluate_policy(run, tmp_path / "test_full.json", cfg, policy)
    old_result = eval_count.evaluate_policy(subset, tmp_path / "old_test.json", cfg, policy)
    assert full_result["stages"][1]["test"]["overall"]["cAcc"] == pytest.approx(2 / 3)
    assert old_result["stages"][1]["test"]["overall"]["cAcc"] == pytest.approx(1.0)

    save_json(tmp_path / "other.json", {**full, "images": [make_image(9)], "annotations": []})
    with pytest.raises(ValueError, match="lacks images"):
        subset_run(run, tmp_path / "other.json", tmp_path / "bad")


@pytest.mark.parametrize("value,round_id,decision", [
    (0.728, 1, "go"), (0.7279, 1, "stop"), (0.60, 2, "stop"), (0.60, 3, "go"), (0.578, 3, "go"), (0.5779, 3, "stop"),
])
def test_task1_gate_follows_d17_and_d19(value, round_id, decision):
    from autocheckout.full_syn import task1_decision
    assert task1_decision(value, round_id)[1] == decision


def test_an_enforced_gate_stops_the_round_and_the_next_stages(tmp_path):
    from autocheckout.full_syn import GateStop, _finish, _gate, _require
    plan = dict(root=str(tmp_path), dry_run=False, campaign_id="c", round=1, release_config_sha256="x", input_sha256={},
                train_images=[1] * N_TASKS)
    gate = _gate(plan, "D18 FSA early check", "stop", "too low")
    with pytest.raises(GateStop, match="too low"):
        _finish(plan, "fsa", {"cAcc_m2": 0.1}, gate)
    with pytest.raises(GateStop):
        _require(plan, "fsa")
    assert "too low" in (tmp_path / "SUMMARY.md").read_text()


def test_fsa_runs_at_its_own_physical_batch_with_the_same_effective_batch(tmp_path):
    """FSA trains the whole model and runs out of the 4090's memory at the PDP batch of 8; it accumulates instead."""
    from autocheckout import full_syn
    settings = dict(campaign_id="dry", round=1, artifact_root=str(tmp_path), dry_run=True, batch_size=8,
                    effective_batch=8, fsa_batch_size=4, resolution=640, fsa_epochs=8, epochs=16)
    plan = full_syn.prepare(settings)
    fsa = full_syn._environment(plan, full_syn.FSA, tmp_path)
    pdp = full_syn._environment(plan, full_syn.PDP, tmp_path)
    assert (fsa["BATCH_SIZE"], fsa["EFFECTIVE_BATCH"]) == ("4", "8")
    assert (pdp["BATCH_SIZE"], pdp["EFFECTIVE_BATCH"]) == ("8", "8")
    with pytest.raises(ValueError, match="multiple of physical batch"):
        full_syn.prepare({**settings, "campaign_id": "odd", "fsa_batch_size": 3})
    same = full_syn.prepare({**settings, "campaign_id": "same", "fsa_batch_size": 8})
    assert full_syn._environment(same, full_syn.FSA, tmp_path)["BATCH_SIZE"] == "8"


@pytest.mark.slow
def test_dry_round_runs_every_stage_and_resumes_without_training_again(tmp_path):
    """The CPU fixture through smoke, G1-G4 (gates reported, not enforced); a second pass trains nothing and the
    test is not scored again."""
    from autocheckout import full_syn
    settings = dict(campaign_id="dry", round=1, artifact_root=str(tmp_path), dry_run=True, batch_size=1,
                    effective_batch=2, resolution=640, fsa_epochs=8, epochs=16, audit_images=4)
    plan = full_syn.prepare(settings)
    smoke = full_syn.stage_smoke(plan, steps=1)
    assert smoke["fsa"]["optimizer_steps"] == 2 and smoke["fsa"]["resume_verification"]["state_equal"]
    stages = [full_syn.stage_fsa, full_syn.stage_task1, full_syn.stage_tasks, full_syn.stage_final_test]
    first = [stage(plan) for stage in stages]
    assert first[0]["gate"]["decision"] in ("go", "stop") and not first[0]["gate"]["enforced"]
    run = tmp_path / "dry_run/dry/runs/EXP-FULL-SYN"
    for task in range(2, N_TASKS + 1):
        log = json.loads((run / f"task_{task}/task_log.json").read_text())
        assert {"cAcc_m1_groups", "pseudo_labels", "confusions_top20", "cAcc_by_level"} <= set(log)
    assert all((run / f"task_{t}/pred_test.npz").is_file() for t in range(1, N_TASKS + 1))
    assert "compare_b12" in first[3] and (tmp_path / "dry_run/dry/SUMMARY.md").is_file()

    again = full_syn.prepare(settings)
    assert full_syn.stage_fsa(again)["cAcc_m2"] == first[0]["cAcc_m2"]
    assert full_syn.stage_final_test(again)["finished_at"] == first[3]["finished_at"]
    logs = sorted((tmp_path / "dry_run/dry/attempts/fsa").glob("*/*_runner.log"), key=lambda p: p.stat().st_mtime)
    assert "done, skipped" in logs[-1].read_text()
    with pytest.raises(ValueError, match="new campaign_id"):
        full_syn.prepare({**settings, "audit_images": 8})  # a changed setting needs a new campaign (round)


@pytest.mark.slow
def test_m5_pilot_reuses_fsa_and_task1_of_its_seed_and_stops_after_task_2(tmp_path):
    """M5 (analysis 2026-10-29, section 7): FSA and task 1 come from the seed campaign untouched, task 2 trains the
    encoder with feature distillation; past task 2 and the final test are refused."""
    from autocheckout import full_syn
    base = dict(round=1, artifact_root=str(tmp_path), dry_run=True, batch_size=1, effective_batch=2,
                resolution=640, fsa_epochs=8, epochs=16, audit_images=4)
    seed = full_syn.prepare({**base, 'campaign_id': 'seed'})
    full_syn.stage_fsa(seed)
    full_syn.stage_task1(seed)
    seed_task1 = tmp_path / 'dry_run/seed/runs/EXP-FULL-SYN/task_1/task_final.pth'
    before = sha(seed_task1)

    with pytest.raises(ValueError, match='pdp_experiment'):
        full_syn.prepare({**base, 'campaign_id': 'bad', 'pdp_experiment': 'EXP-OTHER'})
    settings = {**base, 'campaign_id': 'm5', 'pdp_experiment': full_syn.M5,
                'seed_campaign': str(tmp_path / 'dry_run/seed')}
    plan = full_syn.prepare(settings)
    with pytest.raises(RuntimeError, match='stage_seed'):
        full_syn.stage_fsa(plan)
    seeded = full_syn.stage_seed(plan)
    assert seeded['task1']['cAcc_m2'] == load_json(tmp_path / 'dry_run/seed/stages/task1.json')['cAcc_m2']
    run = tmp_path / 'dry_run/m5/runs' / full_syn.M5
    assert (run / 'task_1').resolve() == seed_task1.parent.resolve()
    assert set(load_json(run / 'calibration_count_m2.json')['stages']) == {'1'}

    with pytest.raises(RuntimeError, match='up to task 2'):
        full_syn.stage_tasks(plan)
    report = full_syn.stage_tasks(plan, last=2)
    assert set(report['tasks']) == {2} and report['stage'] == 2
    log = (run / 'task_2/train.log').read_text()
    assert 'encoder' in log.split('Trainable parameters')[1].split('\n\n')[0]  # tuned at lr_old
    steps = [json.loads(line) for line in (run / 'task_2/step_metrics.jsonl').read_text().splitlines()]
    assert all('FKD' in step['losses'] for step in steps)
    assert sha(seed_task1) == before
    with pytest.raises(RuntimeError, match='after task 5'):
        full_syn.stage_final_test(plan)


def load_json(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("cacc,old_ap,decision", [
    (0.7664, 0.9496, "go"), (0.7663, 0.9496, "stop"), (0.80, 0.9396, "go"), (0.80, 0.9395, "retry"),
])
def test_pilot_verdict_needs_a_clear_gain_without_forgetting(cacc, old_ap, decision):
    from autocheckout.full_syn import pilot_verdict
    seed = {'cAcc_m2': 0.7364, 'AP50': {'mAP_P': 0.9496}}
    assert pilot_verdict({'cAcc_m2': cacc, 'AP50': {'mAP_P': old_ap}}, seed)[0] == decision
