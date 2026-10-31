"""New research configs resolve native paths and preserve the accepted PDP setup."""

import os
import subprocess

import pytest
from conftest import REPO_ROOT
from test_run_exp import calls, main_calls, run

from pdp_helpers import main_args


@pytest.mark.parametrize(
    "name,backbone",
    [("EXP-B1", "resnet50"), ("EXP-B2", "convnextv2_base.fcmae_ft_in22k_in1k")],
)
def test_native_config_uses_locked_release_and_explicit_model(name, backbone):
    config = REPO_ROOT / "configs/exp/native" / f"{name}.sh"
    command = 'source "$CONFIG"; printf "%s\\0" "${ARGS[@]}"'
    result = subprocess.run(
        ["bash", "-ec", command],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CONFIG": str(config),
            "REPO": str(REPO_ROOT),
            "RAW_ROOT": "/fixture/raw",
            "RELEASE_ROOT": "/fixture/release",
            "RESOLUTION": "800",
            "EFFECTIVE_BATCH": "4",
        },
    )
    assert result.returncode == 0, result.stderr
    args = main_args(result.stdout.split("\0")[:-1])
    assert args.backbone == backbone
    assert args.image_size == args.max_image_size == 800
    assert args.train_img_dir == args.test_img_dir == "/fixture/raw"
    assert args.task_ann_dir == "/fixture/release/tasks/real"
    assert args.task_config == "/fixture/release/task_config.json"
    assert args.eff_batch_size == 4 and args.seed == 0 and args.n_classes == 225
    assert (
        args.require_kernel == args.tf32 == args.augment == args.pseudo_dedup_iou == 0
    )
    assert args.num_prompts == 100 and args.prompt_len == 10
    assert args.freeze == "backbone,encoder,decoder" and args.optim_groups == "prompt"


def test_native_runner_calls_real_entrypoint_with_resume_controls(tmp_path):
    result = run(
        tmp_path,
        config='source "$REPO/configs/exp/native/EXP-B2.sh"\n',
        STOP_AFTER_STEPS="1",
        VERIFY_RESUME="1",
        RAW_ROOT="/fixture/raw",
    )
    assert result.returncode == 0, result.stderr
    entries = calls(tmp_path)
    command = main_calls(entries)
    assert (
        len(command) == 1
        and "--stop_after_steps 1" in command[0]
        and "--verify_resume 1" in command[0]
    )
    assert not any("tools.eval" in line for line in entries)


def test_native_runner_passes_explicit_tf32_setting(tmp_path):
    result = run(
        tmp_path,
        config='source "$REPO/configs/exp/native/EXP-B1.sh"\n',
        TF32="1",
    )
    assert result.returncode == 0, result.stderr
    assert "--tf32 1" in main_calls(calls(tmp_path))[0]


def test_native_runner_uses_val_only_calibration_then_saved_policy(tmp_path):
    result = run(tmp_path, config='source "$REPO/configs/exp/native/EXP-B1.sh"\n', EVALUATE="1")
    assert result.returncode == 0, result.stderr
    entries = calls(tmp_path)
    counting = [line for line in entries if "tools.eval_count" in line]
    assert len(counting) == 2
    assert "--calibrate-only" in counting[0] and "--test-ann" not in counting[0]
    assert "--policy-in" in counting[1] and "--val-ann" not in counting[1]
    assert "--nms-grid 0.45" in counting[0]
    assert all("--oracle" not in line for line in entries)


def test_native_runner_reuses_an_existing_calibration_policy(tmp_path):
    policy = tmp_path / "locked.json"
    policy.write_text("{}")  # the CLI validates content; shell only chooses which mode to call
    result = run(tmp_path, config='source "$REPO/configs/exp/native/EXP-B2.sh"\n',
                 EVALUATE="1", CALIBRATION_POLICY=str(policy))
    assert result.returncode == 0, result.stderr
    counting = [line for line in calls(tmp_path) if "tools.eval_count" in line]
    assert len(counting) == 1 and "--policy-in" in counting[0]


def resolved_args(name, **env):
    config = REPO_ROOT / "configs/exp/native" / f"{name}.sh"
    result = subprocess.run(
        ["bash", "-ec", 'source "$CONFIG"; printf "%s\\0" "${ARGS[@]}"'],
        capture_output=True, text=True,
        env={**os.environ, "CONFIG": str(config), "REPO": str(REPO_ROOT), "RAW_ROOT": "/fixture/raw",
             "TASK_DIR": "/fixture/links", "TASK_CFG": "/fixture/eval/task_config.json",
             "SYNTH_ROOT": "/fixture/synth", **env},
    )
    return result, (main_args(result.stdout.split("\0")[:-1]) if result.returncode == 0 else None)


def test_full_syn_configs_follow_the_plan_table():
    """FULL-SYN F0.1 (plan section 3): B2 everywhere except the training images, the FSA start, F14 and I3."""
    _, b2 = resolved_args("EXP-B2")
    _, pdp = resolved_args("EXP-FULL-SYN", FSA_HF="/fixture/fsa/task_1/hf_model")
    _, fsa = resolved_args("EXP-FULL-SYN-FSA")
    for args in (pdp, fsa):
        assert args.train_img_dir == "/fixture/synth" and args.test_img_dir == "/fixture/raw"
        assert args.task_ann_dir == "/fixture/links" and args.task_config == "/fixture/eval/task_config.json"
        assert args.backbone == b2.backbone and args.n_classes == 225 and args.seed == 0 and args.augment == 0
    assert pdp.repo_name == "/fixture/fsa/task_1/hf_model"
    assert pdp.pseudo_dedup_iou == pdp.pseudo_gt_iou == 0.5 and b2.pseudo_dedup_iou == b2.pseudo_gt_iou == 0
    differ = {k for k in vars(b2) if getattr(pdp, k) != getattr(b2, k)}
    assert differ == {"train_img_dir", "repo_name", "pseudo_dedup_iou", "pseudo_gt_iou"}  # same TASK_DIR/TASK_CFG given
    assert fsa.use_prompts == fsa.local_query == 0 and fsa.pseudo == "none" and fsa.freeze == ""
    assert fsa.optim_groups == "detr" and fsa.save_hf == 1 and fsa.select_best_epoch == 1
    assert fsa.repo_name == b2.repo_name and fsa.backbone_pretrained_file == b2.backbone_pretrained_file


def test_full_syn_configs_refuse_missing_inputs():
    assert resolved_args("EXP-FULL-SYN-FSA", SYNTH_ROOT="")[0].returncode != 0
    assert "FSA_HF" in resolved_args("EXP-FULL-SYN")[0].stderr


def test_full_syn_m5_config_only_adds_feature_tuning():
    _, pdp = resolved_args("EXP-FULL-SYN", FSA_HF="/fixture/fsa/task_1/hf_model")
    _, m5 = resolved_args("EXP-FULL-SYN-M5", FSA_HF="/fixture/fsa/task_1/hf_model")
    assert {k for k in vars(pdp) if getattr(m5, k) != getattr(pdp, k)} == {"tune_after_task1", "feature_kd"}
    assert m5.tune_after_task1 == "stages_3,encoder" and m5.feature_kd == 1


def test_full_syn_rl_config_only_sets_the_class_count():
    _, pdp = resolved_args("EXP-FULL-SYN", FSA_HF="/fixture/fsa/task_1/hf_model")
    _, rl = resolved_args("EXP-FULL-SYN-RL", FSA_HF="/fixture/fsa/task_1/hf_model")
    assert {k for k in vars(pdp) if getattr(rl, k) != getattr(pdp, k)} == {"n_classes"} and rl.n_classes == 275
