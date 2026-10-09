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
