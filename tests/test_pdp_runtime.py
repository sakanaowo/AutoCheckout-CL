"""The real CLI must retain model/processor identity across process restarts."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import REPO_ROOT

from autocheckout.predictions import load_predictions

import main as pdp_main
from models.configuration_deformable_detr import DeformableDetrConfig
from pdp_helpers import TINY_DETR, main_args, make_toy_dataset, write_task_config


def test_cli_runs_outside_repo_without_installing_package_or_setting_pythonpath(
    tmp_path,
):
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "pdp/main.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--backbone" in result.stdout and "--image_size" in result.stdout


@pytest.mark.parametrize(
    "physical,effective,devices", [(3, 4, 1), (2, 1, 1), (1, 3, 2), (0, 4, 1)]
)
def test_effective_batch_cannot_be_silently_rounded(physical, effective, devices):
    args = main_args(
        [
            "--batch_size",
            str(physical),
            "--eff_batch_size",
            str(effective),
            "--n_gpus",
            str(devices),
            "--accelerator",
            "cpu",
        ]
    )
    with pytest.raises(ValueError, match="batch"):
        pdp_main.make_pl_trainer(args)


def test_cli_resume_and_predict_rebuild_saved_architecture_without_source_weights(
    tmp_path,
):
    data = make_toy_dataset(tmp_path / "data", sizes=(3,), n_train=4, n_val=2, n_test=2)
    task_config = write_task_config(tmp_path / "tasks.json", sizes=(3,), reserved=0)
    profile = tmp_path / "profile"
    DeformableDetrConfig(**TINY_DETR).save_pretrained(profile)
    output = tmp_path / "run"
    base = [
        sys.executable,
        str(REPO_ROOT / "pdp/main.py"),
        "--task_config",
        str(task_config),
        "--n_classes",
        "4",
        "--n_tasks",
        "1",
        "--repo_name",
        "",
        "--use_prompts",
        "1",
        "--local_query",
        "1",
        "--num_prompts",
        "4",
        "--prompt_len",
        "2",
        "--lambda_query",
        "0.1",
        "--freeze",
        "backbone,encoder,decoder",
        "--new_params",
        "class_embed,prompts",
        "--accelerator",
        "cpu",
        "--n_gpus",
        "1",
        "--batch_size",
        "1",
        "--eff_batch_size",
        "2",
        "--num_workers",
        "0",
        "--epochs",
        "2",
        "--eval_epochs",
        "1",
        "--shuffle",
        "0",
        "--output_dir",
        str(output),
        "--train_img_dir",
        str(data["images"]),
        "--test_img_dir",
        str(data["images"]),
        "--task_ann_dir",
        str(data["tasks"]),
        "--verify_resume",
        "1",
    ]
    env = {
        **os.environ,
        "USE_TF": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }

    def call(extra):
        return subprocess.run(
            base + extra,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    interrupted = call(
        [
            "--model_config",
            str(profile),
            "--image_size",
            "96",
            "--stop_after_steps",
            "1",
        ]
    )
    assert interrupted.returncode == 75, interrupted.stdout + interrupted.stderr
    task = output / "task_1"
    saved = torch.load(task / "last.ckpt", map_location="cpu")
    assert saved["global_step"] == 1
    assert saved["runtime"]["model_config"]["backbone"] == "resnet18"
    assert saved["runtime"]["processor_config"]["size"] == {
        "shortest_edge": 96,
        "longest_edge": 96,
    }
    assert not (task / "task_final.pth").exists()
    del saved
    (
        profile / "config.json"
    ).unlink()  # restart must use checkpoint metadata, not original bootstrap config

    conflict = call(["--image_size", "128"])
    assert conflict.returncode != 0 and "resolution" in conflict.stderr.lower()
    batch_conflict = call(["--eff_batch_size", "4"])
    assert (
        batch_conflict.returncode != 0
        and "training contract" in batch_conflict.stderr.lower()
    )
    resumed = call([])
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    sessions = json.loads((task / "run_info.json").read_text())["sessions"]
    done = sessions[-1]
    assert done["optimizer_steps"] == 4 and done["optimizer_steps_this_session"] == 3
    assert done["effective_batch"] == 2 and done["accumulate_grad_batches"] == 2
    assert done["resume_verification"]["state_equal"] is True
    assert done["resume_verification"]["restored_step"] == 1
    final = torch.load(task / "task_final.pth", map_location="cpu")
    assert final["runtime"]["model_config"]["backbone"] == "resnet18"
    assert final["runtime"]["processor_config"]["size"]["shortest_edge"] == 96
    assert not (task / "last.ckpt").exists()
    reference = load_predictions(task / "pred_val.npz")
    assert reference.meta["processor_size"] == {"shortest_edge": 96, "longest_edge": 96}
    assert reference.meta["coordinates"] == "native_pixels"
    predicted = call(["--predict_only", "1"])
    assert predicted.returncode == 0, predicted.stdout + predicted.stderr
    actual = load_predictions(task / "pred_val.npz")
    for column in ("image_id", "query", "label", "score", "boxes"):
        np.testing.assert_allclose(
            getattr(actual, column), getattr(reference, column), rtol=1e-5, atol=1e-4
        )
