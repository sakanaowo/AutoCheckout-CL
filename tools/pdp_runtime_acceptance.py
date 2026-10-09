"""Run notebook 04 cases through the actual shell runner and independent CLI processes.

No monkeypatches, dependency installs, downloads or long training. Inputs are audited
subsets prepared by the notebook; metrics on those subsets are smoke diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from autocheckout.io import load_json, save_json
from autocheckout.model_acceptance import now
from autocheckout.model_state import state_digest

REPO = Path(__file__).resolve().parents[1]


def run_case(config, output):
    import numpy as np
    import torch

    from autocheckout.predictions import load_predictions

    assert torch.cuda.is_available(), "CUDA is required for CLI/runner acceptance"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    experiment = config["experiment"]
    task = output / experiment / "task_1"
    environment = {
        **os.environ,
        "USE_TF": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHON": sys.executable,
        "RUNS": str(output),
        "EXP": experiment,
        "RAW_ROOT": config["raw_root"],
        "RELEASE_ROOT": config["release_root"],
        "TASK_DIR": config["subset_dir"],
        "TASK_CFG": config["task_config"],
        "DETECTOR_DIR": config["detector_dir"],
        "BACKBONE_WEIGHTS": config["backbone_file"],
        "RESOLUTION": str(config["resolution"]),
        "BATCH_SIZE": "1",
        "EFFECTIVE_BATCH": "2",
        "NUM_WORKERS": "0",
        "EPOCHS": "2",
        "N_TASKS": "1",
        "SHUFFLE": "0",
        "VERIFY_RESUME": "1",
        "STOP_AFTER_STEPS": "1",
        "REQUIRE_KERNEL": "0",
    }
    environment.pop(
        "PYTHONPATH", None
    )  # runner/entrypoint must establish its own import path
    command = [
        "bash",
        str(REPO / "scripts/run_exp.sh"),
        str(REPO / "configs/exp/native" / f"{experiment}.sh"),
    ]
    invocations = []

    def run(label, expected=0):
        with (output / f"{label}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=REPO,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1200,
            )
        invocations.append(
            {"label": label, "command": command, "exit_code": result.returncode}
        )
        save_json(output / "invocations.json", invocations, indent=2)
        assert (
            result.returncode == expected
        ), f'{label}: exit {result.returncode}; see {output / (label + ".log")}'

    run("01_interrupted", expected=75)
    assert not (task / "task_final.pth").exists() and (task / "last.ckpt").exists()
    saved = torch.load(task / "last.ckpt", map_location="cpu")
    assert saved["global_step"] == 1
    expected_backbone = (
        "resnet50" if experiment == "EXP-B1" else "convnextv2_base.fcmae_ft_in22k_in1k"
    )
    assert saved["runtime"]["model_config"]["backbone"] == expected_backbone
    size = {"shortest_edge": config["resolution"], "longest_edge": config["resolution"]}
    assert saved["runtime"]["processor_config"]["size"] == size
    frozen = {
        name.removeprefix("model."): value
        for name, value in saved["state_dict"].items()
        if set(name.split(".")) & {"backbone", "encoder", "decoder"}
    }
    frozen_before = state_digest(frozen)
    interrupted_state = {
        key: state_digest(saved[key])
        for key in ["state_dict", "optimizer_states", "lr_schedulers", "pdp_state"]
    }
    save_json(
        output / "interrupted_state.json",
        {"global_step": 1, "hashes": interrupted_state, "frozen_sha256": frozen_before},
        indent=2,
    )
    shutil.copy2(task / "loading_report.json", output / "initial_loading_report.json")
    initial = load_json(output / "initial_loading_report.json")
    assert initial["mode"] == (
        "detector" if experiment == "EXP-B1" else "cross_backbone"
    )
    if experiment == "EXP-B2":
        assert initial["backbone_preserved_sha256"] and initial["transfer"]["load_keys"]
    del saved, frozen
    gc.collect()

    # A new process must reconstruct from checkpoint metadata with unavailable bootstrap paths.
    environment.update(
        STOP_AFTER_STEPS="0",
        DETECTOR_DIR=str(output / "unavailable_detector"),
        BACKBONE_WEIGHTS=str(output / "unavailable_backbone.safetensors"),
    )
    run("02_resumed")
    sessions = load_json(task / "run_info.json")["sessions"]
    report = sessions[-1]
    assert len(sessions) == 2 and report["mode"] == "train"
    assert report["train_batches"] == 4 and report["effective_batch"] == 2
    assert report["accumulate_grad_batches"] == 2 and report["optimizer_steps"] == 4
    assert report["optimizer_steps_this_session"] == 3
    assert (
        report["resume_verification"]["state_equal"]
        and report["resume_verification"]["restored_step"] == 1
    )
    assert report["processor_size"] == size and report["backbone"] == expected_backbone
    assert not (task / "last.ckpt").exists()
    final = torch.load(task / "task_final.pth", map_location="cpu")
    assert (
        state_digest(
            {
                name: value
                for name, value in final["model"].items()
                if set(name.split(".")) & {"backbone", "encoder", "decoder"}
            }
        )
        == frozen_before
    )
    assert final["runtime"]["model_config"]["backbone"] == expected_backbone
    assert final["runtime"]["processor_config"]["size"] == size
    del final
    gc.collect()
    losses = []
    for path in task.glob("lightning_logs/version_*/metrics.csv"):
        with path.open() as handle:
            losses.extend(
                float(row["tr"]) for row in csv.DictReader(handle) if row.get("tr")
            )
    assert losses and all(math.isfinite(loss) for loss in losses)

    reference = {
        split: load_predictions(task / f"pred_{split}.npz") for split in ["val", "test"]
    }
    for split, pred in reference.items():
        assert (
            pred.meta["processor_size"] == size
            and pred.meta["backbone"] == expected_backbone
        )
        assert (
            pred.meta["coordinates"] == "native_pixels"
            and pred.meta["seen_classes"] == 100
        )
        ids = {
            row["id"]
            for row in load_json(Path(config["subset_dir"]) / f"{split}_full.json")[
                "images"
            ]
        }
        assert set(pred.image_id.tolist()) == ids and len(pred) == 100 * len(ids)
        assert np.isfinite(pred.boxes).all() and np.isfinite(pred.score).all()
    shutil.copy2(task / "pred_test.npz", output / "pred_test_before_reload.npz")
    (
        task / "pred_test.npz"
    ).unlink()  # only this newly-created smoke artifact; triggers runner prediction recovery
    run("03_predict_only")
    for split, previous in reference.items():
        current = load_predictions(task / f"pred_{split}.npz")
        for key in ["image_id", "query", "label", "score", "boxes"]:
            np.testing.assert_allclose(
                getattr(current, key), getattr(previous, key), rtol=1e-5, atol=1e-4
            )
    sessions = load_json(task / "run_info.json")["sessions"]
    assert len(sessions) == 3 and sessions[-1]["mode"] == "predict"
    run("04_completed_skip")
    assert len(load_json(task / "run_info.json")["sessions"]) == 3

    native = verify_native_boxes(task, config, reference["val"])
    return {
        "status": "PASS",
        "experiment": experiment,
        "resolution": config["resolution"],
        "gpu": torch.cuda.get_device_name(0),
        "physical_batch": 1,
        "effective_batch": 2,
        "optimizer_steps": 4,
        "resume_verification": report["resume_verification"],
        "predict_only_equal": True,
        "completed_task_skipped": True,
        "bootstrap_paths_unavailable_on_restart": True,
        "frozen_parameters_unchanged": True,
        "train_losses": losses,
        "peak_training_gpu_gib": max(s.get("peak_gpu_memory_gb", 0) for s in sessions),
        "native_coordinates": native,
        "native_kernel_accepted": False,
        "full_training_ready": False,
        "finished_at": now(),
    }


def verify_native_boxes(task, config, exported):
    """Rebuild from the final checkpoint and compare unit-coordinate boxes with CLI export."""
    import torch

    sys.path.insert(0, str(REPO / "pdp"))
    from datasets.coco_hug import CocoDetection
    from inference import predict_batch
    from models.configuration_deformable_detr import DeformableDetrConfig
    from models.image_processing_deformable_detr import DeformableDetrImageProcessor
    from models.modeling_deformable_detr import DeformableDetrForObjectDetection

    final = torch.load(task / "task_final.pth", map_location="cpu")
    model_config = DeformableDetrConfig.from_dict(final["runtime"]["model_config"])
    model_config.use_pretrained_backbone = False
    model_config.backbone_pretrained_file = None
    model = DeformableDetrForObjectDetection(model_config)
    model.load_state_dict(final["model"], strict=True)
    model.model.prompts.set_task_id(0)
    model.to("cuda").eval()
    processor = DeformableDetrImageProcessor.from_dict(
        final["runtime"]["processor_config"]
    )
    dataset = CocoDetection(
        config["raw_root"], str(Path(config["subset_dir"]) / "val_full.json"), processor
    )
    batch = dataset.collate_fn([dataset[0]])
    target = batch["labels"][0]
    height, width = target["orig_size"].tolist()
    assert max(height, width) > config["resolution"]
    assert max(batch["pixel_values"].shape[-2:]) == config["resolution"]
    with torch.no_grad():
        unit = predict_batch(
            model,
            batch["pixel_values"].to("cuda"),
            batch["pixel_mask"].to("cuda"),
            torch.ones((1, 2), device="cuda"),
            use_prompts=True,
            local_query=True,
            seen_classes=100,
        )[0]
    selected = exported.image_id == int(target["image_id"])
    torch.testing.assert_close(
        torch.from_numpy(exported.boxes[selected]),
        unit["boxes"].cpu() * torch.tensor([width, height, width, height]),
        rtol=1e-5,
        atol=1e-3,
    )
    assert torch.equal(torch.from_numpy(exported.label[selected]), unit["labels"].cpu())
    return {
        "verified": True,
        "image_id": int(target["image_id"]),
        "orig_size": [height, width],
        "processed_size": list(batch["pixel_values"].shape[-2:]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "started_at": now()}
    save_json(args.output / "result.json", report, indent=2)
    try:
        report.update(run_case(load_json(args.config), args.output))
    except BaseException as error:
        report.update(
            status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL",
            error=str(error),
            traceback=traceback.format_exc(),
            finished_at=now(),
        )
        raise
    finally:
        save_json(args.output / "result.json", report, indent=2)


if __name__ == "__main__":
    main()
