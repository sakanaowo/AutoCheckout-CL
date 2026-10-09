"""Use the real PDP/Lightning path to check acceptance checkpoint verification."""

from types import SimpleNamespace

import full_baseline_acceptance as acceptance
import pytest
import torch

import engine
from datasets.coco_hug import CocoDetection
from main import make_train_loader
from pdp_helpers import make_toy_dataset, pdp_args, small_processor, use_tiny_detr


def test_full_worker_rejects_cpu_instead_of_claiming_cuda_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(AssertionError, match="requires a CUDA GPU"):
        acceptance.run_resolution({}, 640, tmp_path)


@pytest.mark.parametrize("accelerator", ["cpu", "gpu"])
def test_acceptance_checks_real_optimizer_and_lightning_resume(tmp_path, monkeypatch, accelerator):
    if accelerator == "gpu" and not torch.cuda.is_available():
        pytest.skip("CUDA device required for Lightning teardown/resume regression")
    use_tiny_detr(monkeypatch)
    data = make_toy_dataset(tmp_path / "data", n_train=4)
    args = pdp_args(
        tmp_path,
        "--batch_size",
        "1",
        "--n_gpus",
        "1",
        "--accelerator",
        accelerator,
        "--eff_batch_size",
        "1",
        "--epochs",
        "2",
        "--ckpt_every_minutes",
        "0",
    )
    args.output_dir = str(tmp_path / "acceptance")
    processor = small_processor()
    dataset = CocoDetection(str(data["images"]), str(data["tasks"] / "train_task_1.json"), processor)
    args.num_workers = 0
    loader = make_train_loader(dataset, args)
    module = engine.local_trainer(loader, None, None, args, SimpleNamespace(), 1)
    module.model.model.prompts.set_task_id(0)
    module.model.model.prompts.init_task_prompts()
    module.resume()
    # A nonempty prototype makes missing PDP state restoration detectable.
    module.update_class_cache(0, torch.ones(256))
    module.class_prototypes[0] = module.compute_class_prototypes(0)
    report = acceptance.exercise_baseline(module, loader, args, tmp_path / "acceptance")
    assert report["updated_classifier"]
    assert report["frozen_parameters_unchanged"]
    assert report["resume"]["restored_optimizer_scheduler_pdp_state"]
    assert report["resume"]["global_step"] == 2 * len(loader)
    assert report["reload_predictions_equal"]
    args.log_file.close()
