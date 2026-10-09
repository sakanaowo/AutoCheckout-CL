"""Full PDP acceptance worker; heavy ML imports happen only inside execution functions.

Notebook 02 calls this file in one subprocess per resolution. No dependency installation,
weight downloads, split generation, test-set batches or experiment training happen here.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from autocheckout.io import (  # noqa: E402 - direct CLI needs the repo root
    load_json,
    save_json,
)
from autocheckout.model_acceptance import now, validate_loading_info  # noqa: E402


from autocheckout.model_state import state_digest


def exercise_baseline(module, loader, args, output):
    """Real engine losses, optimizer step, two-pass predictions and Lightning resume.

    Tests exercise this function with the existing tiny CPU fixture; the notebook worker
    separately enforces full pretrained architecture and CUDA before calling it.
    """
    import pytorch_lightning as pl
    import torch
    from checkpointing import ResumeCheckpoint
    from inference import predict_batch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.accelerator == "gpu" else "cpu")
    module.to(device)
    batch = next(iter(loader))
    report = {
        "started_at": now(),
        "device": str(device),
        "physical_batch": args.batch_size,
        "precision": "32-true",
        "real_image_ids": [int(t["image_id"]) for t in batch["labels"]],
    }

    def frozen():
        return state_digest(
            {n: p for n, p in module.named_parameters() if not p.requires_grad}
        )

    before_frozen = frozen()
    classifier = next(
        p
        for n, p in module.named_parameters()
        if "class_embed" in n and n.endswith("weight") and p.requires_grad
    )
    classifier_before = classifier.detach().cpu().clone()
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    owned = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(owned) == len(set(owned)) and set(owned) == {
        id(p) for p in module.parameters() if p.requires_grad
    }
    report["optimizer_groups"] = [
        {"lr": g["lr"], "parameters": sum(p.numel() for p in g["params"])}
        for g in optimizer.param_groups
    ]
    module.train()
    optimizer.zero_grad(set_to_none=True)
    begin = time.perf_counter()
    loss, losses = module.common_step(batch, 0)
    assert {"loss_ce", "loss_bbox", "loss_giou", "loss_ddl", "query_loss"} <= set(
        losses
    )
    assert loss.requires_grad and torch.isfinite(loss)
    assert all(torch.isfinite(v).all() for v in losses.values())
    loss.backward()
    module.on_after_backward()
    gradients = {
        n: float(p.grad.norm())
        for n, p in module.named_parameters()
        if p.grad is not None
    }
    assert all(
        torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None
    )
    for token in ["class_embed", "query_tf", "shared_p_", "private_p_"]:
        assert any(
            token in n and v > 0 for n, v in gradients.items()
        ), f"No gradient: {token}"
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in module.parameters() if p.requires_grad], 0.1
    )
    assert torch.isfinite(norm)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    assert not torch.equal(classifier_before, classifier.detach().cpu())
    assert frozen() == before_frozen
    report.update(
        total_loss=float(loss.detach()),
        losses={k: float(v.detach()) for k, v in losses.items()},
        gradient_norm=float(norm),
        nonzero_gradient_tensors=sum(v > 0 for v in gradients.values()),
        updated_classifier=True,
        manual_step_seconds=time.perf_counter() - begin,
    )
    del loss, losses, configured, optimizer, classifier, classifier_before
    module.optimizer = None
    module.lr_scheduler = None
    module.zero_grad(set_to_none=True)
    gc.collect()

    def predictions():
        module.eval()
        result = predict_batch(
            module.model,
            batch["pixel_values"].to(device),
            batch["pixel_mask"].to(device),
            torch.stack([t["orig_size"] for t in batch["labels"]]).to(device),
            use_prompts=bool(args.use_prompts),
            local_query=bool(args.local_query),
            seen_classes=module.seen_classes,
        )
        assert all(
            torch.isfinite(r[k]).all() for r in result for k in ["scores", "boxes"]
        )
        return [{k: v.cpu() for k, v in row.items()} for row in result]

    class PlannedInterruption(Exception):
        pass

    class Interrupt(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if trainer.global_step == 1:
                # ResumeCheckpoint runs first, at the start of the next batch.
                assert (output / "last.ckpt").exists()
                raise PlannedInterruption(
                    "Intentional stop after the first completed optimizer step"
                )

    def lightning(callbacks):
        return pl.Trainer(
            accelerator=args.accelerator,
            devices=1,
            precision="32-true",
            max_epochs=2,
            accumulate_grad_batches=1,
            gradient_clip_val=0.1,
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )

    first = lightning([ResumeCheckpoint(str(output), every_minutes=0), Interrupt()])
    try:
        first.fit(module, train_dataloaders=loader)
    except PlannedInterruption:
        pass
    else:
        raise AssertionError("The controlled interruption did not occur")
    # Lightning's exception teardown moves the module to CPU. Put it back on the
    # requested device before the standalone reference prediction, outside fit().
    module.to(device)
    reference = predictions()
    checkpoint_path = output / "last.ckpt"
    saved = torch.load(checkpoint_path, map_location="cpu")
    assert saved["global_step"] == 1 and saved["optimizer_states"][0]["state"]
    expected = {
        key: state_digest(saved[key])
        for key in ["state_dict", "optimizer_states", "lr_schedulers", "pdp_state"]
    }
    saved_epoch, saved_step = saved["epoch"], saved["global_step"]
    del saved, first
    # Corrupt live trainable weights and memory: verification cannot pass using leftover state.
    with torch.no_grad():
        for p in module.parameters():
            if p.requires_grad:
                p.zero_()
    module.class_query_cache = {k: [] for k in module.class_query_cache}
    module.class_prototypes = {
        k: torch.zeros_like(v) for k, v in module.class_prototypes.items()
    }
    module.class_cache_count = {k: 0 for k in module.class_cache_count}
    module.batch_counter = -1

    restored = {}

    class VerifyRestore(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            assert (
                trainer.global_step == saved_step
                and trainer.current_epoch == saved_epoch
            )
            assert state_digest(pl_module.state_dict()) == expected["state_dict"]
            assert (
                state_digest([o.state_dict() for o in trainer.optimizers])
                == expected["optimizer_states"]
            )
            assert (
                state_digest(
                    [c.scheduler.state_dict() for c in trainer.lr_scheduler_configs]
                )
                == expected["lr_schedulers"]
            )
            memory = {}
            pl_module.on_save_checkpoint(memory)
            assert state_digest(memory["pdp_state"]) == expected["pdp_state"]
            actual = predictions()
            for ours, ref in zip(actual, reference, strict=True):
                for key in ref:
                    if ref[key].is_floating_point():
                        torch.testing.assert_close(
                            ours[key], ref[key], rtol=1e-5, atol=1e-5
                        )
                    else:
                        assert torch.equal(ours[key], ref[key])
            pl_module.train()
            restored.update(
                restored_optimizer_scheduler_pdp_state=True,
                epoch=saved_epoch,
                restored_step=saved_step,
                predictions_equal=True,
            )

    second = lightning(
        [ResumeCheckpoint(str(output), every_minutes=0), VerifyRestore()]
    )
    second.fit(module, train_dataloaders=loader, ckpt_path=str(checkpoint_path))
    assert restored and second.global_step == 2 * len(loader)
    assert frozen() == before_frozen
    # Both checkpoint formats are retained; task_final is weights/memory, not full resume state.
    module.save_task_final(str(output / "task_final.pth"))
    module.model.config.save_pretrained(output / "resolved_model_config")
    module.processor.save_pretrained(output / "processor")
    restored["global_step"] = second.global_step
    report.update(
        resume=restored,
        reload_predictions_equal=True,
        frozen_parameters_unchanged=True,
        frozen_sha256=before_frozen,
        optimizer_steps=1 + second.global_step,
        prototype_counts={str(k): v for k, v in module.class_cache_count.items()},
        finished_at=now(),
    )
    return report


def run_resolution(config, resolution, output):
    from types import SimpleNamespace
    from unittest.mock import patch

    import engine
    import main as pdp_main
    import models.modeling_deformable_detr as detr
    import pytorch_lightning as pl
    import torch
    from datasets.coco_hug import CocoDetection
    from models.configuration_deformable_detr import DeformableDetrConfig
    from models.image_processing_deformable_detr import DeformableDetrImageProcessor
    from torch.utils.data import Subset

    assert torch.cuda.is_available(), "This full-baseline notebook requires a CUDA GPU"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pl.seed_everything(config["seed"], workers=True)
    weights = Path(config["pretrained_dir"])
    profile = DeformableDetrConfig.from_pretrained(str(weights), local_files_only=True)
    expected = {
        "backbone": "resnet50",
        "encoder_layers": 6,
        "decoder_layers": 6,
        "num_queries": 300,
        "d_model": 256,
        "num_feature_levels": 4,
        "encoder_ffn_dim": 1024,
        "decoder_ffn_dim": 1024,
    }
    assert all(
        getattr(profile, k) == v for k, v in expected.items()
    ), "Need the full ResNet50 6/6 checkpoint"
    assert (
        not profile.two_stage
        and not profile.with_box_refine
        and profile.use_timm_backbone
    )
    # The detector snapshot contains the backbone; do not ask timm to download it again.
    profile.use_pretrained_backbone = False
    native = detr.MultiScaleDeformableAttention is not None
    assert native or not config["require_custom_kernel"], str(detr.KERNEL_LOAD_ERROR)
    profile.disable_custom_kernels = not native
    if native:
        # Reuse the existing CUDA forward/backward equivalence test, excluding its speed benchmark.
        sys.path.insert(0, str(REPO / "tests"))
        from test_pdp_f11_kernel import test_kernel_matches_pytorch_forward_and_backward

        test_kernel_matches_pytorch_forward_and_backward()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    args = pdp_main.get_args_parser().parse_args(
        [
            "--task_config",
            str(Path(config["release_root"]) / "task_config.json"),
            "--n_classes",
            "225",
            "--repo_name",
            str(weights),
            "--n_tasks",
            "5",
            "--batch_size",
            str(config["batch_size"]),
            "--eff_batch_size",
            str(config["batch_size"]),
            "--num_workers",
            "0",
            "--n_gpus",
            "1",
            "--accelerator",
            "gpu",
            "--epochs",
            "2",
            "--tf32",
            "0",
            "--augment",
            "0",
            "--use_prompts",
            "1",
            "--local_query",
            "1",
            "--num_prompts",
            "100",
            "--prompt_len",
            "10",
            "--lambda_query",
            "0.1",
            "--ddl_lambda",
            "0.15",
            "--query_loss_grad",
            "1",
            "--freeze",
            "backbone,encoder,decoder",
            "--new_params",
            "class_embed,prompts",
            "--optim_groups",
            "prompt",
            "--freeze_shared_after_task1",
            "0",
            "--pseudo_dedup_iou",
            "0",
            "--lr",
            "0.0001",
            "--lr_old",
            "0.00001",
            "--output_dir",
            str(output),
        ]
    )
    pdp_main.setup_task_info(args)
    processor = DeformableDetrImageProcessor.from_pretrained(
        str(weights),
        local_files_only=True,
        size={"shortest_edge": resolution, "longest_edge": resolution},
    )
    ann = Path(config["release_root"]) / "tasks/real/train_task_1.json"
    dataset = CocoDetection(config["raw_root"], str(ann), processor)
    ids = config["image_ids"]
    subset = Subset(dataset, [dataset.ids.index(i) for i in ids])
    subset.collate_fn = dataset.collate_fn
    loader = pdp_main.make_train_loader(subset, args)
    assert len(loader) >= 2, "Need at least two batches for interruption/resume"
    probe = next(iter(loader))
    assert tuple(probe["pixel_values"].shape[-2:]) == (resolution, resolution)
    assert all(
        ((t["class_labels"] >= 0) & (t["class_labels"] < 100)).all()
        for t in probe["labels"]
    )
    loading = {}
    original_load = detr.DeformableDetrForObjectDetection.from_pretrained

    def audited_load(*a, **kw):
        model, info = original_load(
            *a, **kw, local_files_only=True, output_loading_info=True
        )
        save_json(output / "pretrained_loading_info.json", info, indent=2)
        counters = [
            f"{name}.num_batches_tracked"
            for name, layer in model.named_modules()
            if isinstance(layer, detr.DeformableDetrFrozenBatchNorm2d)
        ]
        validate_loading_info(info, frozen_batch_norm_counters=counters)
        ignored = [n for n in info.get("unexpected_keys", []) if n in counters]
        save_json(output / "ignored_frozen_batch_norm_counters.json", ignored, indent=2)
        loading.update(info)
        loading["ignored_frozen_batch_norm_counters"] = ignored
        return model

    with (output / "model.log").open("w") as model_log:
        args.log_file = model_log
        # Scoped adapters only change full config source and request the HF loading report.
        # No tiny test factory; engine.local_trainer/common_step/optimizer remain the real code.
        with (
            patch.object(engine, "DeformableDetrConfig", lambda: profile),
            patch.object(
                engine.DeformableDetrForObjectDetection,
                "from_pretrained",
                side_effect=audited_load,
            ),
        ):
            module = engine.local_trainer(
                loader, None, None, args, SimpleNamespace(), task_id=1
            )
        module.processor = processor
        module.model.model.prompts.set_task_id(0)
        module.resume()
        module.model.model.prompts.init_task_prompts()
        save_json(
            output / "args.json",
            {k: v for k, v in vars(args).items() if k != "log_file"},
            indent=2,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        report = exercise_baseline(module, loader, args, output)
        fallback = bool(detr._kernel_fallback_logged)
        if config["require_custom_kernel"]:
            assert (
                not fallback
            ), "Custom kernel loaded but failed during the full model run"
        report.update(
            resolution=resolution,
            architecture=expected,
            pretrained=True,
            loading_info=loading,
            gpu_name=torch.cuda.get_device_name(),
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            cuda_runtime=torch.version.cuda,
            custom_kernel_loaded=native,
            custom_kernel_accepted=native and not fallback,
            runtime_fallback=fallback,
            kernel_forward_backward_equivalence=native,
            kernel_load_error=str(detr.KERNEL_LOAD_ERROR),
            duration_seconds=time.perf_counter() - begin,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resolution", type=int, choices=[640, 800], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MPLBACKEND="Agg"
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(output.parent / "torch_extensions")
    # Default build concurrency is bounded for the user's development machine.
    os.environ.setdefault("MAX_JOBS", "2")
    record = {"status": "RUNNING", "started_at": now(), "resolution": args.resolution}
    save_json(output / "result.json", record, indent=2)
    try:
        record.update(
            run_resolution(load_json(args.config), args.resolution, output),
            status="PASS",
        )
    except BaseException as error:
        import torch

        record.update(
            status=(
                "OOM"
                if isinstance(error, torch.cuda.OutOfMemoryError)
                else "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL"
            ),
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        if torch.cuda.is_available():
            record.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        traceback.print_exc()
    record["finished_at"] = now()
    save_json(output / "result.json", record, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["status"] == "PASS" else 2 if record["status"] == "OOM" else 1


if __name__ == "__main__":
    raise SystemExit(main())
