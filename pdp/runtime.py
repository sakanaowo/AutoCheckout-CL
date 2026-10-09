"""Model/processor construction and the portable runtime contract saved in checkpoints."""

from copy import deepcopy
from pathlib import Path

from autocheckout.io import save_json, md5_file
from autocheckout.model_acceptance import validate_loading_info
from models.backbones import plan_detector_transfer
from models.configuration_deformable_detr import DeformableDetrConfig


def accumulation_steps(args):
    batch, devices, effective = args.batch_size, args.n_gpus, args.eff_batch_size
    if min(batch, devices, effective) < 1 or effective % (batch * devices):
        raise ValueError(
            "effective batch must be a positive multiple of physical batch * devices"
        )
    return effective // (batch * devices)


def checkpoint_runtime(path):
    if not path:
        return None
    import torch

    return torch.load(path, map_location="cpu").get("runtime")


def validate_runtime(args, runtime):
    if not runtime:
        return
    config, size = runtime["model_config"], runtime["processor_config"]["size"]
    if getattr(args, "backbone", "") and args.backbone != config["backbone"]:
        raise ValueError("Requested backbone conflicts with checkpoint runtime")
    shortest = getattr(args, "image_size", None)
    longest = getattr(args, "max_image_size", None)
    if (
        shortest is not None
        and shortest != size["shortest_edge"]
        or (longest is not None and longest != size["longest_edge"])
    ):
        raise ValueError("Requested resolution conflicts with checkpoint processor")
    for arg, field in [
        ("n_classes", "num_labels"),
        ("use_prompts", "use_prompts"),
        ("local_query", "local_query"),
        ("num_prompts", "num_prompts"),
        ("prompt_len", "prompt_len"),
        ("use_shared", "use_shared_pool"),
        ("use_private", "use_private_pool"),
        ("task_num_classes", "task_num_classes"),
    ]:
        expected = (
            len(config["id2label"]) if field == "num_labels" else config.get(field)
        )
        if expected is not None and getattr(args, arg) != expected:
            raise ValueError(f"Requested {arg} conflicts with checkpoint runtime")


def model_config(args, default_factory):
    saved = getattr(args, "_runtime_snapshot", None)
    validate_runtime(args, saved)
    if saved:
        config = DeformableDetrConfig.from_dict(deepcopy(saved["model_config"]))
        config.use_pretrained_backbone = False
        config.backbone_pretrained_file = None
        return config
    if getattr(args, "model_config", ""):
        config = DeformableDetrConfig.from_pretrained(args.model_config)
    elif getattr(args, "backbone", "") and args.repo_name:
        config = DeformableDetrConfig.from_pretrained(args.repo_name)
    else:
        config = default_factory()
    if getattr(args, "backbone", ""):
        config.backbone = args.backbone
        config.backbone_feature_info = None
        config.use_pretrained_backbone = bool(args.backbone_pretrained_file)
        config.backbone_pretrained_file = args.backbone_pretrained_file or None
    if args.repo_name and not getattr(args, "backbone_pretrained_file", ""):
        config.use_pretrained_backbone = (
            False  # detector already contains its own backbone
        )
    return config


def processor(args, factory):
    saved = getattr(args, "_runtime_snapshot", None)
    validate_runtime(args, saved)
    if saved:
        from models.image_processing_deformable_detr import DeformableDetrImageProcessor

        return DeformableDetrImageProcessor.from_dict(
            deepcopy(saved["processor_config"])
        )
    result = factory.from_pretrained(args.repo_name) if args.repo_name else factory()
    shortest, longest = getattr(args, "image_size", None), getattr(
        args, "max_image_size", None
    )
    if shortest is not None or longest is not None:
        shortest = shortest if shortest is not None else result.size["shortest_edge"]
        longest = longest if longest is not None else shortest
        if shortest <= 0 or longest < shortest:
            raise ValueError("resolution requires 0 < image_size <= max_image_size")
        result.size = {"shortest_edge": shortest, "longest_edge": longest}
    return result


def _audit_load(info, model):
    from models.modeling_deformable_detr import DeformableDetrFrozenBatchNorm2d

    counters = [
        f"{n}.num_batches_tracked"
        for n, m in model.named_modules()
        if isinstance(m, DeformableDetrFrozenBatchNorm2d)
    ]
    validate_loading_info(info, frozen_batch_norm_counters=counters)


def build_model(args, config, model_class):
    kwargs = dict(default=not args.mask_gradients, log_file=args.log_file)
    saved = getattr(args, "_runtime_snapshot", None)
    if saved or not args.repo_name:
        return model_class(config, **kwargs), {
            "mode": "checkpoint" if saved else "random",
            "classifier_loaded": bool(saved),
        }
    source_config = DeformableDetrConfig.from_pretrained(args.repo_name)
    if source_config.backbone == config.backbone:
        model, info = model_class.from_pretrained(
            args.repo_name,
            config=config,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
            **kwargs,
        )
        _audit_load(info, model)
        return model, {
            "mode": "detector",
            "loading_info": info,
            "classifier_loaded": source_config.num_labels == config.num_labels,
        }
    if not getattr(args, "backbone_pretrained_file", ""):
        raise ValueError("Changing backbone requires --backbone_pretrained_file")
    from autocheckout.model_state import state_digest

    source_config.use_pretrained_backbone = False
    source_config.use_prompts = 0
    source_config.local_query = 0
    source, info = model_class.from_pretrained(
        args.repo_name, config=source_config, output_loading_info=True
    )
    _audit_load(info, source)
    model = model_class(config, **kwargs)
    before = state_digest(model.model.backbone.state_dict())
    state = source.state_dict()
    plan = plan_detector_transfer(
        {n: tuple(t.shape) for n, t in state.items()},
        {n: tuple(t.shape) for n, t in model.state_dict().items()},
    )
    result = model.load_state_dict(
        {n: state[n] for n in plan["load_keys"]}, strict=False
    )
    if (
        set(result.missing_keys) != set(plan["new_target_keys"])
        or result.unexpected_keys
    ):
        raise ValueError("Detector transfer did not match the audited plan")
    if state_digest(model.model.backbone.state_dict()) != before:
        raise ValueError("Detector transfer overwrote the pretrained backbone")
    return model, {
        "mode": "cross_backbone",
        "source_loading_info": info,
        "transfer": plan,
        "backbone_preserved_sha256": before,
        "classifier_loaded": False,
    }


def runtime_record(args, model, image_processor):
    previous = getattr(args, "_runtime_snapshot", None)
    provenance = (
        previous["pretrained"]
        if previous
        else {
            "detector": args.repo_name,
            "backbone_file": getattr(args, "backbone_pretrained_file", ""),
        }
    )
    return {
        "model_config": model.config.to_dict(),
        "processor_config": image_processor.to_dict(),
        "pretrained": provenance,
    }


def training_contract(args):
    """Settings that must not change within an interrupted training task."""
    keys = (
        "batch_size",
        "eff_batch_size",
        "n_gpus",
        "seed",
        "shuffle",
        "freeze",
        "freeze_shared_after_task1",
        "lr",
        "lr_old",
        "lr_drop",
        "weight_decay",
        "optim_groups",
        "augment",
        "augment_flip",
        "ddl_lambda",
        "lambda_query",
        "query_loss_grad",
    )
    result = {key: getattr(args, key) for key in keys}
    files = {"task_config": args.task_config}
    if getattr(args, "task", None) and args.task_ann_dir:
        name = (
            f"train_joint{args.train_suffix}.json"
            if args.joint
            else f"train_task_{args.task}{args.train_suffix}.json"
        )
        files["train"] = str(Path(args.task_ann_dir) / name)
    result["data_md5"] = {
        name: md5_file(path)
        for name, path in files.items()
        if path and Path(path).is_file()
    }
    return result


def save_runtime(directory, runtime, report):
    directory = Path(directory)
    save_json(directory / "runtime.json", runtime, indent=2)
    save_json(directory / "loading_report.json", report, indent=2)
