"""Helpers for tests of the PDP code: toy task configs and a small, fast Deformable DETR on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RPC_SIZES = (100, 25, 25, 25, 25)
RPC_RESERVED = 24


def task_config_dict(sizes=RPC_SIZES, reserved=RPC_RESERVED, name="toy") -> dict:
    """A task config in the docs/formats.md layout with fake SKU names."""
    tasks, offset, rpc_id = [], 0, 1
    for task_id, size in enumerate(sizes, start=1):
        classes = [{"label": offset + i, "name": f"sku_{rpc_id + i}", "rpc_category_id": rpc_id + i,
                    "supercategory": "drink"} for i in range(size)]
        tasks.append({"task_id": task_id, "offset": offset, "reserved": False, "classes": classes})
        offset += size
        rpc_id += size
    if reserved:
        tasks.append({"task_id": len(sizes) + 1, "offset": offset, "reserved": True, "classes": [
            {"label": offset + i, "name": f"reserved_{offset + i}", "rpc_category_id": None}
            for i in range(reserved)]})
    return {"name": name, "seed": 0, "tasks": tasks}


def write_task_config(path: Path, **kwargs) -> Path:
    path.write_text(json.dumps(task_config_dict(**kwargs)))
    return path


def main_args(argv: list[str]) -> argparse.Namespace:
    """Parse PDP main.py arguments exactly as the command line would."""
    import main as pdp_main

    parser = argparse.ArgumentParser(parents=[pdp_main.get_args_parser()])
    return parser.parse_args(argv)


# --- a small Deformable DETR for fast CPU tests ---------------------------------------------------
# d_model and num_queries keep the PDP values (query_tf is hard-coded to 300 x 256); the rest shrinks.
TINY_DETR = dict(use_pretrained_backbone=False, backbone="resnet18", encoder_layers=1, decoder_layers=2,
                 encoder_ffn_dim=64, decoder_ffn_dim=64)
# 96 px keeps the smallest feature level at 2x2. At 64 px it is 1x1, and with a batch of one image
# the CPU GroupNorm backward of input_proj fails on the ambiguous memory format of a [1, C, 1, 1]
# tensor ("Expected memory formats of X and dY are same"); real 800 px inputs never get there.
IMAGE_SIZE = 96


def use_tiny_detr(monkeypatch) -> None:
    """Make engine.local_trainer build the tiny model with random weights (no download)."""
    import engine
    from models.configuration_deformable_detr import DeformableDetrConfig

    monkeypatch.setattr(engine, "DeformableDetrConfig", lambda: DeformableDetrConfig(**TINY_DETR))


def pdp_args(tmp_path: Path, *extra: str, sizes=(3, 2), reserved=0, config=None) -> argparse.Namespace:
    """Arguments for a small PDP run on a toy task config, set up like main() does.

    ``config`` (a TaskConfig) replaces the toy config of ``sizes`` and ``reserved``.
    """
    import main as pdp_main

    if config is None:
        path = write_task_config(tmp_path / "tasks.json", sizes=sizes, reserved=reserved)
        n_classes, n_tasks = sum(sizes) + reserved + 1, len(sizes)
    else:
        path = tmp_path / "tasks.json"
        config.save(path)
        n_classes, n_tasks = config.num_classes, len(config.data_tasks)
    args = main_args([
        "--task_config", str(path), "--n_classes", str(n_classes), "--repo_name", "",
        "--n_tasks", str(n_tasks), "--use_prompts", "1", "--local_query", "1",
        "--num_prompts", "4", "--prompt_len", "2", "--lambda_query", "0.1",
        "--freeze", "backbone,encoder,decoder", "--new_params", "class_embed,prompts",
        "--output_dir", str(tmp_path / "run"), *extra,
    ])
    pdp_main.setup_task_info(args)
    args.log_file = open(tmp_path / "log.txt", "a")  # noqa: SIM115 - closed with the tmp dir
    return args


def make_trainer(args: argparse.Namespace, task_id: int):
    """engine.local_trainer for task_id, prompts set to that task, freezing applied like main()."""
    from types import SimpleNamespace

    import engine

    trainer = engine.local_trainer(train_loader=None, val_loader=None, test_dataset=None, args=args,
                                   local_evaluator=SimpleNamespace(), task_id=task_id)
    if args.use_prompts:
        trainer.model.model.prompts.set_task_id(task_id - 1)
    trainer.resume()  # only applies --freeze when no path is given
    return trainer


def make_batch(labels_per_image: list[list[int]], seed: int = 0) -> dict:
    """A batch in the format of CocoDetection.collate_fn, with random pixels and boxes."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    batch_size = len(labels_per_image)
    labels = []
    for image_id, classes in enumerate(labels_per_image, start=1):
        n = len(classes)
        centers = 0.2 + 0.6 * torch.rand(n, 2, generator=generator)
        sizes = 0.1 + 0.2 * torch.rand(n, 2, generator=generator)
        labels.append({
            "class_labels": torch.tensor(classes, dtype=torch.int64),
            "boxes": torch.cat([centers, sizes], dim=1),
            "image_id": torch.tensor([image_id]),
            "orig_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE]),
            "size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE]),
        })
    return {
        "pixel_values": torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator),
        "pixel_mask": torch.ones(batch_size, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.int64),
        "labels": labels,
    }


# --- a toy dataset in the layout of docs/formats.md, for end-to-end runs of main() -----------------
def make_toy_dataset(root: Path, sizes=(3, 2), n_train=8, n_val=4, n_test=4, seed=0) -> dict:
    """Tiny images with coloured boxes and the per-task JSON files main() reads.

    Every image may contain objects of any class, but train/val task files keep only the labels of
    their task (the class-incremental protocol); val_full/test_full keep every label.
    """
    import random

    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    num_labels = sum(sizes)
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    image_dir, task_dir = root / "images", root / "tasks"
    image_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)

    def make_split(name, count, first_id):
        images, annotations = [], []
        for image_id in range(first_id, first_id + count):
            file_name = f"{name}_{image_id}.jpg"
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (235, 235, 235))
            draw = ImageDraw.Draw(image)
            for _ in range(rng.randint(1, 3)):
                label = rng.randrange(num_labels)
                x, y = rng.randint(0, IMAGE_SIZE - 24), rng.randint(0, IMAGE_SIZE - 24)
                w, h = rng.randint(10, 22), rng.randint(10, 22)
                draw.rectangle([x, y, x + w, y + h], fill=(40 * label % 255, 90, 200 - 30 * label % 200))
                annotations.append({"id": len(annotations) + 1 + 1000 * first_id, "image_id": image_id,
                                    "category_id": label, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            image.save(image_dir / file_name)
            images.append({"id": image_id, "file_name": file_name, "width": IMAGE_SIZE, "height": IMAGE_SIZE,
                           "level": ["easy", "medium", "hard"][image_id % 3]})
        return {"images": images, "annotations": annotations}

    def restrict(coco, labels):
        anns = [a for a in coco["annotations"] if a["category_id"] in labels]
        keep = {a["image_id"] for a in anns}
        return {"images": [i for i in coco["images"] if i["id"] in keep], "annotations": anns,
                "categories": [{"id": c, "name": f"sku_{c + 1}"} for c in sorted(labels)]}

    splits = {"train": make_split("train", n_train, 1), "val": make_split("val", n_val, 100),
              "test": make_split("test", n_test, 200)}
    for task_id, (offset, size) in enumerate(zip(offsets, sizes, strict=True), start=1):
        labels = set(range(offset, offset + size))
        for split in ("train", "val"):
            (task_dir / f"{split}_task_{task_id}.json").write_text(json.dumps(restrict(splits[split], labels)))
        task_images = {i["id"] for i in restrict(splits["train"], labels)["images"]}
        full = restrict(splits["train"], set(range(num_labels)))
        full["images"] = [i for i in full["images"] if i["id"] in task_images]
        full["annotations"] = [a for a in full["annotations"] if a["image_id"] in task_images]
        (task_dir / f"train_task_{task_id}_gt_full.json").write_text(json.dumps(full))
    for split in ("val", "test"):
        (task_dir / f"{split}_full.json").write_text(json.dumps(restrict(splits[split], set(range(num_labels)))))
    (task_dir / "train_joint.json").write_text(json.dumps(restrict(splits["train"], set(range(num_labels)))))
    return {"images": image_dir, "tasks": task_dir}


def small_processor():
    """Image processor that keeps the toy images at their size (the default resizes to 800)."""
    from models.image_processing_deformable_detr import DeformableDetrImageProcessor

    return DeformableDetrImageProcessor(size={"shortest_edge": IMAGE_SIZE, "longest_edge": IMAGE_SIZE})


def run_main(tmp_path: Path, monkeypatch, *extra: str, sizes=(3, 2)) -> argparse.Namespace:
    """Run pdp main() end to end on the toy dataset with the tiny model, on CPU."""
    import main as pdp_main

    use_tiny_detr(monkeypatch)
    monkeypatch.setattr(pdp_main, "DeformableDetrImageProcessor", lambda *a, **k: small_processor())
    data = make_toy_dataset(tmp_path / "data", sizes=sizes)
    config = write_task_config(tmp_path / "tasks.json", sizes=sizes, reserved=0)
    args = main_args([
        "--task_config", str(config), "--n_classes", str(sum(sizes) + 1), "--repo_name", "",
        "--n_tasks", str(len(sizes)), "--use_prompts", "1", "--local_query", "1", "--lambda_query", "0.1",
        "--num_prompts", "4", "--prompt_len", "2", "--freeze", "backbone,encoder,decoder",
        "--new_params", "class_embed,prompts", "--accelerator", "cpu", "--n_gpus", "1", "--batch_size", "2",
        "--num_workers", "0", "--epochs", "1", "--eval_epochs", "1",
        "--train_img_dir", str(data["images"]), "--test_img_dir", str(data["images"]),
        "--task_ann_dir", str(data["tasks"]), "--output_dir", str(tmp_path / "run"), *extra,
    ])
    pdp_main.main(args)
    return args
