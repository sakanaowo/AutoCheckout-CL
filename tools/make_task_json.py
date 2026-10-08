"""DL5/DL6: per-task COCO files for the PDP code (docs/data_preprocessing/formats.md, section 4).

For each data task t of the task config (all of them, or ``--tasks``):

- ``train_task_<t>.json``: train images with at least one object of task t; only task-t objects;
- ``train_task_<t>_capped.json``: at most ``--cap`` images of the file above, drawn with
  ``--seed`` and stratified by level (level shares as in the full file);
- ``train_task_<t>_gt_full.json``: the images of ``train_task_<t>.json`` with every object (V4 only);
- ``val_task_<t>.json``: val images with at least one object of task t; only task-t objects;

plus ``val_full.json``, ``test_full.json`` (every object) and ``manifest.json``. ``category_id``
is the model label of the task config, ``categories`` lists the labels that occur in the file
and image entries keep all their fields. Train images can come from several sources
(``--train-source NAME=PATH``, repeatable, e.g. real photos now and composites later); each
train image gets ``train_source: NAME``, and image and annotation ids must be unique across
sources. The pilot (DL6) is the same tool on ``train_pilot.json`` with ``--tasks 1,2``.

    python -m tools.make_task_json --task-config configs/tasks_100-4x25_seed0.json \\
        --train-source real=/data/rpc/splits/train.json --val /data/rpc/splits/val.json \\
        --test /data/rpc/splits/test.json --out-dir /data/rpc/tasks/100-4x25_seed0
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from autocheckout.io import load_json, md5_file, save_json
from autocheckout.sampling import largest_remainder, take_stratified
from autocheckout.taskcfg import TaskConfig


def parse_source(spec: str) -> tuple[str, Path]:
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, Path(path)


def combine_sources(sources: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """One COCO dict with the images and annotations of every train source; images get ``train_source``."""
    images, annotations = [], []
    for name, coco in sources:
        images += [img | {"train_source": name} for img in coco["images"]]
        annotations += coco["annotations"]
    for kind, records in (("image", images), ("annotation", annotations)):
        if len({r["id"] for r in records}) != len(records):
            raise ValueError(f"{kind} ids are not unique across the train sources")
    return {"images": images, "annotations": annotations}


def images_with(coco: dict[str, Any], rpc_ids: set[int]) -> set[int]:
    """Ids of the images that have at least one object of the given RPC categories."""
    return {ann["image_id"] for ann in coco["annotations"] if ann["category_id"] in rpc_ids}


def build_task_coco(coco: dict[str, Any], config: TaskConfig, image_ids: set[int] | None = None,
                    keep_rpc: set[int] | None = None) -> dict[str, Any]:
    """Images in ``image_ids`` (None: all) with their objects of ``keep_rpc`` (None: all), model labels."""
    rpc_to_label = config.rpc_to_label()
    classes = {c.label: c for task in config.tasks for c in task.classes}
    images = [img for img in coco["images"] if image_ids is None or img["id"] in image_ids]
    kept_ids = {img["id"] for img in images}
    annotations = [ann | {"category_id": rpc_to_label[ann["category_id"]]} for ann in coco["annotations"]
                   if ann["image_id"] in kept_ids and (keep_rpc is None or ann["category_id"] in keep_rpc)]
    labels = sorted({ann["category_id"] for ann in annotations})
    categories = [{"id": label, "name": classes[label].name, "supercategory": classes[label].supercategory,
                   "rpc_category_id": classes[label].rpc_category_id} for label in labels]
    return {"images": images, "annotations": annotations, "categories": categories}


def cap_images(coco: dict[str, Any], image_ids: set[int], cap: int, seed: int, task_id: int) -> set[int]:
    """At most ``cap`` of ``image_ids``, drawn with a fixed seed, level shares as in ``image_ids``."""
    if len(image_ids) <= cap:
        return set(image_ids)
    level_of = {img["id"]: img["level"] for img in coco["images"] if img["id"] in image_ids}
    targets = largest_remainder(Counter(level_of.values()), cap)
    rng = random.Random(f"cap-{seed}-task{task_id}")
    return set(take_stratified([(i, level, 1) for i, level in level_of.items()], targets, rng))


def task_files(config: TaskConfig, task_id: int, train: dict[str, Any], val: dict[str, Any], cap: int,
               seed: int) -> dict[str, dict[str, Any]]:
    """The four per-task files of one data task, by file name."""
    task_rpc = {c.rpc_category_id for c in config.task(task_id).classes}
    train_ids = images_with(train, task_rpc)
    return {
        f"train_task_{task_id}.json": build_task_coco(train, config, train_ids, task_rpc),
        f"train_task_{task_id}_capped.json": build_task_coco(
            train, config, cap_images(train, train_ids, cap, seed, task_id), task_rpc),
        f"train_task_{task_id}_gt_full.json": build_task_coco(train, config, train_ids),
        f"val_task_{task_id}.json": build_task_coco(val, config, images_with(val, task_rpc), task_rpc),
    }


def joint_files(config: TaskConfig, train: dict[str, Any], capped_ids: set[int]) -> dict[str, dict[str, Any]]:
    """B1 / E0: every train image with every label, and the union of the capped task images."""
    data_rpc = {c.rpc_category_id for task in config.data_tasks for c in task.classes}
    return {"train_joint.json": build_task_coco(train, config, images_with(train, data_rpc)),
            "train_joint_capped.json": build_task_coco(train, config, capped_ids)}


def agnostic_coco(coco: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    """Class-agnostic copy: the given images with every box as class 0 ("product")."""
    images = [img for img in coco["images"] if img["id"] in image_ids]
    annotations = [ann | {"category_id": 0} for ann in coco["annotations"] if ann["image_id"] in image_ids]
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 0, "name": "product", "supercategory": "product"}]}


def agnostic_files(train: dict[str, Any], val: dict[str, Any], task1_ids: set[int],
                   task1_capped_ids: set[int]) -> dict[str, dict[str, Any]]:
    """B1c / E5 detector (plan B3, option b): images of task 1 with all their boxes, class-agnostic."""
    return {"train_task_1.json": agnostic_coco(train, task1_ids),
            "train_task_1_capped.json": agnostic_coco(train, task1_capped_ids),
            "val_task_1.json": agnostic_coco(val, {img["id"] for img in val["images"]})}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--train-source", type=parse_source, action="append", required=True,
                        metavar="NAME=PATH", help="train split file; repeat for several sources")
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", help="comma-separated data task ids (default: every data task)")
    parser.add_argument("--cap", type=int, default=6000, help="maximum images of train_task_<t>_capped.json")
    parser.add_argument("--seed", type=int, default=0, help="seed of the capped sampling")
    parser.add_argument("--joint", action="store_true",
                        help="also write train_joint.json and train_joint_capped.json (E0)")
    parser.add_argument("--agnostic-out", type=Path,
                        help="also write class-agnostic task-1 files for the E5 detector into this folder")
    args = parser.parse_args(argv)

    config = TaskConfig.load(args.task_config)
    data_task_ids = [t.task_id for t in config.data_tasks]
    task_ids = [int(t) for t in args.tasks.split(",")] if args.tasks else data_task_ids
    if not set(task_ids) <= set(data_task_ids):
        parser.error(f"--tasks must be data tasks of {config.name} ({data_task_ids}), got {task_ids}")
    names = [name for name, _ in args.train_source]
    if len(set(names)) != len(names):
        parser.error(f"train source names must be unique, got {names}")

    train = combine_sources([(name, load_json(path)) for name, path in args.train_source])
    val = load_json(args.val)
    files: dict[str, Any] = {}

    def write(name: str, data: dict[str, Any]) -> None:
        path = args.out_dir / name
        save_json(path, data)
        files[name] = {"images": len(data["images"]), "objects": len(data["annotations"]),
                       "md5": md5_file(path)}
        print(f"{name:32s} images {files[name]['images']:6d}  objects {files[name]['objects']:7d}")

    capped_ids: set[int] = set()
    for task_id in task_ids:
        for name, data in task_files(config, task_id, train, val, args.cap, args.seed).items():
            write(name, data)
            if name.endswith("_capped.json"):
                capped_ids |= {img["id"] for img in data["images"]}
    if args.joint:
        for name, data in joint_files(config, train, capped_ids).items():
            write(name, data)
    write("val_full.json", build_task_coco(val, config))
    write("test_full.json", build_task_coco(load_json(args.test), config))

    def source(path: Path) -> dict[str, str]:
        return {"path": str(path), "md5": md5_file(path)}

    save_json(args.out_dir / "manifest.json", {
        "task_config": {"name": config.name} | source(args.task_config),
        "train_sources": {name: source(path) for name, path in args.train_source},
        "val": source(args.val),
        "test": source(args.test),
        "tasks": task_ids,
        "cap": args.cap,
        "seed": args.seed,
        "files": files,
    }, indent=1)
    print(f"wrote {len(files)} files and manifest.json to {args.out_dir}")

    if args.agnostic_out:
        task1_rpc = {c.rpc_category_id for c in config.task(1).classes}
        task1_ids = images_with(train, task1_rpc)
        task1_capped = cap_images(train, task1_ids, args.cap, args.seed, 1)
        for name, data in agnostic_files(train, val, task1_ids, task1_capped).items():
            save_json(args.agnostic_out / name, data)
            print(f"{args.agnostic_out.name}/{name:24s} images {len(data['images']):6d}  "
                  f"objects {len(data['annotations']):7d}")


if __name__ == "__main__":
    main()
