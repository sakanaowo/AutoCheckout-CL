"""B3 / experiment E5: detect, then retrieve (plan sections 6.7 and 7).

A class-agnostic detector (B1c) gives the boxes; each box is labelled by nearest-neighbour
search over DINOv2 embeddings of the SKUs learned so far. Learning task t means embedding at
most ``--per-class`` ground-truth crops of each task-t class, read from task t's own
``train_task_<t>.json`` (plan section 1.3: only task t is labelled, and no image of an earlier
task is kept, only its embeddings); nothing is trained. The detection crops of val/test do not
depend on the task, so they are embedded once and cached.

Files (``DET`` = ``--det-dir`` with the detector's ``pred_<split>.npz``, one row per box;
``EMB`` = ``--emb-dir``; ``RUN`` = ``--out-dir``):

- ``EMB/det_<split>.npz``: embedding of every row of ``DET/pred_<split>.npz``;
- ``EMB/memory_task_<t>.npz``: embeddings, labels, image and annotation ids of task t's memory;
- ``RUN/task_<t>/pred_<split>.npz``: docs/data_preprocessing/formats.md section 5, ``producer: "retrieval"``.

A cached file stores the settings it was built with and is only reused with the same settings,
so several ``RUN`` directories (e.g. prototype and kNN) can share one ``EMB``.

Class confidence (``class_confidence``): softmax of cosine similarity / ``temperature``, over
the class prototypes (mean embedding, re-normalised) or over the ``k`` most similar memory
embeddings summed per class (similarity-weighted kNN vote). The defaults k = 20 and
temperature = 0.07 are those of DINO's kNN evaluation (Caron et al. 2021, after Wu et al. 2018).
Row score = detector score x class confidence, in [0, 1]: it only has to rank rows (mAP) and be
thresholded (counting picks the threshold on val).

Embedding: crop grown by ``--margin`` x the box size on each side, square resize to 224
(bicubic), ImageNet normalisation, CLS token after the final layer norm, L2-normalised; computed
in fp32; stored as float16.

Usage (every data task of the config)::

    python -m baselines.retrieval run --det-dir DET --task-dir TASKS --task-config CFG \\
        --image-dir /data/rpc/checkout_800 --emb-dir EMB --out-dir RUN [--capped] [--mode knn]

Single steps: ``embed-dets``, ``memory --task t``, ``predict --task t`` (cached files only, no model).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

from autocheckout.io import default_file_mode, load_json, md5_file
from autocheckout.predictions import Predictions, load_predictions, save_predictions
from autocheckout.taskcfg import TaskConfig

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- cached arrays ------------------------------------------------------------------------


def save_arrays(path: Path, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """Write ``arrays`` and a JSON ``meta`` to an ``.npz`` atomically (like ``save_predictions``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.chmod(tmp, default_file_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_arrays(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["meta"])), {k: data[k] for k in data.files if k != "meta"}


def cached(path: Path, meta: dict[str, Any], compute: Callable[[], dict[str, np.ndarray]]
           ) -> dict[str, np.ndarray]:
    """Arrays of ``path`` if it was built with ``meta``; otherwise compute and save them."""
    if path.exists():
        stored, arrays = load_arrays(path)
        if stored != meta:
            raise ValueError(f"{path} was built with {stored}, not {meta}: "
                             "delete it or use another --emb-dir")
        return arrays
    arrays = compute()
    save_arrays(path, meta, arrays)
    return arrays


# --- crops and embeddings ------------------------------------------------------------------


def crop_box(box: Sequence[float], margin: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel crop ``(left, top, right, bottom)`` of an xyxy box grown by ``margin`` x its width
    (height) on the left and right (top and bottom), clipped to the image, at least 1 px."""
    x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = margin * (x2 - x1), margin * (y2 - y1)
    left = min(max(math.floor(x1 - dx), 0), width - 1)
    top = min(max(math.floor(y1 - dy), 0), height - 1)
    right = max(min(math.ceil(x2 + dx), width), left + 1)
    bottom = max(min(math.ceil(y2 + dy), height), top + 1)
    return left, top, right, bottom


class _Crops(Dataset):
    """Item i: the crops of ``items[i] = (image path, (n, 4) xyxy boxes)``, (n, 3, S, S) uint8."""

    def __init__(self, items: Sequence[tuple[Path, np.ndarray]], margin: float) -> None:
        self.items, self.margin = items, margin

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> torch.Tensor:
        path, boxes = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            crops = [np.asarray(image.crop(crop_box(box, self.margin, *image.size))
                                .resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC))
                     for box in boxes]
        return torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2)


@torch.no_grad()
def embed_crops(model: torch.nn.Module, items: Sequence[tuple[Path, np.ndarray]], *, margin: float,
                device: torch.device, batch_size: int = 256, workers: int = 0) -> np.ndarray:
    """L2-normalised CLS embeddings (float32) of every box of ``items``, in item then box order.
    Batches hold about ``batch_size`` crops (whole images)."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def forward(crops: list[torch.Tensor]) -> np.ndarray:
        pixels = (torch.cat(crops).to(device).float() / 255 - mean) / std
        cls = model(pixel_values=pixels).last_hidden_state[:, 0]
        return F.normalize(cls.float(), dim=1).cpu().numpy()

    out, pending, n_pending = [], [], 0
    loader = DataLoader(_Crops(items, margin), batch_size=None, num_workers=workers)
    for crops in tqdm(loader, desc="embedding", unit="image", mininterval=10):
        pending.append(crops)
        n_pending += len(crops)
        if n_pending >= batch_size:
            out.append(forward(pending))
            pending, n_pending = [], 0
    if pending:
        out.append(forward(pending))
    return np.concatenate(out)


def embed_boxes(model: torch.nn.Module, image_ids: np.ndarray, boxes: np.ndarray, file_names: dict[int, str],
                image_dir: Path, **kwargs: Any) -> np.ndarray:
    """Embedding of each ``(image_ids[i], boxes[i])`` row, in row order; each image is read once
    (``kwargs`` go to ``embed_crops``)."""
    order = np.argsort(image_ids, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(image_ids[order])) + 1)
    items = [(image_dir / file_names[int(image_ids[g[0]])], boxes[g]) for g in groups]
    embeddings = embed_crops(model, items, **kwargs)
    out = np.empty_like(embeddings)
    out[order] = embeddings
    return out


def load_backbone(name: str, device: torch.device) -> torch.nn.Module:
    return AutoModel.from_pretrained(name).to(device).eval()


# --- detections and memory -----------------------------------------------------------------


def load_detector(path: Path, split: str) -> Predictions:
    """Class-agnostic detector predictions: one row per box, with the annotation file's md5."""
    det = load_predictions(path)
    missing = {"ann_file", "ann_md5"} - set(det.meta)
    if missing:
        raise ValueError(f"{path}: meta lacks {sorted(missing)}")
    if det.meta.get("split", split) != split:
        raise ValueError(f"{path}: meta.split={det.meta['split']!r}, expected {split!r}")
    pairs = np.stack([det.image_id, det.query.astype(np.int64)], axis=1)
    if len(np.unique(pairs, axis=0)) != len(det):
        raise ValueError(f"{path}: (image_id, query) pairs repeat; expected one row per box")
    return det


def embed_detections(model: torch.nn.Module, det_path: Path, ann_path: Path, image_dir: Path, split: str,
                     **kwargs: Any) -> dict[str, np.ndarray]:
    """Embeddings (float16) of every row of the detector file, whose image ids are those of
    ``ann_path`` (``kwargs`` go to ``embed_crops``)."""
    det = load_detector(det_path, split)
    if md5_file(ann_path) != det.meta["ann_md5"]:
        raise ValueError(f"{ann_path} is not the annotation file of {det_path} (md5 differs)")
    file_names = {img["id"]: img["file_name"] for img in load_json(ann_path)["images"]}
    embeddings = embed_boxes(model, det.image_id, det.boxes, file_names, image_dir, **kwargs)
    return {"embeddings": embeddings.astype(np.float16)}


def select_annotations(annotations: Sequence[dict[str, Any]], labels: Sequence[int], per_class: int,
                       seed: int) -> list[dict[str, Any]]:
    """At most ``per_class`` annotations of each label in ``labels`` (other labels are ignored),
    drawn with ``seed`` and spread over as many images as possible: in a random order, the
    k-th box of an image comes after the (k-1)-th box of every other image."""
    by_label = defaultdict(list)
    for ann in annotations:
        by_label[ann["category_id"]].append(ann)
    picked = []
    for label in labels:
        anns = by_label[label]
        if not anns:
            raise ValueError(f"label {label} has no box")
        rng = np.random.default_rng([seed, label])
        shuffled = [anns[i] for i in rng.permutation(len(anns))]
        seen: Counter[int] = Counter()
        rank = []
        for ann in shuffled:
            rank.append(seen[ann["image_id"]])
            seen[ann["image_id"]] += 1
        picked += [shuffled[i] for i in np.argsort(rank, kind="stable")[:per_class]]
    return picked


def build_memory(model: torch.nn.Module, train_path: Path, labels: Sequence[int], image_dir: Path, *,
                 per_class: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    """Memory of one task: embeddings of at most ``per_class`` GT crops per label, read from
    ``train_path`` only (``kwargs`` go to ``embed_crops``)."""
    coco = load_json(train_path)
    anns = select_annotations(coco["annotations"], labels, per_class, seed)
    file_names = {img["id"]: img["file_name"] for img in coco["images"]}
    image_ids = np.array([ann["image_id"] for ann in anns], dtype=np.int64)
    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in (ann["bbox"] for ann in anns)], dtype=np.float32)
    embeddings = embed_boxes(model, image_ids, boxes, file_names, image_dir, **kwargs)
    return {
        "embeddings": embeddings.astype(np.float16),
        "labels": np.array([ann["category_id"] for ann in anns], dtype=np.int32),
        "image_id": image_ids,
        "ann_id": np.array([ann["id"] for ann in anns], dtype=np.int64),
    }


# --- classification ------------------------------------------------------------------------


def class_confidence(query: torch.Tensor, memory: torch.Tensor, memory_labels: torch.Tensor, num_classes: int,
                     *, temperature: float, k: int | None = None) -> torch.Tensor:
    """(n, num_classes) confidences summing to 1: softmax of cosine similarity / ``temperature``
    over the memory rows (the ``k`` most similar ones if ``k`` is set), summed per class label.
    Rows are L2-normalised; with one row per class (prototypes) this is a plain softmax."""
    sims = query @ memory.T
    labels = memory_labels.expand(len(query), -1)
    if k is not None and k < sims.shape[1]:
        sims, index = sims.topk(k, dim=1)
        labels = memory_labels[index]
    weights = torch.softmax(sims / temperature, dim=1)
    return torch.zeros(len(query), num_classes, device=query.device).scatter_add_(1, labels, weights)


def classify(query: np.ndarray, memory: np.ndarray, memory_labels: np.ndarray, num_classes: int, *,
             mode: str, k: int, temperature: float, labels_per_box: int, device: torch.device,
             chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Top ``labels_per_box`` (labels, confidences) of each query row, best first. ``mode``:
    ``prototype`` (class means, re-normalised) or ``knn`` (``k`` nearest memory rows)."""
    counts = np.bincount(memory_labels, minlength=num_classes)
    if len(counts) != num_classes or not counts.all():
        raise ValueError(f"memory must hold every label 0..{num_classes - 1} and no other")
    mem = torch.from_numpy(memory).float()
    mem_labels = torch.from_numpy(memory_labels.astype(np.int64))
    if mode == "prototype":
        mem = F.normalize(torch.zeros(num_classes, mem.shape[1]).index_add_(0, mem_labels, mem), dim=1)
        mem_labels, k = torch.arange(num_classes), None
    mem, mem_labels = mem.to(device), mem_labels.to(device)

    labels, confidences = [], []
    for start in range(0, len(query), chunk):
        batch = torch.from_numpy(query[start:start + chunk]).float().to(device)
        conf = class_confidence(batch, mem, mem_labels, num_classes, temperature=temperature, k=k)
        top = conf.topk(min(labels_per_box, num_classes), dim=1)
        labels.append(top.indices.cpu().numpy())
        confidences.append(top.values.cpu().numpy())
    return np.concatenate(labels), np.concatenate(confidences)


def box_rows(det: Predictions, labels: np.ndarray, confidences: np.ndarray,
             topk_per_image: int) -> Predictions:
    """One row per (box, label) with score = detector score x confidence (zero-confidence rows
    dropped), then the ``topk_per_image`` best rows of each image, as in the detectors' files."""
    per_box = labels.shape[1]
    rows = Predictions(
        image_id=np.repeat(det.image_id, per_box),
        query=np.repeat(det.query, per_box),
        label=labels.ravel(),
        score=np.repeat(det.score, per_box) * confidences.ravel(),
        boxes=np.repeat(det.boxes, per_box, axis=0),
    ).subset(confidences.ravel() > 0)
    order = np.lexsort((-rows.score, rows.image_id))
    image_ids = rows.image_id[order]
    starts = np.flatnonzero(np.r_[True, image_ids[1:] != image_ids[:-1]])
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = np.zeros(len(rows), dtype=bool)
    keep[order[rank < topk_per_image]] = True
    return rows.subset(keep)


def predict_task(cfg: TaskConfig, task_id: int, split: str, det_path: Path, emb_dir: Path, *, mode: str,
                 k: int, temperature: float, labels_per_box: int, topk_per_image: int,
                 device: torch.device) -> Predictions:
    """Stage ``task_id`` predictions from the cached detection embeddings and memories of tasks 1..t."""
    det = load_detector(det_path, split)
    det_emb_path = emb_dir / f"det_{split}.npz"
    det_meta, det_arrays = load_arrays(det_emb_path)
    if det_meta["detector_md5"] != md5_file(det_path):
        raise ValueError(f"{det_emb_path} was built from another detector file than {det_path}")
    memories = [load_arrays(emb_dir / f"memory_task_{t}.npz") for t in range(1, task_id + 1)]
    for meta, _ in memories:
        expected = {"backbone": det_meta["backbone"], "margin": det_meta["margin"], "task_config": cfg.name}
        if {key: meta[key] for key in expected} != expected:
            raise ValueError(f"memory of task {meta['task_id']} does not match {expected}")

    seen_classes = cfg.seen_classes(task_id)
    labels, confidences = classify(
        det_arrays["embeddings"],
        np.concatenate([arrays["embeddings"] for _, arrays in memories]),
        np.concatenate([arrays["labels"] for _, arrays in memories]),
        seen_classes, mode=mode, k=k, temperature=temperature, labels_per_box=labels_per_box, device=device,
    )
    preds = box_rows(det, labels, confidences, topk_per_image)
    preds.meta = {
        "task_id": task_id, "seen_classes": seen_classes, "split": split,
        "ann_file": det.meta["ann_file"], "ann_md5": det.meta["ann_md5"],
        "producer": "retrieval", "topk_per_image": topk_per_image, "labels_per_box": labels_per_box,
        "backbone": det_meta["backbone"], "margin": det_meta["margin"],
        "mode": mode, "k": k if mode == "knn" else None, "temperature": temperature,
        "detector": str(det_path), "detector_md5": det_meta["detector_md5"],
        "memory": [meta for meta, _ in memories],
    }
    return preds


# --- CLI -----------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    emb = argparse.ArgumentParser(add_help=False)
    emb.add_argument("--emb-dir", type=Path, required=True, help="cached detection embeddings and memories")
    model = argparse.ArgumentParser(add_help=False)
    model.add_argument("--task-dir", type=Path, required=True, help="train_task_<t>.json, <split>_full.json")
    model.add_argument("--image-dir", type=Path, required=True)
    model.add_argument("--backbone", default="facebook/dinov2-small")
    model.add_argument("--margin", type=float, default=0.1, help="crop margin, fraction of the box size")
    model.add_argument("--batch-size", type=int, default=256)
    model.add_argument("--workers", type=int, default=4, help="image-loading processes")
    det = argparse.ArgumentParser(add_help=False)
    det.add_argument("--det-dir", type=Path, required=True, help="detector's pred_<split>.npz")
    det.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    cfg = argparse.ArgumentParser(add_help=False)
    cfg.add_argument("--task-config", type=Path, required=True)
    mem = argparse.ArgumentParser(add_help=False)
    mem.add_argument("--per-class", type=int, default=100)
    mem.add_argument("--seed", type=int, default=0)
    mem.add_argument("--capped", action="store_true", help="read train_task_<t>_capped.json")
    cls = argparse.ArgumentParser(add_help=False)
    cls.add_argument("--out-dir", type=Path, required=True, help="run directory")
    cls.add_argument("--mode", choices=["prototype", "knn"], default="prototype")
    cls.add_argument("--k", type=int, default=20)
    cls.add_argument("--temperature", type=float, default=0.07)
    cls.add_argument("--labels-per-box", type=int, default=5)
    cls.add_argument("--topk-per-image", type=int, default=100)
    task = argparse.ArgumentParser(add_help=False)
    task.add_argument("--task", type=int, required=True)

    parser = argparse.ArgumentParser(description="B3: detect, then retrieve with DINOv2 embeddings")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("embed-dets", parents=[emb, model, det], help="embed the detection crops (once)")
    sub.add_parser("memory", parents=[emb, model, cfg, mem, task], help="build the memory of one task")
    sub.add_parser("predict", parents=[emb, det, cfg, cls, task], help="predict one stage from cached files")
    sub.add_parser("run", parents=[emb, model, det, cfg, mem, cls], help="all steps, every data task")
    return parser.parse_args(argv)


def _embed_kwargs(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return {"margin": args.margin, "device": device, "batch_size": args.batch_size, "workers": args.workers}


def embed_step(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> None:
    for split in args.splits:
        det_path, out_path = args.det_dir / f"pred_{split}.npz", args.emb_dir / f"det_{split}.npz"
        meta = {"detector_md5": md5_file(det_path), "backbone": args.backbone, "margin": args.margin}
        ann_path = args.task_dir / f"{split}_full.json"
        cached(out_path, meta, partial(embed_detections, model, det_path, ann_path, args.image_dir, split,
                                       **_embed_kwargs(args, device)))
        print(f"detection embeddings of {split}: {out_path}")


def memory_step(args: argparse.Namespace, model: torch.nn.Module, device: torch.device, cfg: TaskConfig,
                task_id: int) -> None:
    train_path = args.task_dir / f"train_task_{task_id}{'_capped' if args.capped else ''}.json"
    meta = {"task_id": task_id, "task_config": cfg.name, "train_file": train_path.name,
            "train_md5": md5_file(train_path), "per_class": args.per_class, "seed": args.seed,
            "backbone": args.backbone, "margin": args.margin}
    memory = cached(args.emb_dir / f"memory_task_{task_id}.npz", meta,
                    partial(build_memory, model, train_path, cfg.task(task_id).labels, args.image_dir,
                            per_class=args.per_class, seed=args.seed, **_embed_kwargs(args, device)))
    print(f"task {task_id}: memory of {len(memory['labels'])} crops")


def predict_step(args: argparse.Namespace, device: torch.device, cfg: TaskConfig, task_id: int) -> None:
    for split in args.splits:
        preds = predict_task(cfg, task_id, split, args.det_dir / f"pred_{split}.npz", args.emb_dir,
                             mode=args.mode, k=args.k, temperature=args.temperature,
                             labels_per_box=args.labels_per_box, topk_per_image=args.topk_per_image,
                             device=device)
        out_path = args.out_dir / f"task_{task_id}" / f"pred_{split}.npz"
        save_predictions(out_path, preds)
        print(f"task {task_id}: {len(preds)} rows -> {out_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None if args.command == "predict" else load_backbone(args.backbone, device)
    if args.command in ("embed-dets", "run"):
        embed_step(args, model, device)
    if args.command == "embed-dets":
        return
    cfg = TaskConfig.load(args.task_config)
    for task_id in [args.task] if args.command != "run" else [t.task_id for t in cfg.data_tasks]:
        if args.command in ("memory", "run"):
            memory_step(args, model, device, cfg, task_id)
        if args.command in ("predict", "run"):
            predict_step(args, device, cfg, task_id)


if __name__ == "__main__":
    main()
