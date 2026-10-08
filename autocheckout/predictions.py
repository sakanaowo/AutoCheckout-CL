"""Detection predictions of one model on one image set, stored as a compressed ``.npz``.

One row per (image, query, class) candidate. Detectors produce up to ``topk_per_image``
rows per image, ranked over (query, class) pairs exactly like the original Deformable DETR
post-processing, so COCO mAP stays comparable with the paper. Counting needs one label per
physical object, which ``top1_per_query`` provides (the best class of each query).

Conventions (see docs/data_preprocessing/formats.md):
- ``label`` is the model label (0..num_slots-1), never the RPC category id;
- ``boxes`` are ``[x1, y1, x2, y2]`` in absolute pixels of the image as stored in the
  annotation file that was evaluated (800x800 after resizing);
- ``meta`` is a small JSON-serialisable dict (task id, seen classes, split, producer, ...).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from autocheckout.io import default_file_mode

_ARRAYS = ("image_id", "query", "label", "score", "boxes")


@dataclass
class Predictions:
    image_id: np.ndarray  # (N,) int64
    query: np.ndarray  # (N,) int32, index of the detector query (or box) that produced the row
    label: np.ndarray  # (N,) int32, model label
    score: np.ndarray  # (N,) float32
    boxes: np.ndarray  # (N, 4) float32, xyxy absolute pixels
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_id = np.asarray(self.image_id, dtype=np.int64).reshape(-1)
        self.query = np.asarray(self.query, dtype=np.int32).reshape(-1)
        self.label = np.asarray(self.label, dtype=np.int32).reshape(-1)
        self.score = np.asarray(self.score, dtype=np.float32).reshape(-1)
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        n = len(self.image_id)
        for name in _ARRAYS:
            if len(getattr(self, name)) != n:
                raise ValueError(f"field {name!r} has {len(getattr(self, name))} rows, expected {n}")

    def __len__(self) -> int:
        return len(self.image_id)

    def subset(self, mask: np.ndarray) -> Predictions:
        return Predictions(*(getattr(self, name)[mask] for name in _ARRAYS), meta=dict(self.meta))

    def top1_per_query(self) -> Predictions:
        """Keep, for every (image, query), only the highest-scoring class."""
        if len(self) == 0:
            return self.subset(np.zeros(0, dtype=bool))
        # Sort by (image, query, -score); the first row of each (image, query) run is its best class.
        order = np.lexsort((-self.score, self.query, self.image_id))
        image_id, query = self.image_id[order], self.query[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = (image_id[1:] != image_id[:-1]) | (query[1:] != query[:-1])
        keep = np.zeros(len(self), dtype=bool)
        keep[order[first]] = True
        return self.subset(keep)

    def to_coco_results(self, label_to_category: dict[int, int] | None = None) -> list[dict[str, Any]]:
        """COCO results list (``bbox`` in xywh). ``label_to_category`` maps labels to category ids."""
        xywh = self.boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
        labels = self.label.tolist()
        if label_to_category is not None:
            labels = [label_to_category[x] for x in labels]
        return [
            {"image_id": int(i), "category_id": int(c), "bbox": [float(v) for v in b], "score": float(s)}
            for i, c, b, s in zip(self.image_id.tolist(), labels, xywh, self.score.tolist(), strict=True)
        ]


def save_predictions(path: str | os.PathLike, predictions: Predictions) -> None:
    """Write atomically (temporary file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(
            tmp,
            meta=np.array(json.dumps(predictions.meta)),
            **{name: getattr(predictions, name) for name in _ARRAYS},
        )
        os.chmod(tmp, default_file_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_predictions(path: str | os.PathLike) -> Predictions:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return Predictions(*(data[name] for name in _ARRAYS), meta=meta)
