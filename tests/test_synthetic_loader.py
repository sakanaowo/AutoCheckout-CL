"""Synthetic v2, T4.3: the PDP loader reads the synthetic train files and the RPC evaluation set at 640.

Runs on the pilot release and the evaluation set when they exist locally (skipped otherwise). The
annotation folder is assembled the way pdp/main.py reads it: train_task_<t>.json from the synthetic
release (images under the release root), val_task_<t>.json from the evaluation set (images under
data/archive).
"""

import json

import numpy as np
import pytest

from conftest import REPO_ROOT

PILOT = REPO_ROOT / "data/processed/rpc_synth_v2_pilot_v1"
EVAL = REPO_ROOT / "data/processed/rpc_eval_official_v1"
ARCHIVE = REPO_ROOT / "data/archive"
pytestmark = pytest.mark.skipif(not (PILOT / "tasks/synth_v2").is_dir() or not (EVAL / "val_task_1.json").is_file(),
                                reason="needs the synthetic pilot release and the evaluation set")


@pytest.fixture(scope="module")
def ann_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("tasks")
    for t in (1, 2):
        (out / f"train_task_{t}.json").symlink_to(PILOT / f"tasks/synth_v2/train_task_{t}.json")
        (out / f"val_task_{t}.json").symlink_to(EVAL / f"val_task_{t}.json")
    return out


def processor():
    from models.image_processing_deformable_detr import DeformableDetrImageProcessor
    return DeformableDetrImageProcessor(size={"shortest_edge": 640, "longest_edge": 640})


def task_labels(t):
    config = json.loads((EVAL / "task_config.json").read_text())
    return {c["label"] for task in config["tasks"] if task["task_id"] == t for c in task["classes"]}


def check(dataset, labels, n=6):
    for idx in np.linspace(0, len(dataset) - 1, n).astype(int):
        pixels, target = dataset[int(idx)]
        anns = dataset.coco.imgToAnns[dataset.ids[idx]]
        assert max(pixels.shape[1:]) == 640
        assert len(target["boxes"]) == len(anns) > 0
        assert set(target["class_labels"].tolist()) <= labels
        boxes = target["boxes"].numpy()
        assert boxes.min() >= 0 and boxes.max() <= 1
        h, w = target["orig_size"].tolist()
        centres = np.array([[a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2] for a in anns])
        assert np.abs(boxes[:, :2] * [w, h] - centres).max() < 1.0


@pytest.mark.parametrize("t", [1, 2])
def test_train_files_load_with_only_current_task_labels(ann_dir, t):
    from datasets.coco_hug import CocoDetection
    dataset = CocoDetection(img_folder=str(PILOT), ann_file=str(ann_dir / f"train_task_{t}.json"),
                            processor=processor())
    check(dataset, task_labels(t))


@pytest.mark.parametrize("t", [1, 2])
def test_val_files_load_from_the_rpc_archive(ann_dir, t):
    from datasets.coco_hug import CocoDetection
    dataset = CocoDetection(img_folder=str(ARCHIVE), ann_file=str(ann_dir / f"val_task_{t}.json"),
                            processor=processor())
    check(dataset, task_labels(t))
