"""Synthetic v2, T2.1: scene plans per task (formats.md section 7.2, decisions D7, D13, D14)."""

import hashlib
import json
from collections import Counter

import pytest

from autocheckout.synthetic import basket_patterns, plan_scenes, source_view, stable_seed
from autocheckout.taskcfg import TaskConfig
from tools.make_relearn_config import relearn_config

LAYOUT = {"new_ratio": [0.4, 0.6]}


def task_config():
    def task(tid, offset, rpc_ids):
        return {"task_id": tid, "offset": offset, "classes": [
            {"label": offset + i, "name": f"sku{r}", "rpc_category_id": r, "supercategory": "x"} for i, r in enumerate(rpc_ids)]}
    return TaskConfig.from_dict({"name": "t", "seed": 0, "tasks": [task(1, 0, [1, 2, 3, 4]), task(2, 4, [5, 6]), task(3, 6, [7, 8])]})


def catalog_and_skus():
    cutouts, skus = [], []
    for sku in range(1, 9):
        cams = {"0": 0.99, "2": 0.01} if sku in (5, 6) else {"2": 1.0}
        skus.append({"rpc_category_id": sku, "camera_weights": cams})
        for cam in (0, 2):
            for k in range(3):
                cutouts.append({"cutout_id": f"c:{sku}{cam}{k}", "rpc_category_id": sku, "camera": cam})
    return cutouts, skus


PATTERNS = {"easy": [(2, 1, 1), (1, 1, 1, 1), (3, 2)], "medium": [(3, 2, 2, 1), (2, 2, 2, 2)]}


def plan(task_id, n=300, patterns=PATTERNS, seed=7):
    cutouts, skus = catalog_and_skus()
    return plan_scenes(task_id, n, task_config=task_config(), cutouts=cutouts, skus=skus, patterns=patterns,
                       backgrounds=["b1.jpg", "b2.jpg", "b3.jpg"], layout=LAYOUT, seed=seed)


def test_basket_patterns_are_sorted_unit_counts_per_level():
    coco = {"images": [{"id": 1, "level": "easy"}, {"id": 2, "level": "hard"}],
            "annotations": [{"image_id": 1, "category_id": c} for c in (9, 4, 9, 9, 4, 2)] +
                           [{"image_id": 2, "category_id": 5}]}
    assert basket_patterns(coco) == {"easy": [(3, 2, 1)], "hard": [(1,)]}


def test_stable_seed_is_deterministic_and_part_sensitive():
    assert stable_seed(1, 2, 3) == stable_seed(1, 2, 3) != stable_seed(1, 2, 4)


def test_task1_scenes_have_only_task1_skus_all_labeled():
    recipes = plan(1)
    assert len(recipes) == 300 and {r["generation_task"] for r in recipes} == {1}
    objs = [o for r in recipes for o in r["objects"]]
    assert {o["rpc_category_id"] for o in objs} <= {1, 2, 3, 4} and all(o["labeled"] for o in objs)


def test_incremental_scene_has_new_and_old_and_never_future():
    for r in plan(2):
        rpc = [o["rpc_category_id"] for o in r["objects"]]
        n, new = len(rpc), sum(o["labeled"] for o in r["objects"])
        assert all(o["labeled"] == (o["rpc_category_id"] in (5, 6)) for o in r["objects"])
        assert not {7, 8} & set(rpc)
        assert 1 <= new <= n - 1


def test_new_share_is_forty_to_sixty_percent_over_the_task():
    objs = [o for r in plan(2) for o in r["objects"]]
    share = sum(o["labeled"] for o in objs) / len(objs)
    assert 0.4 <= share <= 0.6


def test_scene_units_follow_a_level_pattern():
    for r in plan(1):
        units = tuple(sorted(Counter(o["rpc_category_id"] for o in r["objects"]).values(), reverse=True))
        assert units in PATTERNS[r["level"]]


def test_new_sku_object_counts_within_twenty_percent():
    counts = Counter(o["rpc_category_id"] for r in plan(1, n=600) for o in r["objects"])
    mean = sum(counts.values()) / len(counts)
    assert all(abs(c - mean) <= 0.2 * mean for c in counts.values()), counts


def test_old_skus_are_spread_evenly():
    counts = Counter(o["rpc_category_id"] for r in plan(3, n=600) for o in r["objects"] if not o["labeled"])
    assert set(counts) == {1, 2, 3, 4, 5, 6}
    mean = sum(counts.values()) / len(counts)
    assert all(abs(c - mean) <= 0.2 * mean for c in counts.values()), counts


def test_cutout_camera_follows_sku_weights_and_units_use_different_cutouts():
    cutouts, _ = catalog_and_skus()
    camera = {c["cutout_id"]: c["camera"] for c in cutouts}
    for r in plan(2):
        for o in r["objects"]:
            assert camera[o["cutout_id"]] == o["camera"]
            if o["rpc_category_id"] in (1, 2, 3, 4):
                assert o["camera"] == 2
        per_sku = Counter((o["rpc_category_id"], o["cutout_id"]) for o in r["objects"])
        assert all(n == 1 for (sku, _), n in per_sku.items()
                   if sum(o["rpc_category_id"] == sku for o in r["objects"]) <= 3)


def test_plan_is_deterministic_and_ids_are_unique():
    a, b = plan(2), plan(2)
    assert a == b and plan(2, seed=8) != a
    assert len({r["image_id"] for r in a}) == len(a)
    assert all([o["instance_id"] for o in r["objects"]] == list(range(1, len(r["objects"]) + 1)) for r in a)


def test_infeasible_patterns_fail_explicitly():
    with pytest.raises(ValueError, match="no basket pattern"):
        plan(2, patterns={"easy": [(5,)], "medium": [(4,)]})


def test_area_budget_rejects_baskets_that_cannot_fit():
    cutouts, skus = catalog_and_skus()
    area = {sku: (0.3 if sku == 1 else 0.05) for sku in range(1, 9)}
    recipes = plan_scenes(1, 300, task_config=task_config(), cutouts=cutouts, skus=skus, patterns=PATTERNS,
                          backgrounds=["b.jpg"], layout=LAYOUT, seed=3, sku_area=area,
                          max_area={"easy": 0.5, "medium": 0.8})
    for r in recipes:
        assert sum(area[o["rpc_category_id"]] for o in r["objects"]) <= {"easy": 0.5, "medium": 0.8}[r["level"]]
    counts = Counter(o["rpc_category_id"] for r in recipes for o in r["objects"])
    assert counts[1] > 0


def test_relearn_task_scenes_label_the_relearned_skus_and_never_list_them_as_old():
    cfg = relearn_config(task_config(), [[0, 3], [1]], reserved=1, name="t+relearn")  # task 4 relearns rpc 1, 4
    cutouts, skus = catalog_and_skus()
    recipes = plan_scenes(4, 300, task_config=cfg, cutouts=cutouts, skus=skus, patterns=PATTERNS,
                          backgrounds=["b1.jpg", "b2.jpg", "b3.jpg"], layout=LAYOUT, seed=7)
    objs = [o for r in recipes for o in r["objects"]]
    relearned = {c.rpc_category_id for c in cfg.task(4).classes}
    assert relearned == {1, 4} and all(o["labeled"] == (o["rpc_category_id"] in relearned) for o in objs)
    assert {o["rpc_category_id"] for o in objs if not o["labeled"]} == set(range(1, 9)) - relearned


def test_source_view_reads_placement_camera_and_step():
    assert source_view("6909409012031-back_camera0-15.jpg") == ("back", 0, 15)
    assert source_view("038900004095_camera2-3.png") == ("front", 2, 3)


GOLDEN = {1: "baf43a4aca5a20a9f1a6113096121b01ef1f372e08a80233a354788f21390047",  # plan(task, n=60) before
          2: "775af6c540dd22996839fa4211abfbcf4a81171e31fc3f62cb9f1d97dc1e8d6e"}  # view_balance existed


@pytest.mark.parametrize("layout", [LAYOUT, {**LAYOUT, "view_balance": False}])
def test_without_view_balance_the_plan_is_unchanged(layout):
    cutouts, skus = catalog_and_skus()
    for task, digest in GOLDEN.items():
        recipes = plan_scenes(task, 60, task_config=task_config(), cutouts=cutouts, skus=skus, patterns=PATTERNS,
                              backgrounds=["b1.jpg", "b2.jpg", "b3.jpg"], layout=layout, seed=7)
        assert hashlib.sha256(json.dumps(recipes, sort_keys=True).encode()).hexdigest() == digest


def test_view_balance_alternates_placements_and_spreads_cutouts():
    cutouts, skus = catalog_and_skus()  # SKU 1 is flat (camera 2): 6 front and 2 back cutouts
    cutouts = [c for c in cutouts if c["rpc_category_id"] != 1]
    cutouts += [{"cutout_id": f"f:{k}", "rpc_category_id": 1, "camera": 2, "source": f"1_camera2-{k + 1}.jpg"}
                for k in range(6)]
    cutouts += [{"cutout_id": f"b:{k}", "rpc_category_id": 1, "camera": 2, "source": f"1-back_camera2-{k + 1}.jpg"}
                for k in range(2)]
    for c in cutouts:
        c.setdefault("source", f"{c['rpc_category_id']}_camera{c['camera']}-1.jpg")
    recipes = plan_scenes(1, 300, task_config=task_config(), cutouts=cutouts, skus=skus, patterns=PATTERNS,
                          backgrounds=["b1.jpg", "b2.jpg", "b3.jpg"], layout={**LAYOUT, "view_balance": True}, seed=7)
    uses = Counter(o["cutout_id"] for r in recipes for o in r["objects"] if o["rpc_category_id"] == 1)
    front, back = (sum(n for c, n in uses.items() if c.startswith(p)) for p in ("f:", "b:"))
    assert abs(front - back) <= 1 and front + back > 100
    for prefix, k in (("f:", 6), ("b:", 2)):
        n = [uses[f"{prefix}{i}"] for i in range(k)]
        assert max(n) - min(n) <= 2
