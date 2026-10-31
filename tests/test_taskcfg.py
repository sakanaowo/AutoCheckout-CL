import pytest

from autocheckout.taskcfg import TaskConfig


def make_config_dict(sizes=(3, 2), reserved=1):
    tasks, offset, rpc_id = [], 0, 1
    for task_id, size in enumerate(sizes, start=1):
        classes = []
        for i in range(size):
            classes.append({"label": offset + i, "name": f"sku{rpc_id}", "rpc_category_id": rpc_id,
                            "supercategory": "drink"})
            rpc_id += 1
        tasks.append({"task_id": task_id, "offset": offset, "reserved": False, "classes": classes})
        offset += size
    if reserved:
        tasks.append({"task_id": len(sizes) + 1, "offset": offset, "reserved": True, "classes": [
            {"label": offset + i, "name": f"reserved_{offset + i}", "rpc_category_id": None}
            for i in range(reserved)
        ]})
    return {"name": "toy", "seed": 0, "tasks": tasks}


def test_sizes_and_lookups():
    cfg = TaskConfig.from_dict(make_config_dict())
    assert cfg.num_slots == 6 and cfg.num_classes == 7
    assert [t.task_id for t in cfg.data_tasks] == [1, 2]
    assert cfg.seen_classes(1) == 3 and cfg.seen_classes(2) == 5
    assert cfg.task_of_label(4) == 2 and cfg.task_of_label(5) == 3
    assert cfg.rpc_to_label() == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    assert cfg.label_to_rpc()[3] == 4
    assert cfg.label_names()[5] == "reserved_5"


def test_round_trip(tmp_path):
    cfg = TaskConfig.from_dict(make_config_dict())
    path = tmp_path / "tasks.json"
    cfg.save(path)
    assert TaskConfig.load(path) == cfg


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["tasks"][1].update(offset=4), "offset"),
    (lambda d: d["tasks"][0]["classes"][1].update(label=7), "label"),
    (lambda d: d["tasks"][1]["classes"][0].update(rpc_category_id=1), "assigned twice"),
    (lambda d: d["tasks"][2]["classes"][0].update(rpc_category_id=99), "RPC id iff"),
    (lambda d: d["tasks"].insert(0, d["tasks"].pop()), "task ids"),
])
def test_invalid_configs_are_rejected(mutate, message):
    data = make_config_dict()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        TaskConfig.from_dict(data)


def alias_config(**alias):
    """Tasks 1-2 (labels 0,1 | 2; RPC 1,2 | 3) and task 3 relearning RPC 1 under the alias label 3."""
    data = make_config_dict(sizes=(2, 1), reserved=0)
    data["tasks"].append({"task_id": 3, "offset": 3, "reserved": False, "classes": [
        {"label": 3, "name": "sku1@t3", "rpc_category_id": 1, "supercategory": "drink", "alias_of": 0, **alias}]})
    return data


def test_alias_class_shares_the_rpc_id_of_its_original():
    cfg = TaskConfig.from_dict(alias_config())
    assert cfg.rpc_to_label() == {1: 0, 2: 1, 3: 2}
    assert cfg.alias_to_label() == {3: 0}
    assert cfg.rpc_to_label_in(2)[1] == 0 and cfg.rpc_to_label_in(3)[1] == 3
    assert cfg.rpc_to_label_in(3)[2] == 1


def test_a_later_alias_of_the_same_sku_wins():
    data = alias_config()
    data["tasks"].append({"task_id": 4, "offset": 4, "reserved": False, "classes": [
        {"label": 4, "name": "sku1@t4", "rpc_category_id": 1, "supercategory": "drink", "alias_of": 0}]})
    cfg = TaskConfig.from_dict(data)
    assert cfg.rpc_to_label_in(3)[1] == 3 and cfg.rpc_to_label_in(4)[1] == 4
    assert cfg.alias_to_label() == {3: 0, 4: 0}


@pytest.mark.parametrize("alias", [{"alias_of": 3}, {"alias_of": 7}, {"rpc_category_id": 2}])
def test_an_alias_must_point_to_an_existing_earlier_class_with_its_rpc_id(alias):
    with pytest.raises(ValueError):
        TaskConfig.from_dict(alias_config(**alias))


def test_an_alias_of_an_alias_is_refused():
    data = alias_config()
    data["tasks"].append({"task_id": 4, "offset": 4, "reserved": False, "classes": [
        {"label": 4, "name": "sku1@t4", "rpc_category_id": 1, "supercategory": "drink", "alias_of": 3}]})
    with pytest.raises(ValueError):
        TaskConfig.from_dict(data)


def test_an_alias_of_a_class_of_its_own_task_is_refused():
    data = make_config_dict(sizes=(2, 1), reserved=0)
    data["tasks"].append({"task_id": 3, "offset": 3, "reserved": False, "classes": [
        {"label": 3, "name": "sku9", "rpc_category_id": 9, "supercategory": "drink"},
        {"label": 4, "name": "sku9@t3", "rpc_category_id": 9, "supercategory": "drink", "alias_of": 3}]})
    with pytest.raises(ValueError):
        TaskConfig.from_dict(data)


def test_a_repeated_rpc_id_without_alias_is_still_refused():
    data = alias_config()
    del data["tasks"][2]["classes"][0]["alias_of"]
    with pytest.raises(ValueError, match="assigned twice"):
        TaskConfig.from_dict(data)


def test_alias_round_trips_and_plain_configs_serialise_as_before(tmp_path):
    cfg = TaskConfig.from_dict(alias_config())
    cfg.save(tmp_path / "alias.json")
    assert TaskConfig.load(tmp_path / "alias.json") == cfg
    assert cfg.to_dict()["tasks"][2]["classes"][0]["alias_of"] == 0
    plain = TaskConfig.from_dict(make_config_dict()).to_dict()
    assert all(set(c) == {"label", "name", "rpc_category_id", "supercategory"}
               for t in plain["tasks"] for c in t["classes"])
