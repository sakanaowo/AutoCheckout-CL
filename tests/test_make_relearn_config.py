import json

import pytest

from autocheckout.taskcfg import TaskConfig
from pdp_helpers import task_config_dict
from tools.make_relearn_config import main, relearn_config


def base_config():
    return TaskConfig.from_dict(task_config_dict(sizes=(3, 2), reserved=2))   # labels 0-2 | 3-4 | reserved 5-6


def test_relearn_config_appends_alias_tasks_and_keeps_the_learned_tasks():
    base = base_config()
    cfg = relearn_config(base, [[4, 0], [1]], reserved=2, name="toy+relearn")
    assert [t.num_classes for t in cfg.tasks] == [3, 2, 2, 1, 2] and cfg.num_classes == 11
    assert cfg.tasks[:2] == base.tasks[:2] and cfg.name == "toy+relearn" and cfg.seed == base.seed
    assert [(c.label, c.alias_of, c.rpc_category_id) for c in cfg.task(3).classes] == [(5, 0, 1), (6, 4, 5)]
    assert cfg.task(3).classes[0].name == "sku_1@t3" and cfg.task(4).classes[0].name == "sku_2@t4"
    assert cfg.task(3).classes[0].supercategory == "drink"
    assert cfg.task(5).reserved and cfg.task(5).classes[0].name == "reserved_8"


@pytest.mark.parametrize("tasks", [[[5]], [[0], [0]], [[0, 0]], [[0], []]])
def test_relearn_config_refuses_unlearned_repeated_or_empty(tasks):
    with pytest.raises(ValueError):
        relearn_config(base_config(), tasks, reserved=2, name="x")


def test_cli_reads_the_tasks_of_a_selection_file_in_id_order(tmp_path):
    base_config().save(tmp_path / "base.json")
    (tmp_path / "selection.json").write_text(json.dumps({"tasks": {"4": [1], "3": [4, 0]}}))
    main(["--base", str(tmp_path / "base.json"), "--selection", str(tmp_path / "selection.json"),
          "--reserved", "1", "--name", "toy+relearn", "--out", str(tmp_path / "out.json")])
    cfg = TaskConfig.load(tmp_path / "out.json")
    assert [t.num_classes for t in cfg.tasks] == [3, 2, 2, 1, 1]
    assert cfg.alias_to_label() == {5: 0, 6: 4, 7: 1}
