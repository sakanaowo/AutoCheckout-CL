"""Relearn: append tasks of alias classes that relearn SKUs already learned in earlier tasks
(docs/project/task6-relearn-plan-2026-10-30.md, Task 3).

- The data tasks of ``--base`` are kept as they are; its reserved task is dropped.
- Each group of ``--selection`` (``tasks``, in task id order) becomes a task of alias classes: class i
  relearns the i-th smallest label of the group, is named ``<name>@t<task id>`` and keeps the RPC id
  and supercategory of the class it relearns.
- ``--reserved`` slots form a last, reserved task.

    python -m tools.make_relearn_config \\
        --base data/processed/rpc_eval_official_v1/task_config.json \\
        --selection configs/relearn/selection.json --reserved 24 \\
        --name 100-4x25_seed0_relearn2x25 --out configs/relearn/task_config_relearn.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

from autocheckout.io import load_json
from autocheckout.taskcfg import ClassInfo, Task, TaskConfig


def relearn_config(base: TaskConfig, tasks: Sequence[Sequence[int]], *, reserved: int,
                   name: str) -> TaskConfig:
    learned = {c.label: c for t in base.data_tasks for c in t.classes}
    if not all(tasks):
        raise ValueError("a relearn task has no classes")
    labels = [label for group in tasks for label in group]
    unknown = sorted(set(labels) - set(learned))
    if unknown:
        raise ValueError(f"labels {unknown} are not data classes of {base.name}")
    repeated = sorted(label for label, n in Counter(labels).items() if n > 1)
    if repeated:
        raise ValueError(f"labels {repeated} are relearned more than once")

    out = list(base.data_tasks)
    offset = sum(t.num_classes for t in out)
    for group in tasks:
        task_id = len(out) + 1
        classes = tuple(ClassInfo(label=offset + i, name=f"{learned[label].name}@t{task_id}",
                                  rpc_category_id=learned[label].rpc_category_id,
                                  supercategory=learned[label].supercategory, alias_of=label)
                        for i, label in enumerate(sorted(group)))
        out.append(Task(task_id=task_id, offset=offset, classes=classes))
        offset += len(classes)
    if reserved:
        classes = tuple(ClassInfo(label=offset + i, name=f"reserved_{offset + i}") for i in range(reserved))
        out.append(Task(task_id=len(out) + 1, offset=offset, classes=classes, reserved=True))
    return TaskConfig(name=name, seed=base.seed, tasks=tuple(out))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="task config of the learned tasks")
    parser.add_argument("--selection", type=Path, required=True,
                        help="JSON whose 'tasks' maps a task id to the labels it relearns")
    parser.add_argument("--reserved", type=int, default=24, help="reserved slots in a last task")
    parser.add_argument("--name", required=True, help="config name, e.g. 100-4x25_seed0_relearn2x25")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    groups = load_json(args.selection)["tasks"]
    tasks = [groups[task_id] for task_id in sorted(groups, key=int)]
    config = relearn_config(TaskConfig.load(args.base), tasks, reserved=args.reserved, name=args.name)
    config.save(args.out)
    print(f"{config.num_slots} slots ({args.reserved} reserved), {config.num_classes} classifier outputs; "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
