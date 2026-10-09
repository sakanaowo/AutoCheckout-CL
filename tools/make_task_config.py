"""DL4: split the RPC SKUs into class-incremental tasks (IMPLEMENTATION_PLAN.md, section 4.4).

- ``--sizes`` (default 100,25,25,25,25) must add up to the number of categories; ``--reserved``
  slots (default 24) form a last, reserved task (products outside RPC, no RPC id).
- Stratified by supercategory: task by task, the task size is shared among the supercategories
  in proportion to their SKUs not assigned yet (largest remainder, ties by supercategory name).
  Task totals are exact and each supercategory is spread over the tasks like the task sizes.
- Within a supercategory the SKUs are shuffled with ``--seed`` and dealt to the tasks in order;
  inside a task, classes are ordered by RPC category id. Labels follow docs/data_preprocessing/formats.md, section 3.

    python -m tools.make_task_config \\
        --categories /data/rpc/raw/retail_product_checkout/instances_test2019.json \\
        --name 100-4x25_seed0 --sizes 100,25,25,25,25 --reserved 24 --seed 0
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from autocheckout.io import load_json
from autocheckout.sampling import largest_remainder
from autocheckout.taskcfg import ClassInfo, Task, TaskConfig


def allocate(skus_per_group: dict[str, int], sizes: list[int]) -> dict[str, list[int]]:
    """SKUs of each supercategory in each task; rows sum to the SKU counts, columns to ``sizes``."""
    total = sum(skus_per_group.values())
    if sum(sizes) != total:
        raise ValueError(f"task sizes add up to {sum(sizes)}, but there are {total} SKUs")
    remaining = dict(skus_per_group)
    shares: dict[str, list[int]] = {group: [] for group in skus_per_group}
    for size in sizes:
        task_shares = largest_remainder(remaining, size)
        for group, n in task_shares.items():
            shares[group].append(n)
            remaining[group] -= n
    return shares


def build_task_config(categories: list[dict[str, Any]], sizes: list[int], reserved: int, seed: int,
                      name: str) -> TaskConfig:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for category in sorted(categories, key=lambda c: c["id"]):
        by_group[category["supercategory"]].append(category)
    shares = allocate({group: len(skus) for group, skus in by_group.items()}, sizes)

    rng = random.Random(seed)
    task_skus: list[list[dict[str, Any]]] = [[] for _ in sizes]
    for group in sorted(by_group):
        skus = list(by_group[group])
        rng.shuffle(skus)
        start = 0
        for task_index, n in enumerate(shares[group]):
            task_skus[task_index].extend(skus[start:start + n])
            start += n

    tasks, offset = [], 0
    for task_id, skus in enumerate(task_skus, start=1):
        skus.sort(key=lambda c: c["id"])
        classes = tuple(ClassInfo(label=offset + i, name=c["name"], rpc_category_id=c["id"],
                                  supercategory=c["supercategory"]) for i, c in enumerate(skus))
        tasks.append(Task(task_id=task_id, offset=offset, classes=classes))
        offset += len(classes)
    if reserved:
        classes = tuple(ClassInfo(label=offset + i, name=f"reserved_{offset + i}") for i in range(reserved))
        tasks.append(Task(task_id=len(tasks) + 1, offset=offset, classes=classes, reserved=True))
    return TaskConfig(name=name, seed=seed, tasks=tuple(tasks))


def distribution_table(config: TaskConfig) -> str:
    """Tasks x supercategory counts as a plain-text table."""
    data_tasks = config.data_tasks
    groups = sorted({c.supercategory for t in data_tasks for c in t.classes})
    lines = [f"{'supercategory':24s}" + "".join(f"{'T' + str(t.task_id):>6s}" for t in data_tasks)
             + f"{'total':>7s}"]
    for group in groups + ["total"]:
        counts = [sum(group in ("total", c.supercategory) for c in t.classes) for t in data_tasks]
        lines.append(f"{group:24s}" + "".join(f"{n:6d}" for n in counts) + f"{sum(counts):7d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", type=Path, required=True, help="COCO JSON with the SKU categories")
    parser.add_argument("--name", required=True, help="config name, e.g. 100-4x25_seed0")
    parser.add_argument("--sizes", default="100,25,25,25,25", help="number of SKUs per data task")
    parser.add_argument("--reserved", type=int, default=24, help="reserved slots in a last task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="default: configs/tasks_<name>.json")
    args = parser.parse_args(argv)

    categories = load_json(args.categories)["categories"]
    sizes = [int(size) for size in args.sizes.split(",")]
    config = build_task_config(categories, sizes, args.reserved, args.seed, args.name)
    out = args.out or Path("configs") / f"tasks_{args.name}.json"
    config.save(out)
    print(distribution_table(config))
    print(f"{config.num_slots} slots ({args.reserved} reserved), {config.num_classes} classifier outputs; "
          f"wrote {out}")


if __name__ == "__main__":
    main()
