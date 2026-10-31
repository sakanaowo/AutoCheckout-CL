"""Class-incremental task configuration.

A task config says which RPC SKU is learned in which task and fixes the mapping between
RPC category ids (1..200) and model labels (0..num_slots-1). Labels are contiguous and
ordered by task, so the classes learned up to and including task t are exactly the labels
``< seen_classes(t)``. Reserved tasks hold slots for products that are not in RPC (demo);
they have no RPC category id. An alias class (``alias_of`` = label of an earlier class with the same RPC
id) relearns an already learned SKU in a later task under a new label. The classifier has ``num_slots + 1`` outputs; the extra last
slot is never a training target (see IMPLEMENTATION_PLAN.md, section 4.4).

File format: see docs/data_preprocessing/formats.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from autocheckout.io import load_json, save_json


@dataclass(frozen=True)
class ClassInfo:
    label: int
    name: str
    rpc_category_id: int | None = None
    supercategory: str | None = None
    alias_of: int | None = None


@dataclass(frozen=True)
class Task:
    task_id: int
    offset: int
    classes: tuple[ClassInfo, ...]
    reserved: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def labels(self) -> range:
        return range(self.offset, self.offset + self.num_classes)


@dataclass(frozen=True)
class TaskConfig:
    name: str
    seed: int
    tasks: tuple[Task, ...]

    def __post_init__(self) -> None:
        self._validate()

    # --- sizes -------------------------------------------------------------------------
    @property
    def num_slots(self) -> int:
        """Number of class slots (all tasks, including reserved ones)."""
        return sum(t.num_classes for t in self.tasks)

    @property
    def num_classes(self) -> int:
        """Classifier outputs (``--n_classes`` in the PDP code): slots plus one unused slot."""
        return self.num_slots + 1

    @property
    def data_tasks(self) -> tuple[Task, ...]:
        """Tasks that have RPC data (reserved tasks excluded)."""
        return tuple(t for t in self.tasks if not t.reserved)

    # --- lookups -----------------------------------------------------------------------
    def task(self, task_id: int) -> Task:
        if not 1 <= task_id <= len(self.tasks):
            raise KeyError(f"task {task_id} not in config {self.name!r} (1..{len(self.tasks)})")
        return self.tasks[task_id - 1]

    def seen_classes(self, task_id: int) -> int:
        """Labels learned after finishing ``task_id`` are ``range(seen_classes(task_id))``."""
        task = self.task(task_id)
        return task.offset + task.num_classes

    def task_of_label(self, label: int) -> int:
        for task in self.tasks:
            if label in task.labels:
                return task.task_id
        raise KeyError(f"label {label} outside 0..{self.num_slots - 1}")

    def rpc_to_label(self) -> dict[int, int]:
        """RPC id -> label of the class that first learned it (aliases ignored)."""
        return {c.rpc_category_id: c.label for t in self.tasks for c in t.classes
                if c.rpc_category_id is not None and c.alias_of is None}

    def rpc_to_label_in(self, task_id: int) -> dict[int, int]:
        """RPC id -> label used in task ``task_id``: aliases of tasks <= task_id replace the original label
        (a later alias of the same SKU wins)."""
        mapping = self.rpc_to_label()
        for task in self.tasks[:task_id]:
            for c in task.classes:
                if c.alias_of is not None:
                    mapping[c.rpc_category_id] = c.label
        return mapping

    def alias_to_label(self) -> dict[int, int]:
        """Alias label -> original label."""
        return {c.label: c.alias_of for t in self.tasks for c in t.classes if c.alias_of is not None}

    def label_to_rpc(self) -> dict[int, int]:
        return {label: rpc for rpc, label in self.rpc_to_label().items()}

    def label_names(self) -> dict[int, str]:
        return {c.label: c.name for t in self.tasks for c in t.classes}

    # --- (de)serialisation ---------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskConfig:
        tasks = tuple(
            Task(
                task_id=int(t["task_id"]),
                offset=int(t["offset"]),
                reserved=bool(t.get("reserved", False)),
                classes=tuple(
                    ClassInfo(
                        label=int(c["label"]),
                        name=str(c["name"]),
                        rpc_category_id=(None if c.get("rpc_category_id") is None
                                         else int(c["rpc_category_id"])),
                        supercategory=c.get("supercategory"),
                        alias_of=None if c.get("alias_of") is None else int(c["alias_of"]),
                    )
                    for c in t["classes"]
                ),
            )
            for t in data["tasks"]
        )
        return cls(name=str(data["name"]), seed=int(data["seed"]), tasks=tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "num_slots": self.num_slots,
            "num_classes": self.num_classes,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "offset": t.offset,
                    "reserved": t.reserved,
                    "classes": [_class_dict(c) for c in t.classes],
                }
                for t in self.tasks
            ],
        }

    @classmethod
    def load(cls, path: str | os.PathLike) -> TaskConfig:
        return cls.from_dict(load_json(path))

    def save(self, path: str | os.PathLike) -> None:
        save_json(path, self.to_dict(), indent=1)

    # --- checks --------------------------------------------------------------------------
    def _validate(self) -> None:
        if not self.tasks:
            raise ValueError("task config has no tasks")
        expected_offset = 0
        seen_reserved = False
        rpc_ids: set[int] = set()
        earlier: dict[int, ClassInfo] = {}          # classes of the tasks before the current one
        for index, task in enumerate(self.tasks, start=1):
            if task.task_id != index:
                raise ValueError(f"task ids must be 1..N in order, got {task.task_id} at position {index}")
            if task.offset != expected_offset:
                raise ValueError(f"task {index}: offset {task.offset} != expected {expected_offset}")
            if task.num_classes == 0:
                raise ValueError(f"task {index} has no classes")
            if seen_reserved and not task.reserved:
                raise ValueError("reserved tasks must come after all data tasks")
            seen_reserved = seen_reserved or task.reserved
            for position, info in enumerate(task.classes):
                if info.label != task.offset + position:
                    raise ValueError(f"task {index}: class {info.name!r} has label {info.label}, "
                                     f"expected {task.offset + position}")
                if task.reserved != (info.rpc_category_id is None):
                    raise ValueError(f"task {index}: class {info.name!r} must have an RPC id "
                                     "iff not reserved")
                if info.alias_of is not None:
                    original = earlier.get(info.alias_of)
                    if (original is None or original.alias_of is not None
                            or original.rpc_category_id != info.rpc_category_id):
                        raise ValueError(f"task {index}: alias {info.name!r} must point to a non-alias class of "
                                         f"an earlier task with RPC id {info.rpc_category_id}, got {info.alias_of}")
                elif info.rpc_category_id is not None:
                    if info.rpc_category_id in rpc_ids:
                        raise ValueError(f"RPC category {info.rpc_category_id} assigned twice")
                    rpc_ids.add(info.rpc_category_id)
            in_task = [c.rpc_category_id for c in task.classes if c.rpc_category_id is not None]
            if len(set(in_task)) != len(in_task):
                raise ValueError(f"task {index}: an RPC category appears twice in the task")
            earlier.update({c.label: c for c in task.classes})
            expected_offset += task.num_classes


def _class_dict(c: ClassInfo) -> dict[str, Any]:
    data = {"label": c.label, "name": c.name, "rpc_category_id": c.rpc_category_id, "supercategory": c.supercategory}
    if c.alias_of is not None:
        data["alias_of"] = c.alias_of
    return data
