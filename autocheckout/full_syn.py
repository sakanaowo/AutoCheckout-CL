"""FULL-SYN training package (docs/project/full-syn-train-plan-2026-10-22.md): inputs, stages and gates.

Training stays in scripts/run_exp.sh and pdp/main.py; this module prepares a round, runs its stages through the
runner of training_workflow, scores them on real validation images and decides the gates.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from autocheckout.io import load_json

N_TASKS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def link_tasks(release: Path, eval_root: Path, target: Path) -> dict[str, str]:
    """F0.1: the annotation folder of a run, as symlinks: synthetic train_task_<t>.json, real val_task_<t>.json,
    val_full.json and test_full.json. Never the train_task_<t>_gt_full.json audit files (old-class objects of the
    synthetic images would become replay). Every file is checked against its release manifest; returns
    {name: sha256}."""
    synth = load_json(release / 'manifest.json')['tasks']
    real = load_json(eval_root / 'manifest.json')['files']
    sources = {}
    for t in range(1, N_TASKS + 1):
        sources[f'train_task_{t}.json'] = (release / 'tasks/synth_v2' / f'train_task_{t}.json',
                                           synth[str(t)]['files'][f'train_task_{t}.json'])
        sources[f'val_task_{t}.json'] = (eval_root / f'val_task_{t}.json', real[f'val_task_{t}.json']['sha256'])
    for name in ('val_full.json', 'test_full.json'):
        sources[name] = (eval_root / name, real[name]['sha256'])
    target.mkdir(parents=True, exist_ok=True)
    unexpected = {p.name for p in target.iterdir()} - set(sources)
    if unexpected:
        raise ValueError(f'{target} holds files that are not FULL-SYN inputs: {sorted(unexpected)}')
    hashes = {}
    for name, (source, expected) in sources.items():
        digest = sha256_file(source)
        if digest != expected:
            raise ValueError(f'{source}: sha256 {digest} does not match its manifest ({expected})')
        link = target / name
        if link.is_symlink() and link.resolve() != source.resolve():
            raise ValueError(f'{link} points to {link.resolve()}, expected {source}')
        if not link.is_symlink():
            link.symlink_to(source.resolve())
        hashes[name] = digest
    return hashes
