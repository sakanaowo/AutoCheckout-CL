"""FULL-SYN training package (docs/project/full-syn-train-plan-2026-10-22.md): inputs, stages and gates.

Training stays in scripts/run_exp.sh and pdp/main.py; this module prepares a round, runs its stages through the
runner of training_workflow, scores them on real validation images and decides the gates.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from autocheckout.io import load_json, md5_file
from autocheckout.predictions import load_predictions, save_predictions

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


def subset_run(run_dir: Path, ann_path: Path, target: Path, split: str = 'test') -> Path:
    """F0.8: a run folder holding only the images of ``ann_path`` (the 6,003 old test images of B1/B2), so the
    usual evaluators compare FULL-SYN with B1/B2 on the same images. Each task's pred_<split>.npz is filtered by
    image id and re-tagged with the md5 of ``ann_path``; task_final.pth is linked, so a policy calibrated on val
    still matches its checkpoint."""
    ann = load_json(ann_path)
    ids = np.array(sorted(image['id'] for image in ann['images']))
    ann_md5 = md5_file(ann_path)
    for task in sorted(Path(run_dir).glob('task_*')):
        source = task / f'pred_{split}.npz'
        if not source.is_file():
            continue
        preds = load_predictions(source)
        if not np.isin(ids, preds.image_id).all():
            raise ValueError(f'{source} lacks images of {ann_path}')
        subset = preds.subset(np.isin(preds.image_id, ids))
        subset.meta.update(ann_md5=ann_md5, subset_of={'file': str(source), 'md5': md5_file(source),
                                                       'ann_md5': preds.meta.get('ann_md5')})
        out = target / task.name
        save_predictions(out / source.name, subset)
        final = out / 'task_final.pth'
        if not final.is_symlink():
            final.symlink_to((task / 'task_final.pth').resolve())
    return target
