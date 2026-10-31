"""Relearn: tasks 6-7 relearn 50 SKUs of tasks 1-5 under alias classes
(docs/project/task6-relearn-plan-2026-10-30.md).

An alias class has its own label but the RPC id of the class it relearns. Before scoring, predictions are
resolved back to the original labels, so a relearn run is scored like a stage-5 model of the seed config:
``merge`` keeps the old slot and the alias of a relearned SKU (one object counted once), ``alias_only``
keeps only the alias.

The campaign (notebook 07) reuses the stages of autocheckout/full_syn.py: ``prepare`` locks a plan with the keys of a
FULL-SYN plan, ``stage_expand`` widens the seed's task-5 model, ``stage_task`` trains task 6 or 7 and decides its gate
(Q10), ``stage_final_test`` scores test once.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from autocheckout.full_syn import (FSA, N_TASKS, PDP, GateStop, _attempt, _audit, _fsa_hf, _gate, _require, _run,
                                   _toy_releases, _train, score_val, sha256_file)
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import Predictions, load_predictions, save_predictions
from autocheckout.taskcfg import TaskConfig

MODES = ("merge", "alias_only")
GAIN = {6: 0.0165, 7: 0.03}  # cAcc-M2 gain over the seed required after task 6 and task 7 (Q10)
OTHER_DROP = 0.01  # largest fall of the AP50 of the 150 SKUs not relearned


def alias_val_file(val_full: dict[str, Any], cfg: TaskConfig, task_id: int) -> dict[str, Any]:
    """Val images with an object of the SKUs alias task ``task_id`` relearns, only those objects, under the
    alias labels (the val_task file PDP reads for that task)."""
    classes = cfg.task(task_id).classes
    alias = {c.alias_of: c.label for c in classes}
    annotations = [{**a, "category_id": alias[a["category_id"]]} for a in val_full["annotations"]
                   if a["category_id"] in alias]
    keep = {a["image_id"] for a in annotations}
    return {**val_full, "images": [img for img in val_full["images"] if img["id"] in keep],
            "annotations": annotations,
            "categories": [{"id": c.label, "name": c.name, "supercategory": c.supercategory} for c in classes]}


def resolve_predictions(preds: Predictions, cfg: TaskConfig, *, mode: str, as_stage: int) -> Predictions:
    """Predictions of a relearn stage with the aliases mapped back to the original labels, as stage ``as_stage``.

    ``alias_only`` first drops the rows of the old slot of every SKU relearned so far. Rows that become the same
    (image, query, label) keep the highest score.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    task_id = int(preds.meta["task_id"])
    seen = cfg.seen_classes(task_id)
    if ((preds.label < 0) | (preds.label >= seen)).any():
        raise ValueError(f"labels outside the {seen} classes learned by task {task_id}")
    aliases = {c.label: c.alias_of for t in cfg.tasks if t.task_id <= task_id for c in t.classes
               if c.alias_of is not None}
    out = preds.subset(~np.isin(preds.label, list(aliases.values()))) if mode == "alias_only" else preds
    original = np.arange(seen)
    original[list(aliases)] = list(aliases.values())
    out = Predictions(out.image_id, out.query, original[out.label], out.score, out.boxes)
    order = np.lexsort((-out.score, out.label, out.query, out.image_id))
    key = np.stack([out.image_id[order], out.query[order], out.label[order]])
    first = np.ones(len(order), dtype=bool)
    first[1:] = (key[:, 1:] != key[:, :-1]).any(axis=0)
    out = out.subset(order[first])
    out.meta = {**preds.meta, "task_id": as_stage, "seen_classes": cfg.seen_classes(as_stage),
                "resolved_from": {"mode": mode, "task_id": task_id}}
    return out


def resolve_run(run: Path, stage: int, cfg: TaskConfig, target: Path, *, mode: str, as_stage: int,
                splits: Sequence[str] = ("val",)) -> Path:
    """``target/task_<as_stage>/``: the resolved predictions of ``run/task_<stage>`` and links to its checkpoint
    and run_info. The target holds that stage only: evaluate_policy wants the stages of the policy, each with
    the md5 of its checkpoint."""
    run, target = Path(run), Path(target)
    others = sorted(p.name for p in target.glob("task_*") if p.name != f"task_{as_stage}")
    if others:
        raise ValueError(f"{target} already holds {others}; the evaluators need task_{as_stage} only")
    source, out = run / f"task_{stage}", target / f"task_{as_stage}"
    out.mkdir(parents=True, exist_ok=True)
    for split in splits:
        preds = load_predictions(source / f"pred_{split}.npz")
        save_predictions(out / f"pred_{split}.npz", resolve_predictions(preds, cfg, mode=mode, as_stage=as_stage))
    for name in ("task_final.pth", "run_info.json"):
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink((source / name).resolve(), link)
    return target


def relearn_verdict(report: dict[str, Any], seed: dict[str, Any], stage: int) -> tuple[str, str]:
    """Gate after relearn task ``stage``: cAcc-M2 must rise by GAIN[stage] over the seed and the AP50 of the SKUs
    not relearned fall by at most OTHER_DROP. Returns (go | stop, reason naming the failed gates)."""
    gain = round(report["cAcc_m2"] - seed["cAcc_m2"], 6)  # compared after rounding, as pilot_verdict
    drop = round(seed["AP50_other"] - report["AP50_other"], 6)
    scores = (f"cAcc-M2 {seed['cAcc_m2']:.4f} -> {report['cAcc_m2']:.4f} ({gain:+.4f}, needs {GAIN[stage]:+.4f}); "
              f"AP50 of the other SKUs {seed['AP50_other']:.4f} -> {report['AP50_other']:.4f} "
              f"({-drop:+.4f}, may fall {OTHER_DROP:.4f})")
    failed = [name for name, bad in (("cAcc-M2 gain", gain < GAIN[stage]),
                                     ("AP50 of the other SKUs", drop > OTHER_DROP)) if bad]
    if failed:
        return "stop", f"{scores}: failed {' and '.join(failed)}"
    return "go", scores


def per_sku_accuracy(coco, counted: Predictions, thresholds: np.ndarray, seen: int,
                     iou: float = 0.5) -> dict[int, dict[str, Any]]:
    """Recall and precision of every learned SKU, as the 50 SKUs to relearn were chosen (min of both < 0.85).

    ``counted`` holds the boxes left by the counting policy's NMS; a box counts when its score reaches the threshold of
    its label. Objects and counted boxes are paired greedily by IoU (>= ``iou``) per image. recall = objects counted
    as their own SKU / objects of the SKU; precision = those / boxes counted as the SKU (right, on another SKU's
    object, or on no object)."""
    from collections import Counter

    from autocheckout.cl_metrics import box_iou

    counted = counted.subset(counted.score >= np.asarray(thresholds)[counted.label])
    order = np.argsort(counted.image_id, kind="stable")
    ids = counted.image_id[order]
    n, right, wrong, missed, called, extra = (Counter() for _ in range(6))
    for image_id in coco.getImgIds():
        rows = order[np.searchsorted(ids, image_id):np.searchsorted(ids, image_id, "right")]
        anns = [a for a in coco.imgToAnns[image_id] if a["category_id"] < seen]
        pairs, taken = {}, set()
        if anns and len(rows):
            gt = np.array([a["bbox"] for a in anns], float)
            gt[:, 2:] += gt[:, :2]
            overlap = box_iou(gt, counted.boxes[rows])
            for _, i, j in sorted(((overlap[i, j], i, j) for i, j in zip(*np.nonzero(overlap >= iou))), reverse=True):
                if i not in pairs and j not in taken:
                    pairs[i] = j
                    taken.add(j)
        for i, ann in enumerate(anns):
            label = ann["category_id"]
            n[label] += 1
            if i not in pairs:
                missed[label] += 1
            elif int(counted.label[rows[pairs[i]]]) == label:
                right[label] += 1
            else:
                wrong[label] += 1
                called[int(counted.label[rows[pairs[i]]])] += 1
        for j in sorted(set(range(len(rows))) - taken):
            extra[int(counted.label[rows[j]])] += 1
    return {label: {"n_val": n[label], "recall": round(right[label] / n[label], 4) if n[label] else None,
                    "precision": round(right[label] / max(1, right[label] + called[label] + extra[label]), 4),
                    "wrong": wrong[label], "missed": missed[label], "others_called_this": called[label],
                    "extra_boxes": extra[label]} for label in range(seen)}


# --- campaign ------------------------------------------------------------------------------------------------------
RELEARN = 'EXP-FULL-SYN-RL'
RELEARN_TASKS = (6, 7)
SEED_LOCKED = ('resolution', 'batch_size', 'effective_batch', 'epochs', 'tf32', 'num_workers',
               'checkpoint_every_minutes')  # the training recipe of the seed round, kept as it is
DRY_GROUPS = [[1], [3]]  # dry run: tasks 6 and 7 relearn labels 1 and 3 of the six-label fixture


def relearn_tasks(cfg: TaskConfig, seed_cfg: TaskConfig) -> list[int]:
    """The tasks of ``cfg`` made only of aliases (RELEARN_TASKS), once its tasks 1-5 are checked against the seed's."""
    if [t for t in cfg.data_tasks if t.task_id <= N_TASKS] != list(seed_cfg.data_tasks):
        raise ValueError(f'tasks 1-{N_TASKS} of {cfg.name} differ from those of the seed config {seed_cfg.name}')
    tasks = [t.task_id for t in cfg.data_tasks if all(c.alias_of is not None for c in t.classes)]
    if tasks != list(RELEARN_TASKS):
        raise ValueError(f'{cfg.name} has alias tasks {tasks}, expected {list(RELEARN_TASKS)}')
    return tasks


def _relearn_fixture(root: Path, seed_cfg: TaskConfig) -> tuple[Path, Path]:
    """Dry run: the relearn config of the six-label fixture and a synthetic release of its tasks 6-7, made from the
    fixture's train_full.json. Returns (release, task_config)."""
    from tools.make_relearn_config import relearn_config

    release = root / 'relearn_synth'
    config = release / 'task_config.json'
    if not (release / 'manifest.json').is_file():
        cfg = relearn_config(seed_cfg, DRY_GROUPS, reserved=1, name='dry-run-relearn')
        cfg.save(config)
        train_full, to_rpc, names = load_json(root / 'toy/tasks/train_full.json'), seed_cfg.label_to_rpc(), cfg.label_names()
        tasks = {}
        for t in RELEARN_TASKS:
            alias, to_label = {c.alias_of: c.label for c in cfg.task(t).classes}, cfg.rpc_to_label_in(t)
            files = {f'train_task_{t}.json': [{**a, 'category_id': alias[a['category_id']]}
                                               for a in train_full['annotations'] if a['category_id'] in alias],
                     f'train_task_{t}_gt_full.json': [{**a, 'category_id': to_label[to_rpc[a['category_id']]]}
                                                       for a in train_full['annotations']]}
            for name, annotations in files.items():
                keep = {a['image_id'] for a in annotations}
                labels = sorted({a['category_id'] for a in annotations})
                save_json(release / 'tasks/synth_v2' / name, {
                    'images': [img for img in train_full['images'] if img['id'] in keep], 'annotations': annotations,
                    'categories': [{'id': label, 'name': names[label]} for label in labels]})
            tasks[str(t)] = {'files': {name: sha256_file(release / 'tasks/synth_v2' / name) for name in files}}
        save_json(release / 'manifest.json', {'config_sha256': 'dry-run', 'tasks': tasks})
    return release, config


def _link_inputs(release: Path, eval_root: Path, cfg: TaskConfig, tasks: list[int], target: Path) -> dict[str, str]:
    """The annotation folder of the relearn run: train_task_<t>.json of the release and val_full/test_full.json of the
    evaluation set as symlinks checked against their manifests, val_task_<t>.json written by alias_val_file. The
    train_task_<t>_gt_full.json audit files are checked, never linked (F0.1: their old-class objects would become
    replay). Returns {name: sha256}."""
    synth, real = load_json(release / 'manifest.json')['tasks'], load_json(eval_root / 'manifest.json')['files']
    links = {name: (eval_root / name, real[name]['sha256']) for name in ('val_full.json', 'test_full.json')}
    audit = {}
    for t in tasks:
        files = synth[str(t)]['files']
        for name, group in ((f'train_task_{t}.json', links), (f'train_task_{t}_gt_full.json', audit)):
            group[name] = (release / 'tasks/synth_v2' / name, files[name])
    written = {f'val_task_{t}.json' for t in tasks}
    target.mkdir(parents=True, exist_ok=True)
    unexpected = {p.name for p in target.iterdir()} - set(links) - written
    if unexpected:
        raise ValueError(f'{target} holds files that are not relearn inputs: {sorted(unexpected)}')
    hashes = {}
    for name, (source, expected) in {**links, **audit}.items():
        digest = sha256_file(source)
        if digest != expected:
            raise ValueError(f'{source}: sha256 {digest} does not match its manifest ({expected})')
        hashes[name] = digest
    for name, (source, _) in links.items():
        link = target / name
        if link.is_symlink() and link.resolve() != source.resolve():
            raise ValueError(f'{link} points to {link.resolve()}, expected {source}')
        if not link.is_symlink():
            link.symlink_to(source.resolve())
    val = load_json(eval_root / 'val_full.json')
    for t in tasks:
        save_json(target / f'val_task_{t}.json', alias_val_file(val, cfg, t))
        hashes[f'val_task_{t}.json'] = sha256_file(target / f'val_task_{t}.json')
    return hashes


def prepare(settings: dict) -> dict:
    """Plan of the relearn campaign: the keys of a FULL-SYN plan (so its runner, audit and attempts work unchanged),
    the training recipe of the seed round, and the identity of the seed (task-5 model) and of the relearn config. A
    changed setting or input in an existing campaign is refused, as in full_syn.prepare."""
    from autocheckout.training_workflow import BACKBONE_REVISION, DETECTOR_REVISION, campaign_root, resolve

    root = campaign_root(settings)
    dry = bool(settings.get('dry_run', True))
    seed = resolve(settings['seed_campaign'])
    seed_plan = load_json(seed / 'plan.json')
    if seed_plan['dry_run'] != dry:
        raise ValueError('a dry run and a real run cannot seed each other')
    if seed_plan.get('pdp_experiment', PDP) != PDP:
        raise ValueError('the seed campaign must be a plain FULL-SYN run')
    changed = [k for k in SEED_LOCKED if not dry and k in settings and settings[k] != seed_plan[k]]
    if changed:
        raise ValueError(f'settings {changed} differ from the seed campaign; relearn keeps its training recipe')
    if dry:
        _, eval_root, images, model = _toy_releases(root / 'fixture')
        release, task_config = _relearn_fixture(root / 'fixture', TaskConfig.load(eval_root / 'task_config.json'))
        synth_root = raw = images
        detector = backbone = ''
    else:
        release, eval_root = resolve(settings['release_root']), resolve(settings['eval_root'])
        task_config = resolve(settings['task_config'])
        synth_root, raw, model = release, resolve(settings.get('raw_root', 'data/archive')), ''
        cache = resolve(settings.get('pretrained_root', 'data/pretrained/hf-cache'))
        detector = str(cache / f'models--SenseTime--deformable-detr/snapshots/{DETECTOR_REVISION}')
        backbone = str(cache / f'models--timm--convnextv2_base.fcmae_ft_in22k_in1k/snapshots/{BACKBONE_REVISION}'
                       / 'model.safetensors')
        for path in (Path(detector) / 'model.safetensors', Path(backbone)):
            if not path.is_file():
                raise FileNotFoundError(f'pinned pretrained weights missing: {path}')
    seed_task_config = eval_root / 'task_config.json'
    if sha256_file(seed_task_config) != seed_plan['task_config_sha256']:
        raise ValueError(f'{seed_task_config} is not the task config of the seed campaign')
    if sha256_file(task_config) != sha256_file(release / 'task_config.json'):
        raise ValueError(f'task_config.json of {release} is not {task_config}')
    cfg = TaskConfig.load(task_config)
    tasks = relearn_tasks(cfg, TaskConfig.load(seed_task_config))
    task_dir = root / 'inputs/tasks'
    hashes = _link_inputs(release, eval_root, cfg, tasks, task_dir)
    plan = dict(campaign_id=settings['campaign_id'], round=int(settings['round']), dry_run=dry, root=str(root),
                runs=str(root / 'runs'), release_root=str(release), eval_root=str(eval_root),
                synth_root=str(synth_root), raw_root=str(raw), task_dir=str(task_dir), task_config=str(task_config),
                model_config=str(model), detector_dir=detector, backbone_weights=backbone,
                **{k: seed_plan[k] for k in (*SEED_LOCKED, 'fsa_batch_size', 'fsa_epochs')},
                audit_images=int(settings.get('audit_images', 1000)),
                release_config_sha256=load_json(release / 'manifest.json').get('config_sha256'),
                input_sha256=hashes, task_config_sha256=sha256_file(task_config),
                pretrained_md5={} if dry else {'detector': md5_file(Path(detector) / 'model.safetensors'),
                                               'backbone': md5_file(backbone)},
                train_images=[len(load_json(task_dir / f'train_task_{t}.json')['images']) for t in tasks],
                pdp_experiment=RELEARN, seed_root=str(seed),
                seed=dict(campaign_id=seed_plan['campaign_id'], round=seed_plan['round'], dry_run=seed_plan['dry_run'],
                          task5_md5=md5_file(seed / 'runs' / PDP / f'task_{N_TASKS}/task_final.pth')),
                seed_task_config=str(seed_task_config), n_classes=cfg.num_classes, relearn_tasks=tasks,
                relearned=sorted(c.alias_of for t in tasks for c in cfg.task(t).classes))
    path = root / 'plan.json'
    locations = {'root', 'runs', 'release_root', 'eval_root', 'synth_root', 'raw_root', 'task_dir', 'task_config',
                 'model_config', 'detector_dir', 'backbone_weights', 'seed_root', 'seed_task_config'}
    if path.is_file():
        previous = load_json(path)
        if {k: v for k, v in previous.items() if k not in locations} != {k: v for k, v in plan.items()
                                                                         if k not in locations}:
            raise ValueError('settings or inputs changed in an existing campaign; start a new campaign_id')
    save_json(path, plan, indent=2)
    return plan


def _finish(plan: dict, stage: str, report: dict, gate: dict | None) -> dict:
    """full_syn._finish without its SUMMARY.md, whose table reads the stages of a FULL-SYN round."""
    from autocheckout.model_acceptance import now

    result = {**report, 'gate': gate, 'finished_at': now(), 'campaign_id': plan['campaign_id'], 'round': plan['round'],
              'release_config_sha256': plan['release_config_sha256'], 'input_sha256': plan['input_sha256']}
    save_json(Path(plan['root']) / 'stages' / f'{stage}.json', result, indent=2)
    if gate and gate['decision'] == 'stop' and gate['enforced']:
        raise GateStop(f"{gate['name']}: {gate['reason']}")
    return result


def _ap50_other(plan: dict, path: Path) -> float:
    """AP50 of the seed's learned SKUs that are not relearned, on val, without NMS (as eval_cl)."""
    from autocheckout.cl_metrics import group_ap, load_coco

    relearned = set(plan['relearned'])
    labels = [label for label in range(TaskConfig.load(plan['seed_task_config']).seen_classes(N_TASKS))
              if label not in relearned]
    coco = load_coco(load_json(Path(plan['task_dir']) / 'val_full.json'))
    return group_ap(coco, load_predictions(path), labels)['AP50']


def stage_expand(plan: dict) -> dict:
    """Task 5 of the relearn run: the seed's task-5 model with the class slots of the relearn config (Q4); the FSA run
    of the seed is linked, as full_syn.stage_seed does. Also scores the seed on the SKUs not relearned."""
    from autocheckout.training_workflow import REPO

    done = Path(plan['root']) / 'stages/expand.json'
    if done.is_file():
        return load_json(done)
    seed, runs = Path(plan['seed_root']), Path(plan['runs'])
    source = seed / 'runs' / PDP / f'task_{N_TASKS}/task_final.pth'
    if md5_file(source) != plan['seed']['task5_md5']:
        raise ValueError(f'task {N_TASKS} of {seed} changed since this campaign was prepared')
    runs.mkdir(parents=True, exist_ok=True)
    link = runs / FSA
    if not link.is_symlink():
        link.symlink_to((seed / 'runs' / FSA).resolve())
    elif link.resolve() != (seed / 'runs' / FSA).resolve():
        raise ValueError(f'{link} points to {link.resolve()}, expected {seed / "runs" / FSA}')
    out = runs / RELEARN / f'task_{N_TASKS}/task_final.pth'
    learned = TaskConfig.load(plan['seed_task_config']).seen_classes(N_TASKS)
    with _attempt(plan, 'expand') as (record, attempt):
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            _run(record, attempt, 'expand', [sys.executable, str(REPO / 'pdp/expand.py'), '--checkpoint', str(source),
                 '--old-config', plan['seed_task_config'], '--new-config', plan['task_config'],
                 '--learned', str(learned), '--out', str(out)], {})
    seed_tasks = load_json(seed / 'stages/tasks.json')['tasks']
    report = {'stage': 'expand', 'learned': learned, 'checkpoint_md5': md5_file(out),
              'seed_scores': {'cAcc_m2': seed_tasks[str(N_TASKS)]['cAcc_m2'],
                              'AP50_other': _ap50_other(plan, seed / 'runs' / PDP / f'task_{N_TASKS}/pred_val.npz')}}
    return _finish(plan, 'expand', report, None)


def _score(plan: dict, task: int, mode: str) -> dict:
    """Val scores of relearn task ``task`` resolved with ``mode``, as stage 5 of the seed config (its policies are
    saved next to the predictions), plus the AP50 of the SKUs not relearned."""
    target = resolve_run(Path(plan['runs']) / RELEARN, task, TaskConfig.load(plan['task_config']),
                         Path(plan['root']) / 'scored' / f'task{task}_{mode}', mode=mode, as_stage=N_TASKS)
    report = score_val({'task_config': plan['seed_task_config'], 'task_dir': plan['task_dir']}, target, N_TASKS)
    return {**report, 'AP50_other': _ap50_other(plan, target / f'task_{N_TASKS}/pred_val.npz')}


def stage_task(plan: dict, task: int) -> dict:
    """Relearn task 6 or 7: audit of the teacher's pseudo-labels (recorded, not a gate), training, both alias modes
    scored on val, the better one (Q9: higher cAcc-M2, merge on a tie) through the gate of the task (Q10). Task 7 runs
    only after a go at task 6; in a dry run gates are reported, not enforced. A finished task is never trained again."""
    if task not in plan['relearn_tasks']:
        raise ValueError(f"relearn tasks are {plan['relearn_tasks']}, got {task}")
    if (Path(plan['root']) / 'stages' / f'task{task}.json').is_file():
        return _require(plan, f'task{task}')
    _require(plan, 'expand' if task == plan['relearn_tasks'][0] else f'task{task - 1}')
    seed_scores = _require(plan, 'expand')['seed_scores']
    with _attempt(plan, f'task{task}') as (record, attempt):
        audit = _audit(plan, record, attempt, task)
        _train(plan, record, attempt, RELEARN, start=task, n=task, epochs=plan['epochs'], splits='val',
               fsa_hf=_fsa_hf(plan))
    modes = {mode: _score(plan, task, mode) for mode in MODES}
    mode = max(MODES, key=lambda m: modes[m]['cAcc_m2'])
    decision, reason = relearn_verdict(modes[mode], seed_scores, task)
    gate = _gate(plan, f'relearn task {task}', decision, reason, mode=mode, gain=GAIN[task], other_drop=OTHER_DROP)
    report = {'stage': task, 'mode': mode, 'modes': modes, 'seed': seed_scores,
              'pseudo_labels': {key: audit.get(key) for key in ('images', 'earlier_objects', 'recall', 'all',
                                                                'branches', 'filters', 'confusions_top')}}
    return _finish(plan, f'task{task}', report, gate)


def sku_accuracy(plan: dict, task: int) -> dict[int, dict[str, Any]]:
    """per_sku_accuracy of every SKU of the seed config on val after relearn task ``task``, resolved with the mode its
    gate chose and counted with that mode's M2 policy, the policy the SKUs were chosen with. Reads the finished stage
    without its gate, so it also tells which SKUs a stopped task did not relearn."""
    from autocheckout.cl_metrics import load_coco
    from autocheckout.counting import dedup_detections, label_thresholds

    path = Path(plan['root']) / 'stages' / f'task{task}.json'
    if not path.is_file():
        raise RuntimeError(f'run stage task{task} first')
    target = Path(plan['root']) / 'scored' / f"task{task}_{load_json(path)['mode']}"
    policy = load_json(target / 'calibration_count_m2.json')['stages'][str(N_TASKS)]
    cfg = TaskConfig.load(plan['seed_task_config'])
    thresholds = label_thresholds([list(cfg.task(t).labels) for t in range(1, N_TASKS + 1)],
                                  [policy['group_thresholds'][f'task_{t}'] for t in range(1, N_TASKS + 1)])
    preds = load_predictions(target / f'task_{N_TASKS}/pred_val.npz')
    counted = dedup_detections(preds, policy['nms_iou']) if policy['nms_iou'] else preds.top1_per_query()
    return per_sku_accuracy(load_coco(load_json(Path(plan['task_dir']) / 'val_full.json')), counted, thresholds,
                            cfg.seen_classes(N_TASKS))


def stage_final_test(plan: dict, confirm: bool = False) -> dict:
    """Test2019, scored once with the val policies of the mode chosen at the last task, after its go and only with
    ``confirm=True``. An existing result is returned as it is, never recomputed."""
    from tools import eval_cl, eval_count

    root = Path(plan['root'])
    done = root / 'stages/final_test.json'
    if done.is_file():
        print(f'test already scored on {load_json(done)["finished_at"]}; not recomputed')
        return load_json(done)
    last = plan['relearn_tasks'][-1]
    mode = _require(plan, f'task{last}')['mode']
    if not confirm:
        raise RuntimeError('test is scored once: call stage_final_test(plan, confirm=True) to score it')
    with _attempt(plan, 'final_test') as (record, attempt):
        _train(plan, record, attempt, RELEARN, start=last, n=last, epochs=plan['epochs'], splits='test',
               fsa_hf=_fsa_hf(plan))
    target = resolve_run(Path(plan['runs']) / RELEARN, last, TaskConfig.load(plan['task_config']),
                         root / 'scored' / f'task{last}_{mode}', mode=mode, as_stage=N_TASKS, splits=('test',))
    cfg, test = TaskConfig.load(plan['seed_task_config']), Path(plan['task_dir']) / 'test_full.json'
    detection = eval_cl.evaluate_run(target, 'test', test, cfg)
    save_json(target / 'metrics_cl_test.json', detection, indent=1)
    counting = {}
    for label, policy in (('groups', 'calibration_count_groups.json'), ('single', 'calibration_count.json')):
        result = eval_count.evaluate_policy(target, test, cfg, load_json(target / policy))
        save_json(target / f'metrics_count_test_{label}.json', result, indent=1)
        counting[policy] = {s: r['test']['overall'] for s, r in result['stages'].items()}
    report = {'stage': 'final_test', 'mode': mode,
              'test2019': {'AP50': {s: v['m1']['mAP_A']['AP50'] for s, v in detection['stages'].items()},
                           'counting': counting}}
    return _finish(plan, 'final_test', report, None)
