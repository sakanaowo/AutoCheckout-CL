"""FULL-SYN training package (docs/project/full-syn-train-plan-2026-10-22.md): inputs, stages and gates.

Training stays in scripts/run_exp.sh and pdp/main.py; this module prepares a round, runs its stages through the
runner of training_workflow, scores them on real validation images and decides the gates.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import uuid
from collections import Counter
from contextlib import contextmanager
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


# --- rounds, stages and gates ------------------------------------------------------------------------------------
# Gates of the plan (section 5) and of synthetic-data-v2-decisions-2026-10-21.md (D16-D19); B2 task 1 = 85.79%.
FSA_EARLY_CHECK = 0.528                   # D18: val cAcc-M2 of FSA
TASK1_GREEN, TASK1_RED = 0.728, 0.578     # D17: B2 - 13 / - 28 points
PSEUDO_PRECISION_MIN = 0.5                # plan 4.1 (user, 25/10): teacher pseudo-labels on task 2 images
MAX_ROUNDS = 3                            # D19
UNLEARNED_IOU = 0.5                       # D16: cAcc-M2
FSA, PDP = 'EXP-FULL-SYN-FSA', 'EXP-FULL-SYN'
DRY_MODEL_ARGS = ['--backbone', '', '--backbone_pretrained_file', '', '--n_classes', '7', '--accelerator', 'cpu',
                  '--num_prompts', '4', '--prompt_len', '2', '--proto_correct_only', '0', '--shuffle', '0']


class GateStop(RuntimeError):
    """A gate of the plan stops the round; the notebook cell fails with the decision."""


def _toy_releases(root: Path) -> tuple[Path, Path, Path, Path]:
    """Dry run: the six-label CPU fixture of training_workflow, laid out as a synthetic release and an evaluation
    set with manifests. Returns (release, eval_root, images, model_config)."""
    from autocheckout.io import save_json
    from autocheckout.training_workflow import _toy_inputs

    toy, release, eval_root = root / 'toy', root / 'synth', root / 'eval'
    _toy_inputs(toy)
    if not (eval_root / 'manifest.json').is_file():
        train_full = load_json(toy / 'tasks/train_full.json')
        tasks = {}
        for t in range(1, N_TASKS + 1):
            save_json(release / 'tasks/synth_v2' / f'train_task_{t}.json',
                      load_json(toy / f'tasks/train_task_{t}.json'))
            save_json(release / 'tasks/synth_v2' / f'train_task_{t}_gt_full.json', train_full)
            tasks[str(t)] = {'files': {n: sha256_file(release / 'tasks/synth_v2' / n)
                                       for n in (f'train_task_{t}.json', f'train_task_{t}_gt_full.json')}}
        save_json(release / 'manifest.json', {'config_sha256': 'dry-run', 'tasks': tasks})
        test = load_json(toy / 'tasks/test_full.json')
        names = [f'val_task_{t}.json' for t in range(1, N_TASKS + 1)] + ['val_full.json', 'test_full.json']
        for name in names:
            save_json(eval_root / name, load_json(toy / 'tasks' / name))
        first = test['images'][0]['id']
        save_json(eval_root / 'compare_b12/test_full.json', {**test, 'images': test['images'][:1],
                  'annotations': [a for a in test['annotations'] if a['image_id'] == first]})
        shutil.copy2(toy / 'task_config.json', eval_root / 'task_config.json')
        shutil.copy2(toy / 'task_config.json', release / 'task_config.json')
        save_json(eval_root / 'manifest.json', {'files': {n: {'sha256': sha256_file(eval_root / n)}
                                                          for n in names}})
    return release, eval_root, toy / 'images', toy / 'model'


def prepare(settings: dict) -> dict:
    """Plan of one round (one campaign): inputs linked and hashed, settings locked. A changed setting or input in an
    existing campaign is refused: a new round is a new campaign and a new notebook file (plan 4.1)."""
    from autocheckout.io import save_json
    from autocheckout.taskcfg import TaskConfig
    from autocheckout.training_workflow import BACKBONE_REVISION, DETECTOR_REVISION, campaign_root, resolve

    root = campaign_root(settings)
    dry = bool(settings.get('dry_run', True))
    round_id = int(settings['round'])
    if not 1 <= round_id <= MAX_ROUNDS:
        raise ValueError(f'round must be 1..{MAX_ROUNDS} (D19)')
    physical, effective = int(settings['batch_size']), int(settings['effective_batch'])
    # FSA trains the whole model, so its activations do not fit at the PDP batch: batch 8 runs out of the 4090's
    # 24 GB at step 0, batch 4 peaks at 15.8 GiB. FSA keeps the effective batch with gradient accumulation.
    fsa_physical = int(settings.get('fsa_batch_size', physical))
    for batch in (physical, fsa_physical):
        if batch < 1 or effective < batch or effective % batch:
            raise ValueError('effective batch must be a positive multiple of physical batch (one GPU)')
    if dry:
        release, eval_root, images, model = _toy_releases(root / 'fixture')
        synth_root = raw = images
        detector = backbone = ''
    else:
        release, eval_root = resolve(settings['release_root']), resolve(settings['eval_root'])
        synth_root, raw, model = release, resolve(settings.get('raw_root', 'data/archive')), ''
        cache = resolve(settings.get('pretrained_root', 'data/pretrained/hf-cache'))
        detector = str(cache / f'models--SenseTime--deformable-detr/snapshots/{DETECTOR_REVISION}')
        backbone = str(cache / f'models--timm--convnextv2_base.fcmae_ft_in22k_in1k/snapshots/{BACKBONE_REVISION}'
                       / 'model.safetensors')
        for path in (Path(detector) / 'model.safetensors', Path(backbone)):
            if not path.is_file():
                raise FileNotFoundError(f'pinned pretrained weights missing: {path}')
    task_dir = root / 'inputs/tasks'
    hashes = link_tasks(release, eval_root, task_dir)
    task_config = eval_root / 'task_config.json'
    if sha256_file(task_config) != sha256_file(release / 'task_config.json'):
        raise ValueError('task_config.json of the synthetic release and of the evaluation set differ')
    cfg = TaskConfig.load(task_config)
    if not dry and [t.num_classes for t in cfg.data_tasks] != [100, 25, 25, 25, 25]:
        raise ValueError('FULL-SYN requires the locked 100+4x25 protocol')
    plan = dict(campaign_id=settings['campaign_id'], round=round_id, dry_run=dry, root=str(root),
                runs=str(root / 'runs'), release_root=str(release), eval_root=str(eval_root),
                synth_root=str(synth_root),
                raw_root=str(raw), task_dir=str(task_dir), task_config=str(task_config), model_config=str(model),
                detector_dir=detector, backbone_weights=backbone,
                resolution=96 if dry else int(settings['resolution']), batch_size=physical, effective_batch=effective,
                fsa_batch_size=fsa_physical,
                num_workers=0 if dry else int(settings.get('num_workers', 8)), tf32=bool(settings.get('tf32', False)),
                fsa_epochs=1 if dry else int(settings['fsa_epochs']), epochs=1 if dry else int(settings['epochs']),
                checkpoint_every_minutes=float(settings.get('checkpoint_every_minutes', 2)),
                audit_images=int(settings.get('audit_images', 1000)),
                release_config_sha256=load_json(release / 'manifest.json').get('config_sha256'),
                input_sha256=hashes, task_config_sha256=sha256_file(task_config),
                pretrained_md5={} if dry else {'detector': md5_file(Path(detector) / 'model.safetensors'),
                                               'backbone': md5_file(backbone)},
                train_images=[len(load_json(task_dir / f'train_task_{t}.json')['images'])
                              for t in range(1, N_TASKS + 1)])
    path = root / 'plan.json'
    locations = {'root', 'runs', 'release_root', 'eval_root', 'synth_root', 'raw_root', 'task_dir', 'task_config',
                 'model_config', 'detector_dir', 'backbone_weights'}
    if path.is_file():
        previous = load_json(path)
        if {k: v for k, v in previous.items() if k not in locations} != {k: v for k, v in plan.items()
                                                                         if k not in locations}:
            raise ValueError('settings or inputs changed in an existing campaign; start a new campaign_id')
    save_json(path, plan, indent=2)
    return plan


@contextmanager
def _attempt(plan: dict, stage: str):
    """One locked execution of a stage, with its receipt (git state, commands, status) under attempts/."""
    from autocheckout.io import save_json
    from autocheckout.model_acceptance import now
    from autocheckout.runinfo import git_state
    from autocheckout.training_workflow import REPO, process_start

    root = Path(plan['root'])
    with (root / 'training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another notebook is using this campaign') from error
        for receipt in root.glob('attempts/*/*/*.process.json'):
            previous = load_json(receipt)
            if (previous['status'] == 'RUNNING' and previous.get('host') == socket.gethostname()
                    and previous.get('process_start') is not None
                    and previous['process_start'] == process_start(previous['pid'])):
                raise RuntimeError(f'runner PID {previous["pid"]} is still running; reconnect/stop it before resume')
        attempt = root / 'attempts' / stage / str(uuid.uuid4())
        attempt.mkdir(parents=True)
        record = dict(status='RUNNING', stage=stage, started_at=now(), git=git_state(REPO), commands=[])
        save_json(attempt / 'execution.json', record, indent=2)
        try:
            yield record, attempt
            record.update(status='COMPLETED', finished_at=now())
        except BaseException as error:
            record.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else
                          'GATE_STOP' if isinstance(error, GateStop) else 'FAIL',
                          error_type=type(error).__name__, error=str(error), finished_at=now())
            raise
        finally:
            save_json(attempt / 'execution.json', record, indent=2)


def _environment(plan: dict, experiment: str, runs: Path, **values) -> dict[str, str]:
    from autocheckout.training_workflow import REPO

    root = Path(plan['root'])
    batch = plan.get('fsa_batch_size', plan['batch_size']) if experiment == FSA else plan['batch_size']
    return dict(EXP=experiment, RUNS=str(runs), REPO=str(REPO), RAW_ROOT=plan['raw_root'], TASK_DIR=plan['task_dir'],
                TASK_CFG=plan['task_config'], SYNTH_ROOT=plan['synth_root'], DETECTOR_DIR=plan['detector_dir'],
                BACKBONE_WEIGHTS=plan['backbone_weights'], RESOLUTION=str(plan['resolution']),
                BATCH_SIZE=str(batch), EFFECTIVE_BATCH=str(plan['effective_batch']),
                NUM_WORKERS=str(plan['num_workers']), TF32=str(int(plan['tf32'])),
                CKPT_EVERY_MINUTES=str(plan['checkpoint_every_minutes']), VERIFY_RESUME='1', REQUIRE_KERNEL='0',
                EVALUATE='0', PYTHON=sys.executable, PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1', USE_TF='0', TORCH_EXTENSIONS_DIR=str(root / 'cache/torch_extensions'),
                MPLCONFIGDIR=str(root / 'cache/matplotlib'), **{k: str(v) for k, v in values.items()})


def _config(plan: dict, experiment: str, runs: Path, fsa_hf: str = '') -> Path:
    """resolved_config.sh of a run: the experiment config, plus the tiny CPU model in a dry run."""
    from autocheckout.io import save_json

    output = runs / experiment
    output.mkdir(parents=True, exist_ok=True)
    lines = [f'source "$REPO/configs/exp/native/{experiment}.sh"']
    if plan['dry_run']:
        model = ['--repo_name', '', '--model_config', plan['model_config']] if experiment == FSA else \
            ['--model_config', fsa_hf]
        lines.append(f'ARGS+=({shlex.join(DRY_MODEL_ARGS + model)})')
    config = output / 'resolved_config.sh'
    config.write_text('\n'.join(lines) + '\n')
    environment = _environment(plan, experiment, runs, FSA_HF=fsa_hf)
    resolved = subprocess.run(['bash', '-ec', 'source "$CONFIG"; printf "%s\\0" "${ARGS[@]}"'],
                              env={**os.environ, **environment, 'CONFIG': str(config)},
                              capture_output=True, text=True, check=True)
    save_json(output / 'resolved_cli.json', dict(arguments=resolved.stdout.split('\0')[:-1],
                                                environment=environment), indent=2)
    return config


def _run(record: dict, attempt: Path, name: str, command: list[str], environment: dict,
         allowed: tuple[int, ...] = (0,)) -> int:
    from autocheckout.io import save_json
    from autocheckout.model_acceptance import now
    from autocheckout.training_workflow import _run_logged

    log = attempt / f'{name}_runner.log'
    entry = dict(name=name, command=command, environment=environment, log=str(log), started_at=now())
    record['commands'].append(entry)
    save_json(attempt / 'execution.json', record, indent=2)
    code = _run_logged(command, environment, log)
    entry.update(returncode=code, finished_at=now())
    save_json(attempt / 'execution.json', record, indent=2)
    if code not in allowed:
        raise RuntimeError(f'{name} exited {code}; inspect {log}; rerun the notebook to resume')
    return code


def _train(plan: dict, record: dict, attempt: Path, experiment: str, *, start: int, n: int, epochs: int,
           splits: str, runs: Path | None = None, fsa_hf: str = '', stop_after: int = 0) -> int:
    """Train (or resume, or only predict) tasks start..n of a run through scripts/run_exp.sh."""
    from autocheckout.training_workflow import REPO

    runs = runs or Path(plan['runs'])
    config = _config(plan, experiment, runs, fsa_hf)
    environment = _environment(plan, experiment, runs, FSA_HF=fsa_hf, START_TASK=start, N_TASKS=n, EPOCHS=epochs,
                               PRED_SPLITS=splits, STOP_AFTER_STEPS=stop_after)
    if plan['dry_run']:
        environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return _run(record, attempt, f'{experiment}_task{start}-{n}_{splits}', ['bash', str(REPO / 'scripts/run_exp.sh'),
                str(config)], environment, allowed=(0, 75) if stop_after else (0,))


def _fsa_hf(plan: dict) -> str:
    return str(Path(plan['runs']) / FSA / 'task_1/hf_model')


# --- validation scores -------------------------------------------------------------------------------------------
def _merge_policy(path: Path, policy: dict) -> dict:
    """Policies are chosen stage by stage; the file through task t holds stages 1..t (test needs all of them)."""
    from autocheckout.io import save_json

    if path.is_file():
        merged = load_json(path)
        merged['stages'].update(policy['stages'])
        policy = {**policy, 'stages': merged['stages']}
    save_json(path, policy, indent=2)
    return policy


def confusions(coco, preds, seen: int, threshold, previous: int, top: int = 20, iou: float = 0.5) -> list[dict]:
    """Most frequent SKU confusions at the counting threshold: an annotated learned object whose best-overlapping
    counted box (IoU >= iou) has another label. ``new_to_old``: an object of the current task taken for an earlier
    class (labels < previous), the failure pseudo-labels cause."""
    from autocheckout.cl_metrics import box_iou

    thresholds = np.asarray(threshold)[preds.label] if np.ndim(threshold) else threshold
    counted = preds.subset(preds.score >= thresholds)
    pairs = Counter()
    for image_id in coco.getImgIds():
        anns = [a for a in coco.imgToAnns[image_id] if a['category_id'] < seen]
        rows = np.flatnonzero(counted.image_id == image_id)
        if not anns or not len(rows):
            continue
        gt = np.array([a['bbox'] for a in anns], dtype=np.float64)
        gt[:, 2:] += gt[:, :2]
        overlap = box_iou(gt, counted.boxes[rows])
        for i, ann in enumerate(anns):
            j = int(overlap[i].argmax())
            label = int(counted.label[rows[j]])
            if overlap[i, j] >= iou and label != ann['category_id']:
                pairs[(ann['category_id'], label)] += 1
    return [{'true': t, 'predicted': p, 'count': n, 'new_to_old': t >= previous and p < previous}
            for (t, p), n in pairs.most_common(top)]


def score_val(plan: dict, run: Path, stage: int) -> dict:
    """Validation scores of one stage (plan 5 and handoff 6): cAcc-M2 (gates), cAcc-M1 with one threshold (as B2)
    and with one threshold per task group (D8), AP50, cAcc per level and the top confusions. The policies saved
    here are the ones the final test applies."""
    from autocheckout.cl_metrics import load_coco
    from autocheckout.counting import dedup_detections, label_thresholds, scores_by_level
    from autocheckout.io import save_json
    from autocheckout.predictions import load_predictions
    from autocheckout.taskcfg import TaskConfig
    from tools import eval_cl, eval_count

    cfg = TaskConfig.load(plan['task_config'])
    val = Path(plan['task_dir']) / 'val_full.json'
    m2 = eval_count.calibrate_run(run, val, cfg, stages=[stage], unlearned_iou=UNLEARNED_IOU,
                                  group_thresholds=stage > 1)
    m2 = _merge_policy(run / 'calibration_count_m2.json', m2)['stages'][str(stage)]
    single = eval_count.calibrate_run(run, val, cfg, stages=[stage])
    single = _merge_policy(run / 'calibration_count.json', single)['stages'][str(stage)]
    groups = eval_count.calibrate_run(run, val, cfg, stages=[stage], group_thresholds=True)
    groups = _merge_policy(run / 'calibration_count_groups.json', groups)['stages'][str(stage)]
    detection = eval_cl.evaluate_run(run, 'val', val, cfg)
    save_json(run / 'metrics_cl_val.json', detection, indent=1)
    m1 = detection['stages'][stage]['m1']

    coco = load_coco(load_json(val))
    preds = load_predictions(run / f'task_{stage}/pred_val.npz')
    processed = dedup_detections(preds, groups['nms_iou']) if groups['nms_iou'] else preds.top1_per_query()
    vector = label_thresholds([list(cfg.task(t).labels) for t in range(1, stage + 1)],
                              [groups['group_thresholds'][f'task_{t}'] for t in range(1, stage + 1)])
    labels = list(range(cfg.seen_classes(stage)))
    info = load_json(run / f'task_{stage}/run_info.json')
    return {
        'stage': stage, 'run': str(run),
        'cAcc_m2': m2['val_scores']['cAcc'], 'cAcc_m1_single': single['val_scores']['cAcc'],
        'cAcc_m1_groups': groups['val_scores']['cAcc'], 'val_images_m2': m2['val_images'],
        'threshold_single': single['score_threshold'], 'thresholds_groups': groups['group_thresholds'],
        'AP50': {name: (m1[name] or {}).get('AP50') for name in ('mAP_C', 'mAP_P', 'mAP_A')},
        'forgetting': detection['forgetting'],
        'cAcc_by_level': {level: s['cAcc'] for level, s in scores_by_level(coco, processed, labels, vector).items()},
        'confusions_top20': confusions(coco, processed, len(labels), vector, cfg.seen_classes(stage - 1)
                                       if stage > 1 else 0),
        'train_hours': round(info.get('total_seconds', 0) / 3600, 3),
    }


def _gate(plan: dict, name: str, decision: str, reason: str, **values) -> dict:
    gate = dict(name=name, decision=decision, reason=reason, enforced=not plan['dry_run'], **values)
    print(f"GATE {name}: {decision.upper()} - {reason}" + ('' if gate['enforced'] else ' (dry run: not enforced)'))
    return gate


def _finish(plan: dict, stage: str, report: dict, gate: dict | None) -> dict:
    from autocheckout.io import save_json
    from autocheckout.model_acceptance import now

    result = {**report, 'gate': gate, 'finished_at': now(), 'campaign_id': plan['campaign_id'], 'round': plan['round'],
              'release_config_sha256': plan['release_config_sha256'], 'input_sha256': plan['input_sha256']}
    save_json(Path(plan['root']) / 'stages' / f'{stage}.json', result, indent=2)
    write_summary(plan)
    if gate and gate['decision'] == 'stop' and gate['enforced']:
        raise GateStop(f"{gate['name']}: {gate['reason']}")
    return result


def _require(plan: dict, stage: str) -> dict:
    path = Path(plan['root']) / 'stages' / f'{stage}.json'
    if not path.is_file():
        raise RuntimeError(f'run stage {stage} first')
    result = load_json(path)
    gate = result.get('gate')
    if gate and gate['decision'] == 'stop' and gate['enforced']:
        raise GateStop(f"{stage} stopped this round: {gate['reason']}")
    return result


# --- stages --------------------------------------------------------------------------------------------------------
def stage_fsa(plan: dict) -> dict:
    """G1: FSA on synthetic task 1 (at most fsa_epochs, best val AP50 epoch), then the D18 early check."""
    with _attempt(plan, 'fsa') as (record, attempt):
        _train(plan, record, attempt, FSA, start=1, n=1, epochs=plan['fsa_epochs'], splits='val')
    run = Path(plan['runs']) / FSA
    report = score_val(plan, run, 1)
    report['epoch_selection'] = load_json(run / 'task_1/hf_model/epoch_selection.json')
    value = report['cAcc_m2']
    passed = value >= FSA_EARLY_CHECK
    gate = _gate(plan, 'D18 FSA early check', 'go' if passed else 'stop',
                 f'val cAcc-M2 {value:.4f} {">=" if passed else "<"} {FSA_EARLY_CHECK}' +
                 ('' if passed else '; skip PDP task 1, tune the generator (plan 6)'),
                 value=value, threshold=FSA_EARLY_CHECK)
    return _finish(plan, 'fsa', report, gate)


def task1_decision(value: float, round_id: int) -> tuple[str, str, str]:
    """D17/D19: (colour, decision, reason) of the task-1 gate for a val cAcc-M2 in a given round."""
    colour = 'green' if value >= TASK1_GREEN else 'red' if value < TASK1_RED else 'yellow'
    if colour == 'green':
        return colour, 'go', f'cAcc-M2 {value:.4f} >= {TASK1_GREEN}'
    if round_id < MAX_ROUNDS:
        return colour, 'stop', f'{colour}: cAcc-M2 {value:.4f} < {TASK1_GREEN}; tuning round {round_id + 1} (plan 6)'
    if colour == 'yellow':
        return colour, 'go', 'yellow in the last round: tasks 2-5 run, expected on par with B2 (plan 6)'
    return colour, 'stop', f'red in the last round (cAcc-M2 {value:.4f} < {TASK1_RED}): stop synthetic-only'


def stage_task1(plan: dict) -> dict:
    """G2: PDP task 1 from FSA, then the D17 gate on val cAcc-M2."""
    _require(plan, 'fsa')
    with _attempt(plan, 'task1') as (record, attempt):
        _train(plan, record, attempt, PDP, start=1, n=1, epochs=plan['epochs'], splits='val', fsa_hf=_fsa_hf(plan))
    report = score_val(plan, Path(plan['runs']) / PDP, 1)
    colour, decision, reason = task1_decision(report['cAcc_m2'], plan['round'])
    gate = _gate(plan, 'D17 task 1', decision, reason, colour=colour, value=report['cAcc_m2'],
                 green=TASK1_GREEN, red=TASK1_RED, round=plan['round'])
    return _finish(plan, 'task1', report, gate)


def _audit(plan: dict, record: dict, attempt: Path, task: int) -> dict:
    """V4/F0.6: the teacher of task ``task`` (model after task - 1) and PPG on audit_images synthetic images of the
    task, against train_task_<t>_gt_full.json (read from the release, never from the training folder)."""

    run = Path(plan['runs']) / PDP
    result = run / f'task_{task}/ppg_audit.json'
    if not result.is_file():
        config = _config(plan, PDP, Path(plan['runs']), _fsa_hf(plan))
        environment = _environment(plan, PDP, Path(plan['runs']), FSA_HF=_fsa_hf(plan), CONFIG=config)
        script = 'source "$CONFIG"; cd "$REPO/pdp"; exec "$PYTHON" ppg_audit.py "${ARGS[@]}" "$@"'
        command = ['bash', '-ec', script, 'ppg_audit', '--output_dir', str(run), '--audit_task', str(task),
                   '--audit_images', str(plan['audit_images']),
                   '--prev_ckpt', str(run / f'task_{task - 1}/task_final.pth'),
                   '--task_ann_dir', str(Path(plan['release_root']) / 'tasks/synth_v2')]
        _run(record, attempt, f'ppg_audit_task{task}', command, environment)
    return load_json(result)


def stage_tasks(plan: dict) -> dict:
    """G3: tasks 2-5. Before training task t, the pseudo-labels its teacher gives are audited (at task 2 they must
    reach PSEUDO_PRECISION_MIN); after it, the stage is scored on val and logged in task_<t>/task_log.json."""
    from autocheckout.io import save_json

    _require(plan, 'task1')
    run = Path(plan['runs']) / PDP
    reports = {}
    for task in range(2, N_TASKS + 1):
        with _attempt(plan, f'task{task}') as (record, attempt):
            audit = _audit(plan, record, attempt, task)
            precision = audit['all'].get('precision')
            if task == 2:
                passed = precision is not None and precision >= PSEUDO_PRECISION_MIN
                gate = _gate(plan, 'pseudo-labels task 2', 'go' if passed else 'stop',
                             f'teacher precision {precision} {">=" if passed else "<"} {PSEUDO_PRECISION_MIN}'
                             + ('' if passed else '; report to the team before training task 2'),
                             value=precision, threshold=PSEUDO_PRECISION_MIN)
                if gate['decision'] == 'stop':
                    _finish(plan, 'tasks', {'stage': 2, 'pseudo_labels': audit}, gate)
            _train(plan, record, attempt, PDP, start=task, n=task, epochs=plan['epochs'], splits='val',
                   fsa_hf=_fsa_hf(plan))
        report = score_val(plan, run, task)
        report['pseudo_labels'] = {key: audit.get(key) for key in ('images', 'earlier_objects', 'recall', 'all',
                                                                   'branches', 'filters', 'confusions_top')}
        save_json(run / f'task_{task}/task_log.json', {**report, 'release_config_sha256': plan['release_config_sha256'],
                                                       'input_sha256': plan['input_sha256']}, indent=2)
        reports[task] = report
        print(f"task {task}: val cAcc M1 {report['cAcc_m1_groups']:.4f} (groups) / {report['cAcc_m1_single']:.4f} "
              f"(one threshold), pseudo-label precision {precision}, recall {audit.get('recall')}")
    return _finish(plan, 'tasks', {'stage': N_TASKS, 'tasks': reports}, None)


def stage_final_test(plan: dict) -> dict:
    """G4: test scored once, with the policies chosen on val: test2019 (24,000) and the 6,003 old test images of
    B1/B2 (F0.8). An existing result is returned as it is, never recomputed."""
    from autocheckout.io import save_json
    from autocheckout.taskcfg import TaskConfig
    from tools import eval_cl, eval_count

    done = Path(plan['root']) / 'stages/final_test.json'
    if done.is_file():
        print(f'test already scored on {load_json(done)["finished_at"]}; not recomputed')
        return load_json(done)
    _require(plan, 'tasks')
    run = Path(plan['runs']) / PDP
    with _attempt(plan, 'final_test') as (record, attempt):
        _train(plan, record, attempt, PDP, start=1, n=N_TASKS, epochs=plan['epochs'], splits='test',
               fsa_hf=_fsa_hf(plan))
    cfg = TaskConfig.load(plan['task_config'])
    test = Path(plan['task_dir']) / 'test_full.json'
    compare_ann = Path(plan['eval_root']) / 'compare_b12/test_full.json'
    compare = subset_run(run, compare_ann, Path(plan['root']) / 'compare_b12' / PDP)
    report = {'stage': 'final_test'}
    for name, run_dir, ann in (('test2019', run, test), ('compare_b12', compare, compare_ann)):
        detection = eval_cl.evaluate_run(run_dir, 'test', ann, cfg)
        save_json(run_dir / 'metrics_cl_test.json', detection, indent=1)
        counting = {}
        for label, policy in (('groups', 'calibration_count_groups.json'), ('single', 'calibration_count.json')):
            result = eval_count.evaluate_policy(run_dir, ann, cfg, load_json(run / policy))
            save_json(run_dir / f'metrics_count_test_{label}.json', result, indent=1)
            counting[policy] = {s: r['test']['overall'] for s, r in result['stages'].items()}
        report[name] = {'AP50': {s: v['m1']['mAP_A']['AP50'] for s, v in detection['stages'].items()},
                        'forgetting': detection['forgetting'], 'counting': counting}
    return _finish(plan, 'final_test', report, None)


def write_summary(plan: dict) -> Path:
    """SUMMARY.md of the round from stages/*.json: the table the notebook prints and the team reads."""
    stages = {p.stem: load_json(p) for p in sorted((Path(plan['root']) / 'stages').glob('*.json'))}

    def pct(x):
        return '-' if x is None else f'{100 * x:.2f}%'

    lines = [f"# FULL-SYN {plan['campaign_id']} (round {plan['round']}{', dry run' if plan['dry_run'] else ''})", '',
             f"Synthetic config sha256 `{plan['release_config_sha256']}`; train images {plan['train_images']}.", '',
             '| stage | val cAcc-M2 | val cAcc-M1 one / groups | thresholds (groups) | AP50 all | pseudo-label P / R '
             '| I3 / F14 dropped | train h | gate |', '|---|---|---|---|---|---|---|---|---|']
    rows = [('FSA', stages.get('fsa')), ('task 1', stages.get('task1'))]
    rows += [(f'task {t}', r) for t, r in sorted(((int(k), v) for k, v in
                                                  stages.get('tasks', {}).get('tasks', {}).items()))]
    for name, report in rows:
        if not report:
            continue
        pseudo = report.get('pseudo_labels') or {}
        filters = pseudo.get('filters') or {}
        gate = report.get('gate')
        lines.append(f"| {name} | {pct(report.get('cAcc_m2'))} | {pct(report.get('cAcc_m1_single'))} / "
                     f"{pct(report.get('cAcc_m1_groups'))} | "
                     f"{'/'.join(f'{v:.2f}' for v in report.get('thresholds_groups', {}).values())} | "
                     f"{pct((report.get('AP50') or {}).get('mAP_A'))} | "
                     f"{pct((pseudo.get('all') or {}).get('precision'))} / {pct(pseudo.get('recall'))} | "
                     f"{filters.get('dropped_i3', '-')} / {filters.get('dropped_f14', '-')} | "
                     f"{report.get('train_hours', '-')} | {gate['decision'] + ': ' + gate['reason'] if gate else ''} |")
    stop = (stages.get('tasks') or {}).get('gate')
    if stop:
        lines += ['', f"Tasks 2-5: {stop['decision']} - {stop['reason']}"]
    final = stages.get('final_test')
    if final:
        lines += ['', '## Test (scored once, policies chosen on val)', '',
                  '| set | stage | AP50 all | cAcc one threshold | cAcc groups |', '|---|---|---|---|---|']
        for name in ('test2019', 'compare_b12'):
            part = final[name]
            for stage in sorted(part['AP50'], key=int):
                one = part['counting']['calibration_count.json'][stage]['cAcc']
                groups = part['counting']['calibration_count_groups.json'][stage]['cAcc']
                lines.append(f"| {name} | {stage} | {pct(part['AP50'][stage])} | {pct(one)} | {pct(groups)} |")
    path = Path(plan['root']) / 'SUMMARY.md'
    path.write_text('\n'.join(lines) + '\n')
    return path


# --- F0.7: smoke and benchmark on the rented GPU ---------------------------------------------------------------------
def check_data(plan: dict, sample: int = 300) -> dict:
    """The data package on this machine: every image the annotation files list exists, and a random sample of the
    package's SHA256SUMS still matches (hashing all 26 GB takes over an hour)."""
    import random

    from autocheckout.training_workflow import REPO

    missing = {}
    for name in [f'train_task_{t}.json' for t in range(1, N_TASKS + 1)] + ['val_full.json', 'test_full.json']:
        root = Path(plan['synth_root'] if name.startswith('train') else plan['raw_root'])
        images = load_json(Path(plan['task_dir']) / name)['images']
        missing[name] = sum(not (root / image['file_name']).is_file() for image in images)
    sums = sorted(REPO.glob('data/packages/*/SHA256SUMS'))
    checked = mismatched = 0
    if sums:
        lines = sums[-1].read_text().splitlines()
        for line in random.Random(0).sample(lines, min(sample, len(lines))):
            digest, path = line.split(maxsplit=1)
            checked += 1
            mismatched += not (REPO / path).is_file() or sha256_file(REPO / path) != digest
    result = dict(missing_images=missing, package_sums=str(sums[-1]) if sums else None, sample_checked=checked,
                  sample_mismatched=mismatched)
    if any(missing.values()) or mismatched:
        raise RuntimeError(f'data package incomplete or changed: {result}')
    return result


def _speed(task_dir: Path) -> dict:
    rows = [json.loads(line) for line in (task_dir / 'step_metrics.jsonl').read_text().splitlines()]
    last = rows[-1]
    return dict(optimizer_steps=last['optimizer_step'], seconds_per_step=last['seconds_per_optimizer_step'],
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows))


def stage_smoke(plan: dict, steps: int = 200) -> dict:
    """F0.7, in its own folder (smoke/): data check, FSA for ``steps`` optimizer steps, stopped and resumed for as
    many again, then PDP task 1 for ``steps`` steps from the COCO detector (as B2; FSA's export does not exist yet).
    Measures seconds per step and peak VRAM and estimates the training hours of the round (validation excluded)."""
    import math

    import torch

    from autocheckout.io import save_json

    runs = Path(plan['root']) / 'smoke'
    with _attempt(plan, 'smoke') as (record, attempt):
        report = {'data': check_data(plan), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  'gpu_memory_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
                  if torch.cuda.is_available() else None}
        fsa = runs / FSA / 'task_1'
        for stop in (steps, 2 * steps):
            if not (fsa / 'progress.json').is_file() or load_json(fsa / 'progress.json')['optimizer_step'] < stop:
                code = _train(plan, record, attempt, FSA, start=1, n=1, epochs=plan['fsa_epochs'], splits='val',
                              runs=runs, stop_after=stop)
                if code != 75:
                    raise RuntimeError(f'FSA smoke did not stop at step {stop} (exit {code})')
        sessions = load_json(fsa / 'run_info.json')['sessions']
        report['fsa'] = {**_speed(fsa), 'resume_verification': sessions[-1].get('resume_verification')}
        if not plan['dry_run']:  # the dry run has no detector to start PDP from; stage_task1 covers that path
            pdp = runs / PDP / 'task_1'
            if not (pdp / 'progress.json').is_file():
                _train(plan, record, attempt, PDP, start=1, n=1, epochs=plan['epochs'], splits='val', runs=runs,
                       fsa_hf=plan['detector_dir'], stop_after=steps)
            report['pdp'] = _speed(pdp)

    def hours(images, epochs, seconds, batch=plan['batch_size']):
        return round(math.ceil(math.ceil(images / batch) / (plan['effective_batch'] // batch))
                     * epochs * seconds / 3600, 2)

    images = plan['train_images']
    estimate = {'fsa': hours(images[0], plan['fsa_epochs'], report['fsa']['seconds_per_step'],
                             plan.get('fsa_batch_size', plan['batch_size']))}
    if 'pdp' in report:
        seconds = report['pdp']['seconds_per_step']
        # B2 on the 4090: a task >= 2 image took 1.47x a task 1 image (teacher forward), validation included
        estimate.update(task1=hours(images[0], plan['epochs'], seconds),
                        tasks_2_5=round(sum(hours(n, plan['epochs'], 1.47 * seconds) for n in images[1:]), 2))
    report['training_hours_estimate'] = {**estimate, 'note': 'training steps only; validation every epoch '
                                         '(val_task_<t>, 5,562 images at task 1) and test predictions come on top'}
    save_json(runs / 'smoke_report.json', report, indent=2)
    return report
