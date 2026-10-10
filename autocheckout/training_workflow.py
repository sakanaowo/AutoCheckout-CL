"""Shared notebook orchestration; training stays in the native shell runner/PDP CLI."""

from __future__ import annotations

import fcntl
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import uuid
from pathlib import Path

from autocheckout.io import load_json, md5_file, save_json
from autocheckout.model_acceptance import now
from autocheckout.runinfo import git_state
from autocheckout.taskcfg import TaskConfig

REPO = Path(__file__).resolve().parents[1]
DETECTOR_REVISION = '83ecd26945199939cb82806f988debdb71e6f43e'
BACKBONE_REVISION = '9e250ed8f88b436472d2b24af26f82a8aa8c719d'


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def campaign_root(settings):
    root = resolve(settings.get('artifact_root', 'data/training'))
    if settings.get('dry_run', True):
        root /= 'dry_run'
    name = settings['campaign_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('campaign_id must contain only letters, numbers, underscore or dash')
    return root / name


def _toy_inputs(root):
    """Six labels across five tasks, random ResNet18, no download; only for local dry runs."""
    if (root / 'task_config.json').is_file():
        return
    from PIL import Image, ImageDraw

    root.mkdir(parents=True, exist_ok=True)
    offsets, sizes = [0, 2, 3, 4, 5], [2, 1, 1, 1, 1]
    cfg = dict(name='dry-run-six-labels', seed=0, tasks=[dict(task_id=t+1, offset=o, reserved=False,
               classes=[dict(label=c, name=f'sku{c}', rpc_category_id=c+1) for c in range(o, o+n)])
               for t, (o, n) in enumerate(zip(offsets, sizes, strict=True))])
    save_json(root / 'task_config.json', cfg, indent=2)
    save_json(root / 'model/config.json', dict(model_type='deformable_detr', use_pretrained_backbone=False,
              backbone='resnet18', encoder_layers=1, decoder_layers=2, encoder_ffn_dim=64, decoder_ffn_dim=64))
    (root / 'images').mkdir()
    for split, ids in [('train', range(1, 5)), ('val', range(10, 12)), ('test', range(20, 22))]:
        images, annotations = [], []
        for image_id in ids:
            image = Image.new('RGB', (96, 96), (210+image_id, 210, 210))
            draw = ImageDraw.Draw(image)
            filename = f'{split}_{image_id}.jpg'
            images.append(dict(id=image_id, file_name=filename, width=96, height=96, level='easy'))
            for label in range(6):
                x, y = 5+(label % 3)*28, 5+(label//3)*35
                draw.rectangle([x, y, x+15, y+18], fill=(30+label*30, 70+image_id, 110))
                annotations.append(dict(id=len(annotations)+1, image_id=image_id, category_id=label,
                                        bbox=[x, y, 15, 18], area=270, iscrowd=0))
            image.save(root / 'images' / filename)
        categories = [dict(id=c, name=f'sku{c}') for c in range(6)]
        full = dict(images=images, annotations=annotations, categories=categories)
        save_json(root / 'tasks' / f'{split}_full.json', full)
        if split in ('train', 'val'):
            for t, (offset, size) in enumerate(zip(offsets, sizes, strict=True), 1):
                save_json(root / 'tasks' / f'{split}_task_{t}.json', dict(full,
                          annotations=[a for a in annotations if offset <= a['category_id'] < offset+size]))


def _subset(source, output, count):
    original = load_json(source)
    ids = {i['id'] for i in original['images'][:count]}
    save_json(output, {**original, 'images': [i for i in original['images'] if i['id'] in ids],
                      'annotations': [a for a in original['annotations'] if a['image_id'] in ids]})


def _portable_plan(plan):
    locations = {'root', 'phase_dir', 'raw_root', 'task_dir', 'task_config',
                 'detector_dir', 'backbone_weights', 'model_config', 'input_hashes'}
    identity = {k: v for k, v in plan.items() if k not in locations}
    identity['input_hashes'] = {Path(p).name: digest for p, digest in plan['input_hashes'].items()}
    identity.setdefault('pretrained_hashes', {})
    return identity


def prepare(settings, phase):
    if phase not in ('pilot', 'transition', 'baseline'):
        raise ValueError('phase must be pilot, transition or baseline')
    physical, effective = int(settings['batch_size']), int(settings['effective_batch'])
    if physical < 1 or effective < physical or effective % physical:
        raise ValueError('effective batch must be a positive multiple of physical batch (one GPU)')
    root = campaign_root(settings)
    dry = bool(settings.get('dry_run', True))
    experiments = settings.get('experiments', ['EXP-B1', 'EXP-B2'])
    if not experiments or set(experiments)-{'EXP-B1', 'EXP-B2'} or len(set(experiments)) != len(experiments):
        raise ValueError('experiments must be distinct EXP-B1/EXP-B2 entries')
    for parent in ({'pilot': [], 'transition': ['pilot'], 'baseline': ['pilot', 'transition']}[phase]):
        receipt = root / parent / 'summary.json'
        if not receipt.is_file() or load_json(receipt)['status'] != 'COMPLETED':
            raise ValueError(f'{phase} requires completed {parent} notebook in this campaign')
    epochs = 1 if dry else settings.get(f'{phase}_epochs')
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f'Set {phase}_epochs to a positive integer; choose baseline budget from pilot val curves')
    if dry:
        inputs = root / 'fixture'
        _toy_inputs(inputs)
        raw, tasks, task_config = inputs/'images', inputs/'tasks', inputs/'task_config.json'
        detector, backbone = '', ''
    else:
        raw = resolve(settings.get('raw_root', 'data/archive'))
        release = resolve(settings.get('release_root', 'data/processed/rpc_100-4x25_seed0_native_v1'))
        tasks, task_config = release/'tasks/real', release/'task_config.json'
        cache = resolve(settings.get('pretrained_root', 'data/pretrained/hf-cache'))
        detector = str(cache/f'models--SenseTime--deformable-detr/snapshots/{DETECTOR_REVISION}')
        backbone = str(cache/f'models--timm--convnextv2_base.fcmae_ft_in22k_in1k/snapshots/{BACKBONE_REVISION}'
                       /'model.safetensors')
        if not (Path(detector)/'model.safetensors').is_file():
            raise FileNotFoundError(f'Copy/download pinned detector weights under data first: {detector}')
        if 'EXP-B2' in experiments and not Path(backbone).is_file():
            raise FileNotFoundError(f'Copy/download pinned ConvNeXt weights under data first: {backbone}')
    cfg = TaskConfig.load(task_config)
    if not dry and [t.num_classes for t in cfg.data_tasks] != [100, 25, 25, 25, 25]:
        raise ValueError('real training requires the locked 100+4x25 protocol')
    parent_checkpoint_hashes = {}
    pretrained_hashes = {} if dry else {'detector': md5_file(Path(detector)/'model.safetensors')}
    if not dry and 'EXP-B2' in experiments:
        pretrained_hashes['backbone'] = md5_file(backbone)
    if phase == 'transition':
        derived = root / 'transition_inputs/tasks'
        if not (derived/'train_task_2.json').is_file():
            for name, count in [('train_task_2.json', settings.get('transition_samples', 16)),
                                ('val_task_2.json', 2), ('val_full.json', 2), ('test_full.json', 2)]:
                _subset(tasks/name, derived/name, count)
        tasks = derived
        for name in experiments:
            parent = root/'pilot'/name/'task_1/task_final.pth'
            if not parent.is_file():
                raise ValueError(f'pilot checkpoint missing: {parent}')
            parent_checkpoint_hashes[name] = md5_file(parent)
    phase_dir = root / phase
    files = [task_config, *tasks.glob('*.json')]
    for task in ([2] if phase == 'transition' else range(1, 6) if phase == 'baseline' else [1]):
        ann = load_json(tasks/f'train_task_{task}.json')
        if not ann['images'] or any(a['category_id'] not in cfg.task(task).labels for a in ann['annotations']):
            raise ValueError(f'task {task} training input empty or includes previous-task labels')
    n_images = len(load_json(tasks/('train_task_2.json' if phase == 'transition' else 'train_task_1.json'))['images'])
    accumulation = effective // physical
    plan = dict(root=str(root), phase_dir=str(phase_dir), phase=phase, dry_run=dry, epochs=epochs,
                experiments=experiments, raw_root=str(raw), task_dir=str(tasks), task_config=str(task_config),
                detector_dir=detector, backbone_weights=backbone, resolution=96 if dry else int(settings['resolution']),
                batch_size=physical, effective_batch=effective, accumulation=accumulation,
                num_workers=0 if dry else int(settings.get('num_workers', 4)),
                checkpoint_every_minutes=float(settings.get('checkpoint_every_minutes', 2)),
                test_interruption=bool(settings.get('test_interruption', dry)),
                actual_backbone='resnet18' if dry else 'per-experiment',
                model_config=str(root/'fixture/model') if dry else '',
                start_task=2 if phase == 'transition' else 1,
                n_tasks=5 if phase == 'baseline' else 2 if phase == 'transition' else 1,
                input_hashes={str(p): md5_file(p) for p in files}, parent_checkpoint_hashes=parent_checkpoint_hashes,
                pretrained_hashes=pretrained_hashes,
                task1_or_task2_images=n_images, steps_per_epoch=math.ceil(math.ceil(n_images/physical)/accumulation),
                count_nms_grid=str(settings.get('count_nms_grid', '0.45')))
    path = phase_dir/'plan.json'
    if path.is_file() and _portable_plan(load_json(path)) != _portable_plan(plan):
        raise ValueError('configuration or input hashes changed in existing phase; use a new campaign_id')
    save_json(path, plan, indent=2)
    return plan


def _case_environment(plan, experiment):
    return dict(EXP=experiment, RUNS=plan['phase_dir'], RAW_ROOT=plan['raw_root'], TASK_DIR=plan['task_dir'],
                TASK_CFG=plan['task_config'], DETECTOR_DIR=plan['detector_dir'],
                BACKBONE_WEIGHTS=plan['backbone_weights'],
                RESOLUTION=str(plan['resolution']), BATCH_SIZE=str(plan['batch_size']),
                EFFECTIVE_BATCH=str(plan['effective_batch']), NUM_WORKERS=str(plan['num_workers']),
                EPOCHS=str(plan['epochs']), START_TASK=str(plan['start_task']), N_TASKS=str(plan['n_tasks']),
                CKPT_EVERY_MINUTES=str(plan['checkpoint_every_minutes']), VERIFY_RESUME='1', REQUIRE_KERNEL='0',
                EVALUATE='1', COUNT_NMS_GRID=plan['count_nms_grid'], PYTHON=sys.executable, PYTHONUNBUFFERED='1',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0',
                TORCH_EXTENSIONS_DIR=str(Path(plan['root'])/'cache/torch_extensions'),
                MPLCONFIGDIR=str(Path(plan['root'])/'cache/matplotlib'))


def process_start(pid):
    stat = Path(f'/proc/{pid}/stat')
    try:
        return stat.read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def _run_logged(command, environment, log_path):
    print(f'{now()} START {shlex.join(command)} -> {log_path}', flush=True)
    with log_path.open('w') as stream:
        process = subprocess.Popen(command, cwd=REPO, env={**os.environ, **environment},
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = log_path.with_suffix('.process.json')
        record = dict(status='RUNNING', pid=process.pid, host=socket.gethostname(),
                      process_start=process_start(process.pid), started_at=now(), command=command)
        save_json(receipt, record, indent=2)
        try:
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGINT)
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                code = process.wait(timeout=10)
            record.update(status='INTERRUPTED', returncode=code, finished_at=now())
            save_json(receipt, record, indent=2)
            raise
        record.update(status='COMPLETED' if code == 0 else 'INTERRUPTED' if code == 75 else 'FAIL',
                      returncode=code, finished_at=now())
        save_json(receipt, record, indent=2)
    print(f'{now()} EXIT {code} log={log_path}', flush=True)
    return code


def run_phase(plan):
    directory = Path(plan['phase_dir'])
    with (Path(plan['root'])/'training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another training notebook is using this campaign') from error
        for path in Path(plan['root']).glob('*/attempts/*/*.process.json'):
            previous = load_json(path)
            if (previous['status'] == 'RUNNING' and previous.get('host') == socket.gethostname()
                    and previous.get('process_start') is not None
                    and previous['process_start'] == process_start(previous['pid'])):
                raise RuntimeError(f'runner PID {previous["pid"]} is still running; reconnect/stop it before resume')
        attempt = directory/'attempts'/str(uuid.uuid4())
        attempt.mkdir(parents=True)
        record = dict(status='RUNNING', started_at=now(), attempt_id=attempt.name,
                      phase=plan['phase'], dry_run=plan['dry_run'], git=git_state(REPO), commands=[])
        receipt = attempt/'execution.json'
        save_json(receipt, record, indent=2)
        try:
            for experiment in plan['experiments']:
                output = directory/experiment
                output.mkdir(exist_ok=True)
                if plan['phase'] == 'transition':
                    parent = Path(plan['root'])/'pilot'/experiment/'task_1/task_final.pth'
                    target = output/'task_1/task_final.pth'
                    target.parent.mkdir(exist_ok=True)
                    if not target.exists():
                        shutil.copy2(parent, target)
                    if md5_file(target) != plan['parent_checkpoint_hashes'][experiment]:
                        raise ValueError('transition parent checkpoint changed')
                config = output/'resolved_config.sh'
                lines = [f'source "$REPO/configs/exp/native/{experiment}.sh"']
                if plan['dry_run']:
                    overrides = ['--backbone', '', '--backbone_pretrained_file', '', '--repo_name', '',
                                 '--model_config', plan['model_config'], '--n_classes', '7', '--accelerator', 'cpu',
                                 '--num_prompts', '4', '--prompt_len', '2', '--proto_correct_only', '0',
                                 '--shuffle', '0']
                    lines.append(f'ARGS+=({shlex.join(overrides)})')
                config.write_text('\n'.join(lines)+'\n')
                environment = _case_environment(plan, experiment)
                if plan['dry_run']:
                    environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                resolved = subprocess.run(['bash', '-ec', 'source "$CONFIG"; printf "%s\\0" "${ARGS[@]}"'],
                          env={**os.environ, **environment, 'REPO': str(REPO), 'CONFIG': str(config)},
                          capture_output=True, text=True, check=True)
                save_json(output/'resolved_cli.json', dict(arguments=resolved.stdout.split('\0')[:-1],
                          environment=environment), indent=2)
                command = ['bash', str(REPO/'scripts/run_exp.sh'), str(config)]
                for task_id in range(plan['start_task'], plan['n_tasks']+1):
                    task = output/f'task_{task_id}'
                    snapshot = output/f'stage_metrics/task_{task_id}'
                    if plan['phase'] == 'baseline':
                        environment.update(START_TASK=str(task_id), N_TASKS=str(task_id),
                                           CALIBRATION_POLICY=str(output/f'calibration_count_through_task_{task_id}.json'))
                        if ((snapshot/'completed.json').is_file() and
                                all((task/n).is_file() for n in ['task_final.pth', 'pred_val.npz', 'pred_test.npz'])):
                            print(f'{now()} {experiment} task {task_id}: completed, skipped', flush=True)
                            continue
                    demonstration = (plan['dry_run'] and plan['test_interruption'] and plan['phase'] == 'pilot'
                                     and not (task/'task_final.pth').exists() and not (task/'last.ckpt').exists())
                    for index, stop in enumerate([1, 0] if demonstration else [0]):
                        env = {**environment, 'STOP_AFTER_STEPS': str(stop)}
                        log_path = attempt/f'{experiment}_task{task_id}_{index}_runner.log'
                        entry = dict(experiment=experiment, task_id=task_id, command=command, environment=env,
                                     log=str(log_path), started_at=now())
                        record['commands'].append(entry)
                        save_json(receipt, record, indent=2)
                        code = _run_logged(command, env, log_path)
                        entry.update(returncode=code, finished_at=now())
                        save_json(receipt, record, indent=2)
                        if stop == 1:
                            if code == 75 and (task/'last.ckpt').exists():
                                continue
                            raise RuntimeError(f'dry-run interruption did not save a resume checkpoint: {log_path}')
                        if code:
                            raise RuntimeError(f'runner exited {code}; inspect {log_path}; rerun campaign to resume')
                    if plan['phase'] == 'baseline':
                        snapshot.mkdir(parents=True, exist_ok=True)
                        for filename in ['metrics_cl_val.json', 'metrics_cl_test.json', 'metrics_count_test.json']:
                            shutil.copy2(output/filename, snapshot/filename)
                        shutil.copy2(environment['CALIBRATION_POLICY'], snapshot/'calibration_count.json')
                        save_json(snapshot/'completed.json', dict(task_id=task_id, finished_at=now()), indent=2)
            summary = collect(plan)
            record.update(status='COMPLETED', finished_at=now())
            save_json(receipt, record, indent=2)
            save_json(directory/'summary.json', summary, indent=2)
            return summary
        except BaseException as error:
            record.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAIL',
                          error_type=type(error).__name__, error=str(error), finished_at=now())
            save_json(receipt, record, indent=2)
            raise


def collect(plan):
    cases = []
    for experiment in plan['experiments']:
        output = Path(plan['phase_dir'])/experiment
        tasks = []
        for task_id in range(plan['start_task'], plan['n_tasks']+1):
            task = output/f'task_{task_id}'
            for name in ['task_final.pth', 'pred_val.npz', 'pred_test.npz', 'run_info.json']:
                if not (task/name).is_file():
                    raise ValueError(f'incomplete artifact: {task/name}')
            sessions = load_json(task/'run_info.json')['sessions']
            finished = next(s for s in reversed(sessions) if s.get('mode') == 'train')
            start = load_json(task/'training_start.json')
            progress = load_json(task/'progress.json')
            if task_id > 1 and not (start['teacher_present'] and start['teacher_frozen']
                                    and progress.get('teacher_forward_calls', 0) > 0):
                raise ValueError(f'task {task_id} teacher was not exercised')
            tasks.append(dict(task_id=task_id, optimizer_steps=finished['optimizer_steps'],
                              peak_gpu_gib=finished.get('peak_gpu_memory_gb', 0), seconds=finished['seconds'],
                              resume_verification=finished.get('resume_verification'), teacher=start,
                              prototype_diagnostics=progress.get('missing_prototypes', []),
                              ppg_pseudo_labels_total=progress.get('ppg_pseudo_labels_total', 0),
                              checkpoint_md5=md5_file(task/'task_final.pth')))
        metrics = load_json(output/'metrics_count_test.json')
        if metrics['oracle_enabled']:
            raise ValueError('oracle must be disabled for research workflow')
        cases.append(dict(experiment=experiment, tasks=tasks,
                          counting=metrics['stages'], detection=load_json(output/'metrics_cl_test.json')))
    if any(md5_file(p) != expected for p, expected in plan['input_hashes'].items()):
        raise ValueError('source annotation/task config changed during execution')
    return dict(status='COMPLETED', finished_at=now(), phase=plan['phase'], dry_run=plan['dry_run'],
                research_metrics=not plan['dry_run'] and plan['phase'] != 'transition',
                input_hashes_unchanged=True, cases=cases,
                limitation='Dry run uses a tiny random ResNet18 and 6 labels; no ConvNeXt/4090 convergence claim.'
                if plan['dry_run'] else 'Review pilot budget/prototype coverage on val before full baseline.')
