"""Training telemetry must survive interrupted notebook/process sessions."""

import json
from types import SimpleNamespace

import pytest
import torch

from autocheckout.io import load_json

import checkpointing


def test_resume_uses_previous_atomic_checkpoint_when_last_is_missing(tmp_path):
    backup = tmp_path / 'last.ckpt.prev'
    backup.write_bytes(b'previous checkpoint')
    assert checkpointing.resume_path(tmp_path) == str(backup)


def test_progress_logs_step_loss_teacher_and_error_with_timestamps(tmp_path):
    callback_type = getattr(checkpointing, 'TrainingProgress', None)
    assert callback_type is not None, 'missing persistent training telemetry callback'
    callback = callback_type(tmp_path)
    teacher = torch.nn.Linear(2, 2)
    teacher.requires_grad_(False)
    module = SimpleNamespace(task_id=2, teacher=teacher, PREV_INTRODUCED_CLS=2, seen_classes=3,
                             teacher_forward_calls=1, ppg_pseudo_labels_total=0)
    trainer = SimpleNamespace(global_step=0, current_epoch=0, optimizers=[SimpleNamespace(param_groups=[{'lr': 1e-4}])],
                              callback_metrics={'tr': torch.tensor(1.25)}, is_global_zero=True)
    callback.on_train_start(trainer, module)
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, module, {}, {}, 0)
    callback.on_exception(trainer, module, RuntimeError('fixture failure'))
    state = load_json(tmp_path / 'progress.json')
    assert state['status'] == 'FAIL' and state['optimizer_step'] == 1
    assert state['error'] == 'fixture failure'
    start = load_json(tmp_path / 'training_start.json')
    assert start['teacher_present'] and start['teacher_frozen'] and start['task_id'] == 2
    rows = [json.loads(line) for line in (tmp_path / 'step_metrics.jsonl').read_text().splitlines()]
    assert rows[0]['losses']['tr'] == 1.25 and rows[0]['learning_rates'] == [1e-4]
    assert rows[0]['teacher_forward_calls'] == 1 and '+07:00' in rows[0]['timestamp']


def test_native_configs_expose_checkpoint_and_logging_controls(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(['bash', '-ec', 'source "$REPO/configs/exp/native/EXP-B1.sh"; printf "%s\\0" "${ARGS[@]}"'],
                            env={**os.environ, 'REPO': str(repo), 'CKPT_EVERY_MINUTES': '2', 'PRINT_FREQ': '5',
                                 'START_TASK': '2', 'PREV_CKPT': '/fixture/task_final.pth'},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    arguments = result.stdout.split('\0')
    assert '--ckpt_every_minutes' in arguments and arguments[arguments.index('--ckpt_every_minutes')+1] == '2'
    assert '--prev_ckpt' in arguments and arguments[arguments.index('--prev_ckpt')+1] == '/fixture/task_final.pth'
    assert arguments[arguments.index('--print_freq')+1] == '5'


def test_lightning_interruption_does_not_mark_partial_task_as_final(tmp_path, monkeypatch):
    import main as pdp_main
    from pdp_helpers import run_main

    def stopped_fit(module, *args, **kwargs):
        module.resume_verification = dict(restored_step=0, verified=False)

    monkeypatch.setattr(pdp_main, 'make_pl_trainer', lambda args:
                        SimpleNamespace(fit=stopped_fit, global_step=1, interrupted=True))
    with pytest.raises(SystemExit) as stopped:
        run_main(tmp_path, monkeypatch, '--n_tasks', '1')
    assert stopped.value.code == 75
    assert not (tmp_path / 'run/task_1/task_final.pth').exists()


def test_validation_records_AP_each_epoch_for_pilot_budget_selection(tmp_path, monkeypatch):
    from pdp_helpers import run_main

    run_main(tmp_path, monkeypatch, '--n_tasks', '1', '--epochs', '1')
    path = tmp_path / 'run/task_1/val_epoch_0000.json'
    assert path.is_file(), 'missing persistent epoch validation metrics'
    record = load_json(path)
    assert record['scope'] == 'val_task' and record['epoch'] == 0
    assert set(record['metrics']) == {'AP', 'AP50', 'AP75'}
    assert '+07:00' in record['timestamp']
