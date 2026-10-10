"""Notebook workflows isolate dry runs, persist plans, and require task prerequisites."""

import importlib.util
from pathlib import Path

import pytest

from autocheckout.io import load_json, save_json


def workflow():
    spec = importlib.util.find_spec('autocheckout.training_workflow')
    assert spec is not None, 'missing shared training notebook workflow'
    from autocheckout import training_workflow
    return training_workflow


def settings(tmp_path):
    return dict(artifact_root=str(tmp_path / 'data/training'), campaign_id='fixture', dry_run=True,
                experiments=['EXP-B1'], resolution=640, batch_size=1, effective_batch=2,
                pilot_epochs=2, transition_epochs=1, baseline_epochs=None, num_workers=0,
                checkpoint_every_minutes=2, test_interruption=True)


def test_dry_run_has_five_toy_tasks_and_keeps_outputs_under_data(tmp_path):
    plan = workflow().prepare(settings(tmp_path), 'pilot')
    assert Path(plan['root']).is_relative_to(tmp_path / 'data/training/dry_run')
    cfg = load_json(plan['task_config'])
    assert len(cfg['tasks']) == 5
    assert sum(len(t['classes']) for t in cfg['tasks']) == 6
    assert plan['epochs'] == 1 and plan['actual_backbone'] == 'resnet18'
    assert load_json(Path(plan['phase_dir']) / 'plan.json') == plan


def test_resume_refuses_changed_batch_in_existing_phase(tmp_path):
    original = settings(tmp_path)
    workflow().prepare(original, 'pilot')
    changed = {**original, 'effective_batch': 4}
    with pytest.raises(ValueError, match='config|configuration'):
        workflow().prepare(changed, 'pilot')


def test_uploaded_campaign_can_resume_after_root_path_changes(tmp_path):
    import shutil

    original = settings(tmp_path)
    before = workflow().prepare(original, 'pilot')
    moved = tmp_path / 'uploaded/data/training'
    shutil.copytree(tmp_path / 'data/training', moved)
    after = workflow().prepare({**original, 'artifact_root': str(moved)}, 'pilot')
    assert after['root'] != before['root']
    assert sorted(after['input_hashes'].values()) == sorted(before['input_hashes'].values())


@pytest.mark.parametrize('phase', ['transition', 'baseline'])
def test_later_notebooks_require_completed_parent_runs(tmp_path, phase):
    with pytest.raises(ValueError, match='pilot|transition'):
        workflow().prepare(settings(tmp_path), phase)


def test_batch_and_campaign_names_are_validated_before_writing(tmp_path):
    with pytest.raises(ValueError, match='batch'):
        workflow().prepare({**settings(tmp_path), 'effective_batch': 3, 'batch_size': 2}, 'pilot')
    with pytest.raises(ValueError, match='campaign'):
        workflow().prepare({**settings(tmp_path), 'campaign_id': '../escape'}, 'pilot')


def test_notebook_executor_exists_and_requires_explicit_settings():
    spec = importlib.util.find_spec('tools.execute_training_notebook')
    assert spec is not None, 'missing headless notebook executor with persistent outputs'
    import subprocess
    import sys

    result = subprocess.run([sys.executable, '-m', 'tools.execute_training_notebook', '--notebook', 'unused.ipynb'],
                            capture_output=True, text=True)
    assert result.returncode == 2 and '--settings' in result.stderr


def test_baseline_evaluates_and_snapshots_after_each_task(tmp_path, monkeypatch):
    module = workflow()
    config = settings(tmp_path)
    root = module.campaign_root(config)
    for phase in ['pilot', 'transition']:
        save_json(root / phase / 'summary.json', {'status': 'COMPLETED'})
    plan = module.prepare(config, 'baseline')
    stages = []

    def fake_runner(command, environment, log_path):
        stage = int(environment['N_TASKS'])
        stages.append(stage)
        case = root / 'baseline/EXP-B1'
        for name in ['metrics_cl_val.json', 'metrics_cl_test.json', 'metrics_count_test.json']:
            save_json(case / name, {'stages': {str(stage): {}}})
        save_json(environment.get('CALIBRATION_POLICY', case / 'calibration_count.json'), {'stages': {str(stage): {}}})
        return 0

    monkeypatch.setattr(module, '_run_logged', fake_runner)
    monkeypatch.setattr(module, 'collect', lambda plan: {'status': 'COMPLETED'})
    module.run_phase(plan)
    assert stages == [1, 2, 3, 4, 5]
    for stage in stages:
        assert (root / f'baseline/EXP-B1/stage_metrics/task_{stage}/metrics_count_test.json').is_file()


def test_process_log_keeps_pid_and_exit_status(tmp_path):
    import sys

    log = tmp_path / 'runner.log'
    assert workflow()._run_logged([sys.executable, '-c', 'print("fixture")'], {}, log) == 0
    receipt = log.with_suffix('.process.json')
    assert receipt.is_file(), 'missing PID/exit receipt for foreground training subprocess'
    metadata = load_json(receipt)
    assert metadata['pid'] > 0 and metadata['returncode'] == 0 and metadata['status'] == 'COMPLETED'


def test_resume_does_not_launch_over_an_orphaned_live_runner(tmp_path):
    import os
    import socket

    module = workflow()
    plan = module.prepare(settings(tmp_path), 'pilot')
    receipt = Path(plan['phase_dir']) / 'attempts/old/runner.process.json'
    save_json(receipt, dict(status='RUNNING', pid=os.getpid(), host=socket.gethostname(),
                           process_start=module.process_start(os.getpid())))
    with pytest.raises(RuntimeError, match='still running'):
        module.run_phase(plan)
