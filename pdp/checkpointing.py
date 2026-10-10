"""Resume checkpoints for Spot VMs (R1).

A Spot VM can be stopped at any moment with at most 30 s notice, so the training state of the
current task is kept in <task dir>/last.ckpt: written at the end of every training epoch and,
within an epoch, every `every_minutes`, between batches, after an optimizer step. The file is written to a
temporary name and renamed, and the previous checkpoint is kept as last.ckpt.prev: Lightning's own
saving writes straight into the target file, so a stop during a write would leave it truncated.
"""

import os
import time
import json
import math
from pathlib import Path

import pytorch_lightning as pl
from autocheckout.model_state import state_digest
from autocheckout.model_acceptance import now
from autocheckout.io import save_json

LAST = 'last.ckpt'


class PlannedTrainingStop(Exception):
    """A requested, checkpointed stop; CLI exit 75 leaves the task resumable."""


class ResumeAudit(pl.Callback):
    def __init__(self, verify=False):
        self.verify = verify

    def on_train_start(self, trainer, pl_module):
        report = {'restored_step': trainer.global_step, 'verified': False}
        if self.verify and trainer.ckpt_path:
            import torch
            saved = torch.load(trainer.ckpt_path, map_location='cpu')
            memory = {}
            pl_module.on_save_checkpoint(memory)
            live = {'state_dict': pl_module.state_dict(),
                    'optimizer_states': [o.state_dict() for o in trainer.optimizers],
                    'lr_schedulers': [c.scheduler.state_dict() for c in trainer.lr_scheduler_configs],
                    'pdp_state': memory['pdp_state']}
            equal = {key: state_digest(value) == state_digest(saved[key]) for key, value in live.items()}
            if not all(equal.values()) or trainer.global_step != saved['global_step']:
                raise ValueError(f'Restored checkpoint state differs: {equal}')
            report.update(verified=True, state_equal=True, components=equal)
        pl_module.resume_verification = report


def resume_path(task_dir):
    """last.ckpt of the task if training was interrupted, else None."""
    path = os.path.join(task_dir, LAST)
    if os.path.exists(path):
        return path
    return path + '.prev' if os.path.exists(path + '.prev') else None


class TrainingProgress(pl.Callback):
    """Append one telemetry row per optimizer step; preserve each resumed session."""

    def __init__(self, task_dir):
        self.directory = Path(task_dir)

    def on_train_start(self, trainer, pl_module):
        self.started = time.monotonic()
        self.initial_step = self.last_step = trainer.global_step
        teacher = pl_module.teacher
        record = dict(timestamp=now(), task_id=pl_module.task_id, optimizer_step=trainer.global_step,
                      teacher_present=teacher is not None,
                      teacher_frozen=teacher is not None and not any(p.requires_grad for p in teacher.parameters()),
                      previous_classes=pl_module.PREV_INTRODUCED_CLS, seen_classes=pl_module.seen_classes)
        save_json(self.directory / 'training_start.json', record, indent=2)
        self.persist(dict(record, status='RUNNING'))

    def persist(self, record):
        self.directory.mkdir(parents=True, exist_ok=True)
        save_json(self.directory / 'progress.json', record, indent=2)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == self.last_step or not trainer.is_global_zero:
            return
        import torch
        elapsed = time.monotonic() - self.started
        losses = {str(k): float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                  if hasattr(v, 'numel') and v.numel() == 1}
        if not all(math.isfinite(value) for value in losses.values()):
            raise FloatingPointError(f'Nonfinite training metrics: {losses}')
        record = dict(timestamp=now(), task_id=pl_module.task_id, epoch=trainer.current_epoch,
                      batch_idx=batch_idx, optimizer_step=trainer.global_step, losses=losses,
                      learning_rates=[g['lr'] for o in trainer.optimizers for g in o.param_groups],
                      session_seconds=round(elapsed, 3),
                      seconds_per_optimizer_step=round(elapsed / (trainer.global_step-self.initial_step), 3),
                      teacher_forward_calls=getattr(pl_module, 'teacher_forward_calls', 0),
                      ppg_pseudo_labels_total=getattr(pl_module, 'ppg_pseudo_labels_total', 0),
                      peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30, 3) if torch.cuda.is_available() else 0,
                      peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30, 3) if torch.cuda.is_available() else 0)
        with (self.directory / 'step_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        self.persist(dict(record, status='RUNNING'))
        self.last_step = trainer.global_step

    def on_train_end(self, trainer, pl_module):
        self.persist(dict(timestamp=now(), status='TRAINING_FINISHED', task_id=pl_module.task_id,
                          optimizer_step=trainer.global_step,
                          teacher_forward_calls=getattr(pl_module, 'teacher_forward_calls', 0),
                          ppg_pseudo_labels_total=getattr(pl_module, 'ppg_pseudo_labels_total', 0),
                          missing_prototypes=pl_module.missing_prototypes()))

    def on_exception(self, trainer, pl_module, exception):
        import torch
        status = ('INTERRUPTED' if isinstance(exception, (KeyboardInterrupt, PlannedTrainingStop)) else
                  'OOM' if isinstance(exception, torch.cuda.OutOfMemoryError) else 'FAIL')
        self.persist(dict(timestamp=now(), status=status, task_id=pl_module.task_id,
                          optimizer_step=trainer.global_step, error_type=type(exception).__name__, error=str(exception)))


def remove_resume_checkpoints(task_dir):
    """Called once the task's final weights are saved; resume checkpoints are 0.5 GiB each."""
    for name in (LAST, LAST + '.prev', LAST + '.tmp'):
        path = os.path.join(task_dir, name)
        if os.path.exists(path):
            os.remove(path)


class ResumeCheckpoint(pl.Callback):
    def __init__(self, task_dir, every_minutes=30, stop_after_steps=0):
        self.path = os.path.join(task_dir, LAST)
        self.every_seconds = every_minutes * 60
        self._last_save = time.monotonic()
        self._global_step = None
        self.stop_after_steps = stop_after_steps

    def on_train_start(self, trainer, pl_module):
        self._global_step = trainer.global_step
        self._last_save = time.monotonic()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Saved at the start of the next batch, when Lightning has already counted the previous
        # batch as completed: a checkpoint written in on_train_batch_end makes the resumed run
        # repeat that batch.
        stepped = trainer.global_step != self._global_step  # the previous batch ran an optimizer step
        self._global_step = trainer.global_step
        if self.stop_after_steps and trainer.global_step >= self.stop_after_steps:
            self.save(trainer)
            raise PlannedTrainingStop(f'Saved at optimizer step {trainer.global_step}')
        if stepped and time.monotonic() - self._last_save >= self.every_seconds:
            self.save(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        self.save(trainer)
        if self.stop_after_steps and trainer.global_step >= self.stop_after_steps:
            raise PlannedTrainingStop(f'Saved at optimizer step {trainer.global_step}')

    def save(self, trainer):
        tmp = self.path + '.tmp'
        trainer.save_checkpoint(tmp)
        if os.path.exists(self.path):
            os.replace(self.path, self.path + '.prev')
        os.replace(tmp, self.path)
        self._last_save = time.monotonic()
        save_json(Path(self.path).parent / 'checkpoint_status.json',
                  dict(timestamp=now(), optimizer_step=trainer.global_step, path=self.path,
                       bytes=os.path.getsize(self.path), previous_exists=os.path.exists(self.path+'.prev')), indent=2)
