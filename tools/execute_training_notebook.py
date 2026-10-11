"""Execute a training notebook headlessly and persist outputs while cells run."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

from autocheckout.io import load_json, save_json
from autocheckout.model_acceptance import now
from autocheckout.training_workflow import REPO, campaign_root, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', required=True, type=Path)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true', help='force isolated CPU fixture mode')
    parser.add_argument('--cell-timeout', type=int, default=-1, help='-1: no timeout for long training cells')
    parser.add_argument('--save-interval', type=float, default=15,
                        help='seconds between snapshots when notebook output arrives')
    parser.add_argument('--live-notebook', type=Path,
                        help='mirror outputs here until the notebook is edited externally')
    args = parser.parse_args()
    if args.save_interval <= 0:
        parser.error('--save-interval must be positive')
    import nbformat
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    settings = load_json(resolve(args.settings))
    if args.dry_run:
        settings['dry_run'] = True
    root = campaign_root(settings)
    notebook_path = resolve(args.notebook)
    attempt = root/'notebook_executions'/notebook_path.stem/str(uuid.uuid4())
    attempt.mkdir(parents=True)
    settings_path = attempt/'settings.json'
    save_json(settings_path, settings, indent=2)
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    nbformat.write(notebook, attempt/'notebook_source.ipynb')
    kernel_dir = attempt/'kernels/ac-training'
    save_json(kernel_dir/'kernel.json',
              dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
              display_name='AutoCheckout training', language='python'))
    (attempt/'kernel_runtime').mkdir()
    kernel_manager = AsyncKernelManager(kernel_name='ac-training',
                                        connection_file=str(attempt/'kernel_runtime/connection.json'),
                                        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    environment = {**os.environ, 'AUTOCHECKOUT_TRAINING_SETTINGS': str(settings_path), 'PYTHONUNBUFFERED': '1',
                   'JUPYTER_RUNTIME_DIR': str(attempt/'kernel_runtime'), 'IPYTHONDIR': str(attempt/'ipython'),
                   'MPLCONFIGDIR': str(root/'cache/matplotlib'), 'USE_TF': '0'}
    record = dict(status='RUNNING', started_at=now(), notebook=str(notebook_path), attempt_id=attempt.name,
                  dry_run=settings.get('dry_run', True), python=sys.executable, completed_cells=0)
    receipt = attempt/'execution.json'
    save_json(receipt, record, indent=2)
    console = (attempt/'console.log').open('a', buffering=1)
    live_path = resolve(args.live_notebook) if args.live_notebook else None
    live_expected = live_path.read_bytes() if live_path and live_path.exists() else None
    last_saved = 0.0
    snapshot_timer = None
    record.update(live_notebook=str(live_path) if live_path else None,
                  live_mirror_status='ACTIVE' if live_path else 'DISABLED')

    def log(message):
        text = f'{now()} {message}'
        print(text, flush=True)
        console.write(text+'\n')

    def save_snapshot(force=False):
        nonlocal last_saved, live_path, live_expected
        if not force and time.monotonic() - last_saved < args.save_interval:
            return
        text = nbformat.writes(notebook)

        def write_atomic(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f'.{path.name}.{attempt.name}.tmp')
            temporary.write_text(text)
            temporary.replace(path)

        write_atomic(attempt/'notebook_executed.ipynb')
        if live_path:
            current = live_path.read_bytes() if live_path.exists() else None
            if current == live_expected:
                write_atomic(live_path)
                live_expected = text.encode()
            else:
                record['live_mirror_status'] = 'STOPPED_EXTERNAL_EDIT'
                log(f'Live notebook edited externally; outputs continue in {attempt}/notebook_executed.ipynb')
                live_path = None
        last_saved = time.monotonic()
        record['last_notebook_saved_at'] = now()
        save_json(receipt, record, indent=2)

    class LiveNotebookClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            output = super().process_message(msg, cell, cell_index)
            if msg['msg_type'] == 'stream':
                text = msg['content']['text']
                console.write(text)
                console.flush()
                sys.stdout.write(text)
                sys.stdout.flush()
                outputs = cell.get('outputs', [])
                if (output is not None and len(outputs) >= 2 and outputs[-1] is output
                        and outputs[-2].output_type == 'stream'
                        and outputs[-2].name == output.name):
                    outputs[-2].text += output.text
                    outputs.pop()
            elif msg['msg_type'] == 'error':
                text = '\n'.join(msg['content'].get('traceback', []))+'\n'
                console.write(text)
                console.flush()
                sys.stdout.write(text)
                sys.stdout.flush()
            save_snapshot()
            return output

    def periodic_snapshot():
        nonlocal snapshot_timer
        save_snapshot(force=True)
        snapshot_timer = asyncio.get_running_loop().call_later(args.save_interval, periodic_snapshot)

    def started(cell, cell_index):
        nonlocal snapshot_timer
        record.update(current_cell=cell_index, cell_started_at=now())
        save_json(receipt, record, indent=2)
        log(f'CELL {cell_index} START')
        if snapshot_timer is None:
            snapshot_timer = asyncio.get_running_loop().call_later(args.save_interval, periodic_snapshot)

    def finished(cell, cell_index, execute_reply):
        record.update(completed_cells=record['completed_cells']+1, last_cell=cell_index, cell_finished_at=now())
        save_snapshot(force=True)
        log(f'CELL {cell_index} FINISH')

    client = LiveNotebookClient(notebook, km=kernel_manager, timeout=args.cell_timeout,
                            resources={'metadata': {'path': str(REPO)}}, on_cell_start=started,
                            on_cell_executed=finished)
    log(f'NOTEBOOK START outputs={attempt}')
    try:
        client.execute(env=environment, cleanup_kc=True)
        record.update(status='COMPLETED', finished_at=now())
        log('NOTEBOOK COMPLETED')
        return 0
    except BaseException as error:
        record.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAIL',
                      error_type=type(error).__name__, error=str(error), finished_at=now())
        (attempt/'error.log').write_text(traceback.format_exc())
        log(f"NOTEBOOK {record['status']} error={error}")
        raise
    finally:
        if snapshot_timer is not None:
            snapshot_timer.cancel()
        save_snapshot(force=True)
        save_json(receipt, record, indent=2)
        console.close()


if __name__ == '__main__':
    raise SystemExit(main())
