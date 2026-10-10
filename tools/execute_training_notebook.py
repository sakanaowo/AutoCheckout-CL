"""Execute a training notebook headlessly and persist outputs after every cell."""

from __future__ import annotations

import argparse
import os
import sys
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
    args = parser.parse_args()
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

    def log(message):
        text = f'{now()} {message}'
        print(text, flush=True)
        console.write(text+'\n')

    def started(cell, cell_index):
        record.update(current_cell=cell_index, cell_started_at=now())
        save_json(receipt, record, indent=2)
        log(f'CELL {cell_index} START')

    def finished(cell, cell_index, execute_reply):
        nbformat.write(notebook, attempt/'notebook_executed.ipynb')
        record.update(completed_cells=record['completed_cells']+1, last_cell=cell_index, cell_finished_at=now())
        save_json(receipt, record, indent=2)
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                console.write(output.text)
                print(output.text, flush=True)
        log(f'CELL {cell_index} FINISH')

    client = NotebookClient(notebook, km=kernel_manager, timeout=args.cell_timeout,
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
        nbformat.write(notebook, attempt/'notebook_executed.ipynb')
        save_json(receipt, record, indent=2)
        console.close()


if __name__ == '__main__':
    raise SystemExit(main())
