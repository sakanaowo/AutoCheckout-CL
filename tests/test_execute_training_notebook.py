"""Live headless outputs persist before a cell ends and on failure."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]


def start_notebook(tmp_path, source):
    notebook = tmp_path/'live.ipynb'
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), notebook)
    settings = tmp_path/'settings.json'
    settings.write_text(json.dumps(dict(artifact_root=str(tmp_path/'data'), campaign_id='live-output', dry_run=True)))
    stream = (tmp_path/'process.log').open('w')
    process = subprocess.Popen([sys.executable, '-m', 'tools.execute_training_notebook',
                               '--notebook', str(notebook), '--settings', str(settings),
                               '--live-notebook', str(notebook), '--save-interval', '0.1'],
                              cwd=REPO, env={**os.environ, 'USE_TF':'0'}, stdout=stream, stderr=subprocess.STDOUT)
    return process, stream, notebook, tmp_path/'data/dry_run/live-output/notebook_executions/live'


def test_live_output_and_protection_of_external_notebook_edit(tmp_path):
    release = tmp_path/'release'
    child = f"""import sys, time
from pathlib import Path
print('live stdout ' + 'probe', flush=True)
print('live stderr ' + 'probe', file=sys.stderr, flush=True)
deadline = time.monotonic() + 25
while not Path({str(release)!r}).exists():
    if time.monotonic() > deadline:
        raise TimeoutError('test did not release the cell')
    time.sleep(0.05)
print('finished probe', flush=True)
"""
    source = f"""import sys
from pathlib import Path
from autocheckout.training_workflow import _run_logged
assert _run_logged([sys.executable, '-c', {child!r}], {{}}, Path({str(tmp_path/'runner.log')!r})) == 0
"""
    process, stream, notebook, root = start_notebook(tmp_path, source)
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            assert process.poll() is None, (tmp_path/'process.log').read_text()
            live = nbformat.read(notebook, as_version=4)
            texts = ''.join(o.get('text', '') for o in live.cells[0].outputs)
            if 'live stdout probe' in texts and 'live stderr probe' in texts:
                break
            time.sleep(0.05)
        else:
            raise AssertionError('missing output while cell is still running')
        attempt = next(root.iterdir())
        console = (attempt/'console.log').read_text()
        assert 'live stdout probe' in console and 'live stderr probe' in console
        assert 'CELL 0 FINISH' not in console
        assert 'live stdout probe' in (attempt/'notebook_executed.ipynb').read_text()
        live.cells[0].source += '\n# user edit while running'
        nbformat.write(live, notebook)
        release.touch()
        assert process.wait(timeout=25) == 0, (tmp_path/'process.log').read_text()
        assert 'user edit while running' in nbformat.read(notebook, as_version=4).cells[0].source
        record = json.loads((attempt/'execution.json').read_text())
        assert record['status'] == 'COMPLETED' and record['live_mirror_status'] == 'STOPPED_EXTERNAL_EDIT'
        assert (attempt/'console.log').read_text().count('live stdout probe') == 1
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        stream.close()


def test_failed_cell_keeps_stream_and_error_in_both_notebooks(tmp_path):
    process, stream, notebook, root = start_notebook(tmp_path, "print('before failure', flush=True)\nraise ValueError('fixture failure')")
    try:
        assert process.wait(timeout=25) != 0
        attempt = next(root.iterdir())
        for path in [notebook, attempt/'notebook_executed.ipynb']:
            saved = nbformat.read(path, as_version=4)
            assert any('before failure' in o.get('text', '') for o in saved.cells[0].outputs)
            assert any(o.output_type == 'error' and o.ename == 'ValueError' for o in saved.cells[0].outputs)
        assert json.loads((attempt/'execution.json').read_text())['status'] == 'FAIL'
        assert 'fixture failure' in (attempt/'error.log').read_text()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        stream.close()
