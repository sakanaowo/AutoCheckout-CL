"""Resumable OpenAI Batch requests for annotation suggestions, separate from labels."""

from __future__ import annotations

import fcntl
import hashlib
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from collections.abc import Callable
from typing import Any

from tools.ai_annotation_review import (
    PROMPT_VERSION, build_request, estimate_cost, file_sha256, parse_review, timestamp, write_json,
)

MODEL = 'gpt-6.1-sol'
TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}


class BatchAPIError(RuntimeError):
    def __init__(self, status: int):
        self.status_code = status
        super().__init__(f'OpenAI Batch API HTTP {status}; credentials and error body omitted')


class BatchClient:
    """Small stdlib client; POST is never automatically retried after uncertain failures."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    def _request(self, method: str, path: str, *, body: bytes | None = None,
                 content_type: str = 'application/json', binary: bool = False) -> Any:
        request = urllib.request.Request(
            'https://api.openai.com/v1/' + path,
            data=body, method=method,
            headers={'Authorization': 'Bearer ' + self._api_key, 'Content-Type': content_type},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read() if binary else json.load(response)
        except urllib.error.HTTPError as error:
            raise BatchAPIError(error.code) from None
        except (urllib.error.URLError, socket.timeout):
            raise RuntimeError('OpenAI Batch connection failed; submission status may be unknown. Resume using saved state.') from None

    def upload(self, path: Path) -> dict[str, Any]:
        boundary = 'rpc_batch_' + uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                'Content-Type: application/jsonl\r\n\r\n').encode()
        body += path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
        return self._request('POST', 'files', body=body, content_type='multipart/form-data; boundary=' + boundary)

    def create(self, file_id: str, metadata: dict[str, str]) -> dict[str, Any]:
        return self._request('POST', 'batches', body=json.dumps({
            'input_file_id': file_id, 'endpoint': '/v1/responses',
            'completion_window': '24h', 'metadata': metadata,
        }).encode())

    def get(self, batch_id: str) -> dict[str, Any]:
        return self._request('GET', 'batches/' + urllib.parse.quote(batch_id, safe=''))

    def list_batches(self) -> list[dict[str, Any]]:
        result = []
        after = None
        while True:
            query = urllib.parse.urlencode({'limit': 100, **({'after': after} if after else {})})
            page = self._request('GET', 'batches?' + query)
            result.extend(page['data'])
            if not page.get('has_more'):
                return result
            after = page['last_id']

    def download(self, file_id: str) -> bytes:
        return self._request('GET', 'files/' + urllib.parse.quote(file_id, safe='') + '/content', binary=True)


def prepare_bundle(cases: list[dict[str, Any]], *, repo: Path, raw_root: Path, out_dir: Path,
                   max_bytes: int = 190_000_000, max_requests: int = 50_000,
                   request_builder: Callable = build_request, prompt_version: str = PROMPT_VERSION,
                   reasoning_effort: str = 'low') -> dict[str, Any]:
    """Freeze self-contained vision JSONL shards below the upload limit, before any API call."""
    if not cases or not 0 < max_bytes <= 200_000_000 or not 0 < max_requests <= 50_000:
        raise ValueError('Nonempty cases and valid Batch size/request limits are required')
    keys = [(r['source_task'], r['annotation_id']) for r in cases]
    if len(keys) != len(set(keys)) or len({r['image_file'] for r in cases}) != len(cases):
        raise ValueError('Duplicate case keys or photos in Batch input')
    out_dir = out_dir.resolve()
    entries = []
    shards: list[list[bytes]] = [[]]
    shard_bytes = 0
    for case in cases:
        preview, cutout = repo / case['preview'], raw_root / case['cutout']
        payload = request_builder(case, preview, cutout, model=MODEL, reasoning_effort=reasoning_effort)
        fingerprint = hashlib.sha256(json.dumps({'payload': payload, 'source_sha256': case['source_sha256']},
                                               sort_keys=True).encode()).hexdigest()
        custom_id = f"rpc-t{case['source_task']}-a{case['annotation_id']}-{fingerprint[:16]}"
        line = (json.dumps({'custom_id': custom_id, 'method': 'POST', 'url': '/v1/responses', 'body': payload},
                           ensure_ascii=False, separators=(',', ':')) + '\n').encode()
        if len(line) > max_bytes:
            raise ValueError(f'Single request exceeds Batch byte limit: {custom_id}')
        if shards[-1] and (shard_bytes + len(line) > max_bytes or len(shards[-1]) == max_requests):
            shards.append([])
            shard_bytes = 0
        shards[-1].append(line)
        shard_bytes += len(line)
        entries.append({'case': case, 'case_key': [case['source_task'], case['annotation_id']],
                        'custom_id': custom_id, 'fingerprint': fingerprint, 'shard_index': len(shards)-1,
                        'preview_sha256': file_sha256(preview), 'cutout_sha256': file_sha256(cutout)})
    descriptions = [{'index': i, 'path': str(out_dir / f'input_{i:03d}.jsonl'),
                     'requests': len(lines), 'bytes': sum(map(len, lines)),
                     'sha256': hashlib.sha256(b''.join(lines)).hexdigest()} for i, lines in enumerate(shards)]
    fingerprint = hashlib.sha256(json.dumps({'cases': entries, 'shards': descriptions}, sort_keys=True).encode()).hexdigest()
    manifest_path = out_dir / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['fingerprint'] != fingerprint:
            raise ValueError('Frozen Batch inputs changed; use a new experiment directory')
        for shard in old['shards']:
            if file_sha256(Path(shard['path'])) != shard['sha256']:
                raise ValueError('Frozen Batch JSONL changed')
        return old
    out_dir.mkdir(parents=True, exist_ok=True)
    for description, lines in zip(descriptions, shards):
        path = Path(description['path'])
        temporary = path.with_suffix('.jsonl.tmp')
        with temporary.open('wb') as target:
            target.writelines(lines)
        temporary.replace(path)
    manifest = {'model': MODEL, 'prompt_version': prompt_version, 'created_at': timestamp(),
                'fingerprint': fingerprint, 'endpoint': '/v1/responses', 'completion_window': '24h',
                'cases': entries, 'shards': descriptions, 'repair_applied': False}
    write_json(manifest_path, manifest)
    return manifest


def submit_bundle(manifest: dict[str, Any], *, out_dir: Path, client: BatchClient) -> dict[str, Any]:
    """Persist IDs and create intent; recover a lost create response instead of submitting twice."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / 'submission.json'
    with (out_dir / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            'manifest_fingerprint': manifest['fingerprint'], 'created_at': timestamp(),
            'shards': [{'index': s['index'], 'file_id': None, 'batch_id': None,
                        'creation_pending': False, 'submission_id': manifest['fingerprint'][:32] + f"-{s['index']}"}
                       for s in manifest['shards']],
        }
        if state['manifest_fingerprint'] != manifest['fingerprint']:
            raise ValueError('Submission state/input fingerprint mismatch')
        for shard, saved in zip(manifest['shards'], state['shards']):
            if file_sha256(Path(shard['path'])) != shard['sha256']:
                raise ValueError('Frozen Batch JSONL changed')
            if saved['batch_id']:
                continue
            if saved['creation_pending']:
                matches = [r for r in client.list_batches()
                           if (r.get('metadata') or {}).get('local_submission_id') == saved['submission_id']]
                if len(matches) != 1 or matches[0]['input_file_id'] != saved['file_id']:
                    raise RuntimeError('Uncertain create is not uniquely recoverable; do not automatically resubmit')
                saved.update(batch_id=matches[0]['id'], batch=matches[0], creation_pending=False)
                write_json(state_path, state)
                continue
            if not saved['file_id']:
                upload = client.upload(Path(shard['path']))
                saved.update(file_id=upload['id'], upload=upload)
                write_json(state_path, state)
            saved['creation_pending'] = True
            saved['create_started_at'] = timestamp()
            write_json(state_path, state)
            try:
                remote = client.create(saved['file_id'], {'local_submission_id': saved['submission_id'],
                    'manifest_fingerprint': manifest['fingerprint'], 'purpose': 'rpc_annotation_suggestions'})
            except BatchAPIError as error:
                if 400 <= error.status_code < 500:
                    saved['creation_pending'] = False
                    saved['last_create_http_status'] = error.status_code
                    write_json(state_path, state)
                raise
            saved.update(batch_id=remote['id'], batch=remote, creation_pending=False)
            write_json(state_path, state)
        return state


def poll_bundle(manifest: dict[str, Any], *, out_dir: Path, client: BatchClient) -> dict[str, Any]:
    """Check once, persist status history, and cache output/error files from terminal batches."""
    state_path = out_dir / 'submission.json'
    state = json.loads(state_path.read_text())
    if state['manifest_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Submission state/input fingerprint mismatch')
    for shard in state['shards']:
        remote = client.get(shard['batch_id'])
        if remote['id'] != shard['batch_id']:
            raise ValueError('Remote Batch ID mismatch')
        shard['batch'] = remote
        shard.setdefault('polls', []).append({'checked_at': timestamp(), 'status': remote['status'],
                                             'request_counts': remote.get('request_counts')})
        if remote['status'] in TERMINAL:
            for name in ['output', 'error']:
                file_id = remote.get(name + '_file_id')
                if not file_id:
                    continue
                path = out_dir / f"{name}_{shard['index']:03d}.jsonl"
                if path.exists():
                    if shard.get(name + '_file_id') != file_id or file_sha256(path) != shard.get(name + '_sha256'):
                        raise ValueError('Downloaded Batch output changed')
                else:
                    temporary = path.with_suffix('.jsonl.tmp')
                    temporary.write_bytes(client.download(file_id))
                    temporary.replace(path)
                shard.update({name + '_file_id': file_id, name + '_path': str(path.resolve()),
                              name + '_sha256': file_sha256(path)})
        write_json(state_path, state)
    state['terminal'] = all(s['batch']['status'] in TERMINAL for s in state['shards'])
    state['last_checked_at'] = timestamp()
    write_json(state_path, state)
    return state


def parse_outputs(manifest: dict[str, Any], rows: list[dict[str, Any]], *,
                  batch_ids: list[str], response_parser: Callable = parse_review
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join unordered receipts, reject missing/error/invalid cases, and retain billed incomplete usage."""
    expected = {r['custom_id']: r for r in manifest['cases']}
    received = {}
    for row in rows:
        key = row['custom_id']
        if key not in expected:
            raise ValueError('Unknown Batch output custom_id')
        if key in received:
            raise ValueError('Duplicate Batch output custom_id')
        received[key] = row
    accepted, rejected = [], []
    for item in manifest['cases']:
        row = received.get(item['custom_id'])
        envelope = (row or {}).get('response') or {}
        response = envelope.get('body') or {}
        usage = response.get('usage')
        cost = estimate_cost(usage, model=response.get('model', MODEL)) if usage else None
        receipt = {'case_key': item['case_key'], 'custom_id': item['custom_id'],
                   'fingerprint': item['fingerprint'], 'prompt_version': manifest['prompt_version'],
                   'model': MODEL, 'source_sha256': item['case']['source_sha256'],
                   'preview': item['case']['preview'], 'cutout': item['case']['cutout'],
                   'preview_sha256': item['preview_sha256'], 'cutout_sha256': item['cutout_sha256'],
                   'response': response, 'response_id': response.get('id'), 'usage': usage,
                   'request_id': envelope.get('request_id'), 'batch_id': batch_ids[item['shard_index']],
                   'collected_at': timestamp(), 'pricing_mode': 'batch',
                   'estimated_cost_usd': cost * .5 if cost is not None else None,
                   'repair_applied': False, 'manual_decision_written': False}
        try:
            if row is None:
                raise ValueError('Missing Batch output')
            if row.get('error') or envelope.get('status_code') != 200:
                raise ValueError('Batch request failed; inspect saved error JSONL')
            if response.get('model') != MODEL:
                raise ValueError('Unexpected response model')
            receipt['review'] = response_parser(response, item['case'])
        except (ValueError, KeyError, TypeError) as error:
            receipt['acceptance_error'] = str(error)
            rejected.append(receipt)
        else:
            accepted.append(receipt)
    return accepted, rejected
