import io
import json
import urllib.error
from pathlib import Path

import pytest
from PIL import Image

from tools import ai_annotation_batch as batch
from tools import ai_annotation_review as ai


def cases(tmp_path, count=3):
    preview = tmp_path / 'preview.png'
    cutout = tmp_path / 'cutout.png'
    Image.new('RGB', (80, 60), 'gray').save(preview)
    Image.new('RGBA', (40, 30), (255, 0, 0, 128)).save(cutout)
    return [{'source_task': 1, 'annotation_id': i, 'image_id': i,
             'rpc_category_id': 72, 'iou': .9, 'bbox': [0, 0, 20, 10],
             'polygon_bbox': [1, 0, 19, 10], 'preview': str(preview),
             'cutout': str(cutout), 'image_file': f'photo{i}.jpg',
             'source_sha256': 'sourcehash'} for i in range(count)]


def prepare(tmp_path, count=3, **kwargs):
    return batch.prepare_bundle(cases(tmp_path, count), repo=tmp_path, raw_root=tmp_path,
                                out_dir=tmp_path / 'batch', **kwargs)


def result_row(item):
    c = item['case']
    review = {'source_task': c['source_task'], 'annotation_id': c['annotation_id'],
              'decision': 'pending', 'bbox_assessment': 'uncertain',
              'polygon_assessment': 'uncertain',
              'observations': ['The red box extends above the visible product.',
                               'Occlusion makes pixel ownership uncertain.'],
              'reason': 'The visible image does not establish ownership, so this case requires review.',
              'confidence': .8, 'needs_human': True}
    return {'custom_id': item['custom_id'], 'response': {'status_code': 200,
        'request_id': 'req_test', 'body': {'id': 'resp_'+str(c['annotation_id']),
        'status': 'completed', 'model': 'gpt-6.1-sol',
        'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(review)}]}],
        'usage': {'input_tokens': 2000, 'input_tokens_details': {'cached_tokens': 500, 'cache_write_tokens': 200},
                  'output_tokens': 300}}}, 'error': None}


class FakeClient:
    def __init__(self):
        self.uploads = []; self.creates = []; self.remote_batches = []
    def upload(self, path):
        self.uploads.append(path)
        return {'id': 'file_test', 'purpose': 'batch'}
    def create(self, file_id, metadata):
        self.creates.append((file_id, metadata))
        result = {'id': 'batch_test', 'status': 'validating', 'input_file_id': file_id,
                  'metadata': metadata}
        self.remote_batches.append(result)
        return result
    def list_batches(self):
        return self.remote_batches


def test_bundle_preserves_vision_payload_and_can_be_reprepared_without_changing_requests(tmp_path):
    first = prepare(tmp_path)
    second = prepare(tmp_path)
    assert first['fingerprint'] == second['fingerprint'] and first['shards'] == second['shards']
    rows = [json.loads(line) for shard in first['shards']
            for line in Path(shard['path']).read_text().splitlines()]
    assert len(rows) == len({r['custom_id'] for r in rows}) == 3
    assert all(r['method'] == 'POST' and r['url'] == '/v1/responses' for r in rows)
    assert all(r['body']['model'] == 'gpt-6.1-sol' and r['body']['store'] is False for r in rows)
    assert all(r['body']['reasoning']['effort'] == 'low' for r in rows)
    assert all(len([p for p in r['body']['input'][0]['content'] if p['type'] == 'input_image']) == 2 for r in rows)


def test_bundle_splits_at_request_limit_and_rejects_a_single_oversized_request(tmp_path):
    manifest = prepare(tmp_path, 5, max_requests=2)
    assert [s['requests'] for s in manifest['shards']] == [2, 2, 1]
    with pytest.raises(ValueError, match='exceeds'):
        batch.prepare_bundle(cases(tmp_path, 1), repo=tmp_path, raw_root=tmp_path,
                             out_dir=tmp_path/'too-large', max_bytes=50)


def test_bundle_rejects_duplicate_cases_and_modified_frozen_images(tmp_path):
    records = cases(tmp_path, 1)
    with pytest.raises(ValueError, match='Duplicate'):
        batch.prepare_bundle(records*2, repo=tmp_path, raw_root=tmp_path, out_dir=tmp_path/'duplicate')
    prepare(tmp_path, 1)
    Image.new('RGB', (80, 60), 'blue').save(tmp_path/'preview.png')
    with pytest.raises(ValueError, match='changed'):
        batch.prepare_bundle(records, repo=tmp_path, raw_root=tmp_path, out_dir=tmp_path/'batch')


def test_resume_does_not_upload_or_create_a_second_batch(tmp_path):
    manifest = prepare(tmp_path, 1); client = FakeClient()
    first = batch.submit_bundle(manifest, out_dir=tmp_path/'batch', client=client)
    second = batch.submit_bundle(manifest, out_dir=tmp_path/'batch', client=client)
    assert len(client.uploads) == len(client.creates) == 1
    assert first['shards'][0]['batch_id'] == second['shards'][0]['batch_id'] == 'batch_test'


def test_uncertain_create_is_recovered_using_remote_metadata_without_rebilling(tmp_path):
    manifest = prepare(tmp_path, 1); client = FakeClient()
    def lose_response(file_id, metadata):
        client.remote_batches.append({'id': 'batch_already_created', 'status': 'validating',
                                      'input_file_id': file_id, 'metadata': metadata})
        raise RuntimeError('Transport failed after submit')
    client.create = lose_response
    with pytest.raises(RuntimeError):
        batch.submit_bundle(manifest, out_dir=tmp_path/'batch', client=client)
    state = batch.submit_bundle(manifest, out_dir=tmp_path/'batch', client=client)
    assert state['shards'][0]['batch_id'] == 'batch_already_created' and len(client.uploads) == 1


def test_out_of_order_outputs_are_matched_by_custom_id_and_use_discounted_cost(tmp_path):
    manifest = prepare(tmp_path, 3)
    rows = [result_row(item) for item in reversed(manifest['cases'])]
    accepted, rejected = batch.parse_outputs(manifest, rows, batch_ids=['batch_test'])
    assert not rejected and [r['case_key'][1] for r in accepted] == [0, 1, 2]
    assert all(r['estimated_cost_usd'] == pytest.approx(.003075) for r in accepted)
    assert all(r['pricing_mode'] == 'batch' and r['repair_applied'] is False for r in accepted)


def test_incomplete_wrong_model_missing_and_api_failures_are_not_accepted(tmp_path):
    manifest = prepare(tmp_path, 4)
    rows = [result_row(item) for item in manifest['cases'][:3]]
    rows[0]['response']['body']['status'] = 'incomplete'
    rows[1]['response']['body']['model'] = 'other-model'
    rows[2] = {'custom_id': rows[2]['custom_id'], 'response': None, 'error': {'code': 'expired'}}
    accepted, rejected = batch.parse_outputs(manifest, rows, batch_ids=['batch_test'])
    assert not accepted and len(rejected) == 4
    # An incomplete response is still billed and must keep usage/cost in its rejected receipt.
    assert rejected[0]['estimated_cost_usd'] == pytest.approx(.003075)


def test_duplicate_or_unknown_output_ids_cannot_pass_coverage(tmp_path):
    manifest = prepare(tmp_path, 1); row = result_row(manifest['cases'][0])
    with pytest.raises(ValueError, match='Duplicate'):
        batch.parse_outputs(manifest, [row, row], batch_ids=['batch_test'])
    with pytest.raises(ValueError, match='Unknown'):
        batch.parse_outputs(manifest, [{**row, 'custom_id': 'unknown'}], batch_ids=['batch_test'])


def test_upload_uses_batch_purpose_and_errors_never_echo_the_key(tmp_path, monkeypatch):
    path = tmp_path/'input.jsonl'; path.write_text('{}\n')
    captured = []
    class Reply(io.BytesIO):
        headers = {}
    def receive(request, timeout):
        captured.append(request)
        return Reply(b'{"id":"file_uploaded","purpose":"batch"}')
    monkeypatch.setattr('urllib.request.urlopen', receive)
    client = batch.BatchClient('unit-test-secret')
    assert client.upload(path)['id'] == 'file_uploaded'
    body = captured[0].data
    assert b'name="purpose"\r\n\r\nbatch' in body
    assert b'filename="input.jsonl"' in body and b'{}\n' in body
    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, 'Unauthorized', None,
                                    io.BytesIO(b'unit-test-secret'))
    monkeypatch.setattr('urllib.request.urlopen', fail)
    with pytest.raises(RuntimeError) as error:
        client.get('batch_test')
    assert 'unit-test-secret' not in str(error.value) and '401' in str(error.value)


def test_completed_outputs_are_downloaded_once_and_preserved_for_offline_acceptance(tmp_path):
    manifest = prepare(tmp_path, 1); client = FakeClient()
    batch.submit_bundle(manifest, out_dir=tmp_path/'batch', client=client)
    client.get = lambda batch_id: {'id': batch_id, 'status': 'completed',
        'output_file_id': 'file_output', 'error_file_id': None,
        'request_counts': {'total': 1, 'completed': 1, 'failed': 0}}
    downloads = []
    def download(file_id):
        downloads.append(file_id)
        return (json.dumps(result_row(manifest['cases'][0]))+'\n').encode()
    client.download = download
    first = batch.poll_bundle(manifest, out_dir=tmp_path/'batch', client=client)
    second = batch.poll_bundle(manifest, out_dir=tmp_path/'batch', client=client)
    assert downloads == ['file_output'] and first['terminal'] and second['terminal']
    assert Path(second['shards'][0]['output_path']).is_file()


def test_unconfirmed_create_with_no_remote_match_never_submits_another_batch(tmp_path):
    manifest = prepare(tmp_path, 1); client = FakeClient(); attempts = []
    def lose_response(file_id, metadata):
        attempts.append(metadata)
        raise RuntimeError('Unknown create outcome')
    client.create = lose_response
    with pytest.raises(RuntimeError, match='Unknown'):
        batch.submit_bundle(manifest, out_dir=tmp_path/'batch', client=client)
    with pytest.raises(RuntimeError, match='not uniquely recoverable'):
        batch.submit_bundle(manifest, out_dir=tmp_path/'batch', client=client)
    assert len(attempts) == len(client.uploads) == 1


def test_real_client_creates_only_responses_batches_with_24h_window(monkeypatch):
    captured = []
    class Reply(io.BytesIO):
        headers = {}
    def receive(request, timeout):
        captured.append(request)
        return Reply(b'{"id":"batch_uploaded","status":"validating"}')
    monkeypatch.setattr('urllib.request.urlopen', receive)
    client = batch.BatchClient('unit-test-secret')
    client.create('file_uploaded', {'local_submission_id': 'submission_test'})
    assert captured[0].full_url == 'https://api.openai.com/v1/batches'
    assert json.loads(captured[0].data) == {'input_file_id':'file_uploaded', 'endpoint':'/v1/responses',
        'completion_window':'24h', 'metadata':{'local_submission_id':'submission_test'}}


def test_batch_accepts_a_new_request_policy_and_matching_response_parser(tmp_path):
    records = cases(tmp_path, 1)
    def builder(case, preview, cutout, *, model, reasoning_effort):
        payload = ai.build_request(case, preview, cutout, model=model, reasoning_effort=reasoning_effort)
        return {**payload, 'instructions': 'Final annotation resolution policy'}
    manifest = batch.prepare_bundle(records, repo=tmp_path, raw_root=tmp_path, out_dir=tmp_path/'custom',
        request_builder=builder, prompt_version='rpc-resolution-v1', reasoning_effort='medium')
    row = json.loads(Path(manifest['shards'][0]['path']).read_text())
    assert row['body']['instructions'] == 'Final annotation resolution policy'
    assert row['body']['reasoning']['effort'] == 'medium' and manifest['prompt_version'] == 'rpc-resolution-v1'
    response = result_row(manifest['cases'][0])
    parsed = {'decision': 'regenerate', 'reason': 'Final decision with independent geometry evidence.'}
    accepted, rejected = batch.parse_outputs(manifest, [response], batch_ids=['batch_test'],
                                              response_parser=lambda body, case: parsed)
    assert not rejected and accepted[0]['review'] == parsed
    assert accepted[0]['prompt_version'] == 'rpc-resolution-v1'
