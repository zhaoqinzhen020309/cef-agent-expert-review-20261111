"""Read bounded host diagnostic evidence with explicit baseline/data identity.

The controller stays task-independent. Optional task metadata registers a
receipt hash; absent or mismatched evidence is never silently reused.
"""
from pathlib import Path
import hashlib
import json
import math


def registered_training_diagnostics(adapter, config):
    protocol = getattr(adapter, 'dataset_protocol', {}) or {}
    registration = protocol.get('metadata', {}).get('training_diagnostics')
    if not registration:
        return None
    path = Path(registration['path'])
    raw = path.read_bytes()
    if len(raw) > 256 * 1024 or hashlib.sha256(raw).hexdigest() != registration['sha256']:
        raise ValueError('training diagnostic receipt identity changed')
    receipt = json.loads(raw)
    baseline = Path(protocol['task']['baseline_path'])
    ids = list(adapter.train_manifest.instance_ids[:config.n_instances])
    if (receipt.get('schema') != 'host_training_diagnostics_v1' or receipt.get('split') != 'train'
            or receipt.get('source_sha256') != hashlib.sha256(baseline.read_bytes()).hexdigest()
            or receipt.get('instance_ids') != ids
            or not receipt.get('seeds') or not set(receipt['seeds']) <= set(config.base_seeds)):
        raise ValueError('training diagnostics do not match baseline, training inputs or seeds')
    manifest_path = Path(adapter.train_manifest.source_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_inputs = {r['id']: r['sha256'] for r in manifest['instances'] if r['id'] in ids}
    rows = receipt.get('rows', [])
    if len(rows) != len(ids) * len(receipt['seeds']):
        raise ValueError('incomplete training diagnostic matrix')
    seen, summaries = set(), []
    for row in rows:
        key = (row['instance_id'], row['seed'])
        if (key in seen or key[0] not in ids or key[1] not in receipt['seeds']
                or row.get('input_sha256') != expected_inputs[key[0]]):
            raise ValueError('training diagnostic row identity mismatch')
        seen.add(key)
        summary = {k: row[k] for k in ('shape', 'iterations', 'incumbent_improvements',
            'longest_stagnation_iterations', 'right_censored_final_stagnation_iterations') if k in row}
        neighborhoods = row.get('neighborhoods', {})
        if not isinstance(neighborhoods, dict) or len(neighborhoods) > 12:
            raise ValueError('invalid diagnostic neighborhoods')
        clean = {}
        for name, values in neighborhoods.items():
            if not isinstance(name, str) or len(name) > 100 or not isinstance(values, dict):
                raise ValueError('invalid diagnostic metric group')
            clean[name] = {}
            for k in ('seconds_with_timer_overhead', 'calls', 'job_attempts', 'strict_improving_moves',
                      'neutral_moves', 'positions_scored', 'strict_move_rate'):
                value = values.get(k)
                if value is not None:
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError('invalid diagnostic measurement')
                    clean[name][k] = value
        summaries.append({'instance_id': key[0], 'seed': key[1], **summary, 'neighborhoods': clean})
    return {'evidence_type': 'host_instrumented_training_probe', 'source_sha256': receipt['source_sha256'],
        'receipt_sha256': registration['sha256'], 'split': 'train', 'rows': summaries,
        'limitations': 'Applies only to the supplied baseline, with instrumentation overhead and separate probe budget. It is not scored performance or evidence about any edited descendant.'}
