from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json
import pytest

from agent_ahd.core.training_diagnostics import registered_training_diagnostics
from agent_ahd.core.research_guidance import guidance_for_candidate, retrieve_training_episodes


def fixture(tmp_path):
    source = tmp_path / 'solver.py'
    source.write_text('baseline')
    manifest = tmp_path / 'train.json'
    manifest.write_text(json.dumps({'instances': [{'id': 'train1', 'sha256': 'input1'}]}))
    receipt = {'schema': 'host_training_diagnostics_v1', 'split': 'train',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'instance_ids': ['train1'], 'seeds': [3],
        'rows': [{'instance_id': 'train1', 'seed': 3, 'input_sha256': 'input1',
                  'neighborhoods': {'insert': {'calls': 50, 'strict_move_rate': .02}}}]}
    path = tmp_path / 'receipt.json'
    adapter = NS(dataset_protocol={'task': {'baseline_path': str(source)}, 'metadata': {}},
                 train_manifest=NS(instance_ids=('train1',), source_path=manifest))
    config = NS(n_instances=1, base_seeds=(3, 4))
    def save():
        path.write_text(json.dumps(receipt))
        adapter.dataset_protocol['metadata']['training_diagnostics'] = {
            'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save()
    return adapter, config, receipt, path, save


def test_probe_enters_generation_guidance_without_quality_or_validation_leak(tmp_path):
    a, c, receipt, _, save = fixture(tmp_path)
    receipt['rows'][0]['best_objective'] = 1234567
    receipt['rows'][0]['improvement_trace'] = [{'objective': 1234567}]
    save()
    diagnostic = registered_training_diagnostics(a, c)
    assert '1234567' not in json.dumps(diagnostic)
    guidance = guidance_for_candidate({'episodes': [], 'host_training_diagnostics': diagnostic},
                                     action='EXPLOIT', slot=0, stagnation=0)
    assert 'evidence_directed_neighborhood' in guidance['skills']


@pytest.mark.parametrize('mutation', ['split', 'source', 'input', 'seed', 'duplicate', 'hash'])
def test_probe_rejects_stale_or_wrong_identity(tmp_path, mutation):
    a, c, r, path, save = fixture(tmp_path)
    if mutation == 'split': r['split'] = 'validation'
    elif mutation == 'source': r['source_sha256'] = 'different'
    elif mutation == 'input': r['rows'][0]['input_sha256'] = 'different'
    elif mutation == 'seed': r['seeds'] = [9]
    elif mutation == 'duplicate': r['rows'] *= 2
    save()
    if mutation == 'hash': path.write_text('{}')
    with pytest.raises(ValueError):
        registered_training_diagnostics(a, c)


def test_diagnostics_prioritize_gap_instead_of_zero_gap_stagnation(tmp_path):
    adapter, config, _, _, _ = fixture(tmp_path)
    from agent_ahd.core.research_records import EvaluationRecord
    baseline = EvaluationRecord('baseline', '__baseline_control__', 'train', (3, 4),
                                ('train1',), ((0.,), (0.,)), (), True)
    state = NS(nodes=[], hypotheses=[], evaluations=[baseline])
    memory = retrieve_training_episodes(state, config, adapter, before_round=0)
    row = memory['host_training_diagnostics']['rows'][0]
    assert row['quality_priority'] == 'already_matches_or_beats_reference'
    assert row['baseline_mean_gap_percent'] == 0
