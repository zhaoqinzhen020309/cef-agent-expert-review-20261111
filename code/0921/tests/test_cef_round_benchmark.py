import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from cef.patch import digest

def test_snapshot_scores_caches_and_preserves_search(tmp_path):
    from cef.round_benchmark import evaluate_snapshot
    from cef.data import receipt
    source="def solve(instance,time_limit_s,seed):\n    return {'solution':[0,1,2]}\n"
    path=tmp_path/'triangle.npz';np.savez(path,distance_matrix=np.ones((3,3))-np.eye(3))
    rows=[dict(id='triangle',reference=3.,**receipt(path))]
    config=dict(solver_time_limit_s=.1,evaluation_workers=1,memory_mb=1024)
    test=dict(instances=rows,test_repeats=1,evaluation_seeds={'triangle:0':2025})
    out=tmp_path/'external'
    with patch('cef.round_benchmark.audit_test'):
        result=evaluate_snapshot(source,config,test,out)
        assert result['valid'] and result['mean_gap_pct']==0.
        with patch('cef.round_benchmark.Runner.case',side_effect=AssertionError('cached')):
            assert evaluate_snapshot(source,config,test,out)==result
    assert not (tmp_path/'search').exists()

def test_manual_config_uses_historical_metric():
    from runCEFAuto import CONFIG
    from scripts.cef_alternating_launcher import build_auto_config
    c,t=build_auto_config(CONFIG,Path('unused_manual_config'),2025)
    assert c['round_benchmark']
    assert {r['evaluation_metric'] for r in t['instances']}=={'tsplib_unrounded_original_reference_v1'}

def test_round_monitor_only_at_requested_complete_boundary(tmp_path):
    from cef.round_benchmark import monitor_round
    from types import SimpleNamespace
    records={'round_complete':[{'round':1}],'manual_step_requested':[{'target_round':2}]}
    engine=SimpleNamespace(archive=SimpleNamespace(of=lambda k:records.get(k,[])),training_epoch=1,completed_iterations=lambda:1)
    with patch('cef.round_benchmark.evaluate_snapshot',side_effect=AssertionError('pending next round')):
        assert monitor_round(engine,{}, {},tmp_path) is None
