import itertools
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np
import pytest
from cef.reference import ReferenceIncomplete
from cef.reference_jobs import validate_reference
from cef.reference_windows import solve_tsp_reference,backend_manifest

@pytest.fixture(autouse=True)
def isolated_reference_cache(tmp_path,monkeypatch):
    monkeypatch.setattr('cef.reference_windows.ROOT',tmp_path)

def test_native_windows_backend_has_no_virtual_machine_dependency():
    m=backend_manifest()
    assert m['architecture']=='windows-x86_64'
    assert 'SCIP' in m['backend']
    assert all('concorde_linux' not in r['path'] for r in m['files'])

def test_disconnected_cheaper_cycles_are_not_accepted_as_reference(tmp_path):
    d=np.ones((6,6));d[:3,3:]=10;d[3:,:3]=10;np.fill_diagonal(d,0)
    r=solve_tsp_reference(d,seconds=20,checkpoint_path=tmp_path/'work.json',console=False)
    assert r['objective']==24 and sorted(r['solution'])==list(range(6))
    validate_reference(d,r,.00001)

@pytest.mark.parametrize('seed',[17,18])
def test_euclidean_reference_matches_exhaustive_optimum(tmp_path,seed):
    xy=np.random.default_rng(seed).random((8,2));d=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    exact=min(math.fsum(float(d[t[i],t[(i+1)%8]]) for i in range(8)) for t in ((0,)+p for p in itertools.permutations(range(1,8))))
    r=solve_tsp_reference(d,seconds=20,checkpoint_path=tmp_path/'work.json',console=False)
    assert abs(r['objective']-exact)<1e-10
    assert r['lower_bound']<=exact+1e-10
    validate_reference(d,r,.00001)
    with patch('cef.reference_windows.subprocess.Popen',side_effect=AssertionError('cached result must not execute again')):
        reused=solve_tsp_reference(d,seconds=20,checkpoint_path=tmp_path/'work.json',console=False)
    assert reused['objective']==r['objective'] and reused['cache_hit']

def test_expired_diagnostic_budget_never_returns_a_reference(tmp_path):
    d=np.ones((11,11))-np.eye(11)
    with pytest.raises(ReferenceIncomplete):
        solve_tsp_reference(d,seconds=1e-12,checkpoint_path=tmp_path/'pending.json',console=False)

def test_nonfinite_asymmetric_or_negative_matrix_is_rejected():
    for d in (np.full((4,4),np.nan),np.eye(4),np.array([[0,1,2],[2,0,1],[2,1,0]]),-np.ones((4,4))):
        with pytest.raises(ValueError):solve_tsp_reference(d,console=False)

def test_same_matrix_parallel_requests_have_one_consistent_reference(tmp_path):
    xy=np.random.default_rng(142).random((9,2));d=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    def work(i):return solve_tsp_reference(d,seconds=20,checkpoint_path=tmp_path/str(i)/'work.json',console=False)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(work,range(2)))
    assert results[0]['objective']==results[1]['objective']
    assert sum(not r['cache_hit'] for r in results)<=1


def test_small_coordinate_scale_still_uses_original_float_objective(tmp_path):
    xy=np.random.default_rng(32).random((7,2))*1e-10
    d=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    exact=min(math.fsum(float(d[t[i],t[(i+1)%7]]) for i in range(7)) for t in ((0,)+p for p in itertools.permutations(range(1,7))))
    r=solve_tsp_reference(d,seconds=20,checkpoint_path=tmp_path/'tiny.json',console=False)
    assert abs(r['objective']-exact)<1e-22
    validate_reference(d,r,.00001)


def test_claimed_optimum_with_wrong_objective_cannot_pass_certification():
    from cef.reference_windows import certify
    from cef.reference_scip_solver import matrix_identity
    d=np.ones((5,5))-np.eye(5)
    raw=dict(matrix_hash=matrix_identity(d),n=5,solution=list(range(5)),model_objective=3.,normalized_lower_bound=3.)
    with pytest.raises(ValueError,match='recomputed'):certify(d,raw,.00001)


def test_cancellation_kills_only_owned_worker_then_same_case_can_continue(tmp_path):
    from concurrent.futures import CancelledError
    from unittest.mock import Mock
    xy=np.random.default_rng(911).random((11,2));d=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    process=Mock();process.poll.return_value=None;process.wait.return_value=0
    checks=[]
    def cancelled():checks.append(1);return len(checks)>1
    with patch('cef.reference_windows.subprocess.Popen',return_value=process),patch('cef.parallel.cancellation_requested',side_effect=cancelled):
        with pytest.raises(CancelledError):solve_tsp_reference(d,checkpoint_path=tmp_path/'cancel.json',console=False)
    process.kill.assert_called_once()
    r=solve_tsp_reference(d,seconds=20,checkpoint_path=tmp_path/'cancel.json',console=False)
    validate_reference(d,r,.00001)
