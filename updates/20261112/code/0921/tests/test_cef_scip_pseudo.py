from pyscipopt import SCIP_RESULT
from unittest.mock import Mock
import itertools
import math
import numpy as np
import pytest
import cef.reference_scip_solver as solver
from cef.reference_scip_solver import SubtourHandler


def test_pseudo_solution_requires_subtour_enforcement(tmp_path):
    handler = SubtourHandler({}, 6, tmp_path / 'state.json', 'identity', [])
    # A pseudo solution has not passed the LP/subtour enforcement. It must
    # never be declared feasible merely to bypass a missing callback.
    handler.consenfolp = Mock(return_value={'result': SCIP_RESULT.CONSADDED})
    assert handler.consenfops([], 0, False, False) == {'result': SCIP_RESULT.CONSADDED}
    handler.consenfolp.assert_called_once_with([], 0, False)


def test_pseudo_subtour_is_cut_and_connected_support_is_checked(tmp_path):
    from pyscipopt import Model
    model=Model()
    variables={e:model.addVar(vtype='B') for e in itertools.combinations(range(6),2)}
    handler=SubtourHandler(variables,6,tmp_path/'state.json','fixture',[])
    handler.model=model
    handler.groups=lambda *args:[{0,1,2},{3,4,5}]
    assert handler.consenfops([],0,False,False)['result']==SCIP_RESULT.CONSADDED
    assert len(handler.cuts)==1
    assert model.getNConss()==1
    handler.groups=lambda *args:[]
    assert handler.consenfops([],0,False,False)['result']==SCIP_RESULT.FEASIBLE


def test_real_solve_with_pseudo_nodes_matches_exhaustive_optimum(tmp_path,monkeypatch):
    from pyscipopt import Model
    original=SubtourHandler.consenfops
    calls=[]
    def counted(self,*args):
        calls.append(1)
        return original(self,*args)
    def factory(*args):
        model=Model(*args)
        model.setIntParam('lp/solvefreq',-1)
        return model
    monkeypatch.setattr(solver,'Model',factory)
    monkeypatch.setattr(SubtourHandler,'consenfops',counted)
    xy=np.random.default_rng(12).random((8,2))
    d=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    result=solver.solve(d,20,.00001,tmp_path/'native.json')
    exact=min(math.fsum(d[t[i],t[(i+1)%8]] for i in range(8))
              for t in ((0,)+p for p in itertools.permutations(range(1,8))))
    tour=result['solution']
    assert tour is not None and sorted(tour)==list(range(8))
    assert sum(d[tour[i],tour[(i+1)%8]] for i in range(8))==pytest.approx(exact,abs=1e-10)
    assert calls, 'regression must exercise the real SCIP pseudo callback'
