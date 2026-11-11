"""Windows x64 reference establishment, verification, and immutable reuse.

Original EoH costs remain frozen. New matrices get independently computed SCIP
references before algorithm evaluation. No WSL, remote service, or legacy DLL.
"""
from contextlib import contextmanager
from concurrent.futures import CancelledError
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np
from .data import receipt
from .reference import ReferenceIncomplete
from .reference_scip_solver import matrix_identity,write_json
from .store import canonical

ROOT=Path(__file__).resolve().parents[1]
BACKEND='SCIP Windows 64-bit'

@lru_cache(maxsize=1)
def _backend_manifest():
    if sys.platform!='win32' or struct.calcsize('P')!=8:raise RuntimeError('64-bit Windows Python required for this native backend')
    import pyscipopt
    from pyscipopt import Model
    package=Path(pyscipopt.__file__).parent;model=Model()
    version=f'{model.getMajorVersion()}.{model.getMinorVersion()}.{model.getTechVersion()}'
    paths=[Path(__file__),Path(__file__).with_name('reference_scip_solver.py'),Path(__file__).with_name('reference_windows_worker.py')]
    paths+=list(package.glob('*.pyd'))+list((package.parent/'pyscipopt.libs').glob('*.dll'))
    return dict(backend=BACKEND,architecture='windows-x86_64',version=f'SCIP {version}; PySCIPOpt {pyscipopt.__version__}',
                source='https://github.com/scipopt/PySCIPOpt/blob/master/tests/test_tsp.py',
                formulation='binary undirected edges, degree 2, lazy DFJ subtour constraints',
                distance_rule='original unrounded matrix',precision='numerical primal/dual bounds; not exact arithmetic',
                files=[receipt(p) for p in paths])

def backend_manifest():return copy.deepcopy(_backend_manifest())

def ensure_backend_ready():
    backend_manifest()
    d=np.ones((6,6));d[:3,3:]=10;d[3:,:3]=10;np.fill_diagonal(d,0)
    result=solve_tsp_reference(d,seconds=20,console=False)
    if result['objective']!=24:raise RuntimeError('native reference startup failed the disconnected-subtour test')

def check_matrix(matrix):
    d=np.asarray(matrix,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or len(d)<3 or not np.isfinite(d).all() or (d<0).any() or np.any(np.diag(d)!=0) or not np.allclose(d,d.T,atol=1e-12,rtol=0):
        raise ValueError('finite symmetric nonnegative distance matrix with zero diagonal required')
    return d

def certify(matrix,raw,gap_pct):
    d=check_matrix(matrix);n=len(d)
    if raw.get('matrix_hash')!=matrix_identity(d) or raw.get('n')!=n:raise ValueError('SCIP result matrix identity differs')
    tour=raw.get('solution')
    if not isinstance(tour,list) or len(tour)!=n or any(type(i)is not int for i in tour) or sorted(tour)!=list(range(n)):
        raise ReferenceIncomplete('SCIP has not established a Hamiltonian incumbent; cuts retained')
    scale=float(d.max());normalized=(d*.5+d.T*.5)/scale
    represented=math.fsum(float(normalized[tour[i],tour[(i+1)%n]]) for i in range(n))
    stated=raw.get('model_objective');lower=raw.get('normalized_lower_bound')
    if not isinstance(stated,(int,float)) or not math.isfinite(stated) or abs(stated-represented)>1e-7*max(1.,represented):
        raise ValueError('SCIP objective differs from independently recomputed tour')
    if not isinstance(lower,(int,float)) or not math.isfinite(lower):raise ValueError('SCIP returned a nonfinite numerical bound')
    if lower>represented+1e-7*max(1.,represented):raise ValueError('SCIP numerical bound exceeds the feasible tour')
    objective=math.fsum(float(d[tour[i],tour[(i+1)%n]]) for i in range(n))
    asymmetry=float(np.max(np.abs(d-(d*.5+d.T*.5))))
    guard=1e-9*max(1.,abs(lower))*scale+n*asymmetry+64*n*np.finfo(float).eps*scale
    lower=max(0.,min(objective,lower*scale-guard))
    gap=0. if objective==0 else 100*(objective/lower-1) if lower>0 else math.inf
    if gap>gap_pct:raise ReferenceIncomplete(f'SCIP numerical reference gap {gap:.9g}% exceeds {gap_pct:.9g}%; cuts and incumbent retained')
    return dict(solution=tour,objective=objective,lower_bound=lower,bound_gap_pct=gap,
                reference_kind='SCIP_numerically_bounded_unrounded_tour',backend=BACKEND,
                distance_rule='original unrounded matrix',numerical_guard=guard,
                certificate='SCIP numerical MIP bound and independently recomputed unrounded Hamiltonian tour; not an exact-arithmetic proof',
                solver_status=raw['status'],subtour_cuts=raw['subtour_cuts'],nodes=raw['nodes'])

@contextmanager
def own_cache(path,deadline):
    import msvcrt
    from .parallel import cancellation_requested
    with Path(path).open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        while True:
            if cancellation_requested():raise CancelledError('reference cache wait cancelled')
            if deadline is not None and time.monotonic()>=deadline:raise ReferenceIncomplete('reference deadline waiting for matching cached work')
            stream.seek(0)
            try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(.1)
        try:yield
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)

def solve_tsp_reference(matrix,seconds=None,gap_pct=.00001,checkpoint_path=None,console=True):
    from .parallel import cancellation_requested
    d=check_matrix(matrix);start=time.monotonic()
    if not math.isfinite(gap_pct) or gap_pct<=0 or seconds is not None and (not math.isfinite(seconds) or seconds<=0):raise ValueError('positive reference precision and time required')
    deadline=None if seconds is None else start+seconds
    if not d.any():return dict(solution=list(range(len(d))),objective=0.,lower_bound=0.,bound_gap_pct=0.,backend='nonnegative zero-cost identity',reference_kind='analytical_zero_reference',seconds=0.,cache_hit=True)
    manifest=backend_manifest();backend_hash=hashlib.sha256(canonical(manifest).encode()).hexdigest()
    identity=dict(matrix_hash=matrix_identity(d),backend_hash=backend_hash,gap_pct=gap_pct)
    key=hashlib.sha256(canonical(identity).encode()).hexdigest()
    work=ROOT/'experiments/reference_work_windows'/backend_hash[:16]/key[:24];work.mkdir(parents=True,exist_ok=True)
    with own_cache(work/'cache.lock',deadline):
        identity_path=work/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('reference cache identity changed')
        if not identity_path.exists():write_json(identity_path,identity)
        completed=work/'completed.json';cache_hit=completed.is_file()
        if cache_hit:
            saved=json.loads(completed.read_text(encoding='utf-8'))
            if receipt(saved['output']['path'])!=saved['output']:raise ValueError('cached native reference evidence changed')
            raw=json.loads(Path(saved['output']['path']).read_text(encoding='utf-8'))
        else:
            index=len(list(work.glob('request_*.json')))
            request=work/f'request_{index:03d}.json';output=work/f'output_{index:03d}.json'
            remaining=None if deadline is None else deadline-time.monotonic()
            if remaining is not None and remaining<=0:raise ReferenceIncomplete('reference deadline before native worker launch')
            write_json(request,dict(matrix=d.tolist(),seconds=remaining,gap_pct=gap_pct,checkpoint=str(work/'state.json')))
            env={k:v for k,v in os.environ.items() if not any(word in k.upper() for word in ('KEY','TOKEN','SECRET','PASSWORD','PROXY'))}
            with (work/f'stdout_{index:03d}.log').open('wb') as out,(work/f'stderr_{index:03d}.log').open('wb') as err:
                process=subprocess.Popen([sys.executable,'-I','-u',str(Path(__file__).with_name('reference_windows_worker.py')),str(request),str(output)],
                                         stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=env)
                try:
                    while True:
                        if cancellation_requested():raise CancelledError('native Windows reference cancelled')
                        remaining=None if deadline is None else deadline-time.monotonic()
                        if remaining is not None and remaining<=0:raise ReferenceIncomplete('native Windows reference deadline; cuts retained')
                        try:process.wait(timeout=30 if remaining is None else min(30,max(.001,remaining)));break
                        except subprocess.TimeoutExpired:
                            if console:print(f'Reference {Path(checkpoint_path).stem if checkpoint_path else key[:8]} | SCIP Windows64 | elapsed={time.monotonic()-start:.0f}s',flush=True)
                finally:
                    if process.poll() is None:process.kill()
                    process.wait()
            if process.returncode or not output.is_file():
                error=(work/f'stderr_{index:03d}.log').read_text(encoding='utf-8',errors='replace')[-3000:]
                raise RuntimeError(f'native Windows SCIP worker failed (exit_code={process.returncode}): '+error)
            raw=json.loads(output.read_text(encoding='utf-8'))
            if raw.get('error'):raise RuntimeError('native Windows SCIP: '+raw['error'])
        result=certify(d,raw,gap_pct)
        if not cache_hit:write_json(completed,dict(identity=identity,output=receipt(output),solver_seconds=raw['seconds']))
        result.update(seconds=0. if cache_hit else raw['seconds'],proof_solver_seconds=raw['seconds'],cache_hit=cache_hit,
                      elapsed_this_call_s=time.monotonic()-start,backend_version=manifest['version'],
                      reference_work=dict(native_work=str(work),proof=receipt(completed),backend_hash=backend_hash))
        if checkpoint_path is not None:
            path=Path(checkpoint_path);path.parent.mkdir(parents=True,exist_ok=True)
            write_json(path,dict(identity=identity,proof=receipt(completed),backend=BACKEND))
        return result
