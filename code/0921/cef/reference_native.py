"""Certified unrounded references through one TSP-specific native solver.

Concorde solves high-precision integer costs. The independent floating scorer
uses a conservative edge-representation error bound; integer scores are never
substituted for the original distances. Incomplete native proofs are not scores.
"""
from concurrent.futures import CancelledError
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np

from .data import receipt
from .reference import ReferenceIncomplete
from .store import canonical

ROOT=Path(__file__).resolve().parents[1]
VENDOR=Path(__file__).with_name('vendor')/'concorde_linux64'
WSL_DISTRIBUTION='Ubuntu-24.04'


def linux_path(path):
    path=Path(path).resolve()
    return '/mnt/'+path.drive[0].lower()+path.as_posix()[2:]


def decode_native_output(output):
    """WSL launch errors are Windows UTF-16; Linux output is UTF-8."""
    encoding='utf-16' if output.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-16-le' if b'\x00' in output else 'utf-8'
    return output.decode(encoding,errors='replace').strip()


def ensure_backend_ready():
    backend_manifest()
    command=['wsl.exe','-d',WSL_DISTRIBUTION,'--exec',linux_path(VENDOR/'concorde'),'-h']
    try:
        result=subprocess.run(command,capture_output=True,timeout=30)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('TSP backend startup timed out after 30s in '+WSL_DISTRIBUTION) from error
    except OSError as error:
        raise RuntimeError('Cannot launch WSL for TSP backend in '+WSL_DISTRIBUTION+': '+str(error)) from error
    if b'Usage:' not in result.stderr+result.stdout:
        details='\n'.join(part for part in (decode_native_output(result.stdout),decode_native_output(result.stderr)) if part)
        raise RuntimeError('TSP backend startup failed in '+WSL_DISTRIBUTION+
                           f' (exit_code={result.returncode})\n'+(details or 'WSL returned no diagnostic output'))


def cancel_native(process,request,work):
    Path(work/request['cancel_name']).touch()
    try:
        # No worker PID means the shared cancellation flag is sufficient: a late
        # worker checks it before exec. Do not boot WSL just to cancel a failed boot.
        pid_file=Path(work)/Path(request['pid_path']).name
        if pid_file.exists():
            result=subprocess.run(['wsl.exe','-d',WSL_DISTRIBUTION,'--exec','python3',
                linux_path(Path(__file__).with_name('reference_native_worker.py'))],
                input=canonical(dict(request,action='cancel')).encode('utf-8'),capture_output=True,timeout=15)
            if result.returncode:
                details='\n'.join(filter(None,(decode_native_output(result.stdout),decode_native_output(result.stderr))))
                raise RuntimeError('native cancellation failed: '+details[-2000:])
    finally:
        if process.poll() is None:process.kill()
        process.wait()


def backend_manifest():
    source=json.loads((VENDOR/'SOURCE.json').read_text(encoding='utf-8'))
    files=[]
    for row in source['files']:
        path=VENDOR/row['path'];r=receipt(path)
        if r['sha256']!=row['sha256']:raise ValueError('native reference dependency changed: '+str(path))
        files.append(r)
    files.extend([receipt(VENDOR/'SOURCE.json'),receipt(Path(__file__)),receipt(Path(__file__).with_name('reference_native_worker.py'))])
    return dict(backend='Concorde Linux 64-bit',architecture='linux-x86_64',gain_bits=64,proof_bits=128,source=source['concorde_source'],
        version=source['label'],files=files,distance_rule='original unrounded matrix',
        precision='integer optimum minus verified representation-error bound')


def prepare_problem(matrix,scale=1e8):
    d=np.asarray(matrix,dtype=float)
    if (d.ndim!=2 or d.shape[0]!=d.shape[1] or len(d)<3 or not np.isfinite(d).all()
            or (d<0).any() or not np.allclose(d,d.T,atol=1e-12,rtol=0) or np.any(np.diag(d)!=0)):
        raise ValueError('finite symmetric nonnegative distance matrix required')
    maximum=float(d.max());scale=min(float(scale),(2**29-1)/maximum) if maximum else float(scale)
    if not math.isfinite(scale) or scale<=0:raise ValueError('positive finite representation scale required')
    costs=np.rint((d*.5+d.T*.5)*scale).astype(np.int64)
    # The actual representation errors, rounded outward, cover every tour.
    error=float(np.max(np.abs(d-costs/scale)))
    error=math.nextafter(error+8*np.finfo(float).eps*maximum,math.inf)
    return dict(matrix=d.tolist(),integer_matrix=costs.tolist(),scale=scale,edge_error_bound=error,
        matrix_hash=hashlib.sha256(np.asarray(d,dtype='<f8').tobytes()).hexdigest())


def certify(matrix,record,log,gap_pct):
    if record.get('returncode')!=0:raise ValueError('native reference did not finish normally')
    markers=re.findall(r'Optimal Solution:\s*([0-9]+(?:\.[0-9]+)?)',log)
    if len(markers)!=1:raise ValueError('native optimality proof is missing or ambiguous')
    p=prepare_problem(matrix,record['scale']);d=np.asarray(matrix,dtype=float);n=len(d)
    tour=record['solution']
    if (len(tour)!=n or any(type(i)is not int for i in tour) or sorted(tour)!=list(range(n))):
        raise ValueError('native reference tour is not a permutation')
    optimum=float(markers[0]);costs=p['integer_matrix']
    integer_length=sum(costs[tour[i]][tour[(i+1)%n]] for i in range(n))
    if not optimum.is_integer() or int(optimum)!=integer_length:
        raise ValueError('native proved objective differs from recomputed integer tour')
    objective=math.fsum(float(d[tour[i],tour[(i+1)%n]]) for i in range(n))
    guard=float(32*n*np.finfo(float).eps*float(d.max()))
    lower=max(0.,optimum/p['scale']-n*p['edge_error_bound']-guard)
    if lower>objective+guard:raise ValueError('native bound exceeds original feasible tour')
    gap=float(0. if objective==0 else 100*(objective/lower-1) if lower>0 else math.inf)
    return dict(solution=tour,objective=objective,lower_bound=min(lower,objective),bound_gap_pct=gap,
        reference_kind='Concorde_numerically_bounded_unrounded_tour',backend='Concorde Linux 64-bit',
        distance_rule='original unrounded matrix',integer_optimum=int(optimum),scale=p['scale'],
        edge_error_bound=p['edge_error_bound'],matrix_hash=p['matrix_hash'],
        accepted=gap<=gap_pct+1e-8,certificate='native integer optimality proof and outward floating representation-error bound; numerical certificate')


def solve_tsp_reference(matrix,seconds=None,gap_pct=.00001,checkpoint_path=None,console=True):
    from .parallel import cancellation_requested
    if (not math.isfinite(gap_pct) or gap_pct<=0 or
            seconds is not None and (not math.isfinite(seconds) or seconds<=0)):
        raise ValueError('positive reference time and precision required')
    initial=prepare_problem(matrix);d=np.asarray(matrix,dtype=float);started=time.monotonic()
    if not d.any():
        return dict(solution=list(range(len(d))),objective=0.,lower_bound=0.,bound_gap_pct=0.,
            backend='nonnegative zero-cost identity',reference_kind='analytical_zero_reference',seconds=0.)
    if len(d)<10:
        # Concorde CLI itself dispatches <10 nodes to a bounded-int Held-Karp
        # routine. Exhaustive floating tours avoid that routine's edge limit;
        # formal TSP100 instances always go through native branch-and-cut.
        n=len(d)
        def value(t):return math.fsum(float(d[t[i],t[(i+1)%n]]) for i in range(n))
        tour=min(((0,)+p for p in itertools.permutations(range(1,n))),key=value)
        objective=value(tour)
        return dict(solution=list(tour),objective=objective,lower_bound=objective,bound_gap_pct=0.,
            backend='exhaustive floating tours (<10 cities)',reference_kind='enumerated_small_reference',
            seconds=time.monotonic()-started)
    if checkpoint_path is None:
        with tempfile.TemporaryDirectory(prefix='cef_native_') as td:
            return solve_tsp_reference(d,seconds,gap_pct,Path(td)/'work.json',console)
    checkpoint=Path(checkpoint_path).resolve();checkpoint.parent.mkdir(parents=True,exist_ok=True)
    backend=backend_manifest();backend_hash=hashlib.sha256(canonical(backend).encode()).hexdigest()
    # Short native paths avoid MAX_PATH and ANSI path limitations in the library.
    jobid=hashlib.sha256(str(checkpoint).encode()).hexdigest()[:24]
    base=ROOT/'experiments/reference_work_native'/jobid;base.mkdir(parents=True,exist_ok=True)
    scale=initial['scale'];attempts=[];native_seconds=0.
    while True:
        p=prepare_problem(d,scale);key=hashlib.sha256(canonical(dict(matrix=p['matrix_hash'],scale=p['scale'])).encode()).hexdigest()[:16]
        work=base/key;work.mkdir(exist_ok=True)
        identity=dict(matrix_hash=p['matrix_hash'],scale=p['scale'],backend_hash=backend_hash,tolerance_pct=gap_pct)
        identity_path=work/'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text())!=identity:raise ValueError('native reference work identity changed')
        else:identity_path.write_text(canonical(identity)+'\n',encoding='utf-8',newline='\n')
        problem='NAME: case\nTYPE: TSP\nDIMENSION: '+str(len(d))+'\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n'
        problem+='\n'.join(' '.join(map(str,row)) for row in p['integer_matrix'])+'\nEOF\n'
        input_path=work/'case.tsp'
        if input_path.exists() and input_path.read_text(encoding='ascii')!=problem:raise ValueError('native reference input changed')
        if not input_path.exists():input_path.write_text(problem,encoding='ascii',newline='\n')
        finished=work/'completed.json'
        if finished.exists():
            done=json.loads(finished.read_text());raw=done['raw'];stdout=Path(done['stdout']['path'])
            if raw.get('scale')!=p['scale']:raise ValueError('cached native reference scale changed')
            if receipt(stdout)!=done['stdout']:raise ValueError('native proof log changed')
            result=certify(d,raw,stdout.read_text(encoding='utf-8',errors='replace'),gap_pct)
        else:
            index=len(list(work.glob('stdout_*.log')))
            stdout=work/f'stdout_{index:03d}.log';stderr=work/f'stderr_{index:03d}.log';output=work/f'output_{index:03d}.json'
            request=dict(binary=linux_path(VENDOR/'concorde'),pid_path=linux_path(work/f'pid_{index:03d}.json'),
                cancel_path=linux_path(work/f'cancel_{index:03d}.flag'),cancel_name=f'cancel_{index:03d}.flag')
            env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('KEY','TOKEN','SECRET','PASSWORD','PROXY'))}
            with stdout.open('w',encoding='utf-8') as out,stderr.open('w',encoding='utf-8') as err:
                native_started=time.monotonic()
                process=subprocess.Popen(['wsl.exe','-d',WSL_DISTRIBUTION,'--cd',linux_path(work),'--exec','python3',
                    linux_path(Path(__file__).with_name('reference_native_worker.py'))],
                    cwd=work,stdin=subprocess.PIPE,stdout=out,stderr=err,text=True,encoding='utf-8',env=env)
                try:
                    process.stdin.write(canonical(request));process.stdin.close()
                    while True:
                        if cancellation_requested():raise CancelledError('native reference cancelled')
                        elapsed=time.monotonic()-started
                        if seconds is not None and elapsed>=seconds:raise ReferenceIncomplete('native reference time limit; branch files retained')
                        try:process.wait(timeout=min(30,max(.01,seconds-elapsed)) if seconds is not None else 30);break
                        except subprocess.TimeoutExpired:
                            if console:print(f'Reference {checkpoint.stem} | Concorde | elapsed={time.monotonic()-started:.0f}s',flush=True)
                except BaseException as error:
                    try:cancel_native(process,request,work)
                    except Exception as cleanup:error.add_note('Native cleanup failed: '+str(cleanup))
                    raise
                finally:
                    (work/f'attempt_{index:03d}.json').write_text(canonical(dict(elapsed_s=time.monotonic()-native_started))+'\n',encoding='utf-8',newline='\n')
            if process.returncode:
                details='\n'.join(filter(None,(decode_native_output(stdout.read_bytes()),decode_native_output(stderr.read_bytes()))))
                error=RuntimeError(f'native reference worker failed (exit_code={process.returncode}): '+details[-4000:])
                try:cancel_native(process,request,work)
                except Exception as cleanup:error.add_note('Native cleanup failed: '+str(cleanup))
                raise error
            if not (work/'tour.sol').exists():raise ValueError('native reference produced no tour')
            tokens=list(map(int,(work/'tour.sol').read_text(encoding='ascii').split()))
            raw=dict(returncode=process.returncode,n=tokens[0],solution=tokens[1:],scale=p['scale'])
            output.write_text(canonical(raw),encoding='utf-8',newline='\n')
            result=certify(d,raw,stdout.read_text(encoding='utf-8',errors='replace'),gap_pct)
            finished.write_text(canonical(dict(raw=raw,stdout=receipt(stdout)))+'\n',encoding='utf-8',newline='\n')
        attempts.append(dict(scale=p['scale'],bound_gap_pct=result['bound_gap_pct'] if math.isfinite(result['bound_gap_pct']) else None,
            proof=receipt(finished)))
        native_seconds+=sum(json.loads(path.read_text())['elapsed_s'] for path in work.glob('attempt_*.json'))
        if result.pop('accepted'):
            result.update(seconds=native_seconds,elapsed_this_call_s=time.monotonic()-started,history=attempts,backend_version=backend['version'],
                reference_work=dict(native_work=str(work),proof=receipt(finished),backend_hash=backend_hash))
            return result
        # Numerical precision refinement, not a new Agent proposal or relaxed score.
        new_scale=min((2**29-1)/float(d.max()),p['scale']*max(2.,2*result['bound_gap_pct']/gap_pct))
        if new_scale<=p['scale']*(1+1e-12):raise ValueError('requested reference precision exceeds native representation range')
        scale=new_scale
