"""Continuous checkpointed reference work, independent of all LLM decisions."""
import json
import hashlib
import math
import os
from pathlib import Path
import tempfile
import time
from concurrent.futures import CancelledError

import numpy as np

from .data import receipt
from .store import canonical
from .tasks import TaskAdapter


class ReferencePending(RuntimeError):
    """Work can continue from a checkpoint; no training score is available yet."""


def _snapshot(path):
    """Returned references cite immutable evidence, not the appendable work files."""
    path=Path(path);data=path.read_bytes();sha=hashlib.sha256(data).hexdigest()
    folder=Path(__file__).resolve().parents[1]/'experiments/reference_snapshots'
    folder.mkdir(parents=True,exist_ok=True)
    # Full hashes stay in receipts; short names avoid Windows MAX_PATH failures.
    frozen=folder/(sha+path.suffix)
    if frozen.exists():
        if frozen.read_bytes()!=data:raise ValueError('reference work snapshot changed')
    else:
        with tempfile.NamedTemporaryFile(dir=folder,delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        try:os.replace(temporary,frozen)
        finally:temporary.unlink(missing_ok=True)
    return receipt(frozen)


def _attempt_history(journal):
    if not journal.exists():return []
    raw=journal.read_bytes();lines=raw.splitlines(keepends=True);records=[];committed=[]
    for index,line in enumerate(lines):
        try:records.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):
            if index!=len(lines)-1:raise ValueError('reference attempt log has a corrupt committed record')
            _snapshot(journal)
            temp=journal.with_suffix(journal.suffix+'.tmp');temp.write_bytes(b''.join(committed))
            os.replace(temp,journal);break
        committed.append(line)
    if journal.stat().st_size and not journal.read_bytes().endswith(b'\n'):
        with journal.open('ab') as stream:stream.write(b'\n');stream.flush();os.fsync(stream.fileno())
    return records


def run_reference_slice(matrix, seconds, gap_pct, checkpoint):
    from .reference_windows import solve_tsp_reference
    from .reference import ReferenceIncomplete
    try:return dict(valid=True,result=solve_tsp_reference(matrix,seconds,gap_pct,checkpoint))
    except ReferenceIncomplete as error:return dict(valid=False,pending=True,error=str(error))


def validate_reference(matrix, result, gap_pct):
    matrix=np.asarray(matrix,dtype=float);tour=result['solution'];n=len(matrix)
    if len(tour)!=n or any(type(v)is not int for v in tour) or sorted(tour)!=list(range(n)):
        raise ValueError('reference tour is not a permutation')
    objective=TaskAdapter('tsp').objective(dict(distance_matrix=matrix.tolist()),tour)
    if not np.isclose(objective,result['objective'],rtol=0,atol=1e-9):
        raise ValueError('reference objective differs from independently recomputed tour')
    lower=result['lower_bound']
    if not math.isfinite(lower) or lower<0 or lower>objective+1e-7:
        raise ValueError('reference lower bound is invalid')
    gap=0. if objective==0 else 100*(objective/min(objective,lower)-1) if lower>0 else math.inf
    if gap>gap_pct+1e-8 or not math.isfinite(result['bound_gap_pct']) or abs(gap-result['bound_gap_pct'])>1e-7:
        raise ValueError('reference bound gap exceeds requested tolerance')
    return result


def independent_reference(matrix, seconds, gap_pct, checkpoint=None, max_slices=None):
    if max_slices is not None and (type(max_slices)is not int or max_slices<1):
        raise ValueError('positive reference work slices or None required')
    if checkpoint is None:
        with tempfile.TemporaryDirectory(prefix='cef_reference_') as td:
            result=independent_reference(matrix,seconds,gap_pct,Path(td)/'work.json',max_slices)
            result.pop('reference_work',None)
            return result
    matrix=np.asarray(matrix,dtype=float);checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    journal=checkpoint.with_suffix('.attempts.jsonl')
    _attempt_history(journal)
    # Default is one persistent native solve. Explicit finite slices remain a
    # diagnostic interface; their time cap never governs normal Agent search.
    budget=None if max_slices is None else seconds
    def record(value):
        with journal.open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    index=0
    while index<(max_slices or 1):
        from .parallel import cancellation_requested
        if cancellation_requested():raise CancelledError('reference work cancelled by experiment owner')
        started=time.monotonic();record(dict(kind='start',slice_seconds=budget,tolerance_pct=gap_pct,time=time.time()))
        try:output=run_reference_slice(matrix,budget,gap_pct,checkpoint)
        except Exception as error:
            record(dict(kind='error',error=type(error).__name__+': '+str(error),elapsed_s=time.monotonic()-started));raise
        record(dict(kind='finished' if output.get('valid') else 'pending' if output.get('pending') else 'error',
            elapsed_s=time.monotonic()-started,error=output.get('error')))
        if output.get('valid') is True:
            result=validate_reference(matrix,output['result'],gap_pct)
            result['reference_work']=dict(result.get('reference_work',{}),attempts=_snapshot(journal),
                backend=result.get('backend','SCIP Windows 64-bit'),time_limit_s=budget)
            if checkpoint.is_file():result['reference_work']['checkpoint']=_snapshot(checkpoint)
            return result
        if not output.get('pending'):raise RuntimeError(output.get('error','invalid reference result'))
        index+=1
    raise ReferencePending(f'reference pending after {max_slices} work slices; checkpoint retained: {checkpoint}')
