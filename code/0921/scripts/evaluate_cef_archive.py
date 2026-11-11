"""Post-search TSPLIB evaluation of every generated complete candidate. No LLM."""
import argparse,csv,hashlib,json,os,statistics,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cef.config import file_hash
from cef.data import audit_test
from cef.tasks import TaskAdapter
from cef.patch import check_source
from cef.parallel import completed_tasks
from cef.store import canonical


def evaluate(source,instance,seed,output):
    from agent_ahd.core.windows_job import WindowsJob
    from agent_ahd.core import windows_job as wj
    import ctypes
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('KEY','TOKEN','SECRET','PASSWORD','PROXY'))}
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED=str(seed),PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic();process=None;job=None
    record=dict(valid=False,error=None,objective=None,gap_pct=None,elapsed=None,requested_solver_limit_s=60,external_worker_timeout_s=600)
    try:
        process=subprocess.Popen([sys.executable,'-I','-u',str(ROOT/'cef/worker.py')],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',env=env,cwd=output)
        job=WindowsJob.create();job.assign_pid(process.pid)
        info=wj._JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags=0x2000|0x100;info.ProcessMemoryLimit=1024*1024*1024
        if not wj._kernel32.SetInformationJobObject(job._handle,9,ctypes.byref(info),ctypes.sizeof(info)):
            raise RuntimeError('worker memory cap failed')
        payload=dict(source=source,instance=TaskAdapter('tsp').solver_input(instance),limit=60,seed=seed)
        stdout,stderr=process.communicate(json.dumps(payload)+'\nGO\n',timeout=600)
        ready,separator,body=stdout.partition('\n')
        if ready.strip()!='READY' or not separator:raise RuntimeError('worker failed: '+stderr[-1000:])
        result=json.loads(body)
        record.update(elapsed=result['elapsed'],cpu_seconds=result.get('cpu_seconds'))
        if result.get('error'):raise RuntimeError(result['error'])
        objective=TaskAdapter('tsp').objective(instance,result['answer']['solution'])
        record.update(valid=True,objective=objective,solution=result['answer']['solution'],
                      gap_pct=100*(objective/instance['reference']-1),late_return=result['elapsed']>60.000001)
    except Exception as error:record['error']=type(error).__name__+': '+str(error)
    finally:
        if job:
            try:job.terminate_and_wait(timeout_sec=2)
            finally:job.close()
        if process:
            if process.poll() is None:process.kill();process.wait()
            for pipe in (process.stdin,process.stdout,process.stderr):
                if pipe and not pipe.closed:pipe.close()
    record['wallclock_s']=time.monotonic()-start
    return record


def summarize(output,candidates,records):
    summaries=[]
    for c in candidates:
        cases=[r for r in records if r['candidate_id']==c['candidate_id']]
        valid=len(cases)==29 and all(r['valid'] for r in cases)
        summaries.append(dict(round=c['round'],candidate_id=c['candidate_id'],parent_id=c.get('parent_id'),
            source_hash=c['source_hash'],count=len(cases),valid_count=sum(r['valid'] for r in cases),valid=valid,
            mean_gap_pct=statistics.fmean(r['gap_pct'] for r in cases) if valid else None,
            late_returns=sum(r.get('late_return',False) for r in cases),
            max_elapsed_s=max((r.get('elapsed') or 0 for r in cases),default=0)))
    temporary=output/'summary.tmp'
    temporary.write_text(json.dumps(summaries,indent=2),encoding='utf-8',newline='\n');temporary.replace(output/'summary.json')
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--output',type=Path);parser.add_argument('--resume',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    freeze=json.loads((run/'test/freeze.json').read_text(encoding='utf-8'))
    audit_test('tsp',freeze['benchmark'])
    candidates=[]
    with (run/'search/facts.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if '"kind": "candidate"' not in line:continue
            event=json.loads(line)
            if event['kind']=='candidate' and event['data'].get('variant')=='PAB':candidates.append(event['data'])
    if len(candidates)!=10 or len({c['source_hash'] for c in candidates})!=10:
        raise ValueError('expected ten distinct complete generated candidates')
    sources={}
    for number,c in enumerate(candidates,1):
        c['round']=number
        source=(run/'search/sources'/ (c['source_hash']+'.py')).read_text(encoding='utf-8')
        if hashlib.sha256(source.encode()).hexdigest()!=c['source_hash']:raise ValueError('candidate source changed')
        check_source(source);sources[c['candidate_id']]=source
    identity=dict(search_journal_sha256=file_hash(run/'search/facts.jsonl'),run=str(run),candidates=candidates,
        benchmark=freeze['benchmark'],seeds=freeze['seeds'],metric=freeze['evaluation_metric'],
        workers=args.workers,requested_solver_limit_s=60,external_worker_timeout_s=600,
        purpose='post-search all-candidate comparison; does not replace training-selected final solver',
        protocol='fresh full TSPLIB29 per candidate, unrounded distance/original references; late returns separately flagged',
        evaluator_sha256=file_hash(__file__),worker_sha256=file_hash(ROOT/'cef/worker.py'))
    output=args.resume or args.output or run/('all_candidates_tsplib29_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if args.resume:
        if json.loads((output/'freeze.json').read_text(encoding='utf-8'))!=identity:raise ValueError('resume evaluation identity differs')
    else:
        output.mkdir();(output/'freeze.json').write_text(canonical(identity),encoding='utf-8',newline='\n')
    records=[json.loads(s) for s in (output/'results.jsonl').read_text(encoding='utf-8').splitlines()] if (output/'results.jsonl').exists() else []
    done={(r['candidate_id'],r['instance_id']) for r in records}
    jobs=[(c,row) for c in candidates for row in freeze['benchmark'] if (c['candidate_id'],row['id']) not in done]
    print('Output:',output,flush=True);print('Candidates=10 | cases=290 | workers='+str(args.workers),flush=True)
    def work(item):
        c,row=item;instance=TaskAdapter('tsp').load(row);seed=freeze['seeds'][row['id']+':0']
        result=evaluate(sources[c['candidate_id']],instance,seed,output)
        result.update(round=c['round'],candidate_id=c['candidate_id'],source_hash=c['source_hash'],instance_id=row['id'],reference=row['reference'],seed=seed)
        return result
    start=time.monotonic()
    for item,result in completed_tasks(jobs,work,args.workers):
        records.append(result)
        with (output/'results.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(canonical(result)+'\n');stream.flush();os.fsync(stream.fileno())
        summaries=summarize(output,candidates,records)
        print(f'Progress {len(records)}/290 | round={result["round"]} | {result["instance_id"]} | valid={result["valid"]} | gap={result["gap_pct"]} | elapsed_total={time.monotonic()-start:.0f}s',flush=True)
        current=summaries[result['round']-1]
        if current['count']==29:print('CANDIDATE COMPLETE',json.dumps(current),flush=True)
    assert file_hash(run/'search/facts.jsonl')==identity['search_journal_sha256']
    print('COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
