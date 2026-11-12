"""Controller owns timing, scoring, fresh-state isolation and process lifetime."""
import copy,json,math,os,subprocess,sys,tempfile,time
from pathlib import Path
from .patch import check_source

DEADLINE_TOLERANCE_S=1e-6

def deadline_exceeded(elapsed,limit):
    """Ignore floating subtraction noise, never millisecond-scale overtime."""
    return elapsed>limit+DEADLINE_TOLERANCE_S

class Runner:
    def __init__(self,time_limit_s=60,memory_mb=1024):
        self.limit,self.memory_mb=time_limit_s,memory_mb
    def case(self,source,adapter,instance,seed,on_started=None,total_timeout_s=None,*,research_request=None):
        start=time.monotonic(); p=None; job=None
        limit=self.limit
        record=dict(valid=False,objective=None,fitness=None,error=None,error_location=[],elapsed=0.,overhead_s=0.,seed=seed,started=False)
        try:
            check_source(source,require_research=research_request is not None)
            request=copy.deepcopy(research_request)
            if request is not None:
                if not isinstance(request,dict) or request.get('kind') not in ('trace','common_start'):
                    raise ValueError('invalid research request kind')
                record.update(research_kind=request['kind'],trace=[],initial_solution_verified=False,cpu_seconds=None)
                local=request.get('local_limit_s',self.limit)
                if isinstance(local,bool) or not isinstance(local,(int,float)) or not math.isfinite(local) or not 0<local<=self.limit:
                    raise ValueError('invalid research local_limit_s')
                if request['kind']=='trace' and local!=self.limit:
                    raise ValueError('training trace must use the formal solve limit')
                limit=local
                if request['kind']=='common_start':
                    if 'initial_solution' not in request: raise ValueError('initial solution required')
                    adapter.objective(instance,request['initial_solution'])
                    request['initial_solution']=[int(x) for x in request['initial_solution']]
                policy=request.get('trace_policy',{})
                if not isinstance(policy,dict): raise ValueError('invalid trace_policy')
                maximum=policy.get('max_snapshots',6)
                fractions=policy.get('fractions',[0.1,0.25,0.5,0.75])
                if isinstance(maximum,bool) or not isinstance(maximum,int) or not 2<=maximum<=6:
                    raise ValueError('trace max_snapshots must be 2..6')
                if (not isinstance(fractions,(list,tuple)) or len(fractions)>4 or
                    any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<x<1 for x in fractions) or
                    list(fractions)!=sorted(set(fractions))):
                    raise ValueError('trace fractions must be increasing finite fractions')
            env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('KEY','TOKEN','SECRET','PASSWORD','PROXY'))}
            env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED=str(seed),PYTHONDONTWRITEBYTECODE='1')
            with tempfile.TemporaryDirectory(prefix='cef_case_') as cwd:
                p=subprocess.Popen([sys.executable,'-I','-u',str(Path(__file__).with_name('worker.py'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',cwd=cwd,env=env)
                record['started']=True
                if on_started: on_started()
                if sys.platform=='win32':
                    from agent_ahd.core.windows_job import WindowsJob
                    job=WindowsJob.create(); job.assign_pid(p.pid)
                    # Set a hard per-process memory limit in addition to kill-on-close.
                    import ctypes
                    from ctypes import wintypes
                    from agent_ahd.core import windows_job as wj
                    info=wj._JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                    info.BasicLimitInformation.LimitFlags=0x2000|0x100
                    info.ProcessMemoryLimit=int(self.memory_mb*1024*1024)
                    if not wj._kernel32.SetInformationJobObject(job._handle,9,ctypes.byref(info),ctypes.sizeof(info)):
                        raise RuntimeError('worker memory cap failed')
                payload=dict(source=source,instance=adapter.solver_input(instance),seed=seed,limit=limit)
                if request is not None: payload['research_request']=request
                # communicate sends request and GO together. Worker initializes all trusted
                # imports before READY; parent excludes that overhead from solve timer.
                p.stdin.write(json.dumps(payload)+'\n'); p.stdin.flush()
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=1) as pool:
                    ready=pool.submit(p.stdout.readline)
                    try: line=ready.result(timeout=min(20,total_timeout_s-(time.monotonic()-start)) if total_timeout_s else 20)
                    except Exception:
                        p.kill(); p.wait(); raise TimeoutError('worker startup timeout')
                if line.strip()!='READY': raise RuntimeError('worker failed to initialize')
                record['overhead_s']=time.monotonic()-start
                t=time.monotonic()
                remaining_total=total_timeout_s-(time.monotonic()-start) if total_timeout_s else limit+0.05
                if remaining_total<=0: raise subprocess.TimeoutExpired('analysis',total_timeout_s)
                try:
                    stdout,stderr=p.communicate('GO\n',timeout=min(limit+0.05,remaining_total))
                except subprocess.TimeoutExpired:
                    # Release the worker's current directory before TemporaryDirectory
                    # cleans it on Windows; the Job is still closed in finally.
                    p.kill(); p.wait()
                    raise
                record['elapsed']=time.monotonic()-t
                if len(stdout)>2_000_000: raise ValueError('oversized worker result')
                result=json.loads(stdout)
                record['elapsed']=result['elapsed']
                record['cpu_seconds']=result.get('cpu_seconds')
                record['error_location']=result.get('error_location',[])
                if request is not None:
                    trace=result.get('trace')
                    if not isinstance(trace,list) or len(trace)>policy.get('max_snapshots',6):
                        raise ValueError('invalid worker research trace')
                    verified=[]; previous=-1.
                    for snapshot in trace:
                        elapsed=snapshot['elapsed_s']
                        if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<previous or elapsed<0 or elapsed>result['elapsed']:
                            raise ValueError('invalid research trace timestamp')
                        objective=adapter.objective(instance,snapshot['solution'])
                        verified.append(dict(solution=list(snapshot['solution']),elapsed_s=elapsed,objective=objective))
                        previous=elapsed
                    record['trace']=verified
                    if request['kind']=='common_start':
                        record['initial_solution_verified']=bool(verified and verified[0]['solution']==request['initial_solution'] and result.get('initial_solution_verified'))
                        if not result['error'] and not record['initial_solution_verified']:
                            raise ValueError('initial solution was not verified')
                    if verified:
                        baseline=adapter.objective(instance,request['initial_solution']) if request['kind']=='common_start' else verified[0]['objective']
                        # Sparse retained observations do not identify the actual
                        # first improvement between callback sampling positions.
                        record['first_retained_improvement_s']=next((row['elapsed_s'] for row in verified if row['objective']<baseline),None)
                if result['error']: raise ValueError(result['error'])
                record['deadline_tolerance_s']=DEADLINE_TOLERANCE_S
                if deadline_exceeded(result['elapsed'],limit): raise TimeoutError('candidate late return')
                answer=result['answer']
                if not isinstance(answer,dict) or 'solution' not in answer: raise ValueError('missing solution')
                objective,fitness=adapter.score(instance,answer['solution'])
                record.update(valid=True,objective=objective,fitness=fitness,solution=answer['solution'])
        except subprocess.TimeoutExpired:
            record.update(error='TimeoutError: external solve deadline',elapsed=limit)
        except Exception as e:
            record['error']=type(e).__name__+': '+str(e)
        finally:
            if job:
                try:
                    job.terminate_and_wait(timeout_sec=2)
                    import ctypes
                    from agent_ahd.core import windows_job as wj
                    info=wj._JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                    if wj._kernel32.QueryInformationJobObject(job._handle,9,ctypes.byref(info),ctypes.sizeof(info),None):
                        record['memory_peak_bytes']=int(info.PeakProcessMemoryUsed)
                finally: job.close()
            if p and p.poll() is None: p.kill(); p.wait()
            if p:
                for pipe in (p.stdin,p.stdout,p.stderr):
                    if pipe and not pipe.closed: pipe.close()
            record['wallclock_s']=time.monotonic()-start
            record['isolation']='fresh process, sanitized env, AST/import/builtin restrictions, audit deny filesystem/network/process; Windows Job memory+tree cap. Not a hostile-code OS sandbox.'
        return record
