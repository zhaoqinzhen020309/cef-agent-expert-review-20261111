"""Fresh restricted worker. Defense in depth, NOT a hostile-code OS sandbox."""
import ast,builtins,collections,contextlib,functools,heapq,itertools,json,math,random,sys,time,traceback
import numpy as np
from numbers import Integral

SAFE_MODULES={'math':math,'random':random,'time':time,'itertools':itertools,'functools':functools,'collections':collections,'heapq':heapq,'numpy':np}

def main():
    request=json.loads(sys.stdin.readline())
    sys.stdout.write('READY\n'); sys.stdout.flush()
    if sys.stdin.readline().strip()!='GO': return
    start=time.monotonic(); cpu_start=time.process_time(); error_location=[]
    research=request.get('research_request'); trace=[]; observation_error=None
    initial_verified=False; observations=0
    def capture(solution, final=False):
        nonlocal observation_error,initial_verified,observations
        try:
            matrix=request['instance'].get('distance_matrix',request['instance'].get('processing_times'))
            n=len(matrix)
            if (not isinstance(solution,(list,tuple,np.ndarray)) or len(solution)!=n or
                any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral) for x in solution) or
                sorted(solution)!=list(range(n))):
                raise ValueError('invalid research observation permutation')
            elapsed=time.monotonic()-start
            if not final:
                observations+=1
                if observations==1 and research['kind']=='common_start':
                    if list(solution)!=research['initial_solution']:
                        raise ValueError('initial solution observation does not match requested start')
                    initial_verified=True
            if final or not trace or (len(trace)<max_snapshots-1 and
                                      fractions and elapsed>=fractions[0]*request['limit']):
                trace.append(dict(solution=[int(x) for x in solution],elapsed_s=elapsed))
                while fractions and elapsed>=fractions[0]*request['limit']:
                    fractions.pop(0)
        except BaseException as e:
            # Sticky failure survives a candidate catching the callback exception.
            observation_error=type(e).__name__+': '+str(e)
            raise
    def observe(*args,**kwargs):
        nonlocal observation_error
        if len(args)!=1 or kwargs:
            observation_error='TypeError: observer(solution) requires exactly one positional argument'
            raise TypeError(observation_error)
        capture(args[0])
    try:
        policy=research.get('trace_policy',{}) if research else {}
        max_snapshots=policy.get('max_snapshots',6)
        fractions=list(policy.get('fractions',[0.1,0.25,0.5,0.75]))
        source=request['source']; tree=ast.parse(source)
        for n in ast.walk(tree):
            if isinstance(n,ast.Attribute) and n.attr.startswith('__'):
                error_location=[dict(filename='solver.py',lineno=n.lineno,function='<validation>',source_line=source.splitlines()[n.lineno-1])]
                raise ValueError('dunder introspection denied: '+n.attr)
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module]
                if any(name not in SAFE_MODULES for name in names): raise ValueError('import denied')
        random.seed(request['seed']); np.random.seed(request['seed'] % (2**32))
        def audit(event,args):
            if event=='open' or event.startswith(('socket.','subprocess.','os.','ctypes.','winreg.','shutil.')):
                raise PermissionError('candidate capability denied: '+event)
        sys.addaudithook(audit)
        def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
            # Array methods resolve loaded NumPy implementation modules through
            # this hook. Resolve only existing objects, without loading modules;
            # source import statements remain limited to SAFE_MODULES above.
            if not level and name.startswith('numpy.') and name in sys.modules:
                return sys.modules[name] if fromlist else np
            if name not in SAFE_MODULES or level: raise ImportError('import denied: '+str(name))
            return SAFE_MODULES[name]
        allowed={k:v for k,v in vars(builtins).items() if k not in ('open','eval','exec','compile','input','breakpoint','help','globals','locals','vars','getattr','setattr','delattr','__import__')}
        allowed['__import__']=safe_import
        scope={'__builtins__':allowed,'__name__':'cef_candidate'}
        exec(compile(tree,'solver.py','exec'),scope)
        remaining=request['limit']-(time.monotonic()-start)
        if remaining<=0: raise TimeoutError('import exhausted solve budget')
        if research:
            kwargs={'observer':observe}
            if research['kind']=='common_start':
                kwargs['initial_solution']=list(research['initial_solution'])
            answer=scope['solve'](request['instance'],remaining,request['seed'],**kwargs)
            if observation_error: raise ValueError(observation_error)
            if research['kind']=='common_start' and (not observations or not initial_verified):
                raise ValueError('initial solution was not observed')
            if not observations:
                raise ValueError('research trace requires a solver observation before final return')
            if not isinstance(answer,dict) or 'solution' not in answer: raise ValueError('missing solution')
            capture(answer['solution'],final=True)
        else:
            answer=scope['solve'](request['instance'],remaining,request['seed'])
        result={'answer':answer,'elapsed':time.monotonic()-start,'error':None}
    except BaseException as e:
        if not error_location:
            for frame,line in traceback.walk_tb(e.__traceback__):
                if frame.f_code.co_filename=='solver.py':
                    error_location.append(dict(filename='solver.py',lineno=line,function=frame.f_code.co_name,source_line=source.splitlines()[line-1] if 0<line<=len(source.splitlines()) else ''))
        result={'answer':None,'elapsed':time.monotonic()-start,'error':type(e).__name__+': '+str(e),'error_location':error_location}
    result['cpu_seconds']=time.process_time()-cpu_start
    if research:
        result.update(trace=trace,research_kind=research['kind'],initial_solution_verified=initial_verified)
    sys.stdout.write(json.dumps(result,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else int(x) if isinstance(x,np.integer) else float(x) if isinstance(x,np.floating) else None)+'\n'); sys.stdout.flush()

if __name__=='__main__': main()
