"""Bounded no-IO analysis over training events only, separately charged."""
import json,os,subprocess,sys,time
from pathlib import Path

def analyze(source,events,seconds):
    start=time.monotonic()
    env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('KEY','TOKEN','SECRET','PASSWORD','PROXY'))}
    # Same restricted evaluator, synthetic adapter only returns the analysis JSON.
    from .runner import Runner
    class AnalysisAdapter:
        def solver_input(self,x): return x
        def score(self,instance,solution): return 0.,0.
    wrapped=source+'\n\ndef solve(instance, time_limit_s, seed):\n    return {"solution": analyze(instance["events"])}\n'
    r=Runner(seconds,512).case(wrapped,AnalysisAdapter(),{'events':events},0,total_timeout_s=max(.001,seconds-.05))
    return dict(result=r.get('solution'),error=r['error'],elapsed=time.monotonic()-start,memory_peak_bytes=r.get('memory_peak_bytes'),memory_measurement='Windows Job peak process memory; hard cap 512MiB')
