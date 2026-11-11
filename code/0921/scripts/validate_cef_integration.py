"""B-level validation: actual 64+64 task executions, no model/performance claim."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cef.config import load_json
from cef.engine import Engine
from cef.data import audit_training,audit_test
from cef.reporting import verify_export

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); args=p.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    summary=[]
    for task,name in [('tsp','data/cef_v1_3_20261026/tsp_manifest.json'),('fssp','data/cef_fssp_U01_v1_3_20261026/manifest.json')]:
        data=load_json(ROOT/name); audit_training(task,data['train']); audit_test(task,data['test'])
        c=dict(task=task,mode='CEF',run_dir=str((output/task).resolve()),data_manifest=data,evaluation_seed_manifest={r['id']:2024 for r in data['train']},max_case_runs=64,max_wallclock_s=600,max_llm_requests=1,max_llm_tokens=1,max_analysis_seconds=1,solver_time_limit_s=60,memory_mb=1024,environment_manifest=dict(validation='actual local Python/NumPy; formal manifest separately generated'),validation_level='B actual adapters/data/runner/store/ledger; no model loop')
        e=Engine(c,None); e.register('seed',e.adapter.seed_source()); e.ledger.reserve('initial',64)
        scores,records=e.evaluate({'P':'seed'},'initial','initial'); e.ledger.release('initial')
        e.export('finished','B integration only; no model or performance claim')
        _,result,_=verify_export(e.archive.root)
        if not all(r['valid'] for r in records['P']) or e.ledger.used('case_runs')!=64: raise AssertionError('full64 integration failed')
        summary.append(dict(task=task,instances=64,valid=64,source_hash=result['best']['source_hash'],fitness_unit='gap percent' if task=='tsp' else 'raw makespan',execution_count=e.ledger.used('case_runs'),journal=str(e.archive.path),model_calls=0))
    (output/'summary.json').write_text(json.dumps(dict(level='B',passed=True,tasks=summary,performance_claim=False),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
