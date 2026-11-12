"""Root-safe entry into the same CEF runtime used by run_task_search cef."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# Final testing must execute the runtime frozen with that search, including
# experiments already running while the working checkout is being improved.
bootstrap=argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--quiet',action='store_true')
bootstrap.add_argument('command',nargs='?')
bootstrap.add_argument('--run-dir')
bootstrap.add_argument('--test-config')
entry,_=bootstrap.parse_known_args()
if entry.command=='test' and entry.run_dir:
    # Enforce metric identity before dispatching to the frozen historical runtime.
    # Do not import current cef here: that would defeat snapshot module isolation.
    if entry.test_config:
        try:
            test_path=Path(entry.test_config)
            tests=json.loads(test_path.read_text(encoding='utf-8-sig'))['instances']
            if any(row.get('id')=='eil51' for row in tests):
                metric=tests[0].get('evaluation_metric')
                message='unlabelled or mixed TSPLIB evaluation metrics; historical records are preserved'
                if metric not in ('tsplib_original_distance_v1','tsplib_unrounded_original_reference_v1'):raise ValueError(message)
                if any(row.get('evaluation_metric')!=metric for row in tests): raise ValueError(message)
                search=Path(entry.run_dir)
                with (search/'facts.jsonl').open(encoding='utf-8') as facts:
                    frozen=next(json.loads(line)['data'] for line in facts if json.loads(line).get('kind')=='manifest')
                if any(row.get('evaluation_metric')!=metric for row in frozen['data_manifest']['test']): raise ValueError(message)
        except (ValueError,OSError,KeyError,IndexError,StopIteration) as error:
            print('Error: '+str(error),file=sys.stderr)
            raise SystemExit(1)
    snapshot=Path(entry.run_dir)/'runtime_snapshot'
    if (snapshot/'cef'/'cli.py').is_file(): sys.path.insert(0,str(snapshot))
from cef.cli import main
if __name__=='__main__': raise SystemExit(main())
