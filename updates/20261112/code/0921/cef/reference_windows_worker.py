"""One owned Windows x64 process for the trusted SCIP reference model."""
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_scip_solver import solve,write_json

def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:
        result=solve(request['matrix'],request['seconds'],request['gap_pct'],request['checkpoint'])
    except Exception as error:
        write_json(sys.argv[2],dict(error=type(error).__name__+': '+str(error),error_type=type(error).__name__))
        traceback.print_exc();return 1
    write_json(sys.argv[2],result);return 0

if __name__=='__main__':raise SystemExit(main())
