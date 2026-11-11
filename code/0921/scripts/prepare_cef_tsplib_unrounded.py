"""Preserve historical unrounded TSPLIB artifacts; never change new-run defaults."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cef.config import load_json
from cef.data import (prepare_tsplib_unrounded,UNROUNDED_TSPLIB_METRIC as TSPLIB_METRIC,
                      UNROUNDED_DISTANCE_RULE as TSPLIB_DISTANCE_RULE,UNROUNDED_REFERENCE_RULE as TSPLIB_REFERENCE_RULE)


def main():
    source=load_json(ROOT/'data/cef_v1_3_20261026/tsp_manifest.json')
    destination=ROOT/'data'/TSPLIB_METRIC
    rows=prepare_tsplib_unrounded(ROOT,source['test'],destination/'instances')
    source.update(test=rows,evaluation_metric=TSPLIB_METRIC,
                  distance_rule='continuous Euclidean author TSP100; TSPLIB '+TSPLIB_DISTANCE_RULE,
                  reference_rule=TSPLIB_REFERENCE_RULE)
    manifest=destination/'tsp_manifest.json'
    text=json.dumps(source,ensure_ascii=False,indent=2)+'\n'
    if manifest.exists() and load_json(manifest)!=source:
        raise ValueError('existing unrounded manifest changed; do not overwrite')
    if not manifest.exists(): manifest.write_text(text,encoding='utf-8',newline='\n')
    print(f'Historical TSPLIB artifacts verified | metric={TSPLIB_METRIC} | current original-distance defaults unchanged')


if __name__=='__main__': main()
