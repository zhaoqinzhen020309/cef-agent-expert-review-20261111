"""Formal 64+64 binding tests, beyond individual prepared-file checks.

Synthetic reference records below are plumbing fixtures, never performance data.
"""
import copy
import json
from pathlib import Path
import pytest
from cef.data import audit_supplement,audit_training,bind_supplement,receipt,SUPPLEMENT_PROTOCOL
from test_cef_supplement_binding import supplement_fixture,ROOT

SCIP_KIND='SCIP_numerically_bounded_unrounded_tour'

def scip_fixture(folder):
    path=supplement_fixture(folder);manifest=json.loads(path.read_text())
    for row in manifest['train']:
        record=Path(row['reference_record']['path']);value=json.loads(record.read_text())
        value['reference_kind']=SCIP_KIND;value['certificate']='synthetic audit fixture only; not a measured SCIP optimum'
        record.write_text(json.dumps(value),encoding='utf-8')
        row.update(reference_kind=SCIP_KIND,reference_record=receipt(record))
    path.write_text(json.dumps(manifest),encoding='utf-8')
    return path,manifest

def test_scip_references_pass_full128_formal_binding(tmp_path):
    path,manifest=scip_fixture(tmp_path)
    base=json.loads((ROOT/'data/cef_v1_3_20261026/tsp_manifest.json').read_text())
    result=bind_supplement(base,path)
    assert result['audit']['valid'] and result['audit']['instances']==128
    assert result['train'][:64]==base['train']


def test_scip_formal_binding_reaches_the_real_alternating_config_validator(tmp_path):
    from scripts.cef_alternating_launcher import build_auto_config
    from runCEFAuto import CONFIG
    path,_=scip_fixture(tmp_path)
    config,_=build_auto_config(dict(CONFIG,initial_supplement=path),tmp_path/'run',2025)
    assert len(config['data_manifest']['train'])==128
    assert all(r['reference_kind']==SCIP_KIND for r in config['data_manifest']['train'][64:])

def test_scip_row_and_reference_kind_must_agree(tmp_path):
    _,manifest=scip_fixture(tmp_path);row=manifest['train'][0]
    path=Path(row['reference_record']['path']);value=json.loads(path.read_text())
    value['reference_kind']='Concorde_numerically_bounded_unrounded_tour'
    path.write_text(json.dumps(value),encoding='utf-8');row['reference_record']=receipt(path)
    with pytest.raises(ValueError,match='kind mismatch'):audit_supplement(manifest['train'])

def test_scip_does_not_bypass_duplicate_instance_rule(tmp_path):
    _,manifest=scip_fixture(tmp_path);rows=manifest['train']
    rows[1]=dict(rows[0],id=rows[1]['id'],generator_seed=rows[1]['generator_seed'])
    original=json.loads((ROOT/'data/cef_v1_3_20261026/tsp_manifest.json').read_text())['train']
    with pytest.raises(ValueError,match='duplicate original/supplement instance'):
        audit_training('tsp',original+rows,SUPPLEMENT_PROTOCOL)

@pytest.mark.parametrize('problem',['weak_bound','wrong_objective','provenance'])
def test_scip_still_rejects_invalid_reference_evidence(tmp_path,problem):
    _,manifest=scip_fixture(tmp_path);row=manifest['train'][0]
    path=Path(row['reference_record']['path']);value=json.loads(path.read_text())
    if problem=='weak_bound':value['lower_bound']=value['objective']/2;value['bound_gap_pct']=100.
    if problem=='wrong_objective':value['objective']+=1.
    path.write_text(json.dumps(value)+'\n',encoding='utf-8')
    if problem!='provenance':row['reference_record']=receipt(path)
    messages={'weak_bound':'bound too weak','wrong_objective':'tour/bound mismatch','provenance':'provenance changed'}
    with pytest.raises(ValueError,match=messages[problem]):audit_supplement(manifest['train'])
