"""Merge compatible measured cases, independently of tool-call batch boundaries."""
from .store import canonical
import hashlib
import json
import statistics
from pathlib import Path


def accumulate(measurements, *, candidate_identity, plan, rules, calibration_source_hash, contract):
    selected={}
    expected_ids={p['id'] for p in plan}
    for measurement in measurements:
        if (measurement.get('candidate_identity')!=candidate_identity or
            measurement.get('calibration_source_hash')!=calibration_source_hash or
            canonical(measurement.get('plan'))!=canonical(plan) or
            canonical(measurement.get('rules'))!=canonical(rules)):
            continue
        rows={r['id']:r for r in measurement.get('prepared_rows',[])}
        executions={r['instance_id']:r for r in measurement.get('executions',[])}
        for case in measurement.get('cases',[]):
            key=case['id']
            if key in selected or key not in expected_ids or key not in rows or key not in executions:continue
            if not case.get('valid') or not executions[key].get('valid'):continue
            if canonical(case.get('observation_contract'))!=canonical(contract):continue
            # First valid matching observation is stable under retries; never choose the best repeat.
            selected[key]=(rows[key],executions[key],case)
    ordered=[selected[key] for key in sorted(selected)]
    return dict(complete=set(selected)==expected_ids,covered=len(selected),total=len(plan),
        missing_ids=sorted(expected_ids-set(selected)),rows=[x[0] for x in ordered],
        executions=[x[1] for x in ordered],cases=[x[2] for x in ordered])


def candidate_state(archive, calibration_source_hash=None, contract=None):
    """Authoritative measured candidate view; independent of Agent-written summaries."""
    measurements=[]
    for event in archive.of('data_candidate_measurement'):
        value=json.loads(Path(event['artifact_path']).read_text(encoding='utf-8'))
        if calibration_source_hash and value['calibration_source_hash']!=calibration_source_hash:continue
        measurements.append(value)
    candidates={}
    for value in measurements:
        if not contract:continue  # A best score requires explicitly comparable execution conditions.
        identity=value['candidate_identity']
        if identity in candidates:continue
        coverage=accumulate(measurements,candidate_identity=identity,plan=value['plan'],rules=value['rules'],
            calibration_source_hash=value['calibration_source_hash'],contract=contract)
        candidates[identity]=dict(candidate_identity=identity,calibration_source_hash=value['calibration_source_hash'],
            measured_case_count=coverage['covered'],formal_case_count=coverage['total'],complete=coverage['complete'],
            mean_reference_gap_pct=statistics.fmean(c['reference_gap_pct'] for c in coverage['cases']) if coverage['complete'] else None,
            rules=[dict(name=r['name'],parameters=r['parameters'],weight=r['weight'],
                source_hash=hashlib.sha256(r['source'].encode()).hexdigest()) for r in value['rules']],
            evidence_paths=[e['artifact_path'] for e in archive.of('data_candidate_measurement') if e.get('candidate_identity')==identity])
    complete=[c for c in candidates.values() if c['complete']]
    return dict(best_complete_candidate=max(complete,key=lambda c:c['mean_reference_gap_pct']) if complete else None,
        candidates=list(candidates.values()),
        meaning='Best means largest complete measured gap under the specified calibration conditions. Working source and hypotheses do not overwrite measured candidate identities.')
