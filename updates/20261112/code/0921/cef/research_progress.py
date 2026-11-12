"""Measured research views. These do not rank or promote formal candidates."""
import statistics


def paired_view(reference, candidate):
    left={r['id']:r for r in reference['cases']}
    right={r['id']:r for r in candidate['cases']}
    rows=[]
    for key in sorted(left.keys() & right.keys()):
        a,b=left[key],right[key]
        valid=bool(a.get('valid') and b.get('valid'))
        rows.append(dict(id=key,valid=valid,reference_objective=a.get('objective'),
            candidate_objective=b.get('objective'),
            candidate_minus_reference_objective=b['objective']-a['objective'] if valid else None,
            reference_gap_pct=a.get('reference_gap_pct'),candidate_gap_pct=b.get('reference_gap_pct'),
            reference_error=a.get('error'),candidate_error=b.get('error')))
    return dict(cases=rows,paired_cases=len(rows),valid_pairs=sum(r['valid'] for r in rows),
        candidate_better=sum(r['valid'] and r['candidate_minus_reference_objective'] < -1e-12 for r in rows),
        candidate_worse=sum(r['valid'] and r['candidate_minus_reference_objective'] > 1e-12 for r in rows),
        unit='objective difference; gap fields are percentages',
        scope='Observed complete-program comparison on requested cases, not component attribution or formal ranking')


def progress_view(archive):
    groups={}
    for r in archive.of('experimental_execution'):
        key=(r['source_hash'],r['id'],r['seed'],str(sorted(r['conditions'].items())) + ('' if r['conditions'].get('evaluation_hash') else ':'+r['experiment_id']))
        groups.setdefault(key,[]).append(r)
    candidates={}
    for (source,case,seed,conditions),records in groups.items():
        item=candidates.setdefault(source,dict(source_id='source:'+source,cases=[],executions=0))
        values=[r['objective'] for r in records if r.get('valid')]
        item['executions']+=len(records)
        item['cases'].append(dict(id=case,solver_seed=seed,observations=len(records),valid=len(values),
            objective_min=min(values) if values else None,objective_max=max(values) if values else None,
            objective_mean=statistics.fmean(values) if values else None))
    return dict(candidates=list(candidates.values()),
        questions=archive.of('research_update'),
        interpretation='Ranges describe observed repeats only, not confidence intervals. Conditions remain in raw evidence.')


def data_observations(cases,records):
    import numpy as np
    by_id={}
    for r in records:
        if r.get('valid'):by_id.setdefault(r['instance_id'],[]).append(r)
    result=[]
    for c in cases:
        if c.get('training_group')!='supplement':continue
        observations=by_id.get(c['id'],[])
        xy=c.get('coordinates')
        geometry=None
        if xy is not None:
            points=np.asarray(xy,dtype=float)
            d=np.linalg.norm(points[:,None]-points[None,:],axis=2)
            np.fill_diagonal(d,np.inf)
            nearest=d.min(axis=1)
            geometry=dict(span=np.ptp(points,axis=0).tolist(),nearest_distance_quartiles=np.quantile(nearest,[.25,.5,.75]).tolist())
        result.append(dict(instance_id=c['id'],coordinate_artifact='training_cases.json',coordinates_available=xy is not None,
            generator_seed=c.get('generator_seed'),solver_seed=c['seed'],geometry=geometry,
            measured_gap_pct=[r['fitness'] for r in observations],
            observation_ids=[r.get('observation_id',r['execution_id']) for r in observations]))
    return dict(cases=result,scope='Geometry describes instances; measured gaps describe solver difficulty. Geometry and reference tour length are not substitutes for measured gap.')
