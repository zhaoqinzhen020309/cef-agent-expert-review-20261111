"""Executable rule provenance and deterministic, host-owned mixture dispatch."""
import math
import ast
import random
import statistics

from .patch import digest, check_source
from .store import canonical


def normalize_rules(proposal):
    raw = proposal.get('rules')
    if not isinstance(raw, list) or not raw:
        raise ValueError('rules must be a nonempty list of executable generators')
    rules = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError('each rule must be an object')
        name, source = item.get('name'), item.get('source')
        parameters, weight = item.get('parameters'), item.get('weight')
        if not isinstance(name, str) or not name.strip() or len(name)>120:
            raise ValueError('rule name must be a short readable string')
        if not isinstance(source, str) or not isinstance(parameters, dict):
            raise ValueError('rule source and parameters are required')
        if isinstance(weight, bool) or not isinstance(weight, (int,float)) or not math.isfinite(weight) or weight<=0:
            raise ValueError('rule weight must be positive and finite')
        tree=ast.parse(source)
        entries=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate']
        if len(entries)!=1 or [a.arg for a in entries[0].args.args]!=['n_cities','seed','parameters']:
            raise ValueError('rule requires generate(n_cities, seed, parameters)')
        check_source(source+'\ndef solve(instance, time_limit_s, seed):\n    return {"solution": []}\n')
        version = digest(canonical(dict(source=source, parameters=parameters)))
        if any(r['version']==version for r in rules):
            raise ValueError('duplicate executable rule version')
        rules.append(dict(name=name, source=source, parameters=parameters, weight=float(weight), version=version))
    return rules


def dispatch(rules, seeds):
    plan = []
    for index, seed in enumerate(seeds):
        # Separate streams: selecting a rule never consumes its coordinate RNG.
        choice_seed = int(digest(f'cef_rule_dispatch_v1:{seed}'),16)
        rule = random.Random(choice_seed).choices(rules,weights=[r['weight'] for r in rules],k=1)[0]
        plan.append(dict(id=f'supplement_{index:04d}', index=index, instance_seed=seed,
                         rule_version=rule['version'], rule_name=rule['name']))
    return plan


def summarize_rules(rows):
    groups = {}
    for row in rows:
        version = row['rule_version']
        group = groups.setdefault(version,dict(rule_version=version,rule_name=row['rule_name'],rows=[]))
        group['rows'].append(row)
    result=[]
    for group in groups.values():
        values=[r['reference_gap_pct'] for r in group['rows'] if r['valid']]
        result.append(dict(rule_version=group['rule_version'],rule_name=group['rule_name'],
            count=len(group['rows']),valid_count=len(values),
            positive_count=sum(v>1e-8 for v in values),
            mean_reference_gap_pct=statistics.fmean(values) if values else None,
            max_reference_gap_pct=max(values) if values else None,
            mean_solver_seconds=statistics.fmean(r['solver_seconds'] for r in group['rows']) if all('solver_seconds' in r for r in group['rows']) else None,
            max_reference_bound_gap_pct=max((r.get('reference_bound_gap_pct',0.) for r in group['rows']),default=None)))
    return result


def prefer_new(old_mean, new_mean):
    return new_mean is not None and new_mean>old_mean+1e-9
