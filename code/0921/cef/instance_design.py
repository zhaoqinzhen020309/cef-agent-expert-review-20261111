"""Independent data-design episode; the full-solver search is a separate consumer."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics

import numpy as np

from .config import file_hash
from .data import receipt, SUPPLEMENT_PROTOCOL
from .parallel import completed_tasks
from .patch import digest
from .runner import Runner
from .reference_jobs import independent_reference, ReferencePending, validate_reference
from .store import Archive, canonical
from .tasks import TaskAdapter



def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _case_identity(config, source, seed):
    return digest(canonical(dict(solver=digest(source),seed=seed,n=config['n_cities'],seconds=config['solver_seconds'],
        trace=config.get('calibration_research_trace',False),trace_policy=config.get('calibration_trace_policy',{}),
        reference_tolerance=config['reference_gap_pct'])))


def design_seeds(master, count):
    if type(master) is not int or not 0 <= master < 2**32:
        raise ValueError('data_seed must be uint32')
    seeds=[int.from_bytes(hashlib.sha256(f'cef_data_v1:{master}:{i}'.encode()).digest()[:4], 'big')
           for i in range(count)]
    if len(set(seeds))!=len(seeds): raise ValueError('derived instance seeds collide')
    return seeds


def validate_design_config(c):
    slices=c.get('reference_work_slices')
    if slices is not None and (type(slices) is not int or slices<1):
        raise ValueError('positive integer reference_work_slices or None required')
    for key in ('iterations','instances','n_cities','workers'):
        if type(c[key]) is not int or c[key]<1: raise ValueError('positive integer: '+key)
    for key in ('solver_seconds','generator_seconds','reference_seconds','reference_gap_pct'):
        if isinstance(c[key],bool) or not isinstance(c[key],(int,float)) or not np.isfinite(c[key]) or c[key]<=0:
            raise ValueError('positive finite setting: '+key)
    if c['n_cities']<3: raise ValueError('at least three cities required')
    if 'diagnostic' in c and type(c['diagnostic']) is not bool: raise ValueError('diagnostic must be boolean')
    if not c.get('diagnostic',False) and (c['instances']!=64 or c['n_cities']!=100 or c['solver_seconds']!=60):
        raise ValueError('formal data design requires 64 TSP100 cases and 60-second calibration')
    if type(c['solver_seed']) is not int or not 0<=c['solver_seed']<2**32:
        raise ValueError('solver_seed must be uint32')
    return design_seeds(c['data_seed'],c['instances'])


def validate_coordinates(value, n):
    p = np.asarray(value, dtype=float)
    if p.shape != (n, 2) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('coordinates must be finite n x 2 values in [0,1]')
    if len(np.unique(p, axis=0)) != n: raise ValueError('duplicate cities')
    return p


class CoordinateAdapter:
    def solver_input(self, instance): return instance
    def score(self, instance, solution):
        validate_coordinates(solution, instance['n_cities'])
        return 0., 0.


def generate_coordinates(source, parameters, n, seed, seconds=5):
    tree = ast.parse(source)
    if not any(isinstance(x, ast.FunctionDef) and x.name == 'generate' for x in tree.body):
        raise ValueError('generator must define generate(n_cities, seed, parameters)')
    wrapper = '\ndef solve(instance, time_limit_s, seed):\n    points = generate(instance["n_cities"], seed, instance["parameters"])\n    return {"solution": [[float(x), float(y)] for x, y in points]}\n'
    record = Runner(seconds, 1024).case(source+wrapper, CoordinateAdapter(),
                                        dict(n_cities=n, parameters=parameters), seed)
    if not record['valid']: raise ValueError('generator execution: '+str(record['error']))
    return validate_coordinates(record['solution'], n).tolist()




class DataDesigner:
    def __init__(self, config, model):
        self.config = copy.deepcopy(config); self.model = model
        c = self.config
        self.seeds = validate_design_config(c)
        self.source = Path(c['solver_source']).read_text(encoding='utf-8-sig')
        self.root = Path(c['output_dir']).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.archive = Archive(self.root)
        from .reference_windows import backend_manifest
        self.archive.add('data_design_manifest', dict(config=c, instance_seeds=self.seeds,
            reference_backend=backend_manifest(),
            reference_work_policy=dict(backend='SCIP Windows 64-bit',continuous=c.get('reference_work_slices') is None,
                checkpoint='subtour cuts, incumbent and completed reference proofs retained; heartbeats do not restart search'),
            solver=receipt(c['solver_source']), implementation=[receipt(Path(__file__).with_name(name))
                for name in ('instance_design.py','reference.py','reference_jobs.py','reference_windows.py','reference_windows_worker.py','reference_scip_solver.py','runner.py','worker.py','tasks.py','model.py')]))
        (self.root/'calibration_solver.py').write_text(self.source, encoding='utf-8', newline='\n')
        write_json(self.root/'design_config.json', dict(c, instance_seeds=self.seeds))

    @classmethod
    def recover(cls, root):
        """Finish a known, fully returned proposal; never make a model request.

        Only recorded reference-backend failures are recoverable here. Completed
        successful measurements are hash checked and reused; old failures remain.
        """
        from .config import load_json
        root=Path(root).resolve();archive=Archive(root)
        manifests=archive.of('data_design_manifest')
        if len(manifests)!=1:raise ValueError('one data design manifest required for recovery')
        manifest=manifests[0];c=manifest['config'];seeds=validate_design_config(c)
        if c['iterations']!=1:raise ValueError('reference recovery requires one frozen data proposal')
        responses=archive.of('data_model_response');evaluations=archive.of('generator_evaluation')
        if len(responses)!=1:
            raise ValueError('complete known proposal and reference failure required; no model resend')
        proposal=responses[0]['response'];folder=root/'iteration_000'
        source=(folder/'generator.py').read_text(encoding='utf-8')
        parameters=load_json(folder/'parameters.json')
        if source!=proposal['generator_source'] or canonical(parameters)!=canonical(proposal['parameters']):
            raise ValueError('frozen generator proposal changed')
        solver=(root/'calibration_solver.py').read_text(encoding='utf-8-sig')
        if (file_hash(manifest['solver']['path'])!=manifest['solver']['sha256'] or
                Path(manifest['solver']['path']).read_text(encoding='utf-8-sig')!=solver):
            raise ValueError('frozen calibration solver changed')
        for row in manifest['implementation']:
            if Path(row['path']).name in ('runner.py','worker.py','tasks.py') and file_hash(row['path'])!=row['sha256']:
                raise ValueError('calibration execution implementation changed')
        latest={r['id']:r for r in archive.of('data_case') if r['iteration']==0}
        for path in (folder/'case_results').glob('*.json'):
            committed=load_json(path)
            if committed.get('measurement_identity')!=_case_identity(c,solver,committed['instance_seed']):
                raise ValueError('committed case evaluation conditions changed')
            if committed['id'] not in latest or not latest[committed['id']]['valid']:
                latest[committed['id']]=dict(iteration=0,**committed)
        expected={f'supplement_{i:04d}':seed for i,seed in enumerate(seeds)}
        if not set(latest)<=set(expected):raise ValueError('unexpected per-case reference results')
        recovered={}
        for identifier,r in latest.items():
            if r['instance_seed']!=expected[identifier]:raise ValueError('instance seed changed')
            if not r['valid']:
                if r.get('failure_stage')!='reference' and not r.get('pending') and not r.get('error','').startswith(('RuntimeError: TimeoutError: reference ',
                        'RuntimeError: RuntimeError: reference ', 'RuntimeError: reference worker:')):
                    raise ValueError('only recorded reference backend failures can be recovered: '+r.get('error',''))
                continue
            row=r['row']
            for artifact in [row,row['reference_record'],*row['source'].values()]:
                if file_hash(artifact['path'])!=artifact['sha256']:raise ValueError('completed data artifact changed')
            # A recovery fork has identical generator bytes at a new location.
            value=copy.deepcopy(r)
            value['row']['source']=dict(generator=receipt(folder/'generator.py'),parameters=receipt(folder/'parameters.json'))
            recovered[identifier]=value
        if (root/'result.json').exists():
            result=load_json(root/'result.json')
            if result.get('status')!='finished':raise ValueError('existing recovery result is not finished')
            return result
        if len(recovered)==len(seeds) and evaluations and evaluations[-1]['valid']:
            best=evaluations[-1]
            best=dict(best,evaluation_record_path=best.get('evaluation_record_path',str(folder/'evaluation.json')))
            obj=cls.__new__(cls);obj.config=copy.deepcopy(c);obj.root=root;obj.archive=archive;obj.seeds=seeds;obj.source=solver
            return obj.export_data(best,[recovered[k]['row'] for k in sorted(recovered)])
        obj=cls.__new__(cls);obj.config=copy.deepcopy(c);obj.root=root;obj.archive=archive;obj.seeds=seeds;obj.source=solver
        archive.add('data_recovery_start',dict(reused_cases=len(recovered),pending_cases=len(seeds)-len(recovered),
            reference_work_policy=dict(backend='SCIP Windows 64-bit',continuous=c.get('reference_work_slices') is None),
            generator=receipt(folder/'generator.py'),reference_implementation=receipt(Path(__file__).with_name('reference_windows.py')),
            instance_design_implementation=receipt(Path(__file__)),no_model_request=True))
        best,rows=obj.evaluate(source,parameters,0,recovered=recovered)
        if not best['valid']:
            if all(r.get('pending') for r in best['per_instance'] if not r['valid']):
                raise ReferencePending('reference recovery still pending; checkpoints retained: '+str(root))
            raise RuntimeError('reference recovery incomplete; measured successes retained; '+
                '; '.join(r['id']+': '+r['error'] for r in best['per_instance'] if not r['valid']))
        return obj.export_data(best,rows)

    def measure_case(self, folder, source, parameters, item):
        c=self.config;adapter=TaskAdapter('tsp')
        index, seed = item; identifier = f'supplement_{index:04d}'
        stage='generation'
        try:
            points = generate_coordinates(source, parameters, c['n_cities'], seed, c['generator_seconds'])
            repeated = generate_coordinates(source, parameters, c['n_cities'], seed, c['generator_seconds'])
            if points != repeated: raise ValueError('generator is not reproducible with a fixed seed')
            p = np.asarray(points); matrix = np.linalg.norm(p[:, None]-p[None, :], axis=2)
            stage='reference'
            ref_path = folder/f'{identifier}.reference.json'
            if ref_path.exists():
                reference=validate_reference(matrix,json.loads(ref_path.read_text(encoding='utf-8')),c['reference_gap_pct'])
            else:
                reference = independent_reference(matrix, c['reference_seconds'], c['reference_gap_pct'],
                    folder/'reference_work'/f'{identifier}.json',c.get('reference_work_slices'))
            stage='calibration'
            instance = dict(distance_matrix=matrix.tolist(), reference=reference['objective'])
            extra={}
            if c.get('calibration_research_trace'):
                extra['research_request']=dict(kind='trace',trace_policy=c.get('calibration_trace_policy',{}))
            execution = Runner(c['solver_seconds'], 1024).case(self.source, adapter, instance, c['solver_seed'],**extra)
            if not execution['valid']: raise ValueError('calibration solver: '+str(execution['error']))
            path = folder/f'{identifier}.npz'
            np.savez(path, coordinates=p, distance_matrix=matrix)
            if not ref_path.exists():
                pending_ref=ref_path.with_suffix(ref_path.suffix+'.tmp')
                with pending_ref.open('w',encoding='utf-8',newline='\n') as stream:
                    json.dump(reference,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
                os.replace(pending_ref,ref_path)
            row = dict(id=identifier, split='train', training_group='supplement',
                **receipt(path), reference=reference['objective'], reference_kind=reference['reference_kind'],
                reference_record=receipt(ref_path), generator_seed=seed,
                reference_gap_tolerance_pct=c['reference_gap_pct'],
                source=dict(generator=receipt(folder/'generator.py'), parameters=receipt(folder/'parameters.json')))
            result=dict(id=identifier, valid=True, row=row, reference_gap_pct=execution['fitness'],
                objective=execution['objective'], solver_seconds=execution['elapsed'],
                reference_bound_gap_pct=reference['bound_gap_pct'],
                instance_seed=seed, matrix_hash=hashlib.sha256(matrix.tobytes()).hexdigest(),
                measurement_identity=_case_identity(c,self.source,seed))
            # Commit each completed measurement before the outer batch journal.
            commits=folder/'case_results';commits.mkdir(exist_ok=True)
            pending=commits/(identifier+'.tmp');target=commits/(identifier+'.json')
            with pending.open('w',encoding='utf-8',newline='\n') as stream:
                json.dump(result,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
            os.replace(pending,target)
            return result
        except Exception as error:
            return dict(id=identifier, valid=False, error=type(error).__name__+': '+str(error), instance_seed=seed,
                failure_stage=stage,pending=isinstance(error,ReferencePending))

    def evaluate(self, source, parameters, iteration, recovered=None):
        c = self.config; folder = self.root/f'iteration_{iteration:03d}'; folder.mkdir(exist_ok=recovered is not None)
        label=f'round {c["global_round"]}/{c["global_total_rounds"]}' if c.get('global_round') else f'{iteration+1}/{c["iterations"]}'
        measured = list((recovered or {}).values())
        if c.get('console_progress'):
            print(f'Evaluate data {label} | {len(measured)}/{len(self.seeds)} | workers={c["workers"]} | reused={len(measured)}',flush=True)
        if recovered is None:
            (folder/'generator.py').write_text(source, encoding='utf-8', newline='\n')
            write_json(folder/'parameters.json', parameters)
        rows = []; seen = set()

        def work(item):
            return self.measure_case(folder, source, parameters, item)

        pending=[(i,seed) for i,seed in enumerate(self.seeds) if f'supplement_{i:04d}' not in (recovered or {})]
        for item, result in completed_tasks(pending, work, c['workers']):
            measured.append(result)
            self.archive.add('data_case', dict(iteration=iteration, **result))
            if c.get('console_progress') and (len(measured)%8==0 or len(measured)==len(self.seeds)):
                print(f'Evaluate data {label} | {len(measured)}/{len(self.seeds)} | valid={sum(r["valid"] for r in measured)} | failed={sum(not r["valid"] for r in measured)}', flush=True)
        measured.sort(key=lambda r:r['id'])
        for r in measured:
            if r['valid']:
                if r['matrix_hash'] in seen:
                    r.update(valid=False, error='duplicate generated instance within batch')
                    self.archive.add('data_case', dict(iteration=iteration, **r))
                seen.add(r['matrix_hash'])
        valid = all(r['valid'] for r in measured)
        if valid: rows = [r['row'] for r in measured]
        observation = dict(iteration=iteration, valid=valid, solver_seed=c['solver_seed'],
            calibration_source_hash=digest(self.source),global_round=c.get('global_round'),
            comparison='fixed complete calibration solver versus independently bounded unrounded reference tours',
            unit='percent', direction='larger reference gap means harder for this calibration solver',
            solver_time_limit_s=c['solver_seconds'],
            mean_reference_gap_pct=statistics.fmean(r['reference_gap_pct'] for r in measured) if valid else None,
            per_instance=[{k:v for k,v in r.items() if k not in ('row','matrix_hash')} for r in measured])
        name='evaluation.json' if recovered is None else f'evaluation.recovery_{len(self.archive.of("data_recovery_start")):03d}.json'
        observation['evaluation_record_path']=str(folder/name)
        write_json(folder/name, observation)
        self.archive.add('generator_evaluation', observation)
        return observation, rows

    def export_data(self, best, rows):
        c=self.config
        if not c.get('diagnostic',False):
            from .data import audit_training
            from .config import load_json
            original=load_json(Path(__file__).resolve().parents[1]/'data/cef_v1_3_20261026/tsp_manifest.json')['train']
            audit_training('tsp',original+rows,SUPPLEMENT_PROTOCOL)
        manifest = dict(protocol=SUPPLEMENT_PROTOCOL, diagnostic=bool(c.get('diagnostic',False)),
            train=rows, selected_iteration=best['iteration'], selected_mean_reference_gap_pct=best['mean_reference_gap_pct'],
            instance_seeds=self.seeds, calibration_solver=receipt(self.root/'calibration_solver.py'),
            selection='highest measured mean reference gap; ties keep the earlier generator',
            limitations='Fixed training samples, not held-out validation; no claim of TSPLIB gain or exact-arithmetic optimality.',
            design_config=receipt(self.root/'design_config.json'),
            evaluation_record=receipt(best['evaluation_record_path']))
        path = self.root/'supplement_manifest.json'; write_json(path, manifest)
        result = dict(status='finished', supplement_manifest=str(path),
                      selected_iteration=best['iteration'], instance_seeds=self.seeds,
                      mean_reference_gap_pct=best['mean_reference_gap_pct'])
        write_json(self.root/'result.json', result)
        return result

    def run(self):
        c = self.config; observations = copy.deepcopy(c.get('prior_results',[])); candidates = []
        initial = c.get('initial_generator_source') or (Path(__file__).parent/'seeds/tsp_instances.py').read_text(encoding='utf-8')
        source = initial; parameters = copy.deepcopy(c.get('initial_parameters',{}))
        try:
            for iteration in range(c['iterations']):
                payload = dict(task='Design a reproducible supplemental Euclidean TSP training distribution',
                    n_cities=c['n_cities'], instances=c['instances'], instance_seeds=self.seeds,
                    solver_seed=c['solver_seed'], solver_seconds=c['solver_seconds'],
                    calibration_solver=self.source, current_generator=source, current_parameters=parameters,
                    previous_results=copy.deepcopy(observations), iteration=iteration, total_iterations=c['iterations'],
                    global_round=c.get('global_round'),global_total_rounds=c.get('global_total_rounds'),calibration_source_hash=digest(self.source),
                    objective='Increase the measured mean reference gap of the fixed complete calibration solver on valid instances.',
                    generator_interface='generate(n_cities, seed, parameters) -> n_cities by 2 coordinates in [0,1]',
                    tool='The controller executes the submitted generator, constructs independent references, runs the fixed solver, and returns actual per-instance results.',
                    initial_generator_provenance='ASRO Appendix H.1 geometric families; local executable implementation, not a claimed new distribution')
                self.archive.add('data_model_start', dict(iteration=iteration, payload=payload))
                if c.get('console_progress'):
                    label=f'round {c["global_round"]}/{c["global_total_rounds"]}' if c.get('global_round') else f'{iteration+1}/{c["iterations"]}'
                    print(f'Data Agent {label} | generating code',flush=True)
                proposal = self.model.call('data_designer', payload)
                self.archive.add('data_model_response', dict(iteration=iteration, response=proposal))
                if not isinstance(proposal.get('generator_source'), str) or not isinstance(proposal.get('parameters'), dict):
                    raise ValueError('data_designer requires generator_source string and parameters object')
                source = proposal['generator_source']; parameters = proposal['parameters']
                observation, rows = self.evaluate(source, parameters, iteration)
                observations.append(observation)
                if observation['valid']: candidates.append((observation, rows))
            if not candidates:
                failures=[r for o in observations[-c['iterations']:] for r in o.get('per_instance',[]) if not r['valid']]
                if failures and all(r.get('pending') for r in failures):
                    raise ReferencePending(f'{len(failures)} reference jobs pending; existing cases and proposal retained: {self.root}')
                detail='; '.join(r['id']+': '+r['error'] for r in failures[:3])
                raise RuntimeError(f'no valid measured generator; failed_cases={len(failures)}; '+detail+'; no default data substituted')
            best, rows = max(candidates, key=lambda item:item[0]['mean_reference_gap_pct'])
            return self.export_data(best, rows)
        except ReferencePending as error:
            self.archive.add('data_design_pending',dict(error=str(error)))
            raise
        except Exception as error:
            self.archive.add('data_design_failure', dict(error=type(error).__name__+': '+str(error)))
            raise
