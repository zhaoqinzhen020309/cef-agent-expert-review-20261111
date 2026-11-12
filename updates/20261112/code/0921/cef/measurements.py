"""One durable solver observation shared by data design and formal selection."""
import copy
from pathlib import Path
import threading

from .patch import digest
from .store import Archive, canonical
from .parallel import completed_tasks


def observation_key(source, instance, seed, contract):
    return digest(canonical(dict(source_hash=digest(source),instance=instance,seed=seed,contract=contract)))


class SharedMeasurements:
    def __init__(self, root):
        self.archive=Archive(root)
        self._lock=threading.RLock()
        self._keys={}

    def obtain(self, key, identity, execute):
        with self._lock:
            lock=self._keys.setdefault(key,threading.RLock())
        with lock:
            previous=[r for r in self.archive.of('observation') if r['key']==key]
            if previous:
                saved=previous[0]
                if saved['identity']!=identity or digest(canonical(saved['record']))!=saved['record_hash']:
                    raise ValueError('shared observation identity or content changed')
                return copy.deepcopy(saved['record']),True
            if any(r['key']==key for r in self.archive.of('observation_start')):
                raise RuntimeError('unfinished shared observation; result unknown; no automatic rerun')
            self.archive.add('observation_start',dict(key=key,identity=identity))
            record=execute()
            self.archive.add('observation',dict(key=key,identity=identity,record=record,
                                              record_hash=digest(canonical(record))))
            return copy.deepcopy(record),False

    def contract(self, engine):
        from .config import file_hash
        files=('runner.py','worker.py','tasks.py','patch.py','measurements.py')
        return dict(version='cef_solver_observation_v1',task=engine.config['task'],
            score='100*(objective/reference-1)' if engine.config['task']=='tsp' else 'makespan',
            seconds=engine.config['solver_time_limit_s'],memory_mb=engine.config.get('memory_mb',1024),
            workers=engine.workers,threads=1,scheduling='solver_only',
            trace=bool(engine.research),trace_policy=engine.config.get('trace_policy',{}) if engine.research else {},
            environment=engine.config.get('environment_manifest'),
            implementation={name:file_hash(Path(__file__).with_name(name)) for name in files})

    def execute_batch(self, engine, jobs, reservation, purpose):
        jobs=list(jobs)
        contract=self.contract(engine)
        if engine.config.get('measured_research_loop_v1'):
            contract['active_parallel_cases']=min(engine.workers,len(jobs))
        sources={cid:engine.archive.read_source(c['source_hash']) for cid,c in engine.candidates().items()}
        def prepare():
            for cid,row,eid,order in jobs:
                source=sources[cid];instance=engine.adapter.load(row)
                seed=engine.config['evaluation_seed_manifest'][row['id']]
                identity=dict(source_hash=digest(source),instance_hash=digest(canonical(instance)),seed=seed,contract=contract)
                yield cid,row,eid,order,source,instance,seed,identity,observation_key(source,instance,seed,contract)
        def work(item):
            cid,row,eid,order,source,instance,seed,identity,key=item
            def execute():
                extra={'research_request':dict(kind='trace',trace_policy=engine.config.get('trace_policy',{}))} if engine.research else {}
                return engine.runner.case(source,engine.adapter,instance,seed,
                    lambda:engine.ledger.start_case(eid,reservation,cid),**extra)
            record,reused=self.obtain(key,identity,execute)
            return record,reused
        for item,(record,reused) in completed_tasks(prepare(),work,engine.workers):
            cid,row,eid,order,source,instance,seed,identity,key=item
            record.update(execution_id=eid,candidate_id=cid,instance_id=row['id'],seed=seed,
                config_hash=engine.eval_hash,group=engine.adapter.group(instance),purpose=purpose,order=order,
                observation_id=key,reused_observation=reused)
            if reused:record['wallclock_s']=0.
            if engine.augmented:record['training_group']=row.get('training_group','original')
            engine.archive.add('execution',record)
            # Calibration occurs before the new epoch exists. Its trace is kept
            # in the observation and captured only by the formal epoch view.
            if engine.research and purpose=='evaluation':engine.research.capture(cid,row,record)
            yield record
