"""A frozen supplement changes actual search data without changing original bindings."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cef.data import receipt, audit_training, bind_supplement, SUPPLEMENT_PROTOCOL
from cef.store import canonical

ROOT = Path(__file__).resolve().parents[1]


def supplement_fixture(folder):
    source=folder/'generate.py'; source.write_text('fixture generator')
    params=folder/'parameters.json'; params.write_text('{}')
    rows=[]
    for i in range(64):
        p=np.random.default_rng(10000+i).random((100,2))
        a=np.linalg.norm(p[:,None]-p[None,:],axis=2)
        path=folder/f'{i}.npz'; np.savez(path,coordinates=p,distance_matrix=a)
        tour=list(range(100)); cost=float(sum(a[j,(j+1)%100] for j in range(100)))
        ref=folder/f'{i}.ref.json'
        ref.write_text(json.dumps(dict(solution=tour,objective=cost,lower_bound=cost,
            bound_gap_pct=0,certificate='offline fixture, never a performance result')))
        rows.append(dict(id=f'supplement_{i:04d}',split='train',training_group='supplement',
            **receipt(path),reference=cost,reference_kind='HiGHS_numerically_bounded_tour',
            reference_record=receipt(ref),generator_seed=i,
            source=dict(generator=receipt(source),parameters=receipt(params))))
    manifest=folder/'supplement_manifest.json'
    manifest.write_text(json.dumps(dict(protocol=SUPPLEMENT_PROTOCOL,diagnostic=False,train=rows)))
    return manifest


class SupplementBindingTests(unittest.TestCase):
    def test_new_native_references_bind_to_formal_original64_plus64(self):
        with tempfile.TemporaryDirectory() as td:
            path=supplement_fixture(Path(td));manifest=json.loads(path.read_text())
            for row in manifest['train']:
                row['reference_kind']='Concorde_numerically_bounded_unrounded_tour'
                record=Path(row['reference_record']['path']);ref=json.loads(record.read_text())
                ref['reference_kind']=row['reference_kind'];record.write_text(json.dumps(ref))
                row['reference_record']=receipt(record)
            path.write_text(json.dumps(manifest))
            base=json.loads((ROOT/'data/cef_v1_3_20261026/tsp_manifest.json').read_text())
            merged=bind_supplement(base,path)
            self.assertEqual(merged['audit']['instances'],128)
            self.assertEqual(merged['train'][:64],base['train'])

    def test_two_full_pab_iterations_execute128_cases_and_change_champion(self):
        from cef.engine import Engine
        from scripts.cef_launcher import build_config
        class Fake:
            def __init__(self):self.inputs=[];self.edits=0
            def call(self,role,payload):
                self.inputs.append((role,payload))
                if role=='planner':
                    return dict(action='revise',edit_count=2,parent_or_target_id=payload['active_parent']['candidate_id'],
                        issue='offline data binding acceptance',proposed_intents=['marker A','marker B'],
                        evidence_ids=[],previous_event_update=[],expected_observations=[],resource_request={})
                index=self.edits;self.edits+=1
                return dict(package_id=f'offline_{index}',parent_id=payload['parent_id'],parent_hash=payload['parent_hash'],
                    issue='fixture',evidence_ids=[],joint_reason='joint-only acceptance',edits=[
                        dict(edit_id='A',purpose='marker',prediction='measured jointly',exact_replacements=[
                            dict(old='import time',new=f'import time\n# iteration_{index}')]),
                        dict(edit_id='B',purpose='marker',prediction='measured jointly',exact_replacements=[
                            dict(old='def solve(instance, time_limit_s, seed):',new='def solve(instance, time_limit_s, seed):\n    # offline marker')])])
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);manifest=supplement_fixture(folder)
            c,_=build_config(2025,folder/'search',dict(model='deepseek-flash',base_url='https://api.deepseek.com',
                iterations=2,workers=4,evaluation_policy='joint_only',supplement_manifest=manifest))
            c['console_progress']=False
            model=Fake();engine=Engine(c,model)
            original={r['reference'] for r in c['data_manifest']['train'][:64]}
            def case(source,adapter,instance,seed,on_started,**kw):
                on_started();second='# iteration_1' in source
                fitness=(1. if second else 0.) if instance['reference'] in original else (0. if second else 10.)
                return dict(valid=True,fitness=fitness,objective=instance['reference']*(1+fitness/100),
                            started=True,elapsed=.01,wallclock_s=.02)
            with patch.object(engine.runner,'case',side_effect=case) as worker:
                result=engine.run()
            self.assertEqual(result['status'],'finished',result['reason'])
            self.assertEqual(worker.call_count,256)
            self.assertEqual({r['candidate_id'] for r in engine.archive.of('execution')},{'step_0:PAB','step_1:PAB'})
            self.assertEqual(result['best']['candidate_id'],'step_1:PAB')
            planners=[p for role,p in model.inputs if role=='planner']
            self.assertEqual(planners[1]['global_best']['training_group_scores'],{'original':0.,'supplement':10.})
            self.assertEqual(result['best']['training_group_scores'],{'original':1.,'supplement':0.})

    def test_launcher_loads_exact_original64_plus_supplement64(self):
        from scripts.cef_launcher import build_config
        settings=dict(model='deepseek-flash',base_url='https://api.deepseek.com',iterations=2,
                      workers=2,evaluation_policy='joint_only')
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td); path=supplement_fixture(folder)
            old,_=build_config(2025,folder/'old',settings)
            new,test=build_config(2025,folder/'new',dict(settings,supplement_manifest=path))
            self.assertEqual(new['data_manifest']['train'][:64],old['data_manifest']['train'])
            self.assertEqual(test['instances'],old['data_manifest']['test'])
            self.assertEqual(len(new['evaluation_seed_manifest']),128)
            self.assertEqual(set(new['evaluation_seed_manifest'].values()),{2025})
            self.assertEqual(new['data_manifest']['training_protocol'],SUPPLEMENT_PROTOCOL)
            self.assertEqual(audit_training('tsp',new['data_manifest']['train'],SUPPLEMENT_PROTOCOL)['instances'],128)
            from cef.config import validate
            validate(new)
            saved=path.read_text();manifest=json.loads(saved);manifest['diagnostic']=True
            path.write_text(json.dumps(manifest))
            bad=copy.deepcopy(new);bad['data_manifest']['supplement_manifest']=receipt(path)
            bad['reference_manifest']['reference_files']=[r for r in bad['reference_manifest']['reference_files'] if r['path']!=str(path.resolve())]
            bad['reference_manifest']['reference_files'].append(receipt(path))
            with self.assertRaisesRegex(ValueError,'diagnostic'):validate(bad)
            path.write_text(saved)
            with self.assertRaises(ValueError):
                bad=copy.deepcopy(new); bad['data_manifest']['train'][0]['reference']+=1; validate(bad)
            with self.assertRaises(ValueError):
                bad=copy.deepcopy(new); bad['data_manifest'].pop('training_protocol'); validate(bad)
            diagnostic=copy.deepcopy(new);diagnostic.update(diagnostic_run=True,diagnostic_train_limit=2)
            diagnostic=validate(diagnostic,formal=False)
            from cef.engine import Engine
            engine=Engine(diagnostic,None)
            self.assertEqual([r.get('training_group','original') for r in engine.rows],['original','supplement'])
            with self.assertRaises(ValueError):
                diagnostic['diagnostic_train_limit']=1;validate(diagnostic,formal=False)
            reference=new['data_manifest']['train'][-1]['reference_record']['path']
            Path(reference).write_text('{}')
            with self.assertRaises(ValueError): validate(new)

    def test_rejects_diagnostic_data_even_when_it_contains64_cases(self):
        with tempfile.TemporaryDirectory() as td:
            path=supplement_fixture(Path(td)); manifest=json.loads(path.read_text());manifest['diagnostic']=True
            path.write_text(json.dumps(manifest))
            base=json.loads((ROOT/'data/cef_v1_3_20261026/tsp_manifest.json').read_text())
            with self.assertRaises(ValueError): bind_supplement(base,path)

    def test_augmented_scores_reach_planner_editor_and_history(self):
        from cef.engine import Engine
        from cef.feedback import planner_history_observation
        from test_cef_iterations_billing import iteration_config
        with tempfile.TemporaryDirectory() as td:
            c=iteration_config(td,2);c.update(task='tsp',evaluation_policy='joint_only',evaluation_workers=1)
            c['data_manifest']={'training_protocol':SUPPLEMENT_PROTOCOL,'train':[
                dict(id='original',split='train',instance={'distance_matrix':[[0,1,1],[1,0,1],[1,1,0]],'reference':3}),
                dict(id='supplement',split='train',training_group='supplement',instance={'distance_matrix':[[0,1,1],[1,0,1],[1,1,0]],'reference':6})]}
            c['evaluation_seed_manifest']={'original':2025,'supplement':2025}
            e=Engine(c,None);e.register('seed',e.adapter.seed_source());e.register('child',e.adapter.seed_source(),'seed')
            e.register('other',e.adapter.seed_source()+'\n# other','seed')
            def case(source,adapter,instance,seed,on_started,**kw):
                on_started();other='# other' in source
                fitness=(1. if other else 0.) if instance['reference']==3 else (0. if other else 10.)
                return dict(valid=True,fitness=fitness,objective=3.06,started=True,elapsed=.01,wallclock_s=.02)
            e.ledger.reserve('x',4)
            with patch.object(e.runner,'case',side_effect=case): scores,records=e.evaluate({'A':'child','B':'other'},'x','x')
            self.assertEqual(scores,{'A':5.,'B':.5})
            candidate=e.best();self.assertEqual(candidate['candidate_id'],'other')
            self.assertEqual(candidate['training_group_scores'],{'original':1.,'supplement':0.})
            state=e.state('other');self.assertEqual(state['global_best']['training_group_scores'],candidate['training_group_scores'])
            compact=planner_history_observation(state)
            self.assertEqual(compact['global_best']['training_group_scores'],candidate['training_group_scores'])
            editor=e.editor_context('other',dict(evidence_ids=[]),2)
            self.assertEqual(editor['training_group_scores'],candidate['training_group_scores'])


if __name__=='__main__': unittest.main()
