"""Behavioral integration for role submissions, execution feedback and PAB scope."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from cef.tool_agents import run_data_task, run_algorithm_task
from test_cef_integrated import GEN, small_fixture
from test_cef_alternating import AlgorithmFake
from cef.alternating import AlternatingEngine


def call(name, args, number):
    return dict(message=dict(role='assistant', content=None, tool_calls=[dict(
        id=f'call_{number}', type='function', function=dict(name=name, arguments=json.dumps(args)))]),
        finish_reason='tool_calls')


class NativeFake:
    config = {}
    def __init__(self, actions): self.actions=list(actions); self.messages=[]
    def exchange(self, messages, tools, role):
        self.messages.append(copy.deepcopy(messages))
        name,args=self.actions.pop(0)
        return call(name,args,len(self.messages))


class RoleTaskTests(unittest.TestCase):
    def test_accepted_task_resumes_after_host_crash_without_model_redispatch(self):
        from unittest.mock import patch
        from cef.store import Archive
        from cef.tool_agents import _run, binding
        from cef.store import canonical
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            c,_=small_fixture(td,rounds=1)
            c['agent_tool_loop_v1']=True
            e=AlternatingEngine(c,None)
            e.register('warm_start',Path(c['warm_start_source']).read_text(),initialization='explicit warm start')
            value=dict(action='inspect',parent_or_target_id='warm_start',evidence_ids=[],previous_event_update=[])
            model=NativeFake([('submit_candidate',dict(content=json.dumps(value)))])
            e.model=model
            original=e.archive.add
            def interrupted(kind,data):
                if kind=='model_raw_response':raise KeyboardInterrupt('host interrupted after accepted task')
                return original(kind,data)
            with patch.object(e.archive,'add',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):e.request('planner',e.state(),'step_0:planner')
            resumed=AlternatingEngine(c,model)
            self.assertEqual(resumed.request('planner',dict(changed='ignored; original input is frozen'),'step_0:planner'),value)
            self.assertEqual(len(model.messages),1)
            self.assertEqual(len(resumed.archive.of('model_response')),1)

    def test_material_manifest_hashes_match_bound_values(self):
        from cef.tool_agents import _run,binding
        from cef.store import canonical
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            c,_=small_fixture(td,rounds=1)
            e=AlternatingEngine(c,None)
            model=NativeFake([('submit_candidate',dict(content='{}'))])
            materials={'source.py':binding('def f(): return 1\n'),'values.json':binding([1,2])}
            _run(e,model,'planner',{},Path(td)/'task',materials,
                 lambda path:dict(accepted=True,value=json.loads(path.read_text())),{})
            index=json.loads(model.messages[0][1]['content'])['artifacts']
            for item in index:
                value=materials[item['id']]['value'] if item['id'] in materials else {}
                raw=value if isinstance(value,str) else canonical(value)
                self.assertEqual(item['sha256'],hashlib.sha256(raw.encode()).hexdigest())

    def test_native_reference_host_failure_is_pending_not_rejected_data(self):
        from unittest.mock import patch
        from cef.integrated_data import IntegratedDataProvider
        from cef.reference_jobs import ReferencePending
        with tempfile.TemporaryDirectory() as td:
            rule=dict(name='x',source=GEN,parameters={},weight=1)
            from cef.recipes import normalize_rules,dispatch
            normalized=normalize_rules(dict(rules=[rule]))[0]
            item=dispatch([normalized],[71])[0]
            provider=IntegratedDataProvider({},None)
            for failure in (RuntimeError('WSL unavailable'),OSError('WSL unavailable')):
                with patch('cef.integrated_data.independent_reference',side_effect=failure):
                    with self.assertRaisesRegex(ReferencePending,'WSL unavailable'):
                        provider.prepare_case(Path(td),item,normalized,dict(agent_tool_loop_v1=True,n_cities=7,generator_seconds=3,
                            reference_gap_pct=.001,reference_seconds=10))

    def test_two_round_native_roles_reach_real_full_solver_evaluation(self):
        class Roles:
            config={}
            def __init__(self):self.legacy=AlgorithmFake();self.number=0;self.buffer=''
            def exchange(self,messages,tools,role):
                self.number+=1
                if len(messages)==2:
                    self.buffer=''
                    return call('read_artifact',dict(artifact_id='task.json',limit=16000),self.number)
                outcome=json.loads(messages[-1]['content'])
                self.buffer+=outcome['content']
                if outcome.get('has_more'):
                    return call('read_artifact',dict(artifact_id='task.json',offset=len(self.buffer),limit=16000),self.number)
                payload=json.loads(self.buffer)
                if role=='data_designer':value=dict(rules=[dict(name='uniform',source=GEN,parameters={},weight=1)],reason='fixture')
                else:value=self.legacy.call(role,payload)
                first=call('write_file',dict(relative_path='answer.json',content=json.dumps(value)),str(self.number)+'w')
                second=call('submit_candidate',dict(path='answer.json'),str(self.number)+'s')
                first['message']['tool_calls']+=second['message']['tool_calls']
                return first
        with tempfile.TemporaryDirectory() as td:
            from cef.alternating import DataRoundProvider
            c,settings=small_fixture(td,rounds=2)
            c['agent_tool_loop_v1']=True;settings['agent_tool_loop_v1']=True
            model=Roles();e=AlternatingEngine(c,model,DataRoundProvider(settings,model))
            result=e.run()
            self.assertEqual(result['status'],'finished',result['reason'])
            self.assertEqual(len(e.archive.of('round_complete')),2)
            self.assertEqual(len(e.archive.of('package')),2)
            tasks=e.archive.of('agent_task_complete')
            self.assertEqual({r['role'] for r in tasks},{'data_designer','planner','editor'})
            self.assertEqual(len(tasks),8)
            self.assertTrue(all(m['valid'] for m in e.archive.of('measurement')))

    def test_data_rejects_bad_generator_then_repairs_inside_same_task(self):
        with tempfile.TemporaryDirectory() as td:
            c,settings=small_fixture(td,rounds=1)
            c['agent_tool_loop_v1']=True
            e=AlternatingEngine(c,None)
            rules=lambda source:dict(rules=[dict(name='generated',source=source,parameters={},weight=1)],reason='trial')
            bad=rules(GEN.replace('    return np.random',
                '    if seed == 72: return []\n    return np.random'))
            good=rules(GEN)
            model=NativeFake([('write_file',dict(relative_path='proposal.json',content=json.dumps(bad))),
                ('submit_candidate',dict(path='proposal.json')),
                ('write_file',dict(relative_path='proposal.json',content=json.dumps(good))),
                ('submit_candidate',dict(path='proposal.json'))])
            payload=dict(n_cities=7,instances=2,instance_seeds=[71,72],current_rules=[],previous_results=[],
                global_round=1,global_total_rounds=1,calibration_solver='def solve(instance,time_limit_s,seed): pass')
            actual=run_data_task(e,model,payload,Path(td)/'data',settings)
            self.assertEqual(actual,good)
            error=json.loads(model.messages[2][-1]['content'])
            self.assertFalse(error['accepted'])
            self.assertIn('coordinates',str(error).lower())
            self.assertEqual(len(model.messages),4)
            self.assertEqual(len(list((Path(td)/'data/generated').glob('*.json'))),3)

    def test_planner_invalid_action_is_observed_then_corrected(self):
        with tempfile.TemporaryDirectory() as td:
            c,settings=small_fixture(td,rounds=1)
            e=AlternatingEngine(c,None)
            e.register('warm_start',Path(c['warm_start_source']).read_text(),initialization='explicit warm start')
            # The small task tests the original action validator, not a full search.
            valid=dict(action='inspect',parent_or_target_id='warm_start',evidence_ids=[],previous_event_update=[])
            model=NativeFake([('write_file',dict(relative_path='decision.json',content=json.dumps(dict(action='invent')))),
                ('submit_candidate',dict(path='decision.json')),
                ('write_file',dict(relative_path='decision.json',content=json.dumps(valid))),
                ('submit_candidate',dict(path='decision.json'))])
            payload=e.state()
            result=run_algorithm_task(e,model,'planner',payload,'step_0:planner')
            self.assertEqual(result,valid)
            self.assertFalse(json.loads(model.messages[2][-1]['content'])['accepted'])

    def test_editor_cannot_change_parent_and_only_complete_pab_is_checked(self):
        with tempfile.TemporaryDirectory() as td:
            c,settings=small_fixture(td,rounds=1)
            e=AlternatingEngine(c,None)
            source=Path(c['warm_start_source']).read_text()
            e.register('warm_start',source,initialization='explicit warm start')
            parent=e.candidates()['warm_start']
            decision=dict(action='revise',edit_count=2,parent_or_target_id='warm_start',evidence_ids=[],
                proposed_intents=['A','B'],issue='fixture')
            payload=e.editor_context('warm_start',decision,2)
            package=AlgorithmFake().call('editor',payload)
            wrong={k:v for k,v in package.items() if k!='package_id'}
            model=NativeFake([('write_file',dict(relative_path='package.json',content=json.dumps(wrong))),
                ('submit_candidate',dict(path='package.json')),
                ('write_file',dict(relative_path='package.json',content=json.dumps(package))),
                ('submit_candidate',dict(path='package.json'))])
            self.assertEqual(run_algorithm_task(e,model,'editor',payload,'step_0:editor'),package)
            self.assertFalse(json.loads(model.messages[2][-1]['content'])['accepted'])
            self.assertEqual(len(e.archive.of('measurement')),0)


if __name__=='__main__':unittest.main()
