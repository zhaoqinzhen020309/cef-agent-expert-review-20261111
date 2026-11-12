"""A completed model proposal can be recovered without repeating successful cases."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cef.instance_design import DataDesigner, independent_reference
from test_cef_instance_design import GENERATOR,SOLVER,DesignerFake


class MeasuredStageRecoveryTests(unittest.TestCase):
    def test_default_data_stage_uses_one_native_search(self):
        from cef.reference_jobs import run_reference_slice
        calls=[]
        def reference(*args):
            calls.append(args[1])
            return run_reference_slice(*args)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);solver=root/'solver.py';solver.write_text(SOLVER)
            c=dict(output_dir=str(root/'data'),solver_source=str(solver),iterations=1,
                data_seed=42,solver_seed=2025,instances=1,n_cities=7,workers=1,
                solver_seconds=1,generator_seconds=3,reference_seconds=1,
                reference_gap_pct=.001,diagnostic=True)
            with patch('cef.reference_jobs.run_reference_slice',side_effect=reference):
                self.assertEqual(DataDesigner(c,DesignerFake()).run()['status'],'finished')
            self.assertEqual(calls,[None])

    def test_pending_reference_is_preserved_without_exporting_a_training_score(self):
        from cef.reference_jobs import ReferencePending
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);solver=root/'solver.py';solver.write_text(SOLVER)
            c=dict(output_dir=str(root/'data'),solver_source=str(solver),iterations=1,
                data_seed=42,solver_seed=2025,instances=1,n_cities=7,workers=1,
                solver_seconds=1,generator_seconds=3,reference_seconds=15,
                reference_gap_pct=.001,diagnostic=True)
            model=DesignerFake();designer=DataDesigner(c,model)
            with patch('cef.instance_design.independent_reference',side_effect=ReferencePending('saved cut pool')):
                with self.assertRaises(ReferencePending):designer.run()
            observation=designer.archive.of('generator_evaluation')[-1]
            self.assertIsNone(observation['mean_reference_gap_pct'])
            self.assertTrue(designer.archive.of('data_design_pending'))
            self.assertFalse((root/'data/supplement_manifest.json').exists())
            self.assertEqual(DataDesigner.recover(root/'data')['status'],'finished')
            self.assertEqual(len(model.inputs),1)

    def test_completed_case_commit_survives_interruption_before_batch_journal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);solver=root/'solver.py';solver.write_text(SOLVER)
            c=dict(output_dir=str(root/'data'),solver_source=str(solver),iterations=1,
                data_seed=42,solver_seed=2025,instances=1,n_cities=7,workers=1,
                solver_seconds=1,generator_seconds=3,reference_seconds=15,
                reference_gap_pct=.001,diagnostic=True)
            measure=DataDesigner.measure_case
            def interrupt(obj,*args):
                measure(obj,*args)
                raise KeyboardInterrupt('case returned before batch journal append')
            with patch.object(DataDesigner,'measure_case',interrupt):
                with self.assertRaises(KeyboardInterrupt):DataDesigner(c,DesignerFake()).run()
            with patch('cef.instance_design.independent_reference',side_effect=AssertionError('completed case must be reused')):
                self.assertEqual(DataDesigner.recover(root/'data')['status'],'finished')

    def test_known_proposal_resumes_partial_batch_without_a_second_model_call(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);solver=root/'solver.py';solver.write_text(SOLVER)
            c=dict(output_dir=str(root/'data'),solver_source=str(solver),iterations=1,
                data_seed=42,solver_seed=2025,instances=3,n_cities=7,workers=1,
                solver_seconds=1,generator_seconds=3,reference_seconds=15,
                reference_gap_pct=.001,diagnostic=True)
            model=DesignerFake();measure=DataDesigner.measure_case;calls=[]
            def interrupt(obj,*args):
                calls.append(1)
                if len(calls)==2:raise KeyboardInterrupt('interrupted batch')
                return measure(obj,*args)
            with patch.object(DataDesigner,'measure_case',interrupt):
                with self.assertRaises(KeyboardInterrupt):DataDesigner(c,model).run()
            with patch('cef.instance_design.independent_reference',wraps=independent_reference) as reference:
                self.assertEqual(DataDesigner.recover(root/'data')['status'],'finished')
            self.assertEqual(reference.call_count,2)
            self.assertEqual(len(model.inputs),1)

    def test_interrupted_recovery_reuses_newly_completed_checkpoint(self):
        from cef.store import Archive
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);solver=root/'solver.py';solver.write_text(SOLVER)
            c=dict(output_dir=str(root/'data'),solver_source=str(solver),iterations=1,
                data_seed=42,solver_seed=2025,instances=3,n_cities=7,workers=1,
                solver_seconds=1,generator_seconds=3,reference_seconds=15,
                reference_gap_pct=.001,diagnostic=True)
            calls=[]
            def two_failures(*args):
                calls.append(1)
                if len(calls)<=2:raise RuntimeError('TimeoutError: reference deadline before Hamiltonian tour')
                return independent_reference(*args)
            with patch('cef.instance_design.independent_reference',side_effect=two_failures):
                with self.assertRaises(RuntimeError):DataDesigner(c,DesignerFake()).run()
            measure=DataDesigner.measure_case;measured=[]
            def interrupt(obj,*args):
                measured.append(1)
                if len(measured)==2:raise KeyboardInterrupt('simulated power interruption')
                return measure(obj,*args)
            with patch.object(DataDesigner,'measure_case',interrupt):
                with self.assertRaises(KeyboardInterrupt):DataDesigner.recover(root/'data')
            cases={r['id']:r for r in Archive(root/'data').of('data_case')}
            self.assertEqual(sum(r['valid'] for r in cases.values()),2)
            with patch('cef.instance_design.independent_reference',wraps=independent_reference) as reference:
                self.assertEqual(DataDesigner.recover(root/'data')['status'],'finished')
            self.assertEqual(reference.call_count,1)

    def test_recovery_fork_keeps_original_run_and_freezes_explicit_runtime(self):
        from cef.data_recovery import recover_run
        from cef.cli import freeze_runtime
        from cef.config import validate
        from cef.store import Archive
        from scripts.cef_alternating_launcher import build_auto_config
        from runCEFAuto import CONFIG
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'original';source.mkdir();target=root/'recovered'
            c,test=build_auto_config(dict(CONFIG,rounds=10,integrated_feedback_v1=False,
                measured_research_loop_v1=False,exploration_efficiency_v1=False),source,2025)
            (source/'search_config.json').write_text(json.dumps(c),encoding='utf-8')
            (source/'test_config.json').write_text(json.dumps(test),encoding='utf-8')
            archive=Archive(source/'search');archive.add('manifest',c)
            archive.add('round_complete',dict(round=1))
            data=source/'rounds/002/data';data.mkdir(parents=True)
            archive.add('data_round_start',dict(round=2,path=str(data)))
            (source/'data_agent').mkdir();(source/'data_agent/facts.jsonl').write_text('')
            (source/'search/result.json').write_text(json.dumps(dict(status='failed',reason='no valid measured generator')))
            freeze_runtime(c['run_dir'],c)
            before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            result=dict(status='finished',supplement_manifest=str(CONFIG['initial_supplement']))
            with patch('cef.data_recovery.DataDesigner.recover',return_value=result) as recover:
                out=recover_run(source,target)
            self.assertEqual(out,target)
            recover.assert_called_once_with(target/'rounds/002/data')
            self.assertTrue(all(Path(p).read_bytes()==b for p,b in before.items()))
            current=json.loads((target/'search_config.json').read_text())
            self.assertEqual(validate(current)['run_dir'],str(target/'search'))
            journal=Archive(target/'search')
            self.assertEqual(journal.of('manifest')[0],current)
            self.assertEqual(len(journal.of('round_complete')),1)
            self.assertTrue(journal.of('runtime_repair_fork'))
            self.assertTrue((target/'search/runtime_snapshot_before_repair/cef/reference.py').exists())
            self.assertEqual(journal.of('data_round_start')[0]['path'],str(target/'rounds/002/data'))
            retry=root/'recovered_again'
            with patch('cef.data_recovery.DataDesigner.recover',return_value=result) as resumed:
                self.assertEqual(recover_run(target,retry),retry)
            resumed.assert_called_once_with(retry/'rounds/002/data')
            # A round-2 repair is ancestry after the search advances to round 4.
            from cef.data import receipt
            journal.add('round_complete',dict(round=2))
            journal.add('round_complete',dict(round=3))
            journal.add('data_round_complete',dict(round=2,manifest=receipt(CONFIG['initial_supplement'])))
            (target/'rounds/004/data').mkdir(parents=True)
            journal.add('data_round_start',dict(round=4,path=str(target/'rounds/004/data')))
            with patch('cef.data_recovery.DataDesigner.recover',return_value=result):
                next_run=recover_run(target,root/'round4')
            self.assertEqual(Archive(next_run/'search').of('runtime_repair_fork')[-1]['pending_round'],4)

    def test_recovery_cli_never_enters_provider_launcher(self):
        from scripts.cef_alternating_launcher import main
        with patch('cef.data_recovery.recover_run',return_value=Path('recovered')) as recover,patch(
                'scripts.cef_alternating_launcher.balance_snapshot',side_effect=AssertionError('No network')):
            self.assertEqual(main({},['--recover-data','original']),0)
            recover.assert_called_once_with(Path('original'))

    def test_recovery_cli_does_not_call_pending_data_finished(self):
        from scripts.cef_alternating_launcher import main
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'recovery_provenance.json').write_text('{"pending_round":4}')
            stream=io.StringIO()
            with patch('cef.data_recovery.recover_run',return_value=root),redirect_stdout(stream):
                self.assertEqual(main({},['--recover-data','original']),0)
            self.assertIn('Reference work pending',stream.getvalue())

    def test_known_reference_failure_reuses_successes_and_never_calls_model(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);solver=root/'solver.py';solver.write_text(SOLVER)
            c=dict(output_dir=str(root/'data'),solver_source=str(solver),iterations=1,
                data_seed=42,solver_seed=2025,instances=3,n_cities=7,workers=1,
                solver_seconds=1,generator_seconds=3,reference_seconds=15,
                reference_gap_pct=.001,diagnostic=True)
            model=DesignerFake();designer=DataDesigner(c,model);calls=[]
            def first_failure(*args):
                calls.append(1)
                if len(calls)==1:raise RuntimeError('TimeoutError: reference deadline before Hamiltonian tour')
                return independent_reference(*args)
            with patch('cef.instance_design.independent_reference',side_effect=first_failure):
                with self.assertRaisesRegex(RuntimeError,'no valid measured generator'):designer.run()
            failed=(root/'data/iteration_000/evaluation.json').read_bytes()
            rows=[x for x in designer.archive.of('data_case') if x['valid']]
            preserved={r['row']['path']:Path(r['row']['path']).read_bytes() for r in rows}
            with patch('cef.instance_design.independent_reference',wraps=independent_reference) as reference:
                result=DataDesigner.recover(root/'data')
            self.assertEqual(reference.call_count,1)
            self.assertEqual(len(model.inputs),1)
            self.assertEqual(result['status'],'finished')
            self.assertEqual((root/'data/iteration_000/evaluation.json').read_bytes(),failed)
            self.assertTrue(all(Path(p).read_bytes()==b for p,b in preserved.items()))
            manifest=json.loads(Path(result['supplement_manifest']).read_text())
            self.assertEqual(len(manifest['train']),3)
            observation=json.loads(Path(manifest['evaluation_record']['path']).read_text())
            self.assertTrue(observation['valid'])
            self.assertEqual(len(observation['per_instance']),3)

    def test_unknown_or_invalid_generator_is_not_retried_as_reference_failure(self):
        class Invalid:
            def call(self,*args):return dict(generator_source='def generate(n_cities,seed,parameters):\n return [[0,0]]*n_cities',parameters={})
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);solver=root/'solver.py';solver.write_text(SOLVER)
            c=dict(output_dir=str(root/'data'),solver_source=str(solver),iterations=1,
                data_seed=42,solver_seed=2025,instances=1,n_cities=7,workers=1,
                solver_seconds=1,generator_seconds=3,reference_seconds=15,
                reference_gap_pct=.001,diagnostic=True)
            with self.assertRaises(RuntimeError):DataDesigner(c,Invalid()).run()
            with self.assertRaisesRegex(ValueError,'reference'):
                DataDesigner.recover(root/'data')
