"""Offline concurrency checks: no provider calls or formal experiment launches."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine
from cef.store import Archive
from test_cef_engine import config


class ParallelTests(unittest.TestCase):
    def test_bounded_pool_really_overlaps_and_keeps_association(self):
        from cef.parallel import completed_tasks
        barrier = threading.Barrier(3, timeout=5)
        active = 0
        peak = 0
        lock = threading.Lock()

        def work(item):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait()
            with lock:
                active -= 1
            return item * 7

        self.assertEqual(dict(completed_tasks(range(12), work, 3)), {i: i * 7 for i in range(12)})
        self.assertEqual(peak, 3)

    def test_parallel_evaluation_preserves_256_cases_seeds_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            c = config(d)
            c.update(evaluation_workers=4, max_case_runs=300, max_wallclock_s=120)
            c['data_manifest']['train'] = [dict(id=f'x{i}', split='train', instance={'processing_times': [[1, 2], [3, 4]]}) for i in range(64)]
            c['evaluation_seed_manifest'] = {f'x{i}': 100+i for i in range(64)}
            e = Engine(c, None)
            variants = {label: label for label in ('P', 'PA', 'PB', 'PAB')}
            for cid in variants:
                e.register(cid, 'def solve(instance, time_limit_s, seed):\n    return {"solution": [0, 1]}\n')
            barrier = threading.Barrier(4, timeout=5)
            calls = []
            lock = threading.Lock()

            def case(source, adapter, instance, seed, on_started):
                on_started()
                with lock:
                    calls.append((seed, threading.get_ident()))
                barrier.wait()
                return dict(valid=True, objective=6, fitness=6, started=True, elapsed=0.01, wallclock_s=0.02)

            e.ledger.reserve('batch', 256)
            with patch.object(e.runner, 'case', side_effect=case):
                scores, records = e.evaluate(variants, 'batch', 'm')
                e.evaluate(variants, 'batch', 'm')
            e.ledger.release('batch')
            self.assertEqual(scores, dict.fromkeys(variants, 6))
            self.assertEqual(len(calls), 256)
            self.assertEqual(len({thread for _, thread in calls}), 4)
            for rows in records.values():
                self.assertEqual([r['seed'] for r in rows], list(range(100, 164)))
                self.assertEqual([r['instance_id'] for r in rows], [f'x{i}' for i in range(64)])
            restored = Archive(d)
            self.assertEqual(len(restored.of('execution_start')), 256)
            self.assertEqual(len(restored.of('execution')), 256)
            self.assertEqual(len(restored.of('measurement')), 4)
            self.assertEqual([r['seq'] for r in restored.records], list(range(len(restored.records))))
            self.assertEqual(e.ledger.used('case_runs'), 256)
            self.assertFalse(e.ledger.reservations())

    def test_interrupted_execution_is_not_resubmitted(self):
        with tempfile.TemporaryDirectory() as d:
            c = config(d); c['evaluation_workers'] = 2
            e = Engine(c, None); e.register('P', e.adapter.seed_source())
            e.ledger.reserve('batch', 1)
            e.ledger.start_case('m:P:x', 'batch', 'P')
            with patch.object(e.runner, 'case', side_effect=AssertionError('must not rerun')):
                scores, records = e.evaluate({'P': 'P'}, 'batch', 'm')
            self.assertIsNone(scores['P'])
            self.assertIn('unknown', records['P'][0]['error'])
            self.assertEqual(e.ledger.used('case_runs'), 1)

    def test_real_isolated_workers(self):
        with tempfile.TemporaryDirectory() as d:
            c = config(d); c.update(evaluation_workers=2, max_wallclock_s=120)
            c['data_manifest']['train'] *= 1
            c['data_manifest']['train'] += [dict(id='y', split='train', instance={'processing_times': [[2, 1], [3, 2]]})]
            c['evaluation_seed_manifest']['y'] = 11
            e = Engine(c, None); e.register('P', e.adapter.seed_source())
            e.ledger.reserve('batch', 2)
            popen=subprocess.Popen
            launched=[]; barrier=threading.Barrier(2,timeout=5)
            def start(*args,**kwargs):
                process=popen(*args,**kwargs)
                launched.append(process)
                barrier.wait()
                self.assertTrue(all(p.poll() is None for p in launched))
                return process
            with patch('cef.runner.subprocess.Popen',side_effect=start):
                scores, records = e.evaluate({'P': 'P'}, 'batch', 'm')
            self.assertIsNotNone(scores['P'])
            self.assertTrue(all(r['valid'] for r in records['P']))
            self.assertEqual(e.ledger.used('case_runs'), 2)
            self.assertEqual(len({p.pid for p in launched}),2)
            self.assertTrue(all(p.poll() is not None for p in launched))

    def test_parallel_joint_is_evaluated_despite_invalid_single_edits(self):
        from cef.patch import compose,digest
        with tempfile.TemporaryDirectory() as d:
            c=config(d); c.update(evaluation_workers=4,max_wallclock_s=120)
            e=Engine(c,None)
            source='def solve(instance, time_limit_s, seed):\n    marker=0\n    return {"solution": [0,1]}\n'
            package=dict(parent_hash=digest(source),joint_reason='dependent fixture',edits=[
                dict(edit_id='A',purpose='use',prediction='joint',exact_replacements=[dict(old='marker=0',new='marker=helper()')]),
                dict(edit_id='B',purpose='define',prediction='joint',exact_replacements=[dict(old='return {"solution": [0,1]}',new='return {"solution": [0,1] if marker==1 else []}\n\ndef helper():\n    return 1')])])
            variants=compose(source,package)
            for cid,program in variants.items(): e.register(cid,program)
            e.ledger.reserve('batch',4)
            scores,records=e.evaluate(dict(zip(variants,variants)),'batch','m')
            self.assertIsNotNone(scores['P']); self.assertIsNotNone(scores['PAB'])
            self.assertIsNone(scores['PA']); self.assertIsNone(scores['PB'])
            self.assertEqual(len(e.archive.of('execution')),4)
            self.assertEqual(e.ledger.used('case_runs'),4)

    def test_preflight_reuse_ignores_only_solver_concurrency(self):
        from cef.preflight import identity, receipt_matches
        c = dict(environment_manifest=dict(python='p', dependencies={'openai': 'v'}, worker_resources=dict(concurrency=1, threads=1, memory_mb=1024)), model_snapshot='m', model_sampling={'reasoning_effort': 'high'})
        receipt = dict(passed=True, identity=identity(c))
        updated = copy.deepcopy(c); updated['environment_manifest']['worker_resources']['concurrency'] = 4
        self.assertTrue(receipt_matches(updated, receipt))
        self.assertEqual(receipt['identity']['environment_manifest']['worker_resources']['concurrency'], 1)
        for key, value in [('model_snapshot', 'other'), ('model_sampling', {'reasoning_effort': 'low'})]:
            changed = copy.deepcopy(updated); changed[key] = value
            self.assertFalse(receipt_matches(changed, receipt))
        updated['environment_manifest']['dependencies']['openai'] = 'other'
        self.assertFalse(receipt_matches(updated, receipt))

    def test_final_test_uses_frozen_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            module = Path(d)/'runtime_snapshot'/'cef'; module.mkdir(parents=True)
            (module/'__init__.py').write_text('', encoding='utf-8')
            (module/'cli.py').write_text('def main():\n    print("FROZEN_RUNTIME")\n    return 0\n', encoding='utf-8')
            entry = Path(__file__).resolve().parents[1]/'scripts'/'run_cef.py'
            test_config=Path(d)/'test.json'
            test_config.write_text(json.dumps({'instances':[{'id':'ta001'}]}),encoding='utf-8')
            proc = subprocess.run([sys.executable, str(entry), '--quiet', 'test', '--run-dir', d, '--test-config', str(test_config), '--output', 'unused'], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), 'FROZEN_RUNTIME')

    def test_parallel_final_test_keeps_ids_and_search_journal(self):
        from cef.reporting import final_test
        from cef.patch import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); a=Archive(root/'search')
            rows=[dict(id=f't{i}', reference=6, instance={'processing_times': [[1, 2], [3, 4]]}) for i in range(4)]
            c=config(str(a.root)); c.update(evaluation_workers=2, test_repeats=1)
            c['data_manifest']['test']=rows
            a.add('manifest',c)
            source='def solve(instance, time_limit_s, seed):\n    return {"solution": [0, 1]}\n'
            before=digest(a.path.read_text(encoding='utf-8'))
            barrier=threading.Barrier(2,timeout=5)
            def case(self,source,adapter,instance,seed):
                barrier.wait()
                return dict(valid=True, objective=6, fitness=6, seed=seed)
            with patch('cef.reporting.verify_export',return_value=(a,dict(status='finished',best={'source_hash':digest(source)}),source)), patch('cef.reporting.audit_test'), patch('cef.reporting.Runner.case',new=case):
                result=final_test(a.root,root/'test',rows,1,{f't{i}:0':i+10 for i in range(4)})
            self.assertTrue(result['valid']); self.assertTrue(result['search_unchanged'])
            self.assertEqual(result['count'],4); self.assertEqual(result['evaluation_workers'],2)
            records=[json.loads(line) for line in (root/'test/results.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual({r['instance_id']:r['seed'] for r in records},{f't{i}':i+10 for i in range(4)})
            self.assertEqual(before,digest(a.path.read_text(encoding='utf-8')))


if __name__ == '__main__':
    unittest.main()
