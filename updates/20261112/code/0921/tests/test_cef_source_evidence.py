"""Source evidence distinguishes defined names, returns and changed text."""
import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef import feedback
from cef.engine import Engine
from test_cef_common_start import settings


class SourceEvidenceTests(unittest.TestCase):
    def test_review_uses_complete_changes_without_duplicate_unified_diff(self):
        from cef.feedback import decision_evidence,source_comparison
        import copy
        comparison=source_comparison('p','c','def solve():\n    return 1\n','def solve():\n    return 2\n')
        incomplete=dict(comparison,changes_complete=False)
        payload={'recent_events':[{'source_comparisons':{'complete':comparison,'incomplete':incomplete}}]}
        original=copy.deepcopy(payload)
        view=decision_evidence(payload)['recent_events'][0]['source_comparisons']
        self.assertNotIn('diff',view['complete'])
        self.assertEqual(view['complete']['changes'],comparison['changes'])
        self.assertEqual(view['incomplete']['diff'],incomplete['diff'])
        self.assertEqual(payload,original)

    def test_document_exposes_real_scopes_and_return_expression(self):
        code='''def solve(instance,time_limit_s,seed,*,initial_solution=None,observer=None):
    incumbent = list(range(len(instance['distance_matrix'])))
    retained_counter = 0
    def read_counter():
        return retained_counter
    return {'solution': incumbent}
'''
        with tempfile.TemporaryDirectory() as folder:
            e=Engine(settings(folder),None);e.register('seed',code)
            doc=e.program_documents(['seed'])['seed']
            self.assertIn('symbols',doc)
            scope=next(s for s in doc['symbols']['scopes'] if s['name']=='solve')
            self.assertIn('retained_counter',scope['locals'])
            self.assertNotIn('ghost_counter',scope['locals'])
            function=next(f for f in doc['symbols']['functions'] if f['name']=='solve')
            self.assertEqual(function['returns'],["{'solution': incumbent}"])
            self.assertEqual(doc['source'],code)

    def test_actual_change_fragments_exclude_unchanged_context(self):
        before='def f(a,b):\n    both = [a,b]\n    return both\n'
        after='def f(a,b):\n    both = [a,b]\n    boundary = [a+1,b-1]\n    return both + boundary\n'
        view=feedback.source_comparison('p','c',before,after)
        self.assertIn('changes',view)
        added='\n'.join(c['added'] for c in view['changes'])
        self.assertIn('boundary = [a+1,b-1]',added)
        self.assertNotIn('both = [a,b]',added)
        self.assertTrue(view['changes_complete'])

    def test_failed_candidate_source_is_readable_when_symbol_compilation_fails(self):
        code="def solve(instance,time_limit_s,seed,*,initial_solution=None,observer=None):\n    nonlocal absent_binding\n    return {'solution': []}\n"
        with tempfile.TemporaryDirectory() as folder:
            e=Engine(settings(folder),None);e.register('seed',code)
            document=e.program_documents(['seed'])['seed']
            self.assertEqual(document['source'],code)
            self.assertIsNone(document['symbols'])
            self.assertIn('absent_binding',document['symbol_error'])


if __name__=='__main__':unittest.main()
