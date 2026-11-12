"""Only task input, seed, feasibility and objective live here; no edit policy."""
import math
from numbers import Integral
from pathlib import Path
import numpy as np

class TaskAdapter:
    def __init__(self,task,training_protocol=None):
        if task not in ('tsp','fssp'): raise ValueError('unknown task')
        self.task=task
        self.training_protocol=training_protocol
    @property
    def description(self):
        common='Implement the complete solve(instance, time_limit_s, seed) -> {"solution": permutation} module. All internal algorithms, functions and data structures may change; evaluator and resources may not. '
        if self.training_protocol:
            return common+'Symmetric Euclidean TSP. Search uses fixed original64 plus supplemental64. Minimize the arithmetic mean of all 128 per-instance reference-gap percentages; each group has equal weight. Original references are author-supplied Concorde costs; supplemental references are independently numerically bounded unrounded tours with recorded lower bounds. References and group identities are hidden from the solver. Group scores describe the measured tradeoff; original-group performance is not guaranteed to remain unchanged.'
        return common+('Symmetric TSP: instance.distance_matrix is a square nonnegative matrix. Minimize tour length; search fitness is mean gap percentage relative to Concorde reference, hidden from solver.' if self.task=='tsp' else 'Permutation flowshop: instance.processing_times[job][machine] is a nonnegative matrix. All jobs visit machines in order. Minimize mean raw makespan.')
    @property
    def fitness_metric(self):
        if self.training_protocol:
            return dict(name='mean_reference_gap_pct',unit='percent',direction='minimize',
                aggregation='arithmetic_mean; 64 original + 64 supplement, equal group weights',
                groups=['original','supplement'],reference='mixed author references and independently numerically bounded tours')
        return dict(name='mean_gap' if self.task=='tsp' else 'mean_makespan',
                    unit='percent' if self.task=='tsp' else 'processing_time', direction='minimize',
                    aggregation='arithmetic_mean')
    def format_fitness(self,value):
        if value is None: return None
        return format(value,'.10g')+('%' if self.task=='tsp' else '')
    def seed_source(self):
        return (Path(__file__).parent/'seeds'/f'{self.task}.py').read_text(encoding='utf-8')
    def load(self,row):
        if 'instance' in row: return row['instance']
        with np.load(row['path'],allow_pickle=False) as z:
            key='distance_matrix' if self.task=='tsp' else 'processing_times'
            arr=np.asarray(z[key],dtype=float)
        if not np.isfinite(arr).all() or (arr<0).any() or arr.ndim!=2: raise ValueError('invalid instance matrix')
        out={key:arr.tolist()}
        if 'reference' in row: out['reference']=row['reference']
        return out
    def solver_input(self,instance):
        return {k:v for k,v in instance.items() if k!='reference'}
    def objective(self,instance,solution):
        """Validate a permutation and recompute its objective without references."""
        key='distance_matrix' if self.task=='tsp' else 'processing_times'
        p=instance[key]; n=len(p)
        if not isinstance(solution,(list,tuple)) or len(solution)!=n or any(isinstance(x,bool) or not isinstance(x,Integral) for x in solution) or sorted(solution)!=list(range(n)):
            raise ValueError('solution must be a permutation of every index exactly once')
        if self.task=='tsp':
            objective=sum(p[solution[i]][solution[(i+1)%n]] for i in range(n))
        else:
            end=[0.]*len(p[0])
            for j in solution:
                for m in range(len(end)): end[m]=max(end[m],end[m-1] if m else 0)+p[j][m]
            objective=end[-1]
        if not math.isfinite(objective): raise ValueError('nonfinite objective')
        return float(objective)
    def score(self,instance,solution):
        objective=self.objective(instance,solution)
        if self.task=='tsp':
            ref=instance['reference']
            if ref<=0: raise ValueError('positive matched reference required')
            fitness=100*(objective/ref-1)
        else:
            fitness=objective
        return float(objective),float(fitness)
    def group(self,instance):
        return str(len(instance['distance_matrix'])) if self.task=='tsp' else f"{len(instance['processing_times'])}x{len(instance['processing_times'][0])}"
