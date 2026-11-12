"""Trusted SCIP DFJ model, adapted from PySCIPOpt's official test_tsp.py.

Binary undirected edges, degree two, and lazy subtour elimination. No generated
algorithm code is executed here. Bounds are numerical, not exact arithmetic.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
from pyscipopt import Model,Conshdlr,quicksum,SCIP_RESULT


def matrix_identity(matrix):
    return hashlib.sha256(np.asarray(matrix,dtype='<f8').tobytes()).hexdigest()


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False),encoding='utf-8',newline='\n')
    temporary.replace(path)


def components(n,edges):
    adjacency=[[] for _ in range(n)]
    for i,j in edges:adjacency[i].append(j);adjacency[j].append(i)
    remaining=set(range(n));groups=[]
    while remaining:
        stack=[min(remaining)];remaining.remove(stack[0]);group=set(stack)
        while stack:
            for v in adjacency[stack.pop()]:
                if v in remaining:remaining.remove(v);group.add(v);stack.append(v)
        groups.append(group)
    return groups,adjacency


def tour_from_edges(n,edges):
    groups,adjacency=components(n,edges)
    if len(groups)!=1 or any(len(row)!=2 for row in adjacency):return None
    tour=[0];previous=-1
    while len(tour)<n:
        following=min(v for v in adjacency[tour[-1]] if v!=previous)
        if following in tour:return None
        previous=tour[-1];tour.append(following)
    return tour if 0 in adjacency[tour[-1]] else None


class SubtourHandler(Conshdlr):
    def __init__(self,variables,n,checkpoint,identity,saved):
        self.variables=variables;self.n=n;self.checkpoint=checkpoint;self.identity=identity
        self.cuts={tuple(s) for s in saved};self.last_save=0.
    def groups(self,solution=None,threshold=1e-8):
        edges=[(i,j) for (i,j),v in self.variables.items() if self.model.getSolVal(solution,v)>threshold]
        groups,_=components(self.n,edges)
        return groups if len(groups)>1 else []
    def persist(self,tour=None):
        value=dict(matrix_hash=self.identity,cuts=[list(s) for s in sorted(self.cuts)])
        if tour is not None:value['tour']=tour
        write_json(self.checkpoint,value);self.last_save=time.monotonic()
    def conscheck(self,constraints,solution,check_integrality,check_lp_rows,print_reason,completely,**results):
        return {'result':SCIP_RESULT.INFEASIBLE if self.groups(solution,.5) else SCIP_RESULT.FEASIBLE}
    def consenfolp(self,constraints,n_useful_conss,sol_infeasible):
        added=False
        for group in self.groups():
            other=set(range(self.n))-group
            subset=min((tuple(sorted(group)),tuple(sorted(other))),key=lambda x:(len(x),x))
            if 1<len(subset)<self.n-1 and subset not in self.cuts:
                self.model.addCons(quicksum(self.variables[i,j] for i,j in itertools.combinations(subset,2))<=len(subset)-1)
                self.cuts.add(subset);added=True
        if added and time.monotonic()-self.last_save>=2:self.persist()
        return {'result':SCIP_RESULT.CONSADDED if added else SCIP_RESULT.FEASIBLE}
    def consenfops(self,constraints,n_useful_conss,sol_infeasible,obj_infeasible):
        # getSolVal(None, var) reads the current LP *or pseudo* solution.
        # Enforce the same DFJ cuts in both states, including when numerical
        # trouble is the reason SCIP did not produce an LP solution.
        return self.consenfolp(constraints,n_useful_conss,sol_infeasible)
    def conslock(self,constraint,locktype,nlockspos,nlocksneg):pass


def solve(matrix,seconds,gap_pct,checkpoint):
    d=np.asarray(matrix,dtype=float);n=len(d);identity=matrix_identity(d)
    scale=float(d.max());normalized=(d*.5+d.T*.5)/scale
    saved={}
    if Path(checkpoint).is_file():
        saved=json.loads(Path(checkpoint).read_text(encoding='utf-8'))
        if saved.get('matrix_hash')!=identity:raise ValueError('SCIP checkpoint matrix changed')
    cuts=saved.get('cuts',[])
    for s in cuts:
        if len(s)!=len(set(s)) or not 1<len(s)<n-1 or any(type(v)is not int or not 0<=v<n for v in s):
            raise ValueError('invalid saved subtour cut')
    started=time.monotonic();model=Model('CEF_reference_TSP')
    model.hideOutput();model.setBoolParam('misc/allowstrongdualreds',False)
    model.setIntParam('parallel/maxnthreads',1);model.setIntParam('randomization/randomseedshift',2025)
    model.setRealParam('numerics/feastol',1e-9);model.setRealParam('numerics/epsilon',1e-12)
    model.setRealParam('limits/gap',min(1e-8,gap_pct/400.))
    if seconds is not None:model.setRealParam('limits/time',max(.001,float(seconds)))
    x={(i,j):model.addVar(vtype='B',obj=float(normalized[i,j])) for i,j in itertools.combinations(range(n),2)}
    for i in range(n):model.addCons(quicksum(x[min(i,j),max(i,j)] for j in range(n) if i!=j)==2)
    for s in cuts:model.addCons(quicksum(x[i,j] for i,j in itertools.combinations(s,2))<=len(s)-1)
    handler=SubtourHandler(x,n,checkpoint,identity,cuts)
    model.includeConshdlr(handler,'TSP','subtour elimination',chckpriority=-10,enfopriority=-10,needscons=False)
    # A real Hamiltonian incumbent helps the independent solver; it is never
    # accepted as a reference without the independent numerical dual-bound test.
    tour=saved.get('tour',list(range(n)))
    if sorted(tour)!=list(range(n)):raise ValueError('invalid saved Hamiltonian incumbent')
    solution=model.createSol();selected={tuple(sorted((tour[i],tour[(i+1)%n]))) for i in range(n)}
    for edge,variable in x.items():model.setSolVal(solution,variable,int(edge in selected))
    model.addSol(solution);handler.persist(tour)
    model.optimize();best=model.getBestSol();tour=None
    if best is not None:
        edges=[edge for edge,v in x.items() if model.getSolVal(best,v)>.5]
        tour=tour_from_edges(n,edges)
    handler.persist(tour)
    return dict(matrix_hash=identity,n=n,solution=tour,status=str(model.getStatus()),
                scale=scale,model_objective=float(model.getObjVal()) if tour is not None else None,
                normalized_lower_bound=float(model.getDualbound()),seconds=time.monotonic()-started,
                nodes=int(model.getNNodes()),subtour_cuts=len(handler.cuts),
                feasibility_tolerance=1e-9,epsilon=1e-12)
