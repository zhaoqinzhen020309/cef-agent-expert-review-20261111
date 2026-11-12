"""Independent unrounded TSP references: binary edges and separated subtour cuts.

Bounds are numerical HiGHS bounds, not mathematical optimality certificates.
"""
import json
import math
import time

import numpy as np


def _tree_cuts(d):
    """Valid subtour constraints from all bipartitions of a minimum spanning tree.

    They constrain the relaxation before the first integer solve. In particular,
    thin geometric instances otherwise require many solve/separate restarts.
    No edges or feasible Hamiltonian tours are excluded.
    """
    n=len(d);used=np.zeros(n,dtype=bool);distance=np.full(n,np.inf)
    parent=np.full(n,-1,dtype=int);distance[0]=0.;order=[]
    for _ in range(n):
        v=int(np.argmin(np.where(used,np.inf,distance)))
        used[v]=True;order.append(v)
        improve=(~used)&(d[v]<distance)
        distance[improve]=d[v,improve];parent[improve]=v
    subtrees=[{v} for v in range(n)];cuts=[];known=set();whole=frozenset(range(n))
    for v in reversed(order[1:]):
        s=frozenset(subtrees[v]);subtrees[parent[v]].update(s)
        if len(s)>n//2:s=whole-s
        if 1<len(s)<n-1 and s not in known:cuts.append(s);known.add(s)
    return cuts


class ReferenceIncomplete(TimeoutError):
    """A valid reference is pending; saved cuts can be continued."""


def _violated_cuts(capacity):
    """Stoer-Wagner phases find violated connectivity cuts in a fractional tour."""
    w=np.maximum(np.asarray(capacity,dtype=float),0.).copy();n=len(w)
    active=list(range(n));groups=[{v} for v in active];out=[]
    while len(active)>1:
        added=set();weights=np.zeros(n);previous=None
        for step in range(len(active)):
            v=max((x for x in active if x not in added),key=lambda x:(weights[x],-x))
            if step==len(active)-1:
                if weights[v]<2.-1e-7:out.append(frozenset(groups[v]))
                w[previous,:]+=w[v,:];w[:,previous]=w[previous,:]
                groups[previous].update(groups[v]);active.remove(v)
            else:
                added.add(v);weights+=w[v];previous=v
    return out


def _feasible_tour(d, deadline):
    n=len(d);best=list(range(n));value=lambda t:float(sum(d[t[i],t[(i+1)%n]] for i in range(n)))
    score=value(best)
    for start in np.linspace(0,n-1,min(n,8),dtype=int):
        if time.monotonic()>=deadline:break
        unused=set(range(n));tour=[int(start)];unused.remove(int(start))
        while unused:
            v=min(unused,key=lambda x:(d[tour[-1],x],x));tour.append(v);unused.remove(v)
        for _ in range(n):
            if time.monotonic()>=deadline:break
            a=np.asarray(tour);next_a=np.roll(a,-1)
            change=d[a[:,None],a[None,:]]+d[next_a[:,None],next_a[None,:]]-d[a,next_a][:,None]-d[a,next_a][None,:]
            change[np.tril_indices(n,1)]=np.inf;change[0,n-1]=np.inf
            i,j=np.unravel_index(np.argmin(change),change.shape)
            if change[i,j]>=-1e-12:break
            tour[i+1:j+1]=reversed(tour[i+1:j+1])
        candidate=value(tour)
        if candidate<score:best=tour;score=candidate
    return best


def solve_tsp_reference(matrix, seconds=120, gap_pct=0.00001, checkpoint_path=None):
    from scipy import __version__ as scipy_version
    from scipy.optimize import Bounds, LinearConstraint, milp, linprog
    from scipy.sparse import coo_matrix
    from pathlib import Path
    import hashlib,os

    d=np.asarray(matrix,dtype=float);n=len(d)
    if (d.shape!=(n,n) or n<3 or not np.isfinite(d).all() or (d<0).any() or
            not np.allclose(d,d.T,atol=1e-12,rtol=0) or np.any(np.diag(d)!=0)):
        raise ValueError('finite symmetric nonnegative distance matrix required')
    if not math.isfinite(seconds) or seconds<=0 or not math.isfinite(gap_pct) or gap_pct<=0:
        raise ValueError('positive reference time and gap tolerance required')
    started=time.monotonic();deadline=started+seconds
    identity=hashlib.sha256(np.asarray(d,dtype='<f8').tobytes()).hexdigest()
    path=Path(checkpoint_path) if checkpoint_path else None
    state=json.loads(path.read_text(encoding='utf-8')) if path and path.exists() else None
    if state:
        if state.get('version')!='tsp_reference_cuts_v2' or state.get('matrix_hash')!=identity or state.get('n')!=n:
            raise ValueError('reference checkpoint matrix/version mismatch')
        if state['gap_pct']!=gap_pct:raise ValueError('reference checkpoint tolerance mismatch')
    else:
        state=dict(version='tsp_reference_cuts_v2',matrix_hash=identity,n=n,gap_pct=gap_pct,
            cuts=[sorted(c) for c in _tree_cuts(d)],lower_bound=0.,tour=list(range(n)),
            history=[],work_slices=0,work_seconds=0.,phase='lp')
    state['work_slices']+=1;prior_seconds=state['work_seconds'];cuts=[]
    for subset in state['cuts']:
        if len(set(subset))!=len(subset) or not 1<len(subset)<n-1 or any(type(i)is not int or not 0<=i<n for i in subset):
            raise ValueError('invalid reference checkpoint cut')
        cuts.append(frozenset(subset))
    known=set(cuts);edges=np.asarray([(i,j) for i in range(n) for j in range(i+1,n)],dtype=int)
    costs=d[edges[:,0],edges[:,1]];whole=frozenset(range(n))
    def persist():
        state['cuts']=[sorted(c) for c in cuts];state['work_seconds']=prior_seconds+time.monotonic()-started
        if path:
            path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
            with temp.open('w',encoding='utf-8',newline='\n') as stream:
                json.dump(state,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
            os.replace(temp,path)
    def pause(reason):
        state['history'].append(dict(phase='pending',reason=reason,subtour_cuts=len(cuts),
            lower_bound=state['lower_bound'],slice=state['work_slices']))
        persist();raise ReferenceIncomplete('reference work slice exhausted: '+reason)
    def add(subsets):
        count=0
        for subset in subsets:
            if len(subset)>n//2:subset=whole-subset
            if 1<len(subset)<n-1 and subset not in known:
                cuts.append(subset);known.add(subset);count+=1
        return count
    def objective(tour):return float(sum(d[tour[i],tour[(i+1)%n]] for i in range(n)))
    if sorted(state['tour'])!=list(range(n)):raise ValueError('invalid checkpoint feasible tour')
    if not math.isfinite(state['lower_bound']) or state['lower_bound']<0:raise ValueError('invalid checkpoint bound')
    if state['work_slices']==1:state['tour']=_feasible_tour(d,deadline)
    persist()
    while True:
        value=objective(state['tour']);lower=state['lower_bound']
        if lower>value+1e-7:raise ValueError('reference bound exceeds feasible tour')
        gap=0. if value==0 else 100*(value/min(lower,value)-1) if lower>0 else math.inf
        if gap<=gap_pct+1e-8:
            state['status']='finished';persist()
            return dict(solution=state['tour'],objective=value,lower_bound=min(lower,value),bound_gap_pct=gap,
                reference_kind='HiGHS_numerically_bounded_tour',backend='HiGHS; fractional connectivity separation + binary subtour cuts',
                scipy_version=scipy_version,distance_rule='unrounded Euclidean',seconds=state['work_seconds'],
                history=state['history'],work_slices=state['work_slices'],
                certificate='numerical MILP/LP bound; not a rigorous exact-arithmetic certificate')
        remaining=deadline-time.monotonic()
        if remaining<=0:pause('deadline before numerical gap acceptance')
        rr=list(np.r_[edges[:,0],edges[:,1]]);cc=list(np.r_[np.arange(len(edges)),np.arange(len(edges))]);vv=[1.]*len(rr)
        for index,subset in enumerate(cuts,n):
            mask=np.isin(edges[:,0],list(subset))&np.isin(edges[:,1],list(subset));indices=np.flatnonzero(mask)
            rr.extend([index]*len(indices));cc.extend(indices.tolist());vv.extend([1.]*len(indices))
        a=coo_matrix((vv,(rr,cc)),shape=(n+len(cuts),len(edges))).tocsc()
        upper=np.r_[np.full(n,2.),[len(c)-1 for c in cuts]]
        remaining=deadline-time.monotonic()
        if remaining<=0:pause('deadline while constructing constraints')
        if state['phase']=='lp':
            result=linprog(costs,A_eq=a[:n,:],b_eq=np.full(n,2.),A_ub=a[n:,:] if cuts else None,
                b_ub=upper[n:] if cuts else None,bounds=(0,1),method='highs',options=dict(time_limit=remaining))
            if result.status==1:pause('LP time limit')
            if result.status!=0:raise RuntimeError('reference LP: '+result.message)
            state['lower_bound']=max(state['lower_bound'],float(result.fun))
            capacity=np.zeros((n,n));capacity[edges[:,0],edges[:,1]]=result.x;capacity+=capacity.T
            fresh=add(_violated_cuts(capacity))
            state['history'].append(dict(phase='lp',status=int(result.status),lower_bound=state['lower_bound'],
                subtour_cuts=len(cuts),new_cuts=fresh,elapsed_s=prior_seconds+time.monotonic()-started,slice=state['work_slices']))
            if not fresh:state['phase']='integer'
            persist();continue
        result=milp(costs,integrality=np.ones(len(edges)),bounds=Bounds(0,1),
            constraints=LinearConstraint(a,np.r_[np.full(n,2.),np.full(len(cuts),-np.inf)],upper),
            options=dict(time_limit=remaining,mip_rel_gap=gap_pct/100))
        bound=getattr(result,'mip_dual_bound',None)
        if bound is not None and math.isfinite(bound):state['lower_bound']=max(state['lower_bound'],float(bound))
        if result.x is None:
            if result.status==1:pause('integer solve time limit without incumbent')
            raise RuntimeError('reference integer solve: '+result.message)
        if np.max(np.abs(result.x-np.rint(result.x)))>1e-5:raise ValueError('reference incumbent is not integral')
        adjacency=[[] for _ in range(n)]
        for (i,j),x in zip(edges,result.x):
            if x>.5:adjacency[i].append(int(j));adjacency[j].append(int(i))
        if any(len(v)!=2 for v in adjacency):raise ValueError('reference degree violation')
        components=[];unseen=set(range(n))
        while unseen:
            todo=[min(unseen)];component=set()
            while todo:
                v=todo.pop()
                if v in component:continue
                component.add(v);unseen.discard(v);todo.extend(w for w in adjacency[v] if w not in component)
            components.append(frozenset(component))
        if len(components)==1:
            tour=[0];previous=-1
            while len(tour)<n:
                v=tour[-1];nxt=next(w for w in adjacency[v] if w!=previous)
                if nxt in tour:raise ValueError('reference tour closes early')
                tour.append(nxt);previous=v
            if 0 not in adjacency[tour[-1]]:raise ValueError('reference tour is not closed')
            if objective(tour)<objective(state['tour']):state['tour']=tour
        elif not add(components):raise RuntimeError('reference separation made no progress')
        state['history'].append(dict(phase='integer',status=int(result.status),lower_bound=state['lower_bound'],
            components=len(components),subtour_cuts=len(cuts),elapsed_s=prior_seconds+time.monotonic()-started,slice=state['work_slices']))
        state['phase']='lp' if len(components)>1 else 'integer';persist()
        if result.status==1 and len(components)==1:
            # Recheck the certified gap before yielding this bounded work slice.
            if 100*(objective(state['tour'])/max(state['lower_bound'],1e-300)-1)>gap_pct+1e-8:pause('integer bound incomplete')


if __name__=='__main__':
    import sys
    request=json.loads(sys.stdin.read())
    try:output=dict(valid=True,result=solve_tsp_reference(**request))
    except ReferenceIncomplete as error:output=dict(valid=False,pending=True,error=type(error).__name__+': '+str(error))
    except Exception as error:output=dict(valid=False,pending=False,error=type(error).__name__+': '+str(error))
    print(json.dumps(output,allow_nan=False))


class ReferenceBackendFailure(RuntimeError):
    """Backend execution failed; not a normal incomplete numerical proof."""
    def __init__(self,message,error_type='native_failure'):
        super().__init__(message)
        self.error_type=error_type
