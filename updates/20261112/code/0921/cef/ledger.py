"""Durable multi-resource accounting. Reservations never hide actual charges."""
import time

class BudgetError(RuntimeError): pass

class BudgetLedger:
    def __init__(self,archive,limits,*,enforce_limits=True):
        self.archive,self.limits=archive,limits
        self.enforce_limits=enforce_limits
        if not archive.of('clock'): archive.add('clock',dict(start=time.time()))
        self.start=archive.of('clock')[0]['start']
    def used(self,resource):
        if resource=='wallclock_s': return time.time()-self.start
        charges=self.archive.of('charge')
        settled={r.get('hold_id') for r in charges}
        launches=sum(r['amount'] for r in self.archive.of('execution_start') if r.get('resource')==resource)
        return launches+sum(r['amount'] for r in charges if r['resource']==resource)+sum(r['amount'] for r in self.archive.of('hold') if r['resource']==resource and r['id'] not in settled)
    def reservations(self):
        out={}
        for r in self.archive.records:
            d=r['data']
            if r['kind']=='reserve': out[d['id']]=d['cases']
            elif r['kind']=='release': out.pop(d['id'],None)
            elif r['kind']=='charge' and d.get('reservation') in out and d['resource']=='case_runs': out[d['reservation']]-=d['amount']
            elif r['kind']=='execution_start' and d.get('reservation') in out and d.get('resource')=='case_runs': out[d['reservation']]-=d['amount']
        return out
    def remaining(self,resource):
        if not self.enforce_limits: return float('inf')
        rem=self.limits['max_'+resource]-self.used(resource)
        if resource=='case_runs': rem-=sum(self.reservations().values())
        return rem
    def check(self,resource,amount=0):
        if self.remaining('wallclock_s')<=0 or self.remaining(resource)<amount: raise BudgetError('budget exhausted: '+resource)
    def reserve(self,id,cases):
        self.check('case_runs',cases)
        if id in self.reservations(): raise BudgetError('duplicate reservation')
        self.archive.add('reserve',dict(id=id,cases=cases))
    def release(self,id): self.archive.add('release',dict(id=id))
    def charge(self,resource,amount,reservation=None,**details):
        if amount<0: raise ValueError('negative charge')
        self.archive.add('charge',dict(resource=resource,amount=amount,reservation=reservation,**details))
    def hold(self,resource,amount,id):
        self.check(resource,amount)
        self.archive.add('hold',dict(resource=resource,amount=amount,id=id))
    def start_case(self,id,reservation,candidate_id):
        with self.archive.lock:
            if any(r['execution_id']==id for r in self.archive.of('execution_start')): return
            if self.reservations().get(reservation,0)<1: raise BudgetError('no reserved case available')
            self.archive.add('execution_start',dict(execution_id=id,candidate_id=candidate_id,resource='case_runs',amount=1,reservation=reservation))
    def balance(self): return {k[4:]:(self.remaining(k[4:]) if self.enforce_limits else None) for k in self.limits if k.startswith('max_')}
