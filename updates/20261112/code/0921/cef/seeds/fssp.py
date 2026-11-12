import time

def makespan(order,p):
    end=[0.0]*len(p[0])
    for j in order:
        for m in range(len(end)):
            end[m]=max(end[m],end[m-1] if m else 0.0)+p[j][m]
    return end[-1]

def solve(instance, time_limit_s, seed):
    p=instance['processing_times']
    deadline=time.monotonic()+time_limit_s
    order=sorted(range(len(p)),key=lambda j:-sum(p[j]))
    seq=[]
    for idx,job in enumerate(order):
        if time.monotonic()>=deadline-0.01:
            return {'solution':seq+order[idx:]}
        seq=min((seq[:k]+[job]+seq[k:] for k in range(len(seq)+1)),key=lambda s:makespan(s,p))
    return {'solution':seq}
