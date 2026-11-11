import time

def solve(instance, time_limit_s, seed):
    d=instance['distance_matrix']; n=len(d)
    deadline=time.monotonic()+time_limit_s
    route=[0]; remaining=set(range(1,n))
    while remaining:
        j=min(remaining,key=lambda j:(d[route[-1]][j],j))
        route.append(j); remaining.remove(j)
    improved=True
    while improved and time.monotonic()<deadline-0.01:
        improved=False
        for i in range(n-2):
            for j in range(i+2,n if i else n-1):
                if time.monotonic()>=deadline-0.01: return {'solution':route}
                a,b=route[i],route[i+1]; c,e=route[j],route[(j+1)%n]
                if d[a][c]+d[b][e]<d[a][b]+d[c][e]-1e-12:
                    route[i+1:j+1]=reversed(route[i+1:j+1]); improved=True
    return {'solution':route}
