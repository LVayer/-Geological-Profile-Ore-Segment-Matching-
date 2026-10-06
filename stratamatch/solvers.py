"""Segmental DTW, sum-product Markov alignment, and hypergraph MILP."""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.special import logsumexp
from scipy.sparse import coo_matrix

GAP=.72

def sequence(edges,n,m,method='dtw',forbid=None):
    moves={}
    for e in edges:
        g,h=e['s'],e['t']
        if e['id']==forbid or tuple(range(g[0],g[-1]+1))!=g or tuple(range(h[0],h[-1]+1))!=h: continue
        c=e.get('model_cost',e['cost'])
        if method=='markov':
            # Semi-Markov duration prior plus contact/order emission penalties.
            c+=.18*(len(g)+len(h)-2)+.15*sum(e['features'][7:10])
        moves.setdefault((g[0],h[0]),[]).append(((g[-1]+1,h[-1]+1),c,e['id']))
    def transitions(i,j):
        out=list(moves.get((i,j),[]))
        if i<n: out.append(((i+1,j),GAP,None))
        if j<m: out.append(((i,j+1),GAP,None))
        return out
    best=np.full((n+1,m+1),np.inf); best[0,0]=0; back={}
    alpha=np.full_like(best,-np.inf); alpha[0,0]=0; temperature=.22
    for i in range(n+1):
        for j in range(m+1):
            for (u,v),c,eid in transitions(i,j):
                if best[i,j]+c<best[u,v]: best[u,v]=best[i,j]+c; back[u,v]=(i,j,eid)
                alpha[u,v]=np.logaddexp(alpha[u,v],alpha[i,j]-c/temperature)
    selected=[]; p=(n,m)
    while p!=(0,0):
        i,j,eid=back[p]
        if eid is not None: selected.append(eid)
        p=(i,j)
    posterior={}
    if method=='markov':
        beta=np.full_like(best,-np.inf); beta[n,m]=0
        for i in range(n,-1,-1):
            for j in range(m,-1,-1):
                tr=transitions(i,j)
                if tr: beta[i,j]=logsumexp([-c/temperature+beta[u,v] for (u,v),c,_ in tr])
                for (u,v),c,eid in tr:
                    if eid is not None: posterior[eid]=float(np.exp(alpha[i,j]-c/temperature+beta[u,v]-alpha[n,m]))
    return selected,float(best[n,m]),posterior

def crossing(a,b):
    # Only enforce order when both groups are vertically separated in layer order.
    return (max(a['s'])<min(b['s']) and min(a['t'])>max(b['t'])) or (max(b['s'])<min(a['s']) and min(b['t'])>max(a['t']))

def graph(edges,n,m,forbid=None):
    k=len(edges); size=k+n+m
    if size==0: return [],0.,{}
    c=np.array([e.get('model_cost',e['cost']) for e in edges]+[GAP]*(n+m))
    rr=[];cc=[]
    for z,e in enumerate(edges):
        for v in list(e['s'])+[n+j for j in e['t']]:rr.append(v);cc.append(z)
    for v in range(n+m):rr.append(v);cc.append(k+v)
    row_count=n+m
    for i,a in enumerate(edges):
        for j in range(i):
            b=edges[j]
            if not(set(a['s'])&set(b['s']) or set(a['t'])&set(b['t'])) and crossing(a,b):
                rr.extend([row_count,row_count]);cc.extend([i,j]);row_count+=1
    # Sparse constraints avoid quadratic rows times candidate count dense allocation.
    matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(row_count,size)).tocsc()
    low=np.r_[np.ones(n+m),np.zeros(row_count-n-m)];high=np.ones(row_count)
    ub=np.ones(size)
    if forbid is not None:
        for z,e in enumerate(edges):
            if e['id']==forbid: ub[z]=0
    result=milp(c,integrality=np.ones(size),bounds=Bounds(np.zeros(size),ub),constraints=LinearConstraint(matrix,low,high),options={'time_limit':15.,'mip_rel_gap':0.})
    if not result.success: raise RuntimeError('Graph optimizer did not prove optimality; manual review required')
    return [e['id'] for i,e in enumerate(edges) if result.x[i]>.5],float(result.fun),{}
