"""A trainable two-layer candidate-graph GCN, implemented with NumPy backprop.

Nodes are candidate hyperedges; shared layers and neighboring orders define edges.
Synthetic training never authorizes automatic real-world decisions.
"""
import numpy as np
from .io import local

def tensors(edges):
    x=np.array([e['features']+[len(e['s'])-1,len(e['t'])-1,1.] for e in edges],float)
    if not len(edges): return np.empty((0,14)),np.empty((0,0))
    adj=np.eye(len(edges))
    for i,a in enumerate(edges):
        for j in range(i):
            b=edges[j]
            if set(a['s'])&set(b['s']) or set(a['t'])&set(b['t']) or (abs(np.mean(a['s'])-np.mean(b['s']))<=1 and abs(np.mean(a['t'])-np.mean(b['t']))<=1):
                adj[i,j]=adj[j,i]=1
    d=1/np.sqrt(adj.sum(1))
    return np.clip(x,-3,3),d[:,None]*adj*d[None,:]

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-40,40)))

def forward(x,a,w1,w2):
    q=a@x; h=np.tanh(q@w1); r=a@h
    # Residual raw features retain distinctions between mutually exclusive neighbors.
    design=np.column_stack([r,x]); p=sigmoid(design@w2)
    return p,(q,h,design)

def gradients(x,a,y,w1,w2):
    p,(q,h,design)=forward(x,a,w1,w2)
    dz=(p-y)/max(1,len(y)); dw2=design.T@dz
    dh=a.T@(dz[:,None]*w2[:w1.shape[1]][None,:])
    dw1=q.T@(dh*(1-h*h))
    loss=-np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12))
    return loss,dw1,dw2

def train(graphs,path,epochs=250,seed=719,domain='synthetic_only'):
    rng=np.random.default_rng(seed); w1=rng.normal(0,.15,(14,16)); w2=rng.normal(0,.15,30)
    history=[]
    for epoch in range(epochs):
        loss=0.; g1=np.zeros_like(w1); g2=np.zeros_like(w2)
        for edges,y in graphs:
            x,a=tensors(edges)
            if not len(x): continue
            v,d1,d2=gradients(x,a,np.asarray(y),w1,w2); loss+=v; g1+=d1; g2+=d2
        w1-=.12*(g1/max(len(graphs),1)+1e-4*w1); w2-=.12*(g2/max(len(graphs),1)+1e-4*w2)
        if epoch%25==0 or epoch==epochs-1: history.append({'epoch':epoch,'loss':float(loss/max(len(graphs),1))})
    p=local(path); p.parent.mkdir(parents=True,exist_ok=True)
    np.savez(p,w1=w1,w2=w2,domain=np.array(domain),schema=np.array(1),seed=np.array(seed))
    return history

def predict(edges,path):
    if not edges: return np.array([])
    with np.load(local(path),allow_pickle=False) as model:
        if int(model['schema'])!=1 or model['w1'].shape!=(14,16) or model['w2'].shape!=(30,): raise ValueError('GNN checkpoint schema mismatch')
        x,a=tensors(edges); p,_=forward(x,a,model['w1'],model['w2'])
    if not np.all(np.isfinite(p)): raise ValueError('Non-finite GNN predictions')
    return p
