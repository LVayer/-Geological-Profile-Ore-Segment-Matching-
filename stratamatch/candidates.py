"""Shared same-lithology hyperedges, physical features and auditable costs."""
from itertools import combinations
import numpy as np

FEATURES=['position','thickness','area','shape','boundary','direction','order','upper','lower','contact','topology']
WEIGHTS=np.array([.25,.05,.03,.16,.08,.03,.04,.16,.16,.02,.02])

def feature_weights(config=None):
    """Read auditable soft-evidence weights; lithology is not a weight.

    Lithology is enforced separately as the only geological hard gate.  The
    remaining values are deliberately low priors until real reviewed pairs can
    be used for calibration.
    """
    config=config or {}
    values=np.array([float(config.get('weight_'+name,default)) for name,default in zip(FEATURES,WEIGHTS)])
    if not np.all(np.isfinite(values)) or np.any(values<0) or values.sum()<=0:
        raise ValueError('Soft evidence weights must be finite, non-negative, and not all zero')
    return values

def aggregate(layers):
    if len(layers)==1: return layers[0]
    a=layers[0].copy(); areas=np.array([l['area'] for l in layers]); extent=np.array([l['extent'] for l in layers])
    a['centroid']=np.average([l['centroid'] for l in layers],axis=0,weights=areas).tolist()
    a['area']=float(areas.sum()); a['extent']=[*extent[:,:2].min(axis=0),*extent[:,2:].max(axis=0)]
    a['quality']=min(l['quality'] for l in layers); a['order']=float(np.mean([l['order'] for l in layers]))
    a['dip']=float(np.average([l['dip'] for l in layers],weights=areas))
    lo,hi=a['extent'][0],a['extent'][2]; x=np.linspace(lo,hi,64)
    up=[]; down=[]
    for l in layers:
        u=np.array(l['upper_boundary']); d=np.array(l['lower_boundary'])
        iu=np.argsort(u[:,0]); idn=np.argsort(d[:,0])
        up.append(np.interp(x,u[iu,0],u[iu,1],left=np.nan,right=np.nan))
        down.append(np.interp(x,d[idn,0],d[idn,1],left=np.nan,right=np.nan))
    up=np.array(up); down=np.array(down); valid=np.any(np.isfinite(up)&np.isfinite(down),axis=0)
    x=x[valid]; up=up[:,valid]; down=down[:,valid]
    a['upper_boundary']=np.column_stack([x,np.nanmin(up,axis=0)]).tolist()
    a['lower_boundary']=np.column_stack([x,np.nanmax(down,axis=0)]).tolist()
    a['mean_thickness']=float(a['area']/max(hi-lo,1e-9))
    ids={l['id'] for l in layers}
    for k in ['upper_neighbors','lower_neighbors','other_contact_neighbors']:
        a[k]=sorted(set(v for l in layers for v in l[k])-ids)
    return a

def neighbor_lith(a,s,k):
    byid={l['id']:l for l in s['layers']}
    return {byid[i]['lithology'] for i in a[k]}

def jaccard(a,b):
    return 1-len(a&b)/len(a|b) if a|b else 0.

def pair_features(a,b,sa,sb):
    # Shared physical extent avoids silently registering unrelated normalized pictures.
    scale=max(sa['bounds'][3]-sa['bounds'][1],sb['bounds'][3]-sb['bounds'][1])
    def logratio(v,w): return min(abs(np.log(max(v,1e-9)/max(w,1e-9))),3.)
    pos=np.linalg.norm(np.array(a['centroid'])-b['centroid'])/scale
    boundaries=[]; shapes=[]
    for key in ['upper_boundary','lower_boundary']:
        pa=np.asarray(a[key]); pb=np.asarray(b[key]); t=np.linspace(0,1,64)
        aa=np.column_stack([np.interp(t,np.linspace(0,1,len(pa)),pa[:,i]) for i in range(2)])
        bb=np.column_stack([np.interp(t,np.linspace(0,1,len(pb)),pb[:,i]) for i in range(2)])
        boundaries.append(np.sqrt(np.mean((aa-bb)**2))/scale)
        shapes.append(np.sqrt(np.mean(((aa-aa.mean(0))-(bb-bb.mean(0)))**2))/scale)
    keys=['upper_neighbors','lower_neighbors','other_contact_neighbors']
    neigh=[jaccard(neighbor_lith(a,sa,k),neighbor_lith(b,sb,k)) for k in keys]
    topo=sum(abs(len(a[k])-len(b[k])) for k in keys)/max(1,sum(len(a[k])+len(b[k]) for k in keys))
    return np.array([pos,logratio(a['mean_thickness'],b['mean_thickness']),logratio(a['area'],b['area']),
        np.mean(shapes),np.mean(boundaries),abs(a['dip']-b['dip'])/90,
        abs(a['order']/max(1,len(sa['layers'])-1)-b['order']/max(1,len(sb['layers'])-1)),*neigh,topo])

def build(sa,sb,max_group=3,max_edges=1200,large_neighbors=2,weights=None):
    aa,bb=sa['layers'],sb['layers']; edges=[]; rejected=[]
    if not 1<=max_group<=4: raise ValueError('max_group must be in [1,4]')
    if not 1<=large_neighbors<=8:raise ValueError('large_neighbors must be in [1,8]')
    large_mode=max(len(aa),len(bb))>80
    def groups(layers):
        out=[(i,) for i in range(len(layers))]
        for n in range(2,max_group+1):
            # For large sections only adjacent layers may form a split/merge
            # group. Allowing skipped combinations grows quickly and has weak
            # geological support when hundreds of image fragments are present.
            near_groups=(tuple(range(start,start+n)) for start in range(len(layers)-n+1)) if large_mode else (
                previous+(end,) for end in range(len(layers)) for previous in combinations(range(max(0,end-max_group),end),n-1))
            for g in near_groups:
                ls=[layers[i] for i in g]
                if len({l['lithology'] for l in ls})==1 and g[-1]-g[0]<=max_group:
                    # Limit distant disconnected aggregation; do not combine arbitrary repeated units.
                    e=np.array([l['extent'] for l in ls]); gap=max(0,e[:,0].max()-e[:,2].min())
                    if gap<=.1*(sa['bounds'][2]-sa['bounds'][0]): out.append(g)
        return out
    ga=[(g,aggregate([aa[i] for i in g])) for g in groups(aa)]
    gb=[(h,aggregate([bb[j] for j in h])) for h in groups(bb)]
    pairs=[];mandatory=set()
    if large_mode:
        by_lith_a={};by_lith_b={}
        for i,(g,a) in enumerate(ga):by_lith_a.setdefault(a['lithology'],[]).append(i)
        for j,(h,b) in enumerate(gb):by_lith_b.setdefault(b['lithology'],[]).append(j)
        def nearest(source,target,reverse=False):
            for index in source:
                group,layer=(gb if reverse else ga)[index]
                choices=[]
                for other in target.get(layer['lithology'],[]):
                    other_group,other_layer=(ga if reverse else gb)[other]
                    if len(group)>1 and len(other_group)>1:continue
                    distance=float(np.linalg.norm(np.asarray(layer['centroid'])-other_layer['centroid']))
                    choices.append((distance,other))
                for _,other in sorted(choices)[:large_neighbors]:
                    pair=(other,index) if reverse else (index,other);pairs.append(pair)
                    if len(group)==1 and len((ga if reverse else gb)[other][0])==1:mandatory.add(pair)
        nearest(list(range(len(ga))),by_lith_b)
        nearest(list(range(len(gb))),by_lith_a,True)
        pairs=list(dict.fromkeys(pairs))
        rejected.append({'reason':'large_instance_candidate_pruning','status':'manual_review','relation_type':'uncertain',
            'source_instances':len(aa),'target_instances':len(bb),'candidate_neighbors':large_neighbors,
            'message':'同岩性近邻候选有界筛选已启用；所有结果需要人工复核'})
    else:pairs=[(i,j) for i in range(len(ga)) for j in range(len(gb))]
    valid=[]
    for gi,hi in pairs:
            g,a=ga[gi];h,b=gb[hi]
            if len(g)>1 and len(h)>1: continue # many-many is explicitly unresolved
            if a['lithology']!=b['lithology'] or a['lithology'] in ['UNKNOWN','',None]:
                if len(g)==len(h)==1: rejected.append({'source':aa[g[0]]['id'],'target':bb[h[0]]['id'],'reason':'lithology_hard_gate','status':'rejected','relation_type':'unrelated'})
                continue
            f=pair_features(a,b,sa,sb); cost=float(f@(WEIGHTS if weights is None else weights))
            flags=[]
            # Large geometric changes are review evidence, never exclusion:
            # faults, erosion and section direction can legitimately cause them.
            if f[0]>.45:flags.append('large_position_change')
            if f[4]>.4:flags.append('large_boundary_change')
            if f[1]>2.3:flags.append('large_thickness_change')
            for group,ls in [(g,aa),(h,bb)]:
                if any(ls[i]['lithology']!=ls[group[0]]['lithology'] for i in range(group[0],group[-1]+1)):
                    flags.append('group_spans_other_lithology')
                if len(group)>1:
                    ext=np.asarray([ls[i]['extent'] for i in group],float)
                    widths=np.maximum(ext[:,2]-ext[:,0],1e-9)
                    horizontal_overlap=max(0.,np.min(ext[:,2])-np.max(ext[:,0]))/np.min(widths)
                    heights=np.maximum(ext[:,3]-ext[:,1],1e-9)
                    vertical_separation=np.ptp([(ls[i]['centroid'][1]) for i in group])/np.mean(heights)
                    # Vertically stacked same-lithology beds are repeated units,
                    # not sufficient evidence of a lateral split/merge.
                    if horizontal_overlap>.25 and vertical_separation>.55:
                        flags.append('group_geometry_ambiguous_repeated_beds')
            valid.append({'id':0,'s':g,'t':h,'features':f.tolist(),'cost':cost+.12*(len(g)+len(h)-2),'quality':min(a['quality'],b['quality']),'flags':flags,
                '_mandatory':(gi,hi) in mandatory})
    if len(valid)>max_edges:
        if not large_mode:raise ValueError('Candidate budget exceeded; subdivide sections or reduce max_group')
        valid.sort(key=lambda e:(not e['_mandatory'],e['cost']))
        rejected.append({'reason':'candidate_budget_pruned','status':'manual_review','relation_type':'uncertain','discarded_count':len(valid)-max_edges,'retained_count':max_edges})
        valid=valid[:max_edges]
    for e in valid:
        e.pop('_mandatory',None);e['id']=len(edges);edges.append(e)
    return edges,rejected
