"""Traverse actual contacts through transverse units before scoring continuation.

Distances do not determine whether a graph path is a candidate. Cut count and
local normal thickness control the balance of direction and neighbourhood shape.
"""
from collections import deque
import cv2
import numpy as np
from scipy import ndimage
from .neighbour_sequences import neighbourhood, compare_neighbourhoods, short_axis_penalty


def axial_angle(a,b):
    return float(np.degrees(np.arccos(np.clip(abs(float(a@b)),0,1))))


def local_frame(d,point):
    idx=int(np.argmin(np.sum((d['centreline']-point)**2,axis=1)))
    return d['tangents'][idx],float(d['width_profile'][idx])


def interruption_weights(count,relative_thickness):
    """>=2 cutting units, or one thick unit, makes contact evidence dominant."""
    disturbance=max(.78 if count>=2 else .0,relative_thickness/(relative_thickness+2.))
    return {'direction':.60-.45*disturbance,'contact':.20+.30*disturbance,
            'morphology':.12+.13*disturbance,'position':.08+.02*disturbance}


def contact_graph(instances,descriptors):
    """Measure contact lengths and sample points; bridge only raster-width unlabeled gaps."""
    ids={d['id']:i for i,d in enumerate(descriptors)};graph={i:{} for i in range(len(descriptors))}
    if not ids:return graph
    work=instances.copy();unknown=work<0
    distance,nearest=ndimage.distance_transform_edt(unknown,return_indices=True)
    near=unknown&(distance<=2);work[near]=work[nearest[0][near],nearest[1][near]]
    buckets={}
    for dy,dx in ((1,0),(0,1)):
        a=work[:work.shape[0]-dy or None,:work.shape[1]-dx or None]
        b=work[dy:,dx:];valid=(a>=0)&(b>=0)&(a!=b)
        yy,xx=np.nonzero(valid)
        left=a[valid];right=b[valid];lo=np.minimum(left,right);hi=np.maximum(left,right)
        keys=lo.astype(np.int64)*(int(work.max())+1)+hi
        for key in np.unique(keys):
            selection=keys==key;u=int(lo[selection][0]);v=int(hi[selection][0])
            if u not in ids or v not in ids:continue
            pair=tuple(sorted((ids[u],ids[v])))
            points=np.column_stack((xx[selection]+dx*.5,yy[selection]+dy*.5))
            buckets.setdefault(pair,[]).append(points)
    for (i,j),blocks in buckets.items():
        points=np.concatenate(blocks);length=len(points)
        if length<3:continue
        edge={'length':length,'point':np.median(points,axis=0),
              'samples':points[::max(1,length//64)]}
        graph[i][j]=edge;graph[j][i]=edge
    return graph


def interruption_candidates(instances,ds,config=None):
    cfg=config or {};graph=contact_graph(instances,ds)
    # 默认走到恢复层序的出口；层数上限只防止超过本图实际节点数。
    max_layers=int(cfg.get('interruption_search_layers',max(1,len(ds))));budget=int(cfg.get('interruption_search_states',3000))
    candidates={};expanded=truncated=0;signature_cache={}
    def signature(index, excluded):
        key=(index,tuple(sorted(excluded)))
        if key not in signature_cache:
            signature_cache[key]=neighbourhood(index,excluded,graph,ds,local_frame,axial_angle,
                depth=int(cfg.get('neighbourhood_depth',4)))
        return signature_cache[key]
    for source,a in enumerate(ds):
        queue=deque();visits={};states=0
        for node,edge in graph[source].items():
            reference,_=local_frame(a,edge['point']);other,_=local_frame(ds[node],edge['point'])
            if ds[node]['lith']!=a['lith'] and axial_angle(reference,other)>45:
                queue.append(([source,node],reference))
        while queue:
            path,reference=queue.popleft();states+=1
            if states>budget:truncated+=1;break
            node=path[-1]
            if len(path)-1>max_layers:truncated+=1;continue
            # Keep a few alternate paths per node, rather than exponential walks.
            key=(node,round(float(np.arctan2(reference[1],reference[0]))/.3))
            if visits.get(key,0)>=3:continue
            visits[key]=visits.get(key,0)+1;expanded+=1
            for target,edge in graph[node].items():
                if target in path:continue
                b=ds[target];axis,_=local_frame(b,edge['point'])
                if b['lith']==a['lith']:
                    cuts=path[1:];points=[graph[u][v]['point'] for u,v in zip(path+[target],(path+[target])[1:])]
                    thickness=[];angles=[]
                    for k,cut in enumerate(cuts):
                        centre=(points[k]+points[k+1])/2
                        direction,width=local_frame(ds[cut],centre)
                        # Project traversed contact separation onto the cutting unit's normal.
                        normal=np.array([-direction[1],direction[0]])
                        crossed=abs(float((points[k+1]-points[k])@normal))
                        thickness.append(min(width*2,max(width*.25,crossed)))
                        angles.append(axial_angle(reference,direction))
                    _,aw=local_frame(a,points[0]);_,bw=local_frame(b,points[-1])
                    relative=sum(thickness)/max(2.,(aw+bw)/2)
                    weights=interruption_weights(len(cuts),relative)
                    excluded=set(cuts)
                    ac=signature(source,excluded);bc=signature(target,excluded)
                    contact,morphology,similarity,confidence,sequence_audit=compare_neighbourhoods(ac,bc)
                    # 邻域覆盖不足时降低其影响，剩余权重归一化；未知不是相似。
                    weights['contact']*=.35+.65*confidence
                    weights['morphology']*=.5+.5*confidence
                    weight_total=sum(weights.values())
                    weights={k:v/weight_total for k,v in weights.items()}
                    ta,_=local_frame(a,points[0]);tb,_=local_frame(b,points[-1]);delta=points[-1]-points[0]
                    direction_cost=axial_angle(ta,tb)/90
                    if np.linalg.norm(delta)>1:
                        unit=delta/np.linalg.norm(delta)
                        direction_cost=.5*direction_cost+.25*(1-abs(float(unit@ta)))+.25*(1-abs(float(unit@tb)))
                    # 两侧原始厚度可整体变化，不在这里直接惩罚宽度差。
                    shape_cost=morphology
                    # Long-axis and short-axis offsets are soft, scale-free evidence.
                    normal=np.array([-ta[1],ta[0]])
                    offset=abs(float(delta@normal));position=min(1.,offset/max(aw+bw+sum(thickness),1.))
                    side_penalty,axis_ratios=short_axis_penalty(a,b,b['centre']-a['centre'])
                    score=weights['direction']*direction_cost+weights['contact']*contact+weights['morphology']*shape_cost+weights['position']*position+.16*side_penalty
                    mode='contact_dominant' if weights['contact']+weights['morphology']>weights['direction'] else 'direction_dominant'
                    # Large disruption without usable neighbour evidence remains a review candidate.
                    evidence_ok=bool(confidence>=.12 and similarity>=.30) if mode=='contact_dominant' else True
                    ports=[]
                    for d,p in ((a,points[0]),(b,points[-1])):
                        axis,_=local_frame(d,p);v=p-d['centre'];normal=np.array([-axis[1],axis[0]])
                        if abs(float(v@axis))/max(d['length']*.5,1)>=abs(float(v@normal))/max(d['width']*.5,1):
                            ports.append(1 if v@axis>=0 else 0)
                        else:ports.append(3 if v@normal>=0 else 2)
                    candidate={'left_index':source,'right_index':target,'left_part':a['id'],'right_part':b['id'],
                        'left_tip':ports[0],'right_tip':ports[1],'score':round(score,5),'decision':'candidate',
                        'evidence_mode':mode,'contact_evidence_sufficient':evidence_ok,
                        'cut_layer_count':len(cuts),'cut_part_ids':[ds[n]['id'] for n in cuts],
                        'cut_lithologies':[ds[n]['lith'] for n in cuts],'cut_angles_degrees':angles,
                        'cut_thicknesses_pixels':thickness,'total_cut_thickness_pixels':sum(thickness),
                        'relative_cut_thickness':relative,'weights':weights,'contact_similarity':similarity,
                        'neighbour_confidence':confidence,'sequence_alignment':sequence_audit,
                        'short_long_ratios':axis_ratios,'short_axis_penalty':side_penalty,
                        'source_neighbours':ac,'target_neighbours':bc,
                        'costs':{'direction':direction_cost,'contact':contact,'morphology':shape_cost,'position':position},
                        'curve_work_pixels':[p.tolist() for p in points],
                        'gap_pixels':float(np.linalg.norm(delta)),'offset_pixels':offset,
                        'angle_degrees':axial_angle(ta,tb),'width_ratio':max(aw,bw)/max(2.,min(aw,bw))}
                    # 两个方向均搜索，避免区域编号顺序决定能否发现弯曲后的出口。
                    if source>target:
                        for suffix in ('index','part','tip'):
                            candidate['left_'+suffix],candidate['right_'+suffix]=candidate['right_'+suffix],candidate['left_'+suffix]
                        candidate['source_neighbours'],candidate['target_neighbours']=bc,ac
                        candidate['short_long_ratios']=axis_ratios[::-1]
                        for matched in candidate['sequence_alignment']['matched_layers']:
                            matched['left'],matched['right']=matched['right'],matched['left']
                        for field in ('cut_part_ids','cut_lithologies','cut_angles_degrees','cut_thicknesses_pixels','curve_work_pixels'):
                            candidate[field]=list(reversed(candidate[field]))
                    keypair=tuple(sorted((source,target)))
                    priority=(not evidence_ok,score)
                    previous=candidates.get(keypair)
                    if previous is None or priority<(not previous['contact_evidence_sufficient'],previous['score']):
                        candidates[keypair]=candidate
                elif axial_angle(reference,axis)>45:
                    queue.append((path+[target],reference))
                # Other parallel units mark restoration of bedding: do not skip past them.
    return list(candidates.values()),{'contact_edges':sum(len(v) for v in graph.values())//2,
        'expanded_states':expanded,'truncated_searches':truncated,'max_cut_layers':max_layers,
        'candidate_discovery':'contact traversal; no pixel-distance gate',
        'cut_count_definition':'distinct connected instance IDs along a contact path'}
