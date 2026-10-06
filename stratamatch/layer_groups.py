"""Group disconnected same-lithology parts without painting connecting pixels.

The geometric groups are review proposals, not inferred stratigraphic ages.
Endpoint matching rejects side-by-side parallel beds and ambiguous alternatives.
"""
from __future__ import annotations

import cv2
import numpy as np


def _descriptor(instances, record):
    x,y,w,h=record['bbox'];mask=instances[y:y+h,x:x+w]==record['instance']
    yy,xx=np.nonzero(mask);points=np.column_stack((xx+x,yy+y)).astype(float)
    centre=points.mean(axis=0)
    sample=points[::max(1,len(points)//12000)]
    values,vectors=np.linalg.eigh(np.cov(sample.T))
    axis=vectors[:,-1]
    if axis[1]<0:axis=-axis
    normal=np.array([-axis[1],axis[0]])
    along=(sample-centre)@axis;across=(sample-centre)@normal
    lo,hi=np.percentile(along,[2,98]);length=max(1.,hi-lo)
    width=max(2.,float(np.percentile(across,95)-np.percentile(across,5)))
    # 沿区域切片拟合中心线，端部方向取局部切线，允许地层本身弯曲。
    centres=[];widths=[]
    boundaries=np.linspace(lo,hi,19)
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        valid=(along>=left)&(along<=right)
        if np.count_nonzero(valid)<3:continue
        centres.append([float(np.median(along[valid])),float(np.median(across[valid]))])
        widths.append(max(2.,float(np.percentile(across[valid],90)-np.percentile(across[valid],10))))
    line=np.asarray(centres);endpoints=[];tips=[]
    for lower in (True,False):
        section=line[:5] if lower else line[-5:]
        local_widths=widths[:5] if lower else widths[-5:]
        sign=-1 if lower else 1
        if len(section)>=2 and np.ptp(section[:,0])>1:
            slope,intercept=np.polyfit(section[:,0],section[:,1],1)
            end_t=lo if lower else hi
            endpoint=centre+axis*end_t+normal*(slope*end_t+intercept)
            tangent=sign*(axis+normal*slope);tangent/=np.linalg.norm(tangent)
            local_width=float(np.median(local_widths))
        else:
            endpoint=centre+axis*(lo if lower else hi);tangent=sign*axis;local_width=width
        endpoints.append(endpoint)
        tips.append({'point':endpoint,'direction':tangent,'width':max(2.,local_width)})
    centreline=centre+line[:,0,None]*axis+line[:,1,None]*normal if len(line) else centre[None,:]
    tangents=np.gradient(centreline,axis=0) if len(centreline)>1 else axis[None,:]
    tangents/=np.maximum(np.linalg.norm(tangents,axis=1,keepdims=True),1e-8)
    return {'id':record['instance'],'lith':record['lithology_index'],
            'centre':centre,'axis':axis,'width':width,'length':length,
            'elongation':float(values[-1]/max(values[0],1.)),
            'ends':endpoints,'tips':tips,'sample':sample,'centreline':centreline,
            'tangents':tangents,'width_profile':np.asarray(widths if widths else [width])}


def _continuation(a,b,ai,bi,diagonal,instances,labels,domain,same_ids):
    """Evaluate a smooth continuation through an interruption, not a fixed gap cutoff."""
    ta,tb=a['tips'][ai],b['tips'][bi]
    first,last=ta['point'],tb['point'];delta=last-first;gap=float(np.linalg.norm(delta))
    if gap<1 or gap>diagonal*.48:return None
    direction=delta/gap;ua,ub=ta['direction'],tb['direction']
    forward_a=float(ua@direction);forward_b=float(ub@-direction)
    # 两端必须大致朝向中断区。侧向并排的同色层不能仅因距离近就合组。
    if min(forward_a,forward_b)<.25:return None
    dot=abs(float(a['axis']@b['axis']))
    pa=a['sample']@a['axis'];pb=b['sample']@a['axis']
    al,ah=np.percentile(pa,[2,98]);bl,bh=np.percentile(pb,[2,98])
    overlap=max(0.,min(ah,bh)-max(al,bl))
    if dot>.90 and overlap>.65*min(ah-al,bh-bl):return None
    width=max(3.,(ta['width']+tb['width'])/2)
    # 两端外推至缺口中间，横向偏差与当地厚度和缺口共同归一化。
    mismatch=float(np.linalg.norm(first+ua*(gap*.5)-last-ub*(gap*.5)))
    allowance=width*1.4+gap*.42
    if mismatch>allowance*1.35:return None
    ratio=max(ta['width'],tb['width'])/max(2.,min(ta['width'],tb['width']))
    # 三次 Hermite 曲线仅用于判别续接走廊，不写入图像、不覆盖穿插地层。
    t=np.linspace(0,1,max(12,int(gap)))[:,None]
    curve=(2*t**3-3*t**2+1)*first+(t**3-2*t**2+t)*ua*gap+(-2*t**3+3*t**2)*last+(t**3-t**2)*(-ub)*gap
    h,w=instances.shape;path=np.rint(curve).astype(int)
    inside=(path[:,0]>=0)&(path[:,0]<w)&(path[:,1]>=0)&(path[:,1]<h)
    if inside.mean()<.95:return None
    px=np.clip(path[:,0],0,w-1);py=np.clip(path[:,1],0,h-1)
    if domain is not None and np.mean(~domain[py,px])>.12:return None
    between=instances[py,px];values,counts=np.unique(between,return_counts=True)
    if any(int(v) in same_ids and int(v) not in (a['id'],b['id']) and count>max(3,.035*len(path)) for v,count in zip(values,counts)):return None
    other_fraction=0.
    if labels is not None:
        corridor=labels[py,px];other_fraction=float(np.mean((corridor>=0)&(corridor!=a['lith'])))
    # 厚穿插层有颜色证据时降低远距离惩罚；方向、宽度差异保留为软代价。
    distance_cost=gap/(diagonal*(.22+.30*other_fraction))
    score=.49*mismatch/allowance+.26*(1-(forward_a+forward_b)/2)+.25*min(1.5,distance_cost)
    return {'left_tip':ai,'right_tip':bi,'gap_pixels':round(gap,2),
        'offset_pixels':round(mismatch,2),'angle_degrees':round(float(np.degrees(np.arccos(np.clip(ua@-ub,-1,1)))),2),
        'width_ratio':round(ratio,2),'intervening_other_lithology_fraction':round(other_fraction,3),
        'score':round(float(score),5),'curve_work_pixels':curve[::max(1,len(curve)//30)].round(2).tolist(),
        'decision':'candidate'}


def group_layer_parts(instances, records, lithology_labels=None, domain=None, config=None):
    """Return X.M membership and evidence; never mutate the input region map."""
    cfg=config or {};h,w=instances.shape;diagonal=float(np.hypot(h,w))
    descriptors=[_descriptor(instances,r) for r in records]
    from .interruption_groups import interruption_candidates
    candidates,topology_audit=interruption_candidates(instances,descriptors,cfg)
    same_ids={lith:{d['id'] for d in descriptors if d['lith']==lith} for lith in {d['lith'] for d in descriptors}}
    topology_pairs={(e['left_index'],e['right_index']) for e in candidates}
    for edge in candidates:
        if not edge.get('contact_evidence_sufficient',True):
            a,b=descriptors[edge['left_index']],descriptors[edge['right_index']]
            fallback=[_continuation(a,b,ai,bi,diagonal,instances,lithology_labels,domain,same_ids[a['lith']])
                      for ai in (0,1) for bi in (0,1)]
            fallback=[e for e in fallback if e is not None and e['score']<.4]
            if fallback:
                best=min(fallback,key=lambda e:e['score'])
                edge['score']=best['score'];edge['evidence_mode']='direction_supported_neighbours_unavailable'
                edge['left_tip']=best['left_tip'];edge['right_tip']=best['right_tip']
                edge['contact_evidence_sufficient']=True
    for i,a in enumerate(descriptors):
        for j in range(i+1,len(descriptors)):
            b=descriptors[j]
            if a['lith']!=b['lith']:continue
            if (i,j) in topology_pairs:continue
            proposals=[]
            for ai in (0,1):
                for bi in (0,1):
                    edge=_continuation(a,b,ai,bi,diagonal,instances,lithology_labels,domain,same_ids[a['lith']])
                    if edge is not None:proposals.append(edge)
            if not proposals:continue
            edge=min(proposals,key=lambda e:e['score'])
            edge['evidence_mode']='direct_geometric_continuation'
            edge['cut_layer_count']=0
            candidates.append({**edge,'left_index':i,'right_index':j,'left_part':a['id'],'right_part':b['id']})
    # 若原图有可提取的红色 F 线，作为额外切断边界候选；无整套支持的关系不会合组。
    virtual_descriptors=[];fault_audit={'mode':'no annotated line source supplied','lines':[]}
    if cfg.get('annotated_fault_image'):
        from .fault_lines import detect_annotated_fault_lines,fault_line_candidates
        if 'annotated_fault_lines' in cfg:
            lines=[(np.asarray(p,float),np.asarray(q,float)) for p,q in cfg['annotated_fault_lines']]
            detection=cfg.get('annotated_fault_detection',{})
        else:
            lines,detection=detect_annotated_fault_lines(cfg['annotated_fault_image'],cfg['roi'],instances.shape)
        fault_edges,virtual_descriptors,line_info=fault_line_candidates(lines,instances,descriptors)
        existing={tuple(sorted((e['left_part'],e['right_part']))) for e in candidates}
        for edge in fault_edges:
            pair=tuple(sorted((edge['left_part'],edge['right_part'])))
            if pair not in existing:
                candidates.append(edge);existing.add(pair)
        fault_audit={'mode':'red annotation candidate extraction','detection':detection,
                     'lines':line_info,'new_pair_candidates':sum(e['evidence_mode']=='annotated_fault_line' for e in candidates)}
    # 先对切断带两侧整套地层估计共同表观位移，再逐次确定相容对应。
    from .band_registration import register_bands
    band_audit=register_bands(candidates,descriptors,cfg,virtual_descriptors)
    # 先保留跨切断带的竞争解释；每轮只确认一对，之后才固定传递性地层组。
    from .hypothesis_layer_matching import select_hypothesis_groups
    memberships,selected_pairs,global_audit=select_hypothesis_groups(
        candidates,descriptors,band_audit,cfg,lithology_labels,virtual_descriptors)
    groups={}
    for i in range(len(records)):groups.setdefault(memberships[i],[]).append(i)
    numbered=[];counts={}
    ordered=sorted(groups.values(),key=lambda inds:(records[inds[0]]['lithology_index'],
        min(descriptors[i]['centre'][1] for i in inds),min(descriptors[i]['centre'][0] for i in inds)))
    for inds in ordered:
        lith=records[inds[0]]['lithology_index'];counts[lith]=counts.get(lith,0)+1
        codes=cfg.get('lithology_codes');code=codes[lith] if codes is not None else lith+1
        numbered.append({'label':f'{code}.{counts[lith]}','lithology_index':lith,'lithology_code':code,
            'group_number':counts[lith],'parts':[records[i]['instance'] for i in inds],
            'status':'geometric_group_requires_review' if len(inds)>1 else 'single_part'})
    return {'groups':numbered,'pairs':candidates,'part_count':len(records),'group_count':len(numbered),
        'multipart_group_count':sum(len(g['parts'])>1 for g in numbered),
        'method':'local_contact_sequence_global_hypothesis_matching',
        'topology_search':topology_audit,
        'band_registration':band_audit,
        'global_matching':global_audit,
        'fault_annotation':fault_audit,
        'parameters':{'direction_model':'local curved axes','cut_candidate_angle_degrees':45},
        'pixel_geometry_changed':False,'cross_section_correspondence':False}


def recognize_layer_groups(labels, domain, config=None):
    """Recover small annotation gaps only for connectivity; output retains observed pixels."""
    cfg=config or {};h,w=labels.shape
    radius=int(cfg.get('fragment_connection_radius',3))
    minimum=int(cfg.get('fragment_min_area',max(30,round(domain.sum()*.000025))))
    instances=np.full(labels.shape,-1,np.int32);records=[]
    fault_barrier=np.zeros(labels.shape,np.uint8)
    for p,q in cfg.get('annotated_fault_lines',[]):
        p=tuple(np.rint(p).astype(int));q=tuple(np.rint(q).astype(int))
        cv2.line(fault_barrier,p,q,1,3,cv2.LINE_8)
    barrier=(fault_barrier>0)&domain
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    for lith in sorted(int(v) for v in np.unique(labels) if v>=0):
        observed=(labels==lith)&domain
        _,original_components,original_stats,_=cv2.connectedComponentsWithStats(
            observed.astype(np.uint8),8)
        virtual=cv2.morphologyEx(observed.astype(np.uint8),cv2.MORPH_CLOSE,kernel).astype(bool)
        # Connectivity repair cannot traverse another known lithology.
        virtual &= domain & ((labels<0)|(labels==lith)) & ~barrier
        count,cc,stats,_=cv2.connectedComponentsWithStats(virtual.astype(np.uint8),8)
        for k in range(1,count):
            x,y,bw,bh,area=map(int,stats[k])
            mask=(cc[y:y+bh,x:x+bw]==k)&observed[y:y+bh,x:x+bw]
            if mask.sum()<minimum:continue
            idx=len(records);instances[y:y+bh,x:x+bw][mask]=idx
            # 预留断层线附近像素的归属带，供形态描述符读取完整原始区域。
            pad=2 if barrier.any() else 0
            x0=max(0,x-pad);y0=max(0,y-pad);x1=min(w,x+bw+pad);y1=min(h,y+bh+pad)
            records.append({'instance':idx,'lithology_index':lith,'area_pixels':int(mask.sum()),
                            'bbox':[x0,y0,x1-x0,y1-y0]})
        if barrier.any():
            # 细薄层被断层线切成小片时，仍保留切分前已合格连通域的每一个颜色像素。
            valid_parent=(original_components>0)&(original_stats[original_components,cv2.CC_STAT_AREA]>=minimum)
            missing=observed&valid_parent&(instances<0)
            known=observed&(instances>=0)
            if missing.any() and known.any():
                from scipy import ndimage
                distance,nearest=ndimage.distance_transform_edt(~known,return_indices=True)
                # 限定在同一个切分前连通域内，防止越过其他地层给像素错误归属。
                restore=missing&(original_components==original_components[nearest[0],nearest[1]])
                instances[restore]=instances[nearest[0][restore],nearest[1][restore]]
            # 极端情况下原连通域全被分为小片，保留一个实例供人工复核。
            remaining=observed&valid_parent&(instances<0)
            for parent_id in np.unique(original_components[remaining]):
                if parent_id<=0:continue
                region=remaining&(original_components==parent_id)
                if not region.any():continue
                idx=len(records);instances[region]=idx
                records.append({'instance':idx,'lithology_index':lith,
                                'area_pixels':int(np.count_nonzero(region)),'bbox':[0,0,w,h]})
    if records:
        areas=np.bincount(instances[instances>=0],minlength=len(records))
        from scipy import ndimage
        boxes=ndimage.find_objects(instances+1)
        for record in records:
            idx=record['instance'];record['area_pixels']=int(areas[idx])
            box=boxes[idx]
            if box is not None:
                ys,xs=box;record['bbox']=[xs.start,ys.start,xs.stop-xs.start,ys.stop-ys.start]
    audit=group_layer_parts(instances,records,labels,domain,cfg)
    audit['fragment_connection_radius']=radius;audit['fragment_min_area']=minimum
    audit['fault_barrier_pixels']=int(np.count_nonzero(barrier))
    audit['unassigned_observed_pixels']=int(np.count_nonzero((labels>=0)&domain&(instances<0)))
    return instances,records,audit
