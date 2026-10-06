"""沿切断带联合估计表观位移，并逐次确认整套地层对应。"""
from collections import defaultdict
import numpy as np
from .neighbour_sequences import align_costs


def _width(d, point):
    line = d['centreline']
    idx = int(np.argmin(np.sum((line-point)**2, axis=1)))
    return max(2., float(d['width_profile'][min(idx, len(d['width_profile'])-1)]))


def _frame(d, point):
    idx = int(np.argmin(np.sum((d['centreline']-point)**2, axis=1)))
    return d['tangents'][idx]


def _median(values):
    return float(np.median(np.asarray(values, float)))


def _band(cut_id, edges, by_id):
    cutter = by_id[cut_id]
    axis = cutter['axis']
    normal = np.array([-axis[1], axis[0]])
    observations = defaultdict(list)
    usable = {}
    for edge in edges:
        u, v = edge['left_part'], edge['right_part']
        points = edge.get('curve_work_pixels') or []
        if len(points) < 2: continue
        pu, pv = np.asarray(points[0], float), np.asarray(points[-1], float)
        su = float((by_id[u]['centre']-cutter['centre']) @ normal)
        sv = float((by_id[v]['centre']-cutter['centre']) @ normal)
        if su*sv >= 0: continue
        if su > 0: u, v, pu, pv = v, u, pv, pu
        observations[('left', u)].append(pu)
        observations[('right', v)].append(pv)
        usable[(u, v)] = edge
    if not usable: return None
    positions = {key:np.median(points,axis=0) for key,points in observations.items()}
    order = lambda key: float((positions[key]-cutter['centre']) @ axis)
    left = sorted((p for side,p in positions if side=='left'), key=lambda p:order(('left',p)))
    right = sorted((p for side,p in positions if side=='right'), key=lambda p:order(('right',p)))
    return {'cut':cut_id, 'axis':axis, 'centre':cutter['centre'], 'left':left, 'right':right,
            'positions':positions, 'edges':usable}


def _hypotheses(band, by_id):
    """从岩性相同的多个接触候选推导位移；零位移始终是一个假设。"""
    axis = band['axis']; positions = band['positions']
    observations = [float((positions['right',v]-positions['left',u]) @ axis)
                    for (u,v),edge in band['edges'].items() if edge['score'] < .9]
    if not observations: return [0.]
    widths = [_width(by_id[p],positions[side,p]) for side,seq in
              (('left',band['left']),('right',band['right'])) for p in seq]
    tolerance = max(4., .6*_median(widths))
    values = [0.] + sorted(observations)
    clusters = []
    for value in values:
        if clusters and abs(value-_median(clusters[-1])) < tolerance: clusters[-1].append(value)
        else: clusters.append([value])
    centres = [_median(cluster) for cluster in clusters]
    # 保留覆盖最多候选的位移模式，也保留零位移供无走滑的穿切体比较。
    centres = sorted(centres, key=lambda x:(-sum(abs(v-x)<=tolerance for v in observations),abs(x)))[:16]
    if not any(abs(x)<1e-6 for x in centres): centres.append(0.)
    return centres


def _ratio_cost(u, v, anchors, band, by_id):
    """只用已成对的同岩性参考层；厚度比允许成比例变化。"""
    lith = by_id[u]['lith']; positions = band['positions']; axis = band['axis']
    refs = [(a,b) for a,b in anchors if a!=u and b!=v and
            by_id[a]['lith']==lith and by_id[b]['lith']==lith]
    if not refs: return None
    a,b = min(refs,key=lambda q:abs(float((positions['left',u]-positions['left',q[0]])@axis))+
              abs(float((positions['right',v]-positions['right',q[1]])@axis)))
    ru = _width(by_id[u],positions['left',u])/_width(by_id[a],positions['left',a])
    rv = _width(by_id[v],positions['right',v])/_width(by_id[b],positions['right',b])
    # log 比值对放大/缩小对称；约 50% 的相对变幅只形成温和代价。
    return min(1.,abs(float(np.log(max(ru,1e-6)/max(rv,1e-6))))/.7)


def _matrix(band, shift, anchors, by_id, orientation=None):
    left,right=band['left'],band['right'];positions=band['positions'];axis=band['axis']
    costs=np.full((len(left),len(right)),10.,float);details={}
    widths=[_width(by_id[p],positions[side,p]) for side,seq in
            (('left',left),('right',right)) for p in seq]
    scale=max(8.,2.*_median(widths))
    for i,u in enumerate(left):
        for j,v in enumerate(right):
            edge=band['edges'].get((u,v))
            if edge is None:continue
            pu,pv=positions['left',u],positions['right',v]
            # shift 可为常数，也可为沿切断边界缓慢变化的表观位移函数。
            local_shift=float(shift(pu) if callable(shift) else shift)
            delta=pv-pu;corrected=delta-local_shift*axis
            residual=abs(float(delta@axis)-local_shift)/scale
            if orientation is None:
                ta,tb=_frame(by_id[u],pu),_frame(by_id[v],pv)
            else:
                ta,_=orientation.sample(by_id[u],pu)
                tb,_=orientation.sample(by_id[v],pv)
            angle=np.degrees(np.arccos(np.clip(abs(float(ta@tb)),0,1)))/90
            norm=float(np.linalg.norm(corrected))
            if norm>1:
                unit=corrected/norm
                direction=.5*angle+.25*(1-abs(float(unit@ta)))+.25*(1-abs(float(unit@tb)))
            else: direction=angle
            width=_width(by_id[u],pu)+_width(by_id[v],pv)+edge.get('total_cut_thickness_pixels',0)
            normal=np.array([-ta[1],ta[0]])
            position=min(1.,abs(float(corrected@normal))/max(width,1.))
            weights=edge.get('weights',{})
            before=edge.get('costs',{})
            raw=float(edge.get('individual_score',edge['score']))
            cost=raw+weights.get('direction',.3)*(direction-before.get('direction',direction))
            cost+=weights.get('position',.08)*(position-before.get('position',position))
            cost+=.20*min(1.5,residual)
            ratio=_ratio_cost(u,v,anchors,band,by_id)
            if ratio is not None:cost+=.15*ratio
            costs[i,j]=max(0.,cost)
            details[(i,j)]={'corrected_direction':float(direction),'corrected_position':float(position),
                            'shift_residual':float(residual),'relative_thickness_cost':ratio}
    return costs,details


def _solve(band, shift, anchors, by_id, gap, orientation=None):
    costs,details=_matrix(band,shift,anchors,by_id,orientation)
    li={p:i for i,p in enumerate(band['left'])};ri={p:i for i,p in enumerate(band['right'])}
    for u,v in anchors:
        i,j=li[u],ri[v]
        costs[i,:]=10.;costs[:,j]=10.;costs[i,j]=0.
    value,matches=align_costs(costs,gap)
    return value,matches,costs,details


def _fit_band(band,by_id,cfg):
    """先评估整套位移，再按全局替代方案差距逐对暂定匹配并重估。"""
    gap=float(cfg.get('band_missing_layer_cost',.38))
    hypotheses=_hypotheses(band,by_id)
    best=min(((_solve(band,s,[],by_id,gap)[0],s) for s in hypotheses),key=lambda x:x[0])
    shift=best[1];anchors=[];accepted=[]
    # 每轮在当前已接受对应和共同位移下重新求全套序列。
    for _ in range(min(len(band['left']),len(band['right']))):
        value,matches,costs,details=_solve(band,shift,anchors,by_id,gap)
        options=[]
        for i,j in matches:
            pair=(band['left'][i],band['right'][j])
            if pair in anchors or costs[i,j]>=2*gap:continue
            alternatives=costs.copy();alternatives[i,j]=10.
            alternate,_=align_costs(alternatives,gap)
            margin=max(0.,alternate-value)
            # 条件增益高且没有近似等价替代的候选先确认。
            priority=(2*gap-costs[i,j])*(margin/(margin+.08))
            options.append((priority,margin,-costs[i,j],i,j))
        if not options:break
        priority,margin,_,i,j=max(options)
        if priority < float(cfg.get('band_min_evidence_gain',.04)):break
        u,v=band['left'][i],band['right'][j]
        anchors.append((u,v))
        accepted.append((u,v,float(costs[i,j]),float(priority),float(margin),details.get((i,j),{})))
        offsets=[float((band['positions']['right',b]-band['positions']['left',a])@band['axis'])
                 for a,b in anchors]
        shift=_median(offsets)
    # 估计表观位移的稳定性，绝不把边界法向间距解释成断层滑移。
    offsets=[float((band['positions']['right',b]-band['positions']['left',a])@band['axis'])
             for a,b in anchors]
    spread=_median([abs(v-shift) for v in offsets]) if offsets else None
    return {'shift':shift,'initial_shift':best[1],'spread':spread,'hypotheses':len(hypotheses),
            'accepted':accepted,'anchors':anchors,'left_count':len(band['left']),
            'right_count':len(band['right'])}


def register_bands(candidates,descriptors,config=None,virtual_descriptors=()):
    """把整套切断带的位移/层序结果写回候选边，供后续合组使用。"""
    cfg=config or {};by_id={d['id']:d for d in [*descriptors,*virtual_descriptors]};bands=defaultdict(list)
    for edge in candidates:
        for cut in edge.get('cut_part_ids',[]):bands[cut].append(edge)
    votes=defaultdict(list);audit=[];evaluated=set()
    for cut,edges in bands.items():
        band=_band(cut,edges,by_id)
        if band is None or min(len(band['left']),len(band['right']))<2:continue
        result=_fit_band(band,by_id,cfg)
        selected={(u,v):(cost,gain,margin,details) for u,v,cost,gain,margin,details in result['accepted']}
        gains=[item[1] for item in selected.values()]
        # 两对弱对应也可能意外具有相同位移；按层数、替代方案差距和离散度评价整套证据。
        widths=[_width(by_id[p],band['positions'][side,p]) for side,seq in
                (('left',band['left']),('right',band['right'])) for p in seq]
        scale=max(8.,2*_median(widths))
        stability=1/(1+(result['spread'] or 0)/scale)
        quality=(_median(gains)*len(gains)/(len(gains)+2)*stability) if gains else 0.
        strong=len(selected)>=2 and quality>=float(cfg.get('band_min_joint_quality',.08))
        if strong:
            for pair,edge in band['edges'].items():
                key=tuple(sorted(pair));evaluated.add(key)
                if pair in selected:
                    cost,gain,margin,details=selected[pair]
                    votes[key].append({'cut':cut,'selected':True,'cost':cost,'gain':gain,
                                       'margin':margin,'shift':result['shift'],'band_quality':quality,**details})
                else:votes[key].append({'cut':cut,'selected':False,'shift':result['shift'],
                                        'band_quality':quality})
        audit.append({'cut_part':cut,'left_layers':result['left_count'],
                      'right_layers':result['right_count'],'matched_layers':len(selected),
                      'joint_quality':quality,'joint_evidence_sufficient':strong,
                      'apparent_shift_along_cut_pixels':result['shift'],
                      'initial_shift_pixels':result['initial_shift'],
                      'shift_spread_pixels':result['spread'],'hypotheses':result['hypotheses'],
                      'cut_axis':band['axis'].tolist(),'cut_centre':band['centre'].tolist(),
                      'pairs':[{'left_part':u,'right_part':v,'cost':cost,'gain':gain,'margin':margin}
                               for u,v,cost,gain,margin,_ in result['accepted']]})
    for edge in candidates:
        key=tuple(sorted((edge['left_part'],edge['right_part'])))
        records=votes.get(key)
        if not records:continue
        good=[v for v in records if v['selected']]
        edge['band_registration']={'selected':bool(good),'votes':records}
        if good:
            winning=max(good,key=lambda x:x['gain'])
            edge['individual_score']=edge.get('individual_score',edge['score'])
            # 总体条件增益影响排序；位移修正后的代价成为本对的新基础评分。
            edge['score']=round(max(0.,winning['cost']-.12*min(1.,winning['gain'])),5)
            if len(audit)>0 and (edge.get('neighbour_confidence',0)>=.08 or winning['gain']>=.12):
                edge['contact_evidence_sufficient']=True
                edge['evidence_mode']='band_displacement_supported'
        else:edge['band_unselected']=True
    return {'bands':audit,'evaluated_pair_count':len(evaluated),
            'selected_pair_count':sum(bool(v.get('band_registration',{}).get('selected')) for v in candidates),
            'fault_line_mode':'red annotated line candidates plus transverse contact bands' if virtual_descriptors
                else 'transverse contact bands only'}
