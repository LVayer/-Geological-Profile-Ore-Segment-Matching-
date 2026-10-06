"""Partition reconstructed lithologies into geological layer instances.

Colour connectivity is the primary partition.  A fault may split one colour
component only when a strictly legend-supported fault-colour path forms a real
pixel barrier in the original drawing.
"""
from __future__ import annotations

import cv2
import numpy as np

from .staged_reconstruction import (
    _load_and_classify, _outer_domain, _render, _repair_linear_gaps, _seed_pale_regions,
)


def _lab(rgb):
    return cv2.cvtColor(np.float32([[np.asarray(rgb, np.float32) / 255.]]),
                            cv2.COLOR_RGB2LAB)[0, 0]


def _fault_prototypes(cfg):
    """Use explicit fault legend rows; a generic geological contact is excluded."""
    found=[]
    for item in (cfg.get('legend_catalog') or {}).get('items', []):
        name=str(item.get('name') or '')
        line=item.get('line') or {}
        rgb=line.get('dominant_rgb')
        if item.get('category')!='geological_boundary' or not any(word in name for word in ('断层','断裂')):
            continue
        if line.get('present') and isinstance(rgb,list) and len(rgb)==3:
            appearance_rgb=(item.get('appearance') or {}).get('dominant_rgb')
            label_same_colour=bool(item.get('subtype')=='label_line' and item.get('symbol_text') and
                isinstance(appearance_rgb,list) and len(appearance_rgb)==3 and
                np.linalg.norm(_lab(appearance_rgb)-_lab(rgb))<=8.)
            found.append({'id':item.get('id'),'name':name,'rgb':rgb,'lab':_lab(rgb),
                          'width':float(line.get('mean_width_px') or 1.),
                          'confidence':float(item.get('confidence') or 0.),
                          'subtype':item.get('subtype'),'symbol_text':item.get('symbol_text') or '',
                          'label_same_colour':label_same_colour})
    return found


def _other_prototypes(cfg, colours, fault_ids):
    values=[_lab(colour) for colour in colours]
    for item in (cfg.get('legend_catalog') or {}).get('items', []):
        if item.get('id') in fault_ids:continue
        rgb=(item.get('line') or {}).get('dominant_rgb')
        if isinstance(rgb,list) and len(rgb)==3:values.append(_lab(rgb))
    return values


def detect_fault_mask(crop, cfg, colours, scale):
    """Detect fault pixels with a strict colour win over every other legend entry.

    The tolerance is capped and also derived from the nearest competing legend
    colour.  Therefore a pixel cannot become a fault merely because it is
    vaguely red; the fault prototype must be its unambiguous closest legend.
    """
    faults=_fault_prototypes(cfg);h,w=crop.shape[:2]
    empty=np.zeros((h,w),bool)
    if not faults:
        return empty,{'enabled':False,'reason':'no_explicit_fault_legend','pixels':0}
    others=_other_prototypes(cfg,colours,{item['id'] for item in faults})
    mask=np.zeros((h,w),bool);details=[]
    for fault in faults:
        separation=min((float(np.linalg.norm(fault['lab']-other)) for other in others),default=30.)
        # OpenCV Lab uses L=0..100 and approximately a,b=-128..127.  Eight is
        # deliberately tight for exported vector colours, while the fraction
        # of prototype separation prevents overlap with a nearby legend colour.
        tolerance=float(np.clip(separation*.30,3.0,8.0));margin=float(np.clip(separation*.12,2.0,4.0))
        raw=np.zeros((h,w),bool);fault_wins=np.zeros((h,w),bool)
        edge_tolerance=min(14.,separation*.45)
        for row in range(0,h,192):
            end=min(h,row+192)
            lab=cv2.cvtColor(crop[row:end].astype(np.float32)/255.,cv2.COLOR_RGB2LAB)
            fault_distance=np.linalg.norm(lab-fault['lab'],axis=2)
            other_distance=np.full(lab.shape[:2],np.inf,np.float32)
            for prototype in others:
                other_distance=np.minimum(other_distance,np.linalg.norm(lab-prototype,axis=2))
            raw[row:end]=(fault_distance<=tolerance)&(fault_distance+margin<other_distance)
            # This wider colour envelope is never sufficient by itself.  It is
            # used only immediately around an accepted strict-colour core to
            # recover anti-aliased stroke edges.
            fault_wins[row:end]=(fault_distance<=edge_tolerance)&(fault_distance+margin<other_distance)

        # Reconnect only very small breaks between already colour-confirmed
        # pixels.  No new path can be nominated by geometry alone.
        working_width=max(.75,fault['width']*float(sum(scale)/2))
        gap=max(2,min(7,round(working_width*1.5)))
        joined=raw.astype(np.uint8);size=2*gap+1;centre=gap
        for angle in range(0,180,15):
            radians=np.deg2rad(angle);dx=round(np.cos(radians)*gap);dy=round(np.sin(radians)*gap)
            kernel=np.zeros((size,size),np.uint8)
            cv2.line(kernel,(centre-dx,centre-dy),(centre+dx,centre+dy),1,1)
            joined|=cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_CLOSE,kernel)

        count,cc,stats,_=cv2.connectedComponentsWithStats(joined,8)
        accepted_core=np.zeros((h,w),bool);minimum_span=max(18,round(min(h,w)*.012))
        kept=0
        for index in range(1,count):
            x,y,cw,ch,area=map(int,stats[index]);span=float(np.hypot(cw,ch))
            # Fault traces are extended paths. Compact red marks and isolated
            # anti-aliased pixels cannot split a layer.
            if span>=minimum_span and area/max(span,1.)<=max(14.,working_width*8.):
                component=cc==index
                if np.count_nonzero(component&raw)>=max(8,minimum_span//3):
                    accepted_core|=component;kept+=1
        radius=max(1,int(np.ceil(working_width/2.)))
        nearby=cv2.dilate(accepted_core.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))).astype(bool)
        accepted=accepted_core|(nearby&fault_wins)
        label_removed=0;line_support_segments=0
        if fault['label_same_colour']:
            # A same-colour fault number (for example F15) is nominated by the
            # colour mask too. Retain only pixels supported by an extended line
            # segment; compact glyph strokes and branches are discarded.
            support=np.zeros((h,w),np.uint8)
            minimum_line=max(18,round(min(h,w)*.008))
            hough=cv2.HoughLinesP((raw.astype(np.uint8)*255),1,np.pi/360,
                threshold=max(10,minimum_line//2),minLineLength=minimum_line,
                maxLineGap=max(3,gap))
            if hough is not None:
                support_width=max(3,round(working_width*2+2))
                for x1,y1,x2,y2 in np.asarray(hough).reshape(-1,4):
                    cv2.line(support,(int(x1),int(y1)),(int(x2),int(y2)),1,support_width,cv2.LINE_8)
                    line_support_segments+=1
            pure=accepted&(support>0)
            label_removed=int(np.count_nonzero(accepted&~pure))
            accepted=pure
        mask|=accepted
        details.append({'id':fault['id'],'name':fault['name'],'rgb':fault['rgb'],
            'nearest_other_delta_e':round(separation,3),'tolerance_delta_e':round(tolerance,3),
            'required_colour_margin_delta_e':round(margin,3),'raw_pixels':int(raw.sum()),
            'accepted_core_pixels':int(accepted_core.sum()),'accepted_pixels':int(accepted.sum()),
            'anti_alias_edge_pixels':int(np.count_nonzero(accepted&~accepted_core)),
            'accepted_paths':kept,'working_width_px':round(working_width,2),
            'edge_tolerance_delta_e':round(edge_tolerance,3),'max_join_gap_px':gap,
            'legend_subtype':fault['subtype'],'legend_symbol_text':fault['symbol_text'],
            'label_colour_matches_line':fault['label_same_colour'],
            'line_support_segments':line_support_segments,'label_pixels_removed':label_removed})
    return mask,{'enabled':True,'pixels':int(mask.sum()),'prototypes':details,
                 'rule':'fault_delta<=adaptive_cap_8 AND fault_delta+margin<all_other_legend_delta'}


def _colour_components(labels, domain, barrier, minimum_area):
    instances=np.full(labels.shape,-1,np.int32);records=[]
    for lithology in [int(value) for value in np.unique(labels[labels>=0])]:
        mask=(labels==lithology)&domain&~barrier
        count,cc,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        for component in range(1,count):
            area=int(stats[component,cv2.CC_STAT_AREA])
            if area<minimum_area:continue
            instance=len(records);region=cc==component;instances[region]=instance
            x,y,w,h=map(int,stats[component,:4])
            records.append({'instance':instance,'lithology_index':lithology,'area_pixels':area,
                            'bbox':[x,y,w,h]})
    return instances,records


def classify_unassigned_by_neighbours(labels,domain):
    """Assign each remaining internal void from its immediate classified ring.

    Labels are discrete, strictly legend-derived colour classes.  Consequently
    two similar RGB values are never merged here: equality means the same
    reviewed legend class.  A one-pixel ring implements the requested
    imperceptibly-close test.  When several classes touch a void, boundary vote
    length selects the surrounding majority and the whole void joins that class.
    """
    out=labels.copy();unknown=(out<0)&domain
    count,cc,stats,_=cv2.connectedComponentsWithStats(unknown.astype(np.uint8),8)
    audit={'components':max(0,count-1),'pixels_before':int(unknown.sum()),
           'same_colour_components':0,'majority_colour_components':0,
           'filled_pixels':0,'unresolved_pixels':0,'decisions':[]}
    kernel=np.ones((3,3),np.uint8)
    for index in range(1,count):
        component=cc==index;area=int(stats[index,cv2.CC_STAT_AREA])
        ring=cv2.dilate(component.astype(np.uint8),kernel).astype(bool)&~component&domain
        neighbours=out[ring];neighbours=neighbours[neighbours>=0]
        if not len(neighbours):
            audit['unresolved_pixels']+=area;continue
        values,votes=np.unique(neighbours,return_counts=True)
        winner_index=int(np.argmax(votes));winner=int(values[winner_index]);total=int(votes.sum())
        out[component]=winner;audit['filled_pixels']+=area
        method='same_colour' if len(values)==1 else 'immediate_ring_majority'
        if len(values)==1:audit['same_colour_components']+=1
        else:audit['majority_colour_components']+=1
        audit['decisions'].append({'component':index,'area_pixels':area,'class':winner,
            'method':method,'winning_boundary_fraction':round(int(votes[winner_index])/max(1,total),4),
            'touching_classes':[int(v) for v in values]})
    audit['unresolved_pixels']=int(np.count_nonzero((out<0)&domain))
    return out,audit


def smooth_filled_boundaries(labels,domain,fault_mask,passes=3,strength=.78,
                             allowed_lithology_classes=None):
    """Remove one-pixel colour-boundary burrs without erasing thin beds.

    Smoothing is categorical: pixels may only change to an existing legend
    class.  A 3x3 majority is accepted only at a boundary, and only where the
    current class has no straight/diagonal continuation through the pixel.
    This continuation test protects one-pixel and two-pixel genuine thin beds.
    Confirmed faults and their immediate edge are never smoothed across.
    """
    original=labels.copy();out=labels.copy();passes=max(0,min(10,int(passes)));strength=float(np.clip(strength,0.,3.))
    allowed=({int(value) for value in allowed_lithology_classes}
             if allowed_lithology_classes is not None else
             {int(value) for value in np.unique(out[domain]) if value>=0})
    # Strength above 1 enters an ultra-smoothing range with a substantially
    # larger spatial scale. This rounds visible multi-pixel stair-steps rather
    # than merely deleting isolated pixel noise.
    if strength>1:
        kernel_size=min(19,7+4*int(np.ceil(strength-1)))
    else:
        kernel_size=7 if strength>=.9 else (5 if strength>=.65 else 3)
    radius=kernel_size//2
    kernel=np.ones((kernel_size,kernel_size),np.uint8);kernel_area=kernel_size**2
    winning_fraction=(.62-.10*strength) if strength<=1 else max(.505,.52-.006*(strength-1))
    winning_min=max(kernel_area//2+1,int(np.ceil(kernel_area*winning_fraction)))
    current_max=kernel_area-winning_min
    protected=cv2.dilate(fault_mask.astype(np.uint8),kernel).astype(bool)
    changes=[]
    for _ in range(passes):
        classes=[int(value) for value in np.unique(out[domain]) if value in allowed]
        if not classes:break
        thin_protected=np.zeros(out.shape,bool)
        minimum_length=max(6,round(min(out.shape)*.003))
        for value in classes:
            class_mask=(out==value).astype(np.uint8)
            component_count,components,stats,_=cv2.connectedComponentsWithStats(class_mask,8)
            thickness=cv2.distanceTransform(class_mask,cv2.DIST_L2,5)
            for component in range(1,component_count):
                x,y,w,h,area=map(int,stats[component])
                elongation=max(w,h)/max(1,min(w,h))
                region=components==component
                if max(w,h)>=minimum_length and elongation>=3 and float(thickness[region].max())<=1.5:
                    thin_protected|=region
        best_count=np.zeros(out.shape,np.uint16);winner=np.full(out.shape,-1,np.int16)
        current_count=np.zeros(out.shape,np.uint16)
        for value in classes:
            mask=(out==value).astype(np.uint8)
            count=cv2.boxFilter(mask,cv2.CV_16U,(kernel_size,kernel_size),normalize=False,
                                borderType=cv2.BORDER_CONSTANT)
            better=count>best_count
            winner[better]=value;best_count[better]=count[better]
            current_count[out==value]=count[out==value]
        same=lambda dy,dx: np.roll(out,(dy,dx),(0,1))==out
        # A real narrow unit normally continues through the centre pixel in at
        # least one direction.  Burrs and single-pixel dents do not.
        continuous=(same(0,1)&same(0,-1))|(same(1,0)&same(-1,0))|\
                   (same(1,1)&same(-1,-1))|(same(1,-1)&same(-1,1))
        continuity_block=continuous if strength<=1 else np.zeros(out.shape,bool)
        change=domain&~protected&~thin_protected&np.isin(out,np.asarray(list(allowed),np.int16))&\
               (winner>=0)&(winner!=out)&(best_count>=winning_min)&\
               (current_count<=current_max)&~continuity_block
        # np.roll wraps at the raster edges; the domain edge is excluded so a
        # wrapped equality can never trigger a classification change there.
        change&=cv2.erode(domain.astype(np.uint8),kernel).astype(bool)
        count=int(change.sum());changes.append(count)
        if not count:break
        out[change]=winner[change]
    # High-strength rounding must not create a new unit or sever an existing
    # fault-bounded component. Revert only pixels involved in such a topology
    # change; ordinary boundary smoothing remains in place.
    active=domain&~fault_mask;topology_reverted=0;split_components_prevented=0;new_islands_prevented=0
    for value in sorted(allowed):
        original_mask=(original==value)&active
        current_mask=(out==value)&active
        _,original_cc=cv2.connectedComponents(original_mask.astype(np.uint8),8)
        current_count,current_cc=cv2.connectedComponents(current_mask.astype(np.uint8),8)
        for component in range(1,current_count):
            region=current_cc==component
            if not np.any(original_mask&region):
                changed=region&(out!=original)
                topology_reverted+=int(changed.sum());new_islands_prevented+=1
                out[changed]=original[changed]
        _,current_cc=cv2.connectedComponents(((out==value)&active).astype(np.uint8),8)
        for component in range(1,int(original_cc.max())+1):
            region=original_cc==component
            surviving=np.unique(current_cc[region&(out==value)]);surviving=surviving[surviving>0]
            if len(surviving)>1:
                changed=region&(out!=value)
                topology_reverted+=int(changed.sum());split_components_prevented+=1
                out[changed]=value
    net_changed=int(np.count_nonzero((out!=original)&domain))
    return out,{'enabled':passes>0,'strength':round(strength,3),'kernel_size':kernel_size,
                'winning_vote_minimum':winning_min,'passes_requested':passes,'passes_run':len(changes),
                'changed_pixels_per_pass':changes,'changed_pixels_total':int(sum(changes)),
                'net_changed_pixels':net_changed,'topology_reverted_pixels':topology_reverted,
                'split_components_prevented':split_components_prevented,
                'new_islands_prevented':new_islands_prevented,
                'fault_edge_protected_pixels':int((protected&domain).sum()),
                'allowed_lithology_classes':sorted(allowed),
                'non_lithology_colours_used':False,
                'thin_layer_continuation_protection':strength<=1,
                'thin_component_protection':True}


def _micro_water_cleanup(crop,labels,domain,fault_mask,colours,minimum_area):
    """Test tiny layer candidates by colour-constrained injection on the source.

    Candidate nomination uses area only; elongation is deliberately not a
    rejection rule because a short real unit may have no long axis. Water is
    injected from the candidate's source-aligned pixels at three LAB colour
    tolerances. If it reaches a non-candidate neighbouring layer in at least
    two runs, the candidate is treated as an open colour fragment and merges
    into the most strongly reached neighbour. Stable containment preserves it.
    """
    out=labels.copy()
    preliminary,records=_colour_components(out,domain,fault_mask,minimum_area)
    micro_limit=max(48,min(160,round(int(domain.sum())*.00003)))
    candidates=[record for record in records if record['area_pixels']<=micro_limit]
    candidate_ids={record['instance'] for record in candidates}
    by_id={record['instance']:record for record in records}
    details=[];merged_pixels=0;merged_components=0;partitioned_pixels_total=0
    thresholds=(4.,7.,10.)
    for record in candidates:
        instance=record['instance'];region=preliminary==instance
        x,y,w,h=record['bbox'];margin=max(18,min(64,round(np.sqrt(record['area_pixels'])*4)))
        x0=max(0,x-margin);y0=max(0,y-margin);x1=min(out.shape[1],x+w+margin);y1=min(out.shape[0],y+h+margin)
        local_region=region[y0:y1,x0:x1];local_instances=preliminary[y0:y1,x0:x1]
        local_domain=domain[y0:y1,x0:x1];local_fault=fault_mask[y0:y1,x0:x1]
        lab=cv2.cvtColor(crop[y0:y1,x0:x1].astype(np.float32)/255.,cv2.COLOR_RGB2LAB)
        source_colour=np.median(lab[local_region],axis=0)
        colour_distance=np.linalg.norm(lab-source_colour,axis=2)
        reached_votes={};escape_runs=0
        for threshold in thresholds:
            walkable=(colour_distance<=threshold)&local_domain&~local_fault
            walkable|=local_region
            count,cc=cv2.connectedComponents(walkable.astype(np.uint8),8)
            seed_ids=np.unique(cc[local_region]);seed_ids=seed_ids[seed_ids>0]
            flooded=np.isin(cc,seed_ids)
            reached=[int(value) for value in np.unique(local_instances[flooded])
                     if value>=0 and value!=instance and value not in candidate_ids]
            if reached:escape_runs+=1
            for value in reached:
                reached_votes[value]=reached_votes.get(value,0)+int(np.count_nonzero(flooded&(local_instances==value)))
        prototype=_lab(colours[record['lithology_index']])
        direct_support=float(np.mean(np.linalg.norm(lab[local_region]-prototype,axis=1)<=8.))
        # A seed with virtually no matching source pixels was created by later
        # inference and cannot be validated as a real source-image layer. Treat
        # it as an open fragment even when colour flooding itself is contained.
        no_source_support=direct_support<.15
        escaped=(escape_runs>=2 and bool(reached_votes)) or no_source_support
        target=None;target_lithology=None;partitioned_pixels=0;fill_classes=[]
        absorbed_into_fault_edge=False
        if escaped:
            if no_source_support:
                # The inferred speck has no evidence at the same coordinates in
                # the source image.  Dissolve it pixel by pixel into the nearest
                # *non-candidate* classified pixels that can be reached without
                # crossing a confirmed fault.  This is the digital equivalent
                # of injecting water on each fault-bounded side: a fault is an
                # impermeable wall, while a false island has no wall of its own.
                local_out=out[y0:y1,x0:x1]
                passable=local_domain&~local_fault
                side_count,sides=cv2.connectedComponents(passable.astype(np.uint8),8)
                assigned=np.zeros(local_region.shape,bool)
                for side in np.unique(sides[local_region]):
                    if side<=0:continue
                    side_mask=sides==side
                    sources=side_mask&~local_region&(local_instances>=0)&\
                            ~np.isin(local_instances,np.asarray(list(candidate_ids),np.int32))
                    targets=local_region&side_mask
                    if not np.any(sources) or not np.any(targets):continue
                    distance_input=np.ones(local_region.shape,np.uint8)
                    distance_input[sources]=0
                    _,nearest=cv2.distanceTransformWithLabels(distance_input,cv2.DIST_L2,5,
                                                               labelType=cv2.DIST_LABEL_PIXEL)
                    lookup=np.full(int(nearest.max())+1,-1,np.int32)
                    lookup[nearest[sources]]=local_out[sources]
                    filled=lookup[nearest[targets]]
                    valid=filled>=0
                    target_y,target_x=np.where(targets)
                    local_out[target_y[valid],target_x[valid]]=filled[valid]
                    assigned[target_y[valid],target_x[valid]]=True
                if not np.any(assigned):
                    # A thick fault mask can form a tiny closed pocket around
                    # anti-aliased line-edge pixels.  Such a pocket has neither
                    # a source-colour seed nor a same-side source, so it is not
                    # positive evidence for a geological unit.  It is an
                    # anti-aliased residue inside a fault junction, so absorb
                    # it into that barrier.  Borrowing colour across the fault
                    # would silently connect units on opposite sides.
                    local_out[local_region]=-1
                    local_fault[local_region]=True
                    assigned[local_region]=True
                    absorbed_into_fault_edge=True
                partitioned_pixels=int(assigned.sum())
                partitioned_pixels_total+=partitioned_pixels
                if partitioned_pixels:
                    values,counts=np.unique(local_out[assigned],return_counts=True)
                    fill_classes=[{'lithology_index':int(value),'pixels':int(count)}
                                  for value,count in zip(values,counts) if value>=0]
                    merged_pixels+=partitioned_pixels;merged_components+=1
            elif reached_votes:
                target=max(reached_votes,key=reached_votes.get)
            else:
                ring=cv2.dilate(region.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&~region&domain&~fault_mask
                values,counts=np.unique(preliminary[ring],return_counts=True)
                choices=[(int(count),int(value)) for value,count in zip(values,counts)
                         if value>=0 and value not in candidate_ids]
                if choices:
                    # With no matching source-colour seed, selecting another
                    # disconnected component of the same inferred class would
                    # merely rename the artefact and can also cross a fault.
                    # Prefer the strongest different-lithology neighbour on the
                    # same fault-free side; fall back only when none exists.
                    different=[choice for choice in choices
                               if by_id[choice[1]]['lithology_index']!=record['lithology_index']]
                    target=max(different or choices)[1]
            if target is not None:
                target_lithology=by_id[target]['lithology_index']
                out[region]=target_lithology
                merged_pixels+=record['area_pixels'];merged_components+=1
        details.append({'instance':instance,'area_pixels':record['area_pixels'],
            'bbox':record['bbox'],'escape_runs':escape_runs,'flood_runs':len(thresholds),
            'source_direct_colour_support':round(direct_support,3),
            'decision':('partition_no_source_support' if no_source_support and partitioned_pixels else
                        'merge_open_fragment' if escaped and target is not None else 'retain_contained'),
            'target_instance':target,'target_lithology_index':target_lithology,
            'fault_side_partitioned_pixels':partitioned_pixels,
            'fault_side_fill_classes':fill_classes,
            'absorbed_into_fault_edge':absorbed_into_fault_edge})
    return out,{'candidate_area_limit_pixels':micro_limit,'elongation_used_as_rejection':False,
        'colour_tolerances_lab':list(thresholds),'candidates':len(candidates),
        'merged_components':merged_components,'merged_pixels':merged_pixels,
        'fault_side_partitioned_pixels':partitioned_pixels_total,
        'retained_components':len(candidates)-merged_components,'details':details}


def _merge_near_same_lithology(instances,records,labels,fault_mask,max_gap):
    """Join close same-lithology parts only across a fault-free local corridor.

    The merge is semantic: disconnected polygons share one layer id without
    repainting an intervening unit. A confirmed fault in the shortest local
    corridor is an absolute veto. Similar surrounding lithology sets provide a
    conservative continuity check for close but geologically distinct units.
    """
    parent=list(range(len(records)))
    def find(value):
        while parent[value]!=value:
            parent[value]=parent[parent[value]];value=parent[value]
        return value
    def union(a,b):
        a=find(a);b=find(b)
        if a!=b:parent[b]=a
    masks=[instances==record['instance'] for record in records]
    neighbour_sets=[]
    for mask,record in zip(masks,records):
        ring=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)&~mask
        neighbour_sets.append({int(value) for value in np.unique(labels[ring&~fault_mask])
                               if value>=0 and value!=record['lithology_index']})
    candidates=merged=fault_veto=topology_veto=0;details=[]
    for left in range(len(records)):
        if records[left]['area_pixels']<=0:continue
        lx,ly,lw,lh=records[left]['bbox']
        for right in range(left+1,len(records)):
            if records[left]['lithology_index']!=records[right]['lithology_index']:continue
            rx,ry,rw,rh=records[right]['bbox']
            dx=max(0,max(lx,rx)-min(lx+lw,rx+rw));dy=max(0,max(ly,ry)-min(ly+lh,ry+rh))
            if np.hypot(dx,dy)>max_gap+1:continue
            x0=max(0,min(lx,rx)-max_gap-2);y0=max(0,min(ly,ry)-max_gap-2)
            x1=min(instances.shape[1],max(lx+lw,rx+rw)+max_gap+2)
            y1=min(instances.shape[0],max(ly+lh,ry+rh)+max_gap+2)
            a=masks[left][y0:y1,x0:x1];b=masks[right][y0:y1,x0:x1]
            distance=cv2.distanceTransform((~a).astype(np.uint8),cv2.DIST_L2,5)
            gap=float(distance[b].min()) if np.any(b) else float('inf')
            if gap>max_gap:continue
            candidates+=1;radius=max(2,int(np.ceil(gap/2))+1)
            bridge=cv2.dilate(a.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))).astype(bool)&\
                   cv2.dilate(b.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))).astype(bool)
            if np.any(bridge&fault_mask[y0:y1,x0:x1]):
                fault_veto+=1;details.append({'left':left,'right':right,'gap_px':round(gap,2),'decision':'fault_veto'});continue
            first,second=neighbour_sets[left],neighbour_sets[right]
            similarity=len(first&second)/max(1,len(first|second))
            area_ratio=min(records[left]['area_pixels'],records[right]['area_pixels'])/max(
                records[left]['area_pixels'],records[right]['area_pixels'])
            # Comparable part size prevents a tiny speck from acting as a
            # transitive bridge that fuses several otherwise distinct layers.
            if similarity<.6 or area_ratio<.5:
                topology_veto+=1;details.append({'left':left,'right':right,'gap_px':round(gap,2),
                    'neighbour_similarity':round(similarity,3),'area_ratio':round(area_ratio,3),
                    'decision':'topology_or_scale_veto'});continue
            union(left,right);merged+=1
            details.append({'left':left,'right':right,'gap_px':round(gap,2),
                'neighbour_similarity':round(similarity,3),'area_ratio':round(area_ratio,3),
                'decision':'merge_multipart'})
    groups={}
    for index in range(len(records)):groups.setdefault(find(index),[]).append(index)
    compact=np.full(instances.shape,-1,np.int32);rebuilt=[]
    for new_id,members in enumerate(groups.values()):
        mask=np.isin(instances,np.asarray([records[index]['instance'] for index in members],np.int32))
        if not np.any(mask):continue
        yy,xx=np.where(mask);lith=records[members[0]]['lithology_index']
        compact[mask]=len(rebuilt)
        rebuilt.append({'instance':len(rebuilt),'lithology_index':lith,'area_pixels':int(mask.sum()),
            'bbox':[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)],
            'part_count':len(members)})
    return compact,rebuilt,{'max_gap_pixels':max_gap,'candidate_pairs':candidates,
        'merged_pairs':merged,'fault_veto_pairs':fault_veto,'topology_veto_pairs':topology_veto,
        'multipart_layers':sum(record['part_count']>1 for record in rebuilt),'details':details}


def _render_instances(instances, records, colours, domain, width=2):
    image=np.full((*instances.shape,3),255,np.uint8)
    for record in records:
        image[instances==record['instance']]=colours[record['lithology_index']]
    edge=np.zeros(instances.shape,bool)
    for dy,dx in ((1,0),(0,1)):
        a=instances[dy:,dx:];b=instances[:-dy or None,:-dx or None]
        different=(a!=b)&((a>=0)|(b>=0))
        edge[dy:,dx:]|=different;edge[:-dy or None,:-dx or None]|=different
    if width>1:edge=cv2.dilate(edge.astype(np.uint8),np.ones((width,width),np.uint8)).astype(bool)
    image[edge]=0;image[~domain]=255
    contours,_=cv2.findContours(domain.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image,contours,-1,(0,0,0),max(2,width+1),lineType=cv2.LINE_8)
    return image


def _render_plain_labels(labels,colours,domain,fault_mask=None,fault_rgb=(237,23,23)):
    """Render filled legend classes without black layer-interface outlines."""
    image=np.full((*labels.shape,3),255,np.uint8)
    for index,colour in enumerate(colours):image[(labels==index)&domain]=colour
    image[~domain]=255
    if fault_mask is not None:image[fault_mask&domain]=np.asarray(fault_rgb,np.uint8)
    return image


def _render_colour_checked_boundaries(labels,instances,colours,domain,width=2,with_audit=False):
    """Validate short boundary segments from their local two-sided colours.

    Candidate instance contours are split into short arcs.  Each arc obtains a
    local tangent and samples classified pixels along both normal directions at
    several positions.  Equal legend classes cancel only that short arc;
    different classes retain it.  This avoids an all-or-nothing decision for a
    long line whose neighbouring colours change along its path.
    """
    image=np.full((*labels.shape,3),255,np.uint8)
    for index,colour in enumerate(colours):image[(labels==index)&domain]=colour
    segment_pixels=max(4,min(10,round(min(labels.shape)*.0025)))
    start=max(1,int(width));probe=start+4
    line=np.zeros(labels.shape,np.uint8)
    audit={'segment_length_pixels':segment_pixels,'probe_min_pixels':start,
           'probe_max_pixels':probe,'kept_different_colour_segments':0,
           'cancelled_same_colour_segments':0,'unresolved_segments':0}

    def side_class(points,normal,sign):
        votes=[]
        for fraction in (.2,.5,.8):
            index=min(len(points)-1,max(0,round((len(points)-1)*fraction)))
            x,y=points[index]
            # The first classified pixel on the ray is the local side colour;
            # multiple positions make corners and one-pixel noise less decisive.
            for distance in range(start,probe+1):
                xx=int(round(x+normal[0]*distance*sign));yy=int(round(y+normal[1]*distance*sign))
                if 0<=yy<labels.shape[0] and 0<=xx<labels.shape[1] and domain[yy,xx] and labels[yy,xx]>=0:
                    votes.append(int(labels[yy,xx]));break
        if not votes:return -1
        values,counts=np.unique(np.asarray(votes,np.int16),return_counts=True)
        return int(values[int(np.argmax(counts))])

    for instance in [int(value) for value in np.unique(instances[instances>=0])]:
        contours,_=cv2.findContours((instances==instance).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            points=contour[:,0,:]
            for begin in range(0,len(points)-1,segment_pixels):
                arc=points[begin:min(len(points),begin+segment_pixels+1)]
                if len(arc)<2:continue
                tangent=arc[-1].astype(np.float64)-arc[0].astype(np.float64)
                length=float(np.linalg.norm(tangent))
                if length<1:continue
                normal=np.asarray([-tangent[1],tangent[0]])/length
                first=side_class(arc,normal,1);second=side_class(arc,normal,-1)
                if first<0 or second<0:
                    audit['unresolved_segments']+=1;continue
                if first==second:
                    audit['cancelled_same_colour_segments']+=1;continue
                cv2.polylines(line,[arc.reshape(-1,1,2)],False,1,max(1,int(width)),cv2.LINE_8)
                audit['kept_different_colour_segments']+=1
    image[line>0]=0;image[~domain]=255
    contours,_=cv2.findContours(domain.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image,contours,-1,(0,0,0),max(2,width+1),lineType=cv2.LINE_8)
    return (image,audit) if with_audit else image


def partition_layers(cfg):
    """Return a colour-first, strict-fault-split layer partition and audit."""
    crop,stage1,colours,names,pale,pale_candidate,allowed,scale,_,audit1=_load_and_classify(cfg)
    domain,audit2=_outer_domain(stage1,pale,allowed,pale_candidate)
    labels=stage1.copy();labels[~domain]=-1
    labels,pale_audit=_seed_pale_regions(labels,pale_candidate,pale,domain)
    audit2['pale_region_seeding']=pale_audit
    # Preserve the stage-2 labels so the fault evidence can be reviewed on the
    # overall-outline image independently of any later colour-gap repair.
    stage2_labels=labels.copy()
    labels,audit3=_repair_linear_gaps(labels,domain,int(cfg.get('stage3_line_probe_pixels',16)))
    labels,unassigned_audit=classify_unassigned_by_neighbours(labels,domain)
    fault_mask,fault_audit=detect_fault_mask(crop,cfg,colours,scale)
    fault_mask&=domain
    labels,smoothing_audit=smooth_filled_boundaries(
        labels,domain,fault_mask,int(cfg.get('boundary_smoothing_passes',3)),
        float(cfg.get('boundary_smoothing_strength',.78)),range(len(colours)))
    minimum_area=max(12,round(int(domain.sum())*.000006))
    labels,micro_audit=_micro_water_cleanup(crop,labels,domain,fault_mask,colours,minimum_area)
    before,base_records=_colour_components(labels,domain,np.zeros_like(domain),minimum_area)
    instances,records=_colour_components(labels,domain,fault_mask,minimum_area)
    reconnect_gap=max(4,min(10,round(min(labels.shape)*.003)))
    instances,records,reconnect_audit=_merge_near_same_lithology(
        instances,records,labels,fault_mask,reconnect_gap)
    for record in records:record['lithology']=names[record['lithology_index']]
    # 地层组独立于像素实例：允许穿插/错位两侧共享编号，不改写几何。
    from .layer_groups import group_layer_parts
    layer_groups=group_layer_parts(instances,records,labels,domain,cfg)
    memberships={part:group['label'] for group in layer_groups['groups'] for part in group['parts']}
    for record in records:record['group_label']=memberships[record['instance']]
    split_increase=len(records)-len(base_records)
    image,final_boundary_audit=_render_colour_checked_boundaries(
        labels,instances,colours,domain,int(cfg.get('layer_outline_width',2)),True)
    # Confirmed faults remain geological barriers even when their two sides
    # have the same lithology, so overlay them after local colour-side checks.
    image[fault_mask]=0
    colour_only_image,boundary_audit=_render_colour_checked_boundaries(
        labels,before,colours,domain,int(cfg.get('layer_outline_width',2)),True)
    outline_fault_image=_render(stage2_labels,colours,domain,4,0)
    # Use the reviewed fault legend colour here.  Red stays distinct from the
    # black overall outline and makes false/omitted fault paths easy to inspect.
    fault_rgb=np.asarray((_fault_prototypes(cfg)[0]['rgb'] if _fault_prototypes(cfg) else [237,23,23]),np.uint8)
    outline_fault_image[fault_mask]=fault_rgb
    plain_image=_render_plain_labels(labels,colours,domain,fault_mask,fault_rgb)
    return {'image':image,'colour_only_image':colour_only_image,
            'outline_fault_image':outline_fault_image,'plain_image':plain_image,
            'instances':instances,'fault_mask':fault_mask,'domain':domain,'records':records,
            'layer_groups':layer_groups,
            'audit':{'stage1':audit1,'stage2':audit2,'stage3':audit3,
                     'unassigned_classification':unassigned_audit,'fault':fault_audit,
                     'boundary_smoothing':smoothing_audit,
                     'micro_water_cleanup':micro_audit,
                     'same_lithology_reconnection':reconnect_audit,
                     'local_boundary_validation':boundary_audit,
                     'final_boundary_validation':final_boundary_audit,
                     'minimum_component_area_pixels':minimum_area,
                     'colour_components_before_fault':len(base_records),
                     'layers_after_fault':len(records),'fault_complete_splits_added':split_increase,
                     'unassigned_inside_pixels':int(np.count_nonzero((instances<0)&domain&~fault_mask)),
                     'fault_barrier_pixels':int(fault_mask.sum())}}
