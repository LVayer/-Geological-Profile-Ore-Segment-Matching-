"""Deterministic legend mapping. Never infer a geological name from RGB alone."""
import cv2
import numpy as np
from PIL import Image
from .io import local, validate_section
from .geometry import extract,topology
from .artifacts import (detect_masks,restore,restore_text_oriented,fill_enclosed_gaps,
                        classify_line_paths,reconnect_artifact_corridors,
                        expand_confirmed_corridors,inpaint_artifact_labels,
                        classify_catalogue_strokes)
from .domain import infer_domain,enforce_single_outer_domain

def recognize(cfg,diagnostics=None):
    if cfg.get('instance_mask'):
        return from_instances(cfg)
    p=local(cfg['image'])
    if p.suffix.lower() not in ['.png','.jpg','.jpeg','.tif','.tiff']:
        raise ValueError('Unsupported image format')
    if 'roi' not in cfg or not cfg.get('legend'):
        raise ValueError('Provide plot ROI and reviewed legend labels/swatches; OCR alone cannot authorize lithology')
    x,y,roi_w,roi_h=cfg['roi']
    max_pixels=int(cfg.get('max_recognition_pixels',7_500_000))
    if not 250_000<=max_pixels<=40_000_000:raise ValueError('max_recognition_pixels must be between 250000 and 40000000')
    with Image.open(p) as im:
        if getattr(im,'n_frames',1)>1 and 'page' not in cfg:raise ValueError('Multipage TIFF requires explicit page')
        im.seek(cfg.get('page',0));original_w,original_h=im.size
        if min(x,y)<0 or min(roi_w,roi_h)<8 or x+roi_w>original_w or y+roi_h>original_h:raise ValueError('Invalid ROI')
        target=min(1.,np.sqrt(max_pixels/(roi_w*roi_h)))
        # JPEG draft decoding reduces memory before allocating the RGB bitmap.
        if target<1 and p.suffix.lower() in ['.jpg','.jpeg']:
            im.draft('RGB',(max(1,round(original_w*target)),max(1,round(original_h*target))))
        working=im.convert('RGB')
        actual_target=min(1.,np.sqrt(max_pixels/((roi_w*working.width/original_w)*(roi_h*working.height/original_h))))
        if actual_target<.995:
            working=working.resize((max(1,round(working.width*actual_target)),max(1,round(working.height*actual_target))),Image.Resampling.LANCZOS)
        rgb=np.asarray(working)
    sx,sy=rgb.shape[1]/original_w,rgb.shape[0]/original_h
    wx,wy=round(x*sx),round(y*sy);right,bottom=round((x+roi_w)*sx),round((y+roi_h)*sy)
    crop=rgb[wy:bottom,wx:right].copy();h,w=crop.shape[:2]
    prototypes=[]; names=[]; verified=[]
    for entry in cfg['legend']:
        if 'swatch' in entry:
            u,v,sw,sh=entry['swatch']
            if min(u,v)<0 or min(sw,sh)<1 or u+sw>original_w or v+sh>original_h: raise ValueError('Invalid legend swatch')
            u0,v0=max(0,round(u*sx)),max(0,round(v*sy));u1,v1=min(rgb.shape[1],round((u+sw)*sx)),min(rgb.shape[0],round((v+sh)*sy))
            pixels=rgb[v0:max(v0+1,v1),u0:max(u0+1,u1)].reshape(-1,3)
            color=np.median(pixels,axis=0)
        else: color=np.array(entry['rgb'])
        prototypes.append(cv2.cvtColor(np.float32([[color/255]]),cv2.COLOR_RGB2LAB)[0,0])
        names.append(entry.get('lithology') or 'UNKNOWN')
        verified.append(bool(entry.get('verified',False)) and names[-1]!='UNKNOWN')
    del rgb
    # Classify row blocks and retain only the winning prototype and its error.
    # This replaces the former H×W×legend_count matrix that exceeded 1 GB on P11.
    classified=np.full((h,w),-1,np.int16);best_error=np.full((h,w),np.inf,np.float16)
    prototype_array=np.asarray(prototypes,np.float32);block_rows=max(32,min(256,max_pixels//max(1,w*12)))
    # A confirmed near-white legend item needs a controlled exception to the
    # usual second-best margin. Otherwise white geological units are rejected
    # whenever another very pale legend colour is almost as close. The
    # exception is deliberately disabled if several white legend items exist.
    pale_indices=[index for index,(prototype,ok) in enumerate(zip(prototypes,verified))
                  if ok and prototype[0]>=88 and np.linalg.norm(prototype[1:])<8]
    for row in range(0,h,block_rows):
        end=min(h,row+block_rows);lab=cv2.cvtColor(crop[row:end].astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        best=np.full(lab.shape[:2],np.inf,np.float32);second=np.full_like(best,np.inf);idx=np.zeros(lab.shape[:2],np.int16)
        for k,prototype in enumerate(prototype_array):
            distance=np.linalg.norm(lab-prototype,axis=2);better=distance<best
            second=np.where(better,best,np.minimum(second,distance));best=np.where(better,distance,best);idx[better]=k
        eligible=(best<cfg.get('lab_tolerance',12))&((second-best)>cfg.get('lab_margin',4))
        if len(pale_indices)==1:
            pale_index=pale_indices[0];pale_error=np.linalg.norm(lab-prototype_array[pale_index],axis=2)
            near_white=(lab[:,:,0]>=92)&(np.linalg.norm(lab[:,:,1:],axis=2)<5)
            white_unit=near_white&(pale_error<cfg.get('pale_lab_tolerance',7))
            idx[white_unit]=pale_index;best[white_unit]=pale_error[white_unit];eligible|=white_unit
        classified[row:end]=np.where(eligible,idx,-1);best_error[row:end]=best.astype(np.float16)
    def polygon(points):return np.asarray([[round(px*sx)-wx,round(py*sy)-wy] for px,py in points],np.int32)
    rings=cfg.get('plot_content_rings')
    content=None
    if rings:
        content=np.zeros((h,w),np.uint8)
        # Enclosed white regions may be a valid pale/white unit (for example an
        # ore-grade class), not empty paper. Include them only after a reviewer
        # has confirmed a near-white legend entry; otherwise retain them as
        # unclassified holes. This avoids both systematic omission and blind fill.
        pale_verified=bool(pale_indices)
        outer=[polygon(r['points']) for r in rings if not r.get('hole') and len(r.get('points',[]))>=3]
        if outer:cv2.fillPoly(content,outer,1)
        for ring in (r for r in rings if r.get('hole') and len(r.get('points',[]))>=3):
            hole_polygon=polygon(ring['points'])
            contains_legend=False
            for entry in cfg.get('legend',[]):
                box=entry.get('detected_box',entry.get('swatch'))
                if box:
                    u,v,sw,sh=box;contains_legend=cv2.pointPolygonTest(hole_polygon.astype(np.float32),((u+sw/2)*sx-wx,(v+sh/2)*sy-wy),False)>=0
                    if contains_legend:break
            if not pale_verified or contains_legend:cv2.fillPoly(content,[hole_polygon],0)
    elif cfg.get('plot_content_polygons'):
        content=np.zeros((h,w),np.uint8)
        polygons=[polygon(poly) for poly in cfg['plot_content_polygons'] if len(poly)>=3]
        if polygons:cv2.fillPoly(content,polygons,1)
    # Coloured axes and borehole guides can exactly match a legend swatch and
    # therefore evade neutral-colour artifact detection. Revoke only
    # near-pixel-width, very long, axis-aligned components. Wider or inclined
    # geological dykes are deliberately excluded from this rule.
    coloured_axis=np.zeros((h,w),bool)
    max_vertical_width=max(2,round(w*.0012));max_horizontal_height=max(2,round(h*.0012))
    for class_index in range(len(names)):
        count,cc,stats,_=cv2.connectedComponentsWithStats((classified==class_index).astype(np.uint8),8)
        vertical_fragments=[];horizontal_fragments=[]
        for component in range(1,count):
            bx,by,bw,bh,_=map(int,stats[component])
            vertical=bw<=max_vertical_width and bh>=h*.18 and bh/max(bw,1)>=50
            horizontal=bh<=max_horizontal_height and bw>=w*.18 and bw/max(bh,1)>=50
            if vertical or horizontal:coloured_axis|=cc==component
            if bw<=max_vertical_width and bh/max(bw,1)>=10:
                vertical_fragments.append((component,bx+bw/2,by,by+bh,bh))
            if bh<=max_horizontal_height and bw/max(bh,1)>=10:
                horizontal_fragments.append((component,by+bh/2,bx,bx+bw,bw))
        # Labels and crossing contacts can split one technical guide into short
        # collinear pieces. Group those pieces before applying the long-span
        # test; genuine broad geological bodies never enter these lists.
        for fragments,limit,total_limit,span_limit in ((vertical_fragments,max_vertical_width*2,h*.18,h*.35),
                                                       (horizontal_fragments,max_horizontal_height*2,w*.18,w*.35)):
            used=set()
            for seed in fragments:
                if seed[0] in used:continue
                group=[item for item in fragments if abs(item[1]-seed[1])<=limit]
                used.update(item[0] for item in group)
                if sum(item[4] for item in group)>=total_limit and max(item[3] for item in group)-min(item[2] for item in group)>=span_limit:
                    for item in group:coloured_axis|=cc==item[0]
    prior_content=content.copy() if content is not None else None
    artifact_stats={'enabled':bool(cfg.get('artifact_preprocessing',True)),'method':'path_first_then_domain_v2'}
    artifact_context=np.zeros((h,w),bool)
    artifact_mask=np.zeros((h,w),bool)
    inpaint_seams=np.zeros((h,w),bool)
    if artifact_stats['enabled']:
        # First pass deliberately runs before the geological domain is fixed.
        # Otherwise grid/drill paths can become part of the outer contour.
        grid_mask,text_mask=detect_masks(crop,cfg.get('auto_text_boxes',[]),sx,sy,(wx,wy),None,
                                         detect_pale_lines=bool(pale_indices))
        catalogue_boundary,catalogue_technical,catalogue_uncertain,catalogue_audit=classify_catalogue_strokes(
            crop,classified,cfg.get('legend_catalog'),(sx+sy)/2,
            cfg.get('legend_line_colour_tolerance',14),cfg.get('line_side_probe_pixels',12))
        # Catalogue colour is primary evidence. Remove confirmed geological
        # strokes from the generic grid candidates before topology is queried.
        line_candidates=(grid_mask|coloured_axis)&~catalogue_boundary
        first_line,protected_boundary,uncertain_line,path_audit=classify_line_paths(
            classified,line_candidates,cfg.get('line_side_probe_pixels',12))
        first_line|=catalogue_technical
        protected_boundary|=catalogue_boundary
        uncertain_line|=catalogue_uncertain
        first_line,corridor_width_audit=expand_confirmed_corridors(
            crop,first_line,cfg.get('artifact_corridor_half_width',14))
        classified[first_line]=-1
        classified,first_line_healing=restore(classified,first_line,cfg.get('artifact_max_gap_pixels',32),cfg.get('artifact_fill_passes',4))
        # Text is a separate object family and uses a shorter reconstruction.
        classified[text_mask]=-1
        classified,text_healing=restore(classified,text_mask,cfg.get('text_max_gap_pixels',18),cfg.get('artifact_fill_passes',4),extended=True)
        classified,oriented_text=restore_text_oriented(classified,text_mask,cfg.get('text_max_gap_pixels',18))
        text_healing={'candidate_pixels':text_healing['candidate_pixels'],
                      'recovered_pixels':text_healing['recovered_pixels']+oriented_text['recovered_pixels'],
                      'unresolved_pixels':oriented_text['unresolved_pixels'],
                      'oriented_recovery':oriented_text}
        # Repair whole confirmed corridors after the conservative pixel pass.
        # Unique-class proposals reconnect beds at line/text intersections;
        # conflicts remain blank rather than overwriting a real contact.
        classified,corridor_repair=reconnect_artifact_corridors(
            classified,first_line|text_mask,cfg.get('artifact_max_gap_pixels',32),
            cfg.get('artifact_reconnect_angle_step',15))
        # The reviewed/automatic curve is now only a maximum prior. The final
        # content domain is rebuilt from cleaned, non-pale geological support.
        unresolved_pre_domain=(first_line|text_mask)&(classified<0)
        content,domain_audit=infer_domain(classified,prior_content,pale_indices,unresolved_pre_domain,
                                          protected_boundary)
        content,outer_domain_audit=enforce_single_outer_domain(content,prior_content)
        domain_audit['single_outer_domain']=outer_domain_audit
        classified[~content]=-1
        # Reconsider only ambiguous line paths using the provisional domain.
        remaining=uncertain_line&content
        second_line,second_boundary,still_uncertain,second_path_audit=classify_line_paths(
            classified,remaining,cfg.get('line_side_probe_pixels',12))
        second_line,second_width_audit=expand_confirmed_corridors(
            crop,second_line,cfg.get('artifact_corridor_half_width',14))
        classified[second_line]=-1
        classified,second_line_healing=restore(classified,second_line,cfg.get('artifact_max_gap_pixels',32),cfg.get('artifact_fill_passes',4))
        classified,second_corridor_repair=reconnect_artifact_corridors(
            classified,second_line,cfg.get('artifact_max_gap_pixels',32),
            cfg.get('artifact_reconnect_angle_step',15))
        line_mask=first_line|second_line;artifact_mask=line_mask|text_mask
        line_healing={'candidate_pixels':first_line_healing['candidate_pixels']+second_line_healing['candidate_pixels'],
                      'recovered_pixels':first_line_healing['recovered_pixels']+second_line_healing['recovered_pixels'],
                      'unresolved_pixels':int(np.count_nonzero(line_mask&(classified<0)))}
        if content is not None:
            # White drill traces may evade colour-based artefact detection and
            # remain connected to exterior whitespace. Bridge any narrow
            # unclassified band only where opposite classified sides agree.
            # Bridge only pixels supported by a detected artifact corridor.
            # Filling every same-colour gap would erase real split events.
            bridge_mask=(classified<0)&content.astype(bool)&cv2.dilate(artifact_mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
            classified,bridge_fill=restore(classified,bridge_mask,cfg.get('bridge_unclassified_gap_pixels',48),cfg.get('artifact_fill_passes',4))
            # Accepted technical corridors and label strokes must not survive
            # as blank separators. Partition them from verified legend classes;
            # reconstructed class seams retain a separate review mask.
            reliable_classes=[index for index,ok in enumerate(verified) if ok]
            classified,inpaint_seams,artifact_inpaint=inpaint_artifact_labels(
                classified,artifact_mask&content.astype(bool),reliable_classes)
            gap_fill={'filled_pixels':0,'filled_components':0,'max_fraction':float(cfg.get('internal_gap_max_fraction',.08)),
                      'reason':'deferred_until_instance_labels_to_preserve_split_merge'}
        else:
            bridge_fill={'candidate_pixels':0,'recovered_pixels':0,'unresolved_pixels':0,'reason':'content_outline_required'}
            artifact_inpaint={'candidate_pixels':0,'filled_pixels':0,'unresolved_pixels':0,'seam_pixels':0,
                              'reason':'content_outline_required'}
            inpaint_seams=np.zeros((h,w),bool)
            gap_fill={'filled_pixels':0,'filled_components':0,'max_fraction':float(cfg.get('internal_gap_max_fraction',.08)),
                      'reason':'content_outline_required'}
        artifact_context=cv2.dilate(artifact_mask.astype(np.uint8),np.ones((31,31),np.uint8)).astype(bool)
        artifact_stats.update(grid_pixels=int(grid_mask.sum()),text_pixels=int(text_mask.sum()),
                              coloured_axis_pixels=int(coloured_axis.sum()),
                              candidate_pixels=int(np.count_nonzero(artifact_mask)),
                              recovered_pixels=line_healing['recovered_pixels']+text_healing['recovered_pixels'],
                              unresolved_pixels=int(np.count_nonzero(artifact_mask&(classified<0))),
                              line_repair=line_healing,text_repair=text_healing,
                              line_path_classification={'first_pass':path_audit,'second_pass':second_path_audit,
                                  'protected_boundary_pixels':int((protected_boundary|second_boundary).sum()),
                                  'uncertain_line_pixels':int(still_uncertain.sum())},
                              legend_line_classification=catalogue_audit,
                              corridor_width_expansion={'first_pass':corridor_width_audit,'second_pass':second_width_audit},
                              corridor_repair={'first_pass':corridor_repair,'second_pass':second_corridor_repair},
                              artifact_colour_inpaint=artifact_inpaint,
                              domain_inference=domain_audit,
                              same_class_gap_bridge=bridge_fill,internal_gap_fill=gap_fill)
        if diagnostics is not None:
            diagnostics['grid_mask']=grid_mask;diagnostics['line_mask']=line_mask
            diagnostics['protected_boundary_mask']=protected_boundary|second_boundary
            diagnostics['catalogue_boundary_mask']=catalogue_boundary
            diagnostics['catalogue_technical_mask']=catalogue_technical
            diagnostics['catalogue_uncertain_mask']=catalogue_uncertain
            diagnostics['uncertain_line_mask']=still_uncertain
            diagnostics['artifact_inpaint_seams']=inpaint_seams
            diagnostics['text_mask']=text_mask;diagnostics['artifact_mask']=artifact_mask
            if content is not None:diagnostics['content_mask']=content.astype(bool)
    else:
        if content is not None:classified[~content.astype(bool)]=-1
        artifact_stats.update(grid_pixels=0,text_pixels=0,candidate_pixels=0,recovered_pixels=0,unresolved_pixels=0,
                               same_class_gap_bridge={'candidate_pixels':0,'recovered_pixels':0,'unresolved_pixels':0},
                               internal_gap_fill={'filled_pixels':0,'filled_components':0,'max_fraction':0.})
    del crop
    exclusion=np.zeros((h,w),bool)
    # Legend boxes can lie inside the rectangular plot ROI. They are layout
    # annotations and must never become geological regions.
    for entry in cfg.get('legend',[]):
        if 'detected_box' not in entry:continue
        u,v,sw,sh=entry['detected_box'];u=round(u*sx)-wx;v=round(v*sy)-wy;sw=round(sw*sx);sh=round(sh*sy)
        cv2.rectangle(exclusion,(max(0,u),max(0,v)),(min(w-1,u+sw),min(h-1,v+sh)),True,-1)
    # Polygons in original image coordinates mark text, faults, axes and annotations.
    for poly in cfg.get('exclude_polygons',[])+cfg.get('fault_polygons',[]):
        temp=np.zeros((h,w),np.uint8)
        cv2.fillPoly(temp,[polygon(poly)],1)
        exclusion |= temp.astype(bool)
    classified[exclusion]=-1
    labels=np.full((h,w),-1,np.int32);layers=[];rejected_pixels=0
    min_fraction=float(cfg.get('min_component_fraction',0.00005))
    if not 0<=min_fraction<=.01:raise ValueError('min_component_fraction must be between 0 and 0.01')
    # Absolute pixels work for small fixtures; a tiny relative floor suppresses
    # JPEG antialiasing fragments in 20–30 MP plots. Rejected components remain
    # unassigned instead of being merged into a nearby geological unit.
    min_area=max(4,round(cfg.get('min_area_pixels',60)*sx*sy),round(h*w*min_fraction))
    def axis_annotation(cc,stats,index):
        x0,y0,bw,bh=map(int,stats[index,:4]);ratio=max(bw/max(bh,1),bh/max(bw,1))
        if (bw<=max_vertical_width and bh>=h*.15 and bh/max(bw,1)>=50) or \
           (bh<=max_horizontal_height and bw>=w*.15 and bw/max(bh,1)>=50):return True
        if ratio<20 or max(bw/w,bh/h)<.2:return False
        local=cc[y0:y0+bh,x0:x0+bw]==index
        overlap=np.count_nonzero(local&artifact_context[y0:y0+bh,x0:x0+bw])/max(1,np.count_nonzero(local))
        return overlap>=.5
    def thin_layer_evidence(cc,stats,index,class_index):
        """Retain a small region only when it has bed-like geometric support."""
        area=int(stats[index,cv2.CC_STAT_AREA])
        if not verified[class_index] or area<max(20,round(min_area*.12)):return False
        mask=(cc==index);points=np.column_stack(np.where(mask)[::-1]).astype(np.float32)
        if len(points)<5:return False
        _,(rw,rh),_=cv2.minAreaRect(points);length=max(rw,rh);width=max(1.,min(rw,rh))
        if length<min(h,w)*.015 or length/width<5:return False
        x0,y0,bw,bh=map(int,stats[index,:4]);local=mask[y0:y0+bh,x0:x0+bw]
        artifact_overlap=np.count_nonzero(local&artifact_context[y0:y0+bh,x0:x0+bw])/max(1,np.count_nonzero(local))
        return artifact_overlap<.35
    count_prior=cfg.get('expected_layer_count')
    count_audit=None
    if count_prior not in [None,'']:
        target=int(count_prior)
        if not 1<=target<=5000:raise ValueError('expected_layer_count must be between 1 and 5000')
        proposed=[]
        # A rough human count is validation evidence only. It must never change
        # the component floor or decide which geological regions survive.
        for k in range(len(names)):
            if not verified[k]:continue
            count,cc,stats,_=cv2.connectedComponentsWithStats((classified==k).astype('uint8'),8)
            proposed.extend(int(stats[c,cv2.CC_STAT_AREA]) for c in range(1,count)
                            if (int(stats[c,cv2.CC_STAT_AREA])>=min_area or thin_layer_evidence(cc,stats,c,k))
                            and not axis_annotation(cc,stats,c))
        before=len(proposed)
        count_audit={'source':'rough_human_count_estimate','requested':target,'components_before':before,
            'changed_segmentation':False,'effective_min_component_pixels':min_area,
            'warning':'The estimate is audit-only and never changes segmentation thresholds or retained regions'}
    axis_annotation_pixels=0;thin_layer_retained=0;thin_layer_pixels=0
    for k,name in enumerate(names):
        count,cc,stats,_=cv2.connectedComponentsWithStats((classified==k).astype('uint8'),8)
        for c in range(1,count):
            if not verified[k]:
                rejected_pixels+=int(stats[c,cv2.CC_STAT_AREA]);continue
            if axis_annotation(cc,stats,c):axis_annotation_pixels+=int(stats[c,cv2.CC_STAT_AREA]);continue
            thin=stats[c,cv2.CC_STAT_AREA]<min_area and thin_layer_evidence(cc,stats,c,k)
            if stats[c,cv2.CC_STAT_AREA]<min_area and not thin:
                rejected_pixels+=int(stats[c,cv2.CC_STAT_AREA]); continue
            mask=cc==c
            q=float(np.clip(1-np.mean(best_error[mask].astype(np.float32))/40,.0,1.)) if verified[k] else .35
            # Lossy JPEG color bleed moves boundaries even if interior colors agree.
            # Keep uncertain fringe pixels unassigned and require human boundary review.
            if p.suffix.lower() in ['.jpg','.jpeg']: q=min(q,.75)
            if cfg.get('layout_reviewed') is False:q=min(q,.65)
            layer=extract(mask,f"{cfg['id']}_L{len(layers)+1}",name if verified[k] else 'UNKNOWN',cfg['bounds'],q,cfg.get('boundary_samples',64))
            if thin:
                thin_layer_retained+=1;thin_layer_pixels+=int(stats[c,cv2.CC_STAT_AREA])
                layer['geometry_flags']=sorted(set(layer.get('geometry_flags',[])+['thin_layer_multi_evidence']))
            layer['legend_verified']=verified[k]
            layer['_pixel_bbox']=[int(stats[c,cv2.CC_STAT_LEFT]),int(stats[c,cv2.CC_STAT_TOP]),int(stats[c,cv2.CC_STAT_WIDTH]),int(stats[c,cv2.CC_STAT_HEIGHT])]
            labels[mask]=len(layers);layers.append(layer)
    # Area calibration intentionally rejects small components. Some of those
    # components lie completely inside a retained stratum; leaving them blank
    # creates artificial holes in both the review map and downstream geometry.
    # Reassign only enclosed blank components, keep exterior-connected paper
    # empty, then rebuild every affected feature from the final label map.
    post_filter_fill={'filled_pixels':0,'filled_components':0,'max_fraction':float(cfg.get('internal_gap_max_fraction',.08)),
                      'reason':'content_outline_required'}
    if content is not None and layers:
        # Nearest-label partitioning can expose a few one-pixel gaps after its
        # first pass. Iterate to a fixed point, with a small hard bound.
        audits=[]
        for _ in range(4):
            labels,current=fill_enclosed_gaps(labels,content.astype(bool),cfg.get('internal_gap_max_fraction',.08))
            audits.append(current)
            if not current['filled_pixels']:break
        post_filter_fill={'filled_pixels':sum(item['filled_pixels'] for item in audits),
                          'filled_components':sum(item['filled_components'] for item in audits),
                          'passes':len(audits),'max_fraction':float(cfg.get('internal_gap_max_fraction',.08))}
        # A blank pixel can be connected to exterior paper through a diagonal
        # one-pixel path and therefore evade the global component test, while
        # still being a true hole in one stratum's contour. Close only blank
        # pixels inside contour child rings; pixels belonging to another layer
        # are never overwritten.
        contour_hole_pixels=0
        for index in range(len(layers)):
            contours,hierarchy=cv2.findContours((labels==index).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:continue
            children=[contour for contour,node in zip(contours,hierarchy[0]) if node[3]>=0]
            if not children:continue
            enclosed=np.zeros(labels.shape,np.uint8);cv2.drawContours(enclosed,children,-1,1,-1)
            fill=(enclosed>0)&(labels<0);count=int(fill.sum())
            if count:labels[fill]=index;contour_hole_pixels+=count
        post_filter_fill['contour_hole_pixels']=contour_hole_pixels
        post_filter_fill['filled_pixels']+=contour_hole_pixels
        # Component filtering must not reopen a technical corridor that was
        # already removed and colour-filled. Repartition only those accepted
        # artefact pixels from retained instances; ordinary unknown areas stay.
        final_artifact=(artifact_mask&content.astype(bool)&~exclusion)
        labels,final_seams,final_artifact_audit=inpaint_artifact_labels(
            labels,final_artifact,range(len(layers)))
        inpaint_seams|=final_seams
        artifact_stats['post_filter_artifact_inpaint']=final_artifact_audit
        post_filter_fill['artifact_pixels']=final_artifact_audit['filled_pixels']
        post_filter_fill['filled_pixels']+=final_artifact_audit['filled_pixels']
        # Reconnecting a corridor can close a loop around a minute pre-existing
        # blank. Fill such newly enclosed blanks once more so final instances do
        # not contain holes even though the corridor itself is complete.
        post_artifact_holes=0
        for index in range(len(layers)):
            contours,hierarchy=cv2.findContours((labels==index).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:continue
            children=[contour for contour,node in zip(contours,hierarchy[0]) if node[3]>=0]
            if not children:continue
            enclosed=np.zeros(labels.shape,np.uint8);cv2.drawContours(enclosed,children,-1,1,-1)
            fill=(enclosed>0)&(labels<0);count=int(fill.sum())
            if count:labels[fill]=index;post_artifact_holes+=count
        post_filter_fill['post_artifact_contour_hole_pixels']=post_artifact_holes
        post_filter_fill['filled_pixels']+=post_artifact_holes
        if post_filter_fill['filled_pixels']:
            rebuilt=[]
            for index,old in enumerate(layers):
                mask=labels==index
                yy,xx=np.where(mask)
                if not len(xx):continue
                layer=extract(mask,old['id'],old['lithology'],cfg['bounds'],old['quality'],cfg.get('boundary_samples',64))
                layer['legend_verified']=old['legend_verified']
                layer['_pixel_bbox']=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]
                rebuilt.append(layer)
            layers=rebuilt
    # Final non-negotiable domain clip. It runs after every hole fill and
    # artifact repaint so no late operation can recreate a stratum outside the
    # accepted outer contour. Features are rebuilt from this clipped label map.
    final_domain_clip={'applied':content is not None,'removed_pixels':0,'removed_layers':0,'sealed_hole_pixels':0}
    if content is not None and layers:
        outside=(labels>=0)&~content.astype(bool)
        final_domain_clip['removed_pixels']=int(outside.sum());labels[outside]=-1
        # Clipping can expose a one-pixel contour child at the domain edge.
        # Seal only blank child rings of one retained instance; never overwrite
        # another stratum and never fill exterior-connected uncertainty.
        for old_index in range(len(layers)):
            contours,hierarchy=cv2.findContours((labels==old_index).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:continue
            children=[contour for contour,node in zip(contours,hierarchy[0]) if node[3]>=0]
            if not children:continue
            enclosed=np.zeros(labels.shape,np.uint8);cv2.drawContours(enclosed,children,-1,1,-1)
            fill=(enclosed>0)&(labels<0)&content.astype(bool)
            final_domain_clip['sealed_hole_pixels']+=int(fill.sum());labels[fill]=old_index
        compact=np.full(labels.shape,-1,np.int32);rebuilt=[]
        for old_index,old in enumerate(layers):
            mask=labels==old_index
            if not np.any(mask):
                final_domain_clip['removed_layers']+=1;continue
            new_index=len(rebuilt);compact[mask]=new_index
            layer=extract(mask,old['id'],old['lithology'],cfg['bounds'],old['quality'],cfg.get('boundary_samples',64))
            layer['legend_verified']=old['legend_verified']
            layer['geometry_flags']=sorted(set(layer.get('geometry_flags',[])+old.get('geometry_flags',[])))
            yy,xx=np.where(mask);layer['_pixel_bbox']=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]
            rebuilt.append(layer)
        labels=compact;layers=rebuilt
        artifact_stats['final_domain_hard_clip']=final_domain_clip
        if diagnostics is not None:diagnostics['content_mask']=content.astype(bool)
    topology(layers,labels,max(1,round(cfg.get('contact_gap_pixels',3)*(sx+sy)/2)))
    if count_audit is not None:
        tolerance=max(15,round(count_audit['requested']*float(cfg.get('expected_layer_tolerance_fraction',.2))))
        count_audit.update(actual=len(layers),tolerance=tolerance,
                           expected_range=[max(1,count_audit['requested']-tolerance),count_audit['requested']+tolerance],
                           within_estimate=abs(len(layers)-count_audit['requested'])<=tolerance)
    uncertainty={'available':content is not None,'boundary_pixels':0,'artifact_pixels':0,'interior_pixels':0,'components':0}
    if content is not None:
        unexplained=(labels<0)&content.astype(bool)
        boundary_band=content.astype(bool)&~cv2.erode(content.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        artifact_band=cv2.dilate(artifact_context.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        boundary_unknown=unexplained&boundary_band
        artifact_unknown=unexplained&~boundary_unknown&artifact_band
        interior_unknown=unexplained&~boundary_unknown&~artifact_unknown
        uncertainty={'available':True,'boundary_pixels':int(boundary_unknown.sum()),
                     'artifact_pixels':int(artifact_unknown.sum()),'interior_pixels':int(interior_unknown.sum()),
                     'components':int(cv2.connectedComponents(unexplained.astype(np.uint8),8)[0]-1),
                     'reconstructed_seam_pixels':int(np.count_nonzero(inpaint_seams&content.astype(bool)))}
        if diagnostics is not None:
            uncertainty_map=np.zeros((h,w),np.uint8);uncertainty_map[boundary_unknown]=1
            uncertainty_map[artifact_unknown]=2;uncertainty_map[interior_unknown]=3
            uncertainty_map[inpaint_seams&content.astype(bool)]=4
            diagnostics['uncertainty_map']=uncertainty_map
    roi_coverage=float(np.mean(labels>=0))
    if content is not None:
        domain=content.astype(bool);coverage=float(np.count_nonzero((labels>=0)&domain)/max(1,np.count_nonzero(domain)))
        unassigned_inside=int(np.count_nonzero((labels<0)&domain))
    else:
        coverage=roi_coverage;unassigned_inside=int(np.count_nonzero(labels<0))
    result={k:cfg[k] for k in ['id','frame','units','station','bounds']}
    result.update(layers=sorted(layers,key=lambda a:a['order']),image=cfg['image'],roi=cfg['roi'],
        complex_structure=bool(cfg.get('fault_polygons') or cfg.get('complex_structure',False)),
        registration_verified=bool(cfg.get('registration_verified',False)),
        complete=bool(cfg.get('complete',False)) and coverage>=cfg.get('minimum_complete_coverage',.85),
        recognition={'coverage':coverage,'roi_coverage':roi_coverage,'unassigned_pixels':unassigned_inside,
            'outside_content_pixels':int(np.count_nonzero((labels<0)&~content.astype(bool))) if content is not None else 0,
            'small_component_pixels':rejected_pixels,
            'layout_reviewed':cfg.get('layout_reviewed'),
            'status':'manual_review_count_estimate' if count_audit else ('manual_review' if not layers or not all(verified) else 'deterministic_extraction'),
            'label_id_mapping':{str(i):l['id'] for i,l in enumerate(layers)},
            'lossy_boundary_review':p.suffix.lower() in ['.jpg','.jpeg'],
            'content_mask_used':bool(rings or cfg.get('plot_content_polygons')),
            'confirmed_pale_regions_enabled':bool(rings and pale_verified),
            'working_shape':[h,w],'source_to_working_scale':[sx,sy],
            'memory_bounded_classification':True,
            'artifact_preprocessing':artifact_stats,
            'axis_annotation_pixels':axis_annotation_pixels,
            'thin_layer_retained':thin_layer_retained,'thin_layer_pixels':thin_layer_pixels,
            'post_filter_gap_fill':post_filter_fill,
            'effective_min_component_pixels':min_area,
            'count_calibration':count_audit,
            'unclassified_audit':uncertainty,
            'limitations':['OCR does not authorize lithology','Same-color touching beds require reviewed instance masks','Dark or irregular non-text annotations may require supplied polygons','Conflicting artifact-corridor continuations remain unclassified','Texture-only lithology requires reviewed instance masks']})
    return validate_section(result),labels

def from_instances(cfg):
    """Import reviewed instance masks: 0 background, positive integer per stratum.

    Useful when two touching beds share RGB or when texture/fault interpretation
    cannot be established by the deterministic color path.
    """
    mask=np.asarray(Image.open(local(cfg['instance_mask'])))
    if mask.ndim!=2 or not np.issubdtype(mask.dtype,np.integer):
        raise ValueError('Instance mask must be a single-channel integer PNG/TIFF')
    entries=cfg.get('instances',[]); ids=[e['value'] for e in entries]
    if len(ids)!=len(set(ids)) or any(v<=0 for v in ids): raise ValueError('Invalid instance IDs')
    if set(np.unique(mask))-{0}-set(ids): raise ValueError('Mask has undeclared instances')
    layers=[];masks=[];labels=np.full(mask.shape,-1,int)
    for e in entries:
        region=mask==e['value']
        if np.sum(region)<8: raise ValueError('Empty/tiny declared instance')
        verified=bool(e.get('verified',False))
        l=extract(region,e['id'],e['lithology'] if verified else 'UNKNOWN',cfg['bounds'],.98 if verified else .35)
        labels[region]=len(layers);layers.append(l);masks.append(region)
    topology(layers,masks,cfg.get('contact_gap_pixels',3))
    result={k:cfg[k] for k in ['id','frame','units','station','bounds']}
    result.update(layers=sorted(layers,key=lambda l:l['order']),registration_verified=bool(cfg.get('registration_verified',False)),
        complex_structure=bool(cfg.get('complex_structure',False)),
        complete=bool(cfg.get('complete',False)),recognition={'mode':'reviewed_instance_mask','status':'manual_review' if any(l['quality']<.8 for l in layers) else 'reviewed',
            'label_id_mapping':{str(i):l['id'] for i,l in enumerate(layers)}})
    return validate_section(result),labels
