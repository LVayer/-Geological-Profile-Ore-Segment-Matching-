"""Four-stage colour-first reconstruction for exported geological sections.

The stages intentionally separate evidence: legend colours, outer domain,
linear gaps, then regional gaps. This makes every irreversible inference
reviewable and prevents annotations from influencing the outer contour.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .io import local
from .artifacts import (inpaint_artifact_labels,detect_masks,classify_catalogue_strokes,
                        classify_line_paths,expand_confirmed_corridors)
from .domain import enforce_single_outer_domain


def _polygon_mask(shape,rings,sx,sy,origin):
    if not rings:return np.ones(shape,bool)
    ox,oy=origin;mask=np.zeros(shape,np.uint8)
    def points(r):return np.asarray([[round(x*sx)-ox,round(y*sy)-oy] for x,y in r['points']],np.int32)
    outer=[points(r) for r in rings if not r.get('hole') and len(r.get('points',[]))>=3]
    if outer:cv2.fillPoly(mask,outer,1)
    for ring in (r for r in rings if r.get('hole') and len(r.get('points',[]))>=3):cv2.fillPoly(mask,[points(ring)],0)
    return mask.astype(bool)


def _load_and_classify(cfg):
    """Stage 1: retain only pixels supported by a reviewed lithology colour."""
    image=local(cfg['image']);x,y,rw,rh=cfg['roi'];maximum=int(cfg.get('max_recognition_pixels',7_500_000))
    with Image.open(image) as source:
        source=source.convert('RGB');ow,oh=source.size;factor=min(1.,np.sqrt(maximum/(rw*rh)))
        if factor<1:source=source.resize((round(ow*factor),round(oh*factor)),Image.Resampling.LANCZOS)
        full=np.asarray(source)
    sx,sy=full.shape[1]/ow,full.shape[0]/oh;wx,wy=round(x*sx),round(y*sy)
    crop=full[wy:round((y+rh)*sy),wx:round((x+rw)*sx)].copy();h,w=crop.shape[:2]
    colours=[];names=[];verified=[]
    for entry in cfg['legend']:
        if 'swatch' in entry:
            u,v,sw,sh=entry['swatch'];u0,v0=round(u*sx),round(v*sy);u1,v1=round((u+sw)*sx),round((v+sh)*sy)
            patch=full[max(0,v0):max(v0+1,v1),max(0,u0):max(u0+1,u1)]
            patch_lab=cv2.cvtColor(patch.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
            # Ignore dark embedded labels when estimating the geological fill.
            keep=patch_lab[:,:,0]>45;pixels=patch[keep] if np.any(keep) else patch.reshape(-1,3)
            colour=np.median(pixels,axis=0)
        else:colour=np.asarray(entry['rgb'])
        colours.append(np.asarray(colour,np.uint8));names.append(entry.get('lithology','UNKNOWN'))
        verified.append(bool(entry.get('verified')) and names[-1]!='UNKNOWN')
    prototypes=np.asarray([cv2.cvtColor(np.float32([[c/255]]),cv2.COLOR_RGB2LAB)[0,0] for c in colours])
    line_prototypes=[]
    for item in (cfg.get('legend_catalog') or {}).get('items',[]):
        line=item.get('line') or {}
        if not line.get('present'):continue
        colour=line.get('dominant_rgb')
        if isinstance(colour,list) and len(colour)==3:
            line_prototypes.append(cv2.cvtColor(np.float32([[np.asarray(colour)/255.]]),cv2.COLOR_RGB2LAB)[0,0])
    labels=np.full((h,w),-1,np.int16);error=np.full((h,w),np.inf,np.float32);margin=np.zeros((h,w),np.float32)
    for row in range(0,h,192):
        end=min(h,row+192);lab=cv2.cvtColor(crop[row:end].astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        best=np.full(lab.shape[:2],np.inf,np.float32);second=best.copy();winner=np.zeros(lab.shape[:2],np.int16)
        for index,prototype in enumerate(prototypes):
            distance=np.linalg.norm(lab-prototype,axis=2);better=distance<best
            second=np.where(better,best,np.minimum(second,distance));best=np.where(better,distance,best);winner[better]=index
        eligible=(best<float(cfg.get('lab_tolerance',12)))&((second-best)>float(cfg.get('lab_margin',4)))
        if line_prototypes:
            line_best=np.full(lab.shape[:2],np.inf,np.float32)
            for prototype in line_prototypes:line_best=np.minimum(line_best,np.linalg.norm(lab-prototype,axis=2))
            # A clear line-colour win overrides the lithology colour. Close
            # ties remain eligible and are resolved later by geometry.
            eligible&=~(line_best+2<best)
        labels[row:end]=np.where(eligible,winner,-1);error[row:end]=best;margin[row:end]=second-best
    # A pale swatch can be geometrically reliable even when OCR cannot read its
    # lithology name.  P11 is such a case: the white ore-grade swatch was
    # labelled UNKNOWN, so requiring ``verified`` classified page background,
    # grids and text holes as one hundred false white layers.  Treat every
    # actual legend swatch with near-white neutral colour as ambiguous here;
    # stage 2 restores only broad source regions inside the geological domain.
    swatch_backed=[bool(entry.get('swatch')) for entry in cfg['legend']]
    pale=[i for i,p in enumerate(prototypes)
          if swatch_backed[i] and p[0]>=88 and np.linalg.norm(p[1:])<8]
    pale_candidate=np.isin(labels,np.asarray(pale,np.int16)) if pale else np.zeros(labels.shape,bool)
    labels[pale_candidate]=-1
    # Remove compact colour-matched lettering while retaining long thin beds.
    area_floor=max(12,round(h*w*.000004));cleaned=np.full(labels.shape,-1,np.int16)
    for value in range(len(colours)):
        count,cc,stats,_=cv2.connectedComponentsWithStats((labels==value).astype(np.uint8),8)
        for component in range(1,count):
            x0,y0,cw,ch,area=map(int,stats[component]);ratio=max(cw/max(ch,1),ch/max(cw,1))
            if area>=area_floor or (ratio>=5 and max(cw,ch)>=max(18,round(min(h,w)*.01))):cleaned[cc==component]=value
    allowed=_polygon_mask((h,w),cfg.get('plot_content_rings'),sx,sy,(wx,wy))
    # Keep ``cleaned`` as the immutable stage-1 colour-classification result.
    # The copy also makes the pre-clean audit count the pixels that were
    # actually withdrawn, instead of observing the already-mutated array.
    labels=cleaned.copy()
    preclean={'enabled':False,'removed_pixels':0}
    if cfg.get('stage1_artifact_preclean',True) and cfg.get('legend_catalog'):
        # P11 contains technical paths whose anti-aliased pixels can win a
        # lithology colour even though their complete geometry is a grid,
        # borehole guide or reviewed technical symbol.  Classify these paths
        # before the outer contour is inferred; later stages refill the vacated
        # pixels from neighbouring geological classes.
        grid_mask,text_mask=detect_masks(crop,cfg.get('auto_text_boxes',[]),sx,sy,(wx,wy),allowed,
                                         detect_pale_lines=bool(pale))
        catalogue_boundary,catalogue_technical,catalogue_uncertain,catalogue_audit=classify_catalogue_strokes(
            crop,labels,cfg.get('legend_catalog'),(sx+sy)/2,
            cfg.get('legend_line_colour_tolerance',14),cfg.get('line_side_probe_pixels',12))
        path_artifact,path_boundary,path_uncertain,path_audit=classify_line_paths(
            labels,grid_mask&~catalogue_boundary,cfg.get('line_side_probe_pixels',12))
        technical=path_artifact|catalogue_technical
        technical,width_audit=expand_confirmed_corridors(
            crop,technical,cfg.get('artifact_corridor_half_width',14))
        removed=(technical|text_mask)&allowed
        retained_before=int(np.count_nonzero(labels>=0))
        removed_classified=int(np.count_nonzero(removed&(labels>=0)))
        labels[removed]=-1
        preclean={'enabled':True,'retained_pixels_before':retained_before,
                  'retained_pixels_after':int(np.count_nonzero(labels>=0)),
                  'removed_pixels':removed_classified,
                  'grid_candidate_pixels':int(grid_mask.sum()),'text_candidate_pixels':int(text_mask.sum()),
                  'technical_path_pixels':int(technical.sum()),
                  'protected_geological_boundary_pixels':int((catalogue_boundary|path_boundary).sum()),
                  'uncertain_line_pixels':int((catalogue_uncertain|path_uncertain).sum()),
                  'catalogue':catalogue_audit,'path_classification':path_audit,
                  'corridor_width_expansion':width_audit}
    audit={'working_shape':[h,w],'scale':[sx,sy],'classified_pixels':int((labels>=0).sum()),
           'background_pixels':int((labels<0).sum()),'pale_classes':pale,
           'ambiguous_pale_pixels':int(pale_candidate.sum()),'line_colour_prototypes':len(line_prototypes),
           'artifact_preclean':preclean}
    return crop,labels,colours,names,pale,pale_candidate,allowed,(sx,sy),(wx,wy),audit


def _chaikin_closed(points,iterations=1):
    """Smooth a closed polygon without fitting high-curvature splines."""
    current=np.asarray(points,np.float32)
    for _ in range(max(0,int(iterations))):
        following=np.roll(current,-1,axis=0)
        q=.75*current+.25*following;r=.25*current+.75*following
        merged=np.empty((len(current)*2,2),np.float32);merged[0::2]=q;merged[1::2]=r;current=merged
    return np.rint(current).astype(np.int32)


def _prune_hairpins(points,max_path=180.,ratio=.58):
    """Remove short narrow out-and-back excursions from a closed polygon."""
    pts=[np.asarray(p,np.float32) for p in points];removed=0
    # Rebuild the list after every removal.  This also handles a candidate that
    # crosses the first/last vertex of the closed contour without stale indices.
    for _ in range(8):
        changed=False;n=len(pts)
        if n<6:break
        # Test one- and two-vertex excursions. A real corner has a chord close
        # to its path length; a thin spike travels out and nearly back.
        for inside in (2,1):
            for start in range(n):
                ids=[(start+j)%n for j in range(inside+2)]
                seq=[pts[i] for i in ids]
                path=sum(float(np.linalg.norm(seq[j+1]-seq[j])) for j in range(len(seq)-1))
                chord=float(np.linalg.norm(seq[-1]-seq[0]))
                if 10<path<=max_path and chord<path*ratio:
                    discard=set(ids[1:-1])
                    pts=[point for index,point in enumerate(pts) if index not in discard]
                    removed+=inside;changed=True;break
            if changed:break
        if not changed:break
    return np.asarray(pts,np.float32),removed


def _connect_local_short_gaps(clean,hough_lines,allowed):
    """Join outline gaps shorter than 70% of the shortest nearby line segment."""
    direct=clean.copy()
    if hough_lines is None:return direct,{'candidate_gaps':0,'connected_gaps':0,'connections':[]}
    raw=np.asarray(hough_lines).reshape(-1,4);segments=[];midpoints=[]
    for x1,y1,x2,y2 in raw:
        vector=np.asarray([x2-x1,y2-y1],np.float64);length=float(np.linalg.norm(vector))
        if length<=0:continue
        unit=vector/length;index=len(segments)
        segments.append({'p0':(int(x1),int(y1)),'p1':(int(x2),int(y2)),
                         'unit':unit,'length':length,'index':index})
        midpoints.append(((x1+x2)/2.,(y1+y2)/2.))
    if not segments:return direct,{'candidate_gaps':0,'connected_gaps':0,'connections':[]}
    midpoints=np.asarray(midpoints,np.float64);endpoints=[];cell_size=64;grid={}
    for segment in segments:
        # The extension direction points away from the observed segment.
        for side,point,outward in ((0,segment['p0'],-segment['unit']),(1,segment['p1'],segment['unit'])):
            item={'segment':segment['index'],'side':side,'point':point,'outward':outward}
            endpoint_index=len(endpoints);endpoints.append(item)
            cell=(point[0]//cell_size,point[1]//cell_size);grid.setdefault(cell,[]).append(endpoint_index)

    candidates=[];cosine_limit=float(np.cos(np.deg2rad(30.)))
    for left,item in enumerate(endpoints):
        x,y=item['point'];cx,cy=x//cell_size,y//cell_size
        nearby_endpoint_ids=[]
        for gx in range(cx-2,cx+3):
            for gy in range(cy-2,cy+3):nearby_endpoint_ids.extend(grid.get((gx,gy),[]))
        for right in nearby_endpoint_ids:
            if right<=left:continue
            other=endpoints[right]
            if item['segment']==other['segment']:continue
            difference=np.asarray(other['point'],np.float64)-np.asarray(item['point'],np.float64)
            gap=float(np.linalg.norm(difference))
            if gap<=1.5 or gap>128:continue
            direction=difference/gap
            if float(np.dot(item['outward'],direction))<cosine_limit:continue
            if float(np.dot(other['outward'],-direction))<cosine_limit:continue
            midpoint=(np.asarray(item['point'])+np.asarray(other['point']))/2.
            distances=np.linalg.norm(midpoints-midpoint,axis=1)
            nearest=np.argpartition(distances,min(5,len(distances)-1))[:min(6,len(distances))]
            shortest=min(segments[int(index)]['length'] for index in nearest)
            threshold=.7*shortest
            if gap>=threshold:continue
            candidates.append((gap,left,right,threshold))

    candidates.sort(key=lambda value:value[0]);used=set();accepted=[]
    for gap,left,right,threshold in candidates:
        if left in used or right in used:continue
        first=endpoints[left]['point'];second=endpoints[right]['point']
        line=np.zeros_like(clean,np.uint8);cv2.line(line,first,second,1,1,cv2.LINE_8)
        if np.any((line>0)&~allowed):continue
        endpoint_mask=np.zeros_like(clean,np.uint8)
        cv2.circle(endpoint_mask,first,3,1,-1);cv2.circle(endpoint_mask,second,3,1,-1)
        interior=(line>0)&(endpoint_mask==0);pixels=int(interior.sum())
        if pixels and np.count_nonzero((clean==0)&interior)/pixels<.8:continue
        direct|=line;used.update((left,right))
        accepted.append({'gap_px':round(gap,2),'threshold_0_7_px':round(threshold,2)})
    return direct,{'candidate_gaps':len(candidates),'connected_gaps':len(accepted),
                   'connections':accepted,'rule':'gap < 0.7 * shortest of six nearby Hough segments'}


def _outer_domain(labels,pale,allowed,pale_candidate=None):
    """Stage 2: connect confident non-background outer arcs at two scales."""
    support=(labels>=0)&allowed
    if pale:support&=~np.isin(labels,np.asarray(pale,np.int16))
    h,w=support.shape;count,cc,stats,_=cv2.connectedComponentsWithStats(support.astype(np.uint8),8)
    clean=np.zeros_like(support,np.uint8);floor=max(12,round(h*w*.00002))
    for index in range(1,count):
        if stats[index,cv2.CC_STAT_AREA]>=floor:clean[cc==index]=1
    # Outer tracing uses a two-pixel interior core. A one-pixel coloured axis or
    # annotation may survive colour filtering but cannot pull the domain edge.
    core=cv2.erode(clean,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    clean=cv2.dilate(core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    edge=clean&~cv2.erode(clean,np.ones((3,3),np.uint8))
    lines=cv2.HoughLinesP(edge*255,1,np.pi/180,threshold=max(18,round(min(h,w)*.01)),
        minLineLength=max(18,round(min(h,w)*.012)),maxLineGap=max(3,round(min(h,w)*.004)))
    lengths=[]
    if lines is not None:
        for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):lengths.append(float(np.hypot(x2-x1,y2-y1)))
    nearby=float(np.median(lengths)) if lengths else max(16,min(h,w)*.025)
    # First-version rule: short gaps use straight directional closing; gaps of
    # comparable local scale use a restrained rounded connection.
    direct_gap=max(6,min(28,round(nearby*.22)))
    curve_gap=max(direct_gap+4,min(56,round(nearby*.48)))
    direct=clean.copy();size=2*direct_gap+1;centre=direct_gap
    for angle in range(0,180,15):
        a=np.deg2rad(angle);dx=round(np.cos(a)*direct_gap);dy=round(np.sin(a)*direct_gap)
        kernel=np.zeros((size,size),np.uint8)
        cv2.line(kernel,(centre-dx,centre-dy),(centre+dx,centre+dy),1,1)
        direct|=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,kernel)
    # The second connection has rounded ends and therefore represents a low-
    # curvature continuation rather than an unconstrained straight chord.
    radius=max(3,curve_gap//2);rounded=cv2.morphologyEx(direct,cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    rounded=(rounded>0)&allowed
    contours,_=cv2.findContours(rounded.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    domain=np.zeros((h,w),np.uint8)
    if contours:
        cv2.fillPoly(domain,[max(contours,key=cv2.contourArea)],1)
    domain,audit_single=enforce_single_outer_domain(domain.astype(bool),allowed)
    return domain,{'confident_support_pixels':int(clean.sum()),'median_nearby_segment_px':round(nearby,2),
        'direct_gap_px':direct_gap,'curved_gap_px':curve_gap,
        'confident_edge_pixels':int(edge.sum()),
        'single_outer_domain':audit_single,'domain_pixels':int(domain.sum())}


def _seed_pale_regions(labels,pale_candidate,pale,domain):
    """Restore broad source pixels of a background-coloured lithology.

    The exact candidate positions come from the original image before other
    colours are cleared.  A true pale bed can have a sparse bounding box after
    labels, grids and boreholes cut through it, so bounding-box density must
    not reject it.  Local half-width is the safer discriminator: lettering and
    technical strokes stay thin, while a geological area contains a broad core.
    """
    out=labels.copy();audit={'candidate_pixels':int(np.count_nonzero(pale_candidate&domain)),
        'seeded_pixels':0,'seeded_regions':0,'rejected_thin_pixels':0,
        'source_position_preserved':True}
    if not pale:return out,audit
    mask=pale_candidate&domain;count,cc,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    floor=max(80,round(mask.size*.00002));domain_area=max(1,int(domain.sum()))
    for component in range(1,count):
        x,y,w,h,area=map(int,stats[component]);aspect=max(w/max(h,1),h/max(w,1))
        local=cc[y:y+h,x:x+w]==component;radius=float(cv2.distanceTransform(local.astype(np.uint8),cv2.DIST_L2,5).max())
        # Do not cap the area or require a dense axis-aligned bounding box: a
        # legitimate background-coloured bed may be large, inclined and split
        # internally by many removed annotations.  The broad-core requirement
        # still rejects narrow white grid, drill and label strokes.
        if area>=floor and aspect<=24 and radius>=4:
            out[cc==component]=pale[0];audit['seeded_pixels']+=area;audit['seeded_regions']+=1
        else:audit['rejected_thin_pixels']+=area
    return out,audit


def _linear_candidates(labels,domain,probe):
    unknown=(labels<0)&domain;candidate=np.zeros(labels.shape,bool)
    for dy,dx in ((1,0),(0,1),(1,1),(1,-1),(1,2),(2,1),(1,-2),(2,-1)):
        negative=np.full(labels.shape,-1,np.int16);positive=negative.copy()
        for distance in range(1,probe+1):
            yy=abs(dy)*distance;xx=abs(dx)*distance
            if dx==0:
                negative[yy:]=np.where((negative[yy:]<0)&(labels[:-yy]>=0),labels[:-yy],negative[yy:])
                positive[:-yy]=np.where((positive[:-yy]<0)&(labels[yy:]>=0),labels[yy:],positive[:-yy])
            elif dy==0:
                negative[:,xx:]=np.where((negative[:,xx:]<0)&(labels[:,:-xx]>=0),labels[:,:-xx],negative[:,xx:])
                positive[:,:-xx]=np.where((positive[:,:-xx]<0)&(labels[:,xx:]>=0),labels[:,xx:],positive[:,:-xx])
            elif dx>0:
                negative[yy:,xx:]=np.where((negative[yy:,xx:]<0)&(labels[:-yy,:-xx]>=0),labels[:-yy,:-xx],negative[yy:,xx:])
                positive[:-yy,:-xx]=np.where((positive[:-yy,:-xx]<0)&(labels[yy:,xx:]>=0),labels[yy:,xx:],positive[:-yy,:-xx])
            else:
                negative[yy:,:-xx]=np.where((negative[yy:,:-xx]<0)&(labels[:-yy,xx:]>=0),labels[:-yy,xx:],negative[yy:,:-xx])
                positive[:-yy,xx:]=np.where((positive[:-yy,xx:]<0)&(labels[yy:,:-xx]>=0),labels[yy:,:-xx],positive[:-yy,xx:])
        candidate|=unknown&(negative>=0)&(positive>=0)
    return candidate


def _repair_linear_gaps(labels,domain,probe=16):
    out=labels.copy();audits=[]
    for _ in range(4):
        candidate=_linear_candidates(out,domain,probe)
        if not np.any(candidate):break
        before=int((out<0).sum());out,seams,audit=inpaint_artifact_labels(out,candidate,range(int(out.max())+1))
        audit['seam_pixels']=int(seams.sum());audits.append(audit)
        if int((out<0).sum())==before:break
    return out,{'passes':len(audits),'candidate_pixels':sum(a['candidate_pixels'] for a in audits),
                'filled_pixels':sum(a['filled_pixels'] for a in audits),'passes_detail':audits}


def _distance_partition(local_labels,component,neighbours):
    costs=[]
    for value in neighbours:
        costs.append(cv2.distanceTransform((local_labels!=value).astype(np.uint8),cv2.DIST_L2,5))
    winner=np.argmin(np.stack(costs),axis=0)
    result=local_labels.copy()
    for index,value in enumerate(neighbours):result[component&(winner==index)]=value
    return result


def _curve_partition(local_labels,component,a,b):
    """Partition a two-colour void by a capped low-curvature contact fit."""
    ma=local_labels==a;mb=local_labels==b
    contact=cv2.dilate(ma.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)&\
            cv2.dilate(mb.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)&~component
    yy,xx=np.where(contact)
    if len(xx)<18:return None
    points=np.column_stack((xx,yy)).astype(np.float64);centre=points.mean(axis=0)
    _,_,vt=np.linalg.svd(points-centre,full_matrices=False);uv=(points-centre)@vt.T
    if np.ptp(uv[:,0])<8:return None
    coeff=np.polyfit(uv[:,0],uv[:,1],2);span=max(1.,np.ptp(uv[:,0]));limit=.12*span/(span*span)
    coeff[0]=float(np.clip(coeff[0],-limit,limit))
    cy,cx=np.where(component);query=(np.column_stack((cx,cy))-centre)@vt.T
    signed=query[:,1]-np.polyval(coeff,query[:,0])
    ay,ax=np.where(ma&cv2.dilate(component.astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool))
    by,bx=np.where(mb&cv2.dilate(component.astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool))
    if not len(ax) or not len(bx):return None
    asign=np.median(((np.column_stack((ax,ay))-centre)@vt.T)[:,1]-np.polyval(coeff,((np.column_stack((ax,ay))-centre)@vt.T)[:,0]))
    bsign=np.median(((np.column_stack((bx,by))-centre)@vt.T)[:,1]-np.polyval(coeff,((np.column_stack((bx,by))-centre)@vt.T)[:,0]))
    if asign*bsign>=0:return None
    result=local_labels.copy();choose_a=signed*np.sign(asign)>=0
    result[cy[choose_a],cx[choose_a]]=a;result[cy[~choose_a],cx[~choose_a]]=b
    return result


def _repair_regions(labels,domain):
    out=labels.copy();unknown=(out<0)&domain;count,cc,stats,_=cv2.connectedComponentsWithStats(unknown.astype(np.uint8),8)
    audit={'components':max(0,count-1),'single_colour':0,'curve_partition':0,'distance_partition':0,'filled_pixels':0}
    for index in range(1,count):
        x,y,w,h,area=map(int,stats[index]);margin=max(12,min(80,round(max(w,h)*.25)))
        x0=max(0,x-margin);y0=max(0,y-margin);x1=min(out.shape[1],x+w+margin);y1=min(out.shape[0],y+h+margin)
        local=out[y0:y1,x0:x1];component=cc[y0:y1,x0:x1]==index
        ring=cv2.dilate(component.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)&~component
        neighbours=[int(v) for v in np.unique(local[ring]) if v>=0]
        if not neighbours:continue
        if len(neighbours)==1:local[component]=neighbours[0];audit['single_colour']+=1
        elif len(neighbours)==2:
            result=_curve_partition(local,component,*neighbours)
            if result is None:result=_distance_partition(local,component,neighbours);audit['distance_partition']+=1
            else:audit['curve_partition']+=1
            local[:]=result
        else:
            local[:]=_distance_partition(local,component,neighbours);audit['distance_partition']+=1
        out[y0:y1,x0:x1]=local;audit['filled_pixels']+=area
    audit['unfilled_pixels']=int(np.count_nonzero((out<0)&domain))
    return out,audit


def _render(labels,colours,domain,outer_width=4,internal_width=2):
    image=np.full((*labels.shape,3),255,np.uint8)
    for index,colour in enumerate(colours):image[labels==index]=colour
    if internal_width:
        boundary=np.zeros(labels.shape,bool)
        for dy,dx in ((1,0),(0,1)):
            different=(labels[dy:,dx:]>=0)&(labels[:-dy or None,:-dx or None]>=0)&(labels[dy:,dx:]!=labels[:-dy or None,:-dx or None])
            boundary[dy:,dx:]|=different;boundary[:-dy or None,:-dx or None]|=different
        boundary=cv2.dilate(boundary.astype(np.uint8),np.ones((internal_width,internal_width),np.uint8)).astype(bool)
        image[boundary]=0
    if domain is not None:
        contours,_=cv2.findContours(domain.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image,contours,-1,(0,0,0),outer_width,lineType=cv2.LINE_8)
        image[~domain]=255;cv2.drawContours(image,contours,-1,(0,0,0),outer_width,lineType=cv2.LINE_8)
    return image


def _longest_circular_run(flags):
    """Return the largest contiguous angular sector marked True."""
    values=np.asarray(flags,bool)
    if not np.any(values):return 0
    doubled=np.concatenate((values,values));best=current=0
    for value in doubled:
        current=current+1 if value else 0;best=max(best,current)
    return min(best,len(values))


def _background_seed_is_open(walkable,allowed,y,x,radius=36,directions=32):
    """Test whether a nearby background-colour seed has a broad escape sector."""
    h,w=walkable.shape;open_rays=[]
    for angle in np.linspace(0,2*np.pi,directions,endpoint=False):
        dy,dx=np.sin(angle),np.cos(angle);escaped=True;last=(-1,-1)
        for step in range(1,radius+1):
            yy=int(round(y+dy*step));xx=int(round(x+dx*step))
            if (yy,xx)==last:continue
            last=(yy,xx)
            if yy<0 or yy>=h or xx<0 or xx>=w or not allowed[yy,xx]:break
            if not walkable[yy,xx]:escaped=False;break
        open_rays.append(escaped)
    # Eight of 32 adjacent rays form a 90-degree opening.  A thin enclosed bed
    # can be long, but normally exposes only two narrow opposite directions.
    return _longest_circular_run(open_rays)>=max(4,directions//4)


def _direction_anomaly_components(arcs,threshold=20.):
    """Find arcs inconsistent with the axial direction of nearby segments."""
    count,cc,stats,centroids=cv2.connectedComponentsWithStats(arcs.astype(np.uint8),8)
    records=[]
    for index in range(1,count):
        x,y,cw,ch,area=map(int,stats[index])
        if area<12 or max(cw,ch)<10:continue
        yy,xx=np.where(cc[y:y+ch,x:x+cw]==index)
        points=np.column_stack((xx+x,yy+y)).astype(np.float32);centre=points.mean(axis=0)
        _,_,vt=np.linalg.svd(points-centre,full_matrices=False)
        angle=float(np.degrees(np.arctan2(vt[0,1],vt[0,0]))%180.)
        records.append({'index':index,'centre':np.asarray(centroids[index]),'angle':angle})

    def difference(a,b):
        value=abs(float(a)-float(b))%180.
        return min(value,180.-value)

    neighbour_radius=max(80.,min(arcs.shape)*.10);anomalies=set();evaluated=0;deviations=[]
    for position,record in enumerate(records):
        nearby=[]
        for other_position,other in enumerate(records):
            if position==other_position:continue
            distance=float(np.linalg.norm(record['centre']-other['centre']))
            if distance<=neighbour_radius:nearby.append((distance,other))
        neighbours=[item for _,item in sorted(nearby,key=lambda item:item[0])[:6]]
        if len(neighbours)<3:continue
        evaluated+=1;angles=[item['angle'] for item in neighbours]
        majority=min(angles,key=lambda candidate:sum(difference(candidate,angle) for angle in angles))
        deviation=difference(record['angle'],majority);deviations.append(deviation)
        if deviation>threshold:anomalies.add(record['index'])
    return cc,anomalies,{'total_components':len(records),'evaluated_components':evaluated,
        'anomaly_components':len(anomalies),'direction_threshold_deg':threshold,
        'nearby_segment_count':6,'nearby_radius_px':round(neighbour_radius,2),
        'maximum_deviation_deg':round(max(deviations),2) if deviations else 0}


def _background_distance(crop,colours,pale,allowed):
    """Compute LAB distance to page background and reviewed pale lithologies."""
    outside=crop[~allowed]
    if not len(outside):outside=crop.reshape(-1,3)
    # Sampling bounds the cost without changing the coordinate-aligned result.
    step=max(1,len(outside)//200000);page_rgb=np.median(outside[::step],axis=0)
    references=[page_rgb]
    references.extend(np.asarray(colours[index],np.float32) for index in pale)
    reference_lab=[]
    for colour in references:
        reference_lab.append(cv2.cvtColor(np.float32([[colour/255.]]),cv2.COLOR_RGB2LAB)[0,0])
    distance=np.full(crop.shape[:2],np.inf,np.float32)
    for row in range(0,crop.shape[0],160):
        end=min(crop.shape[0],row+160)
        lab=cv2.cvtColor(crop[row:end].astype(np.float32)/255.,cv2.COLOR_RGB2LAB)
        local=np.full(lab.shape[:2],np.inf,np.float32)
        for reference in reference_lab:local=np.minimum(local,np.linalg.norm(lab-reference,axis=2))
        distance[row:end]=local
    return distance,page_rgb


def _exterior_vote_map(distance,thresholds=(5.5,8.5,12.)):
    """Accumulate exterior reachability over colour tolerances and 4/8-neighbour floods."""
    votes=np.zeros(distance.shape,np.uint8);runs=0;run_pixels=[]
    for threshold in thresholds:
        walkable=(distance<=threshold).astype(np.uint8)
        for connectivity in (4,8):
            count,labels,stats,_=cv2.connectedComponentsWithStats(walkable,connectivity)
            border=np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
            border=border[border>0]
            if count>1:
                largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));border=np.unique(np.append(border,largest))
            exterior=np.isin(labels,border)
            votes+=exterior.astype(np.uint8);runs+=1;run_pixels.append(int(exterior.sum()))
    return votes,{'thresholds_lab':list(thresholds),'connectivities':[4,8],
        'flood_runs':runs,'exterior_pixels_per_run':run_pixels}


def _arc_sample_points(component):
    """Sample endpoints, middle, uniform positions and strong bends of a thin arc."""
    yy,xx=np.where(component)
    if not len(xx):return []
    points=np.column_stack((xx,yy)).astype(np.float32);centre=points.mean(axis=0)
    _,_,vt=np.linalg.svd(points-centre,full_matrices=False)
    along=(points-centre)@vt[0];across=np.abs((points-centre)@vt[1])
    order=np.argsort(along);fractions=(0.,.08,.25,.5,.75,.92,1.)
    selected=[points[order[min(len(order)-1,round(value*(len(order)-1)))]] for value in fractions]
    # Large perpendicular deviations are a stable proxy for bends when a
    # one-pixel component cannot be traversed as a simple chain.
    for index in np.argsort(across)[-min(3,len(points)):]:selected.append(points[index])
    unique=[];seen=set()
    for point in selected:
        key=(int(round(point[0])),int(round(point[1])))
        if key not in seen:seen.add(key);unique.append(key)
    return unique


def _water_injection_filter(arcs,crop,colours,pale,allowed,anomaly_ids):
    """Validate immediately adjacent seeds by pixelwise multi-source flooding."""
    distance,page_rgb=_background_distance(crop,colours,pale,allowed)
    votes,flood_audit=_exterior_vote_map(distance)
    stable_exterior=votes>=flood_audit['flood_runs']-1;nominal_background=distance<=8.5
    count,cc,stats,_=cv2.connectedComponentsWithStats(arcs.astype(np.uint8),8)
    kept=np.zeros_like(arcs);details=[];radius=8
    for index in range(1,count):
        x,y,cw,ch,area=map(int,stats[index])
        if area<12 or max(cw,ch)<10:continue
        component=cc==index;samples=_arc_sample_points(component);passed=0;usable=0
        for px,py in samples:
            y0=max(0,py-radius);y1=min(arcs.shape[0],py+radius+1)
            x0=max(0,px-radius);x1=min(arcs.shape[1],px+radius+1)
            yy,xx=np.ogrid[y0:y1,x0:x1];ring=(xx-px)**2+(yy-py)**2
            near=(ring>=1)&(ring<=radius*radius)&nominal_background[y0:y1,x0:x1]
            if not np.any(near):continue
            usable+=1
            if np.any(near&stable_exterior[y0:y1,x0:x1]):passed+=1
        ratio=passed/max(1,usable);required=.70 if index in anomaly_ids else .50
        accepted=usable>=max(2,len(samples)//3) and ratio>=required
        if accepted:kept[component]=True
        details.append((index,accepted,round(ratio,3),usable,len(samples),index in anomaly_ids))
    return kept,{'seed_radius_px':radius,'coordinate_frame':'source-aligned working crop',
        'page_background_rgb':[round(float(value),2) for value in page_rgb],
        'required_stable_flood_votes':flood_audit['flood_runs']-1,
        'normal_minimum_pass_ratio':.50,'direction_anomaly_minimum_pass_ratio':.70,
        'input_components':max(0,count-1),'kept_components':sum(item[1] for item in details),
        'rejected_components':sum(not item[1] for item in details),
        'flood':flood_audit}


def _strict_background_escape(arcs,walkable,allowed):
    """Apply escape validation only to locally direction-inconsistent arcs."""
    count,cc,stats,centroids=cv2.connectedComponentsWithStats(arcs.astype(np.uint8),8)
    accepted=np.zeros_like(arcs);records=[]
    near_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    radius=max(24,min(48,round(min(arcs.shape)*.018)))
    for index in range(1,count):
        x,y,cw,ch,area=map(int,stats[index])
        if area<12 or max(cw,ch)<10:continue
        local_y,local_x=np.where(cc[y:y+ch,x:x+cw]==index)
        points=np.column_stack((local_x+x,local_y+y)).astype(np.float32)
        centre=points.mean(axis=0);_,_,vt=np.linalg.svd(points-centre,full_matrices=False)
        direction=float(np.degrees(np.arctan2(vt[0,1],vt[0,0]))%180.)
        records.append({'index':index,'area':area,'centre':np.asarray(centroids[index]),
                        'direction':direction})

    def axial_difference(a,b):
        difference=abs(float(a)-float(b))%180.
        return min(difference,180.-difference)

    neighbour_radius=max(80.,min(arcs.shape)*.10);anomalies=[];escape_passed=0
    direction_evaluated=0;deviations=[]
    for position,record in enumerate(records):
        distances=[]
        for other_position,other in enumerate(records):
            if other_position==position:continue
            distance=float(np.linalg.norm(record['centre']-other['centre']))
            if distance<=neighbour_radius:distances.append((distance,other_position))
        neighbours=[records[i] for _,i in sorted(distances)[:6]]
        # With fewer than three nearby arcs there is no defensible local
        # majority, so direction alone must not trigger rejection.
        if len(neighbours)<3:
            accepted[cc==record['index']]=True;continue
        direction_evaluated+=1
        neighbour_angles=[item['direction'] for item in neighbours]
        # Axial medoid gives one vote per nearby segment and is resistant to one
        # unusually long or noisy line dominating the local direction.
        majority=min(neighbour_angles,key=lambda candidate:
            sum(axial_difference(candidate,angle) for angle in neighbour_angles))
        deviation=axial_difference(record['direction'],majority);deviations.append(round(deviation,2))
        if deviation<=20.:
            accepted[cc==record['index']]=True;continue
        anomalies.append(record['index'])
        component=cc==record['index']
        seeds=cv2.dilate(component.astype(np.uint8),near_kernel).astype(bool)&walkable
        yy,xx=np.where(seeds)
        if not len(xx):continue
        # Spread samples along the local boundary while bounding runtime.
        selection=np.linspace(0,len(xx)-1,min(9,len(xx)),dtype=int)
        votes=sum(_background_seed_is_open(walkable,allowed,int(yy[i]),int(xx[i]),radius)
                  for i in selection)
        if votes>=max(1,int(np.ceil(len(selection)*.45))):
            accepted[component]=True;escape_passed+=1
    return accepted,{'total_components':len(records),'direction_evaluated_components':direction_evaluated,
        'direction_anomaly_components':len(anomalies),'direction_threshold_deg':20,
        'nearby_segment_count':6,'nearby_radius_px':round(neighbour_radius,2),
        'escape_tested_components':len(anomalies),'escape_passed_components':escape_passed,
        'escape_rejected_components':len(anomalies)-escape_passed,
        'maximum_direction_deviation_deg':max(deviations) if deviations else 0,
        'escape_radius_px':radius,'directions_per_seed':32,'required_open_sector_deg':90}


def _external_band(mask,width=9):
    """Return a narrow band on mask edges that face crop-exterior background."""
    background=(mask==0).astype(np.uint8)
    count,cc=cv2.connectedComponents(background,8)
    border_ids=np.unique(np.concatenate((cc[0],cc[-1],cc[:,0],cc[:,-1])))
    border_ids=border_ids[border_ids>0]
    exterior=np.isin(cc,border_ids) if count>1 else np.zeros(mask.shape,bool)
    edge=(mask>0)&cv2.dilate(exterior.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    return cv2.dilate(edge.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*width+1,2*width+1))).astype(bool)


def _remove_internal_parallel_edges(trusted,clean):
    """Reject comb-like internal flanks using multi-scale virtual envelopes."""
    # These masks are evidence filters only.  They are not rendered and never
    # alter the classified lithology.  Narrow pale gaps vanish at larger scales,
    # while a true section exterior remains exterior at every scale.
    minimum=min(trusted.shape)
    sizes=[]
    for fraction in (.007,.012,.018):
        size=max(11,round(minimum*fraction));size+=1-size%2;sizes.append(size)
    votes=np.zeros(trusted.shape,np.uint8)
    for size in sizes:
        envelope=cv2.morphologyEx(clean.astype(np.uint8),cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
        votes+=_external_band(envelope,max(5,size//5)).astype(np.uint8)

    count,cc,stats,_=cv2.connectedComponentsWithStats(trusted.astype(np.uint8),8)
    kept=np.zeros_like(trusted);removed=[];kept_count=0
    for index in range(1,count):
        component=cc==index;area=int(stats[index,cv2.CC_STAT_AREA])
        # Require at least two of the three scales to regard most of the arc as
        # exterior-facing.  This specifically removes repeated parallel flanks.
        exterior_ratio=float(np.count_nonzero(component&(votes>=2))/max(1,area))
        if exterior_ratio>=.55:kept[component]=True;kept_count+=1
        else:removed.append({'component':index,'pixels':area,'exterior_ratio':round(exterior_ratio,3)})
    return kept,{'scales_px':sizes,'required_scale_votes':2,'minimum_exterior_ratio':.55,
        'input_components':max(0,count-1),'kept_components':kept_count,
        'removed_components':len(removed),'removed_pixels':sum(item['pixels'] for item in removed)}


def _remove_bilateral_material_edges(trusted,clean):
    """Remove arcs with reliable geological colour on both normal sides."""
    count,cc,stats,_=cv2.connectedComponentsWithStats(trusted.astype(np.uint8),8)
    kept=np.zeros_like(trusted);removed=[];probe=max(32,min(72,round(min(trusted.shape)*.028)))
    for index in range(1,count):
        component=cc==index;yy,xx=np.where(component);area=len(xx)
        if area<12:continue
        points=np.column_stack((xx,yy)).astype(np.float32);centre=points.mean(axis=0)
        _,_,vt=np.linalg.svd(points-centre,full_matrices=False)
        normal=np.asarray([-vt[0,1],vt[0,0]],np.float64)
        samples=points[np.linspace(0,area-1,min(13,area),dtype=int)]
        bilateral=valid=0
        for px,py in samples:
            hits=[]
            for sign in (-1.,1.):
                hit=False;last=(-1,-1)
                for distance in range(2,probe+1):
                    qx=int(round(px+sign*normal[0]*distance))
                    qy=int(round(py+sign*normal[1]*distance))
                    if (qy,qx)==last:continue
                    last=(qy,qx)
                    if qy<0 or qy>=clean.shape[0] or qx<0 or qx>=clean.shape[1]:break
                    if clean[qy,qx]>0:hit=True;break
                hits.append(hit)
            valid+=1;bilateral+=int(all(hits))
        ratio=bilateral/max(1,valid)
        # A majority of samples seeing lithology on both normal sides indicates
        # an internal pale corridor between repeated beds, not the outer domain.
        if ratio<.55:kept[component]=True
        else:removed.append({'component':index,'pixels':area,'bilateral_ratio':round(ratio,3)})
    return kept,{'probe_distance_px':probe,'samples_per_component':13,
        'maximum_bilateral_ratio':.55,'input_components':max(0,count-1),
        'kept_components':max(0,cv2.connectedComponents(kept.astype(np.uint8),8)[0]-1),
        'removed_components':len(removed),'removed_pixels':sum(item['pixels'] for item in removed)}


def _component_endpoints(component):
    """Find two end pixels of a thin arc, falling back to PCA extremes."""
    neighbours=cv2.filter2D(component.astype(np.uint8),cv2.CV_16U,
        np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT)-component.astype(np.uint16)
    yy,xx=np.where(component&(neighbours<=2))
    all_y,all_x=np.where(component)
    if len(all_x)<2:return []
    points=np.column_stack((all_x,all_y)).astype(np.float32)
    if len(xx)>=2:candidates=np.column_stack((xx,yy)).astype(np.float32)
    else:candidates=points
    centre=points.mean(axis=0);_,_,vt=np.linalg.svd(points-centre,full_matrices=False)
    projection=(candidates-centre)@vt[0]
    return [tuple(map(int,candidates[np.argmin(projection)])),
            tuple(map(int,candidates[np.argmax(projection)]))]


def _endpoint_extends_toward(component,endpoint,target,radius=32):
    """Require a proposed bridge to continue the local arc rather than turn across it."""
    ex,ey=endpoint;tx,ty=target
    y0=max(0,ey-radius);y1=min(component.shape[0],ey+radius+1)
    x0=max(0,ex-radius);x1=min(component.shape[1],ex+radius+1)
    yy,xx=np.where(component[y0:y1,x0:x1]);xx=xx+x0;yy=yy+y0
    near=(xx-ex)**2+(yy-ey)**2<=radius*radius;xx=xx[near];yy=yy[near]
    if len(xx)<3:return False
    outward=np.asarray([ex-float(np.mean(xx)),ey-float(np.mean(yy))])
    bridge=np.asarray([tx-ex,ty-ey],np.float64)
    denominator=np.linalg.norm(outward)*np.linalg.norm(bridge)
    return denominator>0 and float(np.dot(outward,bridge)/denominator)>=.72


def _connect_short_pale_gaps(trusted,walkable,prior_distance,prior_tolerance):
    """Join only short, aligned gaps fully supported by background-like pixels."""
    count,cc,stats,_=cv2.connectedComponentsWithStats(trusted.astype(np.uint8),8)
    pieces=[]
    for index in range(1,count):
        area=int(stats[index,cv2.CC_STAT_AREA]);component=cc==index
        endpoints=_component_endpoints(component)
        if len(endpoints)==2:pieces.append({'id':index,'area':area,'mask':component,'ends':endpoints})
    candidates=[]
    for left in range(len(pieces)):
        for right in range(left+1,len(pieces)):
            a,b=pieces[left],pieces[right]
            limit=min(40.,max(3.,.1*min(a['area'],b['area'])))
            for ea in a['ends']:
                for eb in b['ends']:
                    distance=float(np.hypot(eb[0]-ea[0],eb[1]-ea[1]))
                    if 1.5<distance<=limit:candidates.append((distance,left,right,ea,eb,limit))
    candidates.sort(key=lambda item:item[0]);used=set();connections=np.zeros_like(trusted,np.uint8);accepted=[]
    for distance,left,right,ea,eb,limit in candidates:
        key_a=(left,ea);key_b=(right,eb)
        if key_a in used or key_b in used:continue
        if not _endpoint_extends_toward(pieces[left]['mask'],ea,eb):continue
        if not _endpoint_extends_toward(pieces[right]['mask'],eb,ea):continue
        line=np.zeros_like(connections);cv2.line(line,ea,eb,1,1,cv2.LINE_8)
        yy,xx=np.where(line>0)
        if len(xx)<2:continue
        # The line must remain on the coarse outer boundary and traverse pale /
        # page-background pixels apart from a two-pixel endpoint allowance.
        if np.max(prior_distance[yy,xx])>prior_tolerance:continue
        interior=(line>0)&~cv2.dilate((trusted&(line>0)).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        interior_count=int(interior.sum())
        if interior_count and np.count_nonzero(walkable&interior)/interior_count<.9:continue
        connections|=line;used.update((key_a,key_b));accepted.append(round(distance,2))
    return connections.astype(bool),{'candidate_pairs':len(candidates),'connected_gaps':len(accepted),
        'connection_lengths_px':accepted,'max_gap_rule':'0.1 * shorter adjacent arc, capped at 40 px'}


def _connect_long_stratum_gaps(trusted,blank,prior_distance,prior_tolerance):
    """Connect aligned outline gaps longer than 10% of nearby segment scale."""
    count,cc,stats,centroids=cv2.connectedComponentsWithStats(trusted.astype(np.uint8),8)
    pieces=[]
    for index in range(1,count):
        area=int(stats[index,cv2.CC_STAT_AREA]);component=cc==index
        endpoints=_component_endpoints(component)
        if area>=12 and len(endpoints)==2:
            pieces.append({'area':area,'mask':component,'ends':endpoints,
                           'centre':np.asarray(centroids[index],np.float64)})
    candidates=[]
    for left in range(len(pieces)):
        for right in range(left+1,len(pieces)):
            a,b=pieces[left],pieces[right];midpoint=(a['centre']+b['centre'])/2
            nearby=sorted(pieces,key=lambda item:float(np.linalg.norm(item['centre']-midpoint)))[:6]
            lengths=[item['area'] for item in nearby]
            shortest=max(1.,float(min(lengths)));typical=float(np.median(lengths))
            lower=.1*shortest
            # The requested 0.1 rule is a lower bound.  A local upper bound is
            # necessary to avoid joining unrelated bodies across the section.
            upper=min(180.,max(lower+2.,typical*.9))
            for ea in a['ends']:
                for eb in b['ends']:
                    distance=float(np.hypot(eb[0]-ea[0],eb[1]-ea[1]))
                    if distance>lower and distance<=upper:
                        candidates.append((distance,left,right,ea,eb,lower,upper))
    candidates.sort(key=lambda item:item[0]);used=set();connections=np.zeros_like(trusted,np.uint8);accepted=[]
    for distance,left,right,ea,eb,lower,upper in candidates:
        key_a=(left,ea);key_b=(right,eb)
        if key_a in used or key_b in used:continue
        if not _endpoint_extends_toward(pieces[left]['mask'],ea,eb):continue
        if not _endpoint_extends_toward(pieces[right]['mask'],eb,ea):continue
        line=np.zeros_like(connections);cv2.line(line,ea,eb,1,1,cv2.LINE_8)
        yy,xx=np.where(line>0)
        if len(xx)<2 or np.max(prior_distance[yy,xx])>prior_tolerance:continue
        interior=(line>0)&~cv2.dilate((trusted&(line>0)).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        pixels=int(interior.sum())
        if pixels and np.count_nonzero(blank&interior)/pixels<.8:continue
        connections|=line;used.update((key_a,key_b))
        accepted.append({'length_px':round(distance,2),'lower_0_1_px':round(lower,2),
                         'local_upper_px':round(upper,2)})
    return connections.astype(bool),{'candidate_pairs':len(candidates),'connected_gaps':len(accepted),
        'connections':accepted,'rule':'gap > 0.1 * shortest nearby segment; aligned/local-extent safeguards'}


def reconstruct_confident_outline(cfg):
    """Render only directly observed, high-confidence outer-boundary arcs.

    No gap connection, pale-bed expansion, curve prediction or closed-domain
    construction is performed here.  A coloured edge is retained only when it
    borders page background reachable from the crop exterior through a region
    wider than ordinary grid/contact lines.
    """
    crop,labels,colours,_,pale,pale_candidate,allowed,_,_,audit1=_load_and_classify(cfg)
    support=(labels>=0)&allowed
    if pale:support&=~np.isin(labels,np.asarray(pale,np.int16))
    h,w=support.shape
    count,cc,stats,_=cv2.connectedComponentsWithStats(support.astype(np.uint8),8)
    clean=np.zeros_like(support,np.uint8);floor=max(12,round(h*w*.00002))
    for index in range(1,count):
        if stats[index,cv2.CC_STAT_AREA]>=floor:clean[cc==index]=1
    core=cv2.erode(clean,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    clean=cv2.dilate(core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    # Close only narrow drawing-line gaps for the exterior/background test.
    # This helper mask is never used to draw or complete an outline.
    barrier=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    background=(barrier==0).astype(np.uint8)
    bg_count,bg_cc=cv2.connectedComponents(background,8)
    border_ids=np.unique(np.concatenate((bg_cc[0],bg_cc[-1],bg_cc[:,0],bg_cc[:,-1])))
    border_ids=border_ids[border_ids>0]
    exterior=np.isin(bg_cc,border_ids) if bg_count>1 else np.zeros_like(support)
    # Require a background core several pixels wide, excluding narrow grid,
    # borehole and text strokes from the trusted outside-background evidence.
    exterior_core=cv2.erode(exterior.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    exterior_near=cv2.dilate(exterior_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))).astype(bool)
    support_edge=(clean>0)&~cv2.erode(clean,np.ones((3,3),np.uint8)).astype(bool)
    arcs=support_edge&exterior_near

    # A directly observed colour edge is still not necessarily the section's
    # outer edge: internal contacts can open into a large white annotation gap.
    # Keep only arcs close to the coarse plot mask's *external* contour.  This
    # mask is used as a rejection gate, never as a source of drawn line pixels.
    prior_contours,_=cv2.findContours(allowed.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    prior_edge=np.zeros_like(clean)
    if prior_contours:
        cv2.drawContours(prior_edge,[max(prior_contours,key=cv2.contourArea)],-1,1,1)
        prior_distance=cv2.distanceTransform((prior_edge==0).astype(np.uint8),cv2.DIST_L2,5)
        prior_tolerance=max(10,round(min(h,w)*.012))
        arcs&=prior_distance<=prior_tolerance
    else:prior_tolerance=0

    # Reject tiny isolated edge fragments; they cannot establish an outer arc.
    arc_count,arc_cc,arc_stats,_=cv2.connectedComponentsWithStats(arcs.astype(np.uint8),8)
    trusted=np.zeros_like(arcs)
    for index in range(1,arc_count):
        x,y,cw,ch,area=map(int,arc_stats[index])
        if area>=12 and max(cw,ch)>=10:trusted[arc_cc==index]=True

    # One common confidence path is used for every drawing.  Background-like
    # lithologies no longer activate a separate water/parallel-edge branch.
    displayed=trusted

    review=labels.copy();review[~allowed]=-1
    image=_render(review,colours,None,0,0)
    line=cv2.dilate(displayed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(bool)
    image[line]=0
    audit={'stage1':audit1,'confident_outline':{
        'support_pixels':int(clean.sum()),'trusted_arc_pixels':int(trusted.sum()),
        'displayed_outline_pixels':int(displayed.sum()),
        'trusted_arc_components':int(max(0,cv2.connectedComponents(trusted.astype(np.uint8),8)[0]-1)),
        'coarse_outer_proximity_px':int(prior_tolerance),
        'special_background_lithology_branch':False,
        'gap_connection':False,'curve_completion':False,'pale_expansion':False,
        'closed_outline':False}}
    return image,displayed,audit


def reconstruct_stages(cfg):
    crop,stage1,colours,names,pale,pale_candidate,allowed,scale,origin,audit1=_load_and_classify(cfg)
    domain,audit2=_outer_domain(stage1,pale,allowed,pale_candidate)
    clipped=stage1.copy();clipped[~domain]=-1
    clipped,pale_audit=_seed_pale_regions(clipped,pale_candidate,pale,domain);audit2['pale_region_seeding']=pale_audit
    stage3,audit3=_repair_linear_gaps(clipped,domain,int(cfg.get('stage3_line_probe_pixels',16)))
    final,audit4=_repair_regions(stage3,domain);final[~domain]=-1
    # A strict invariant checked on the returned data, not just the rendering.
    outside=int(np.count_nonzero((final>=0)&~domain));unfilled=int(np.count_nonzero((final<0)&domain))
    stage1_review=stage1.copy();stage1_review[~allowed]=-1
    return {'stage1':_render(stage1_review,colours,None,0,0),
            'stage2':_render(clipped,colours,domain,4,0),
            'stage3':_render(stage3,colours,domain,4,0),
            'final':_render(final,colours,domain,4,2),
            'labels':final,'domain':domain,
            'audit':{'stage1':audit1,'stage2':audit2,'stage3':audit3,'stage4':audit4,
                     'final_invariants':{'outside_domain_pixels':outside,'unfilled_inside_pixels':unfilled},
                     'lithologies':names,'method':'four_stage_colour_outer_line_region_v1'}}


def reconstruct_stage2(cfg):
    """Run only colour filtering and generalized outer-outline construction."""
    _,stage1,colours,names,pale,pale_candidate,allowed,_,_,audit1=_load_and_classify(cfg)
    domain,audit2=_outer_domain(stage1,pale,allowed,pale_candidate)
    clipped=stage1.copy();clipped[~domain]=-1
    clipped,pale_audit=_seed_pale_regions(clipped,pale_candidate,pale,domain)
    audit2['pale_region_seeding']=pale_audit
    return _render(clipped,colours,domain,4,0),domain,{'stage1':audit1,'stage2':audit2,
        'invariants':{'outer_contours':len(cv2.findContours(domain.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0])}}


def reconstruct_stage3(cfg):
    """Run through linear-gap repair only, leaving region repair untouched."""
    _,stage1,colours,names,pale,pale_candidate,allowed,_,_,audit1=_load_and_classify(cfg)
    domain,audit2=_outer_domain(stage1,pale,allowed,pale_candidate)
    clipped=stage1.copy();clipped[~domain]=-1
    # Keep the exact coordinates cleared by stage 1.  Stage 3 is only allowed
    # to restore pixels at these recorded locations inside the final domain.
    cleared_positions=(clipped<0)&domain
    clipped,pale_audit=_seed_pale_regions(clipped,pale_candidate,pale,domain)
    audit2['pale_region_seeding']=pale_audit
    stage3,audit3=_repair_linear_gaps(clipped,domain,int(cfg.get('stage3_line_probe_pixels',16)))
    audit3['recorded_cleared_pixels']=int(cleared_positions.sum())
    audit3['filled_outside_recorded_positions']=int(np.count_nonzero((stage3!=clipped)&~cleared_positions))
    outside=int(np.count_nonzero((stage3>=0)&~domain))
    unfilled=int(np.count_nonzero((stage3<0)&domain))
    return _render(stage3,colours,domain,4,0),domain,{
        'stage1':audit1,'stage2':audit2,'stage3':audit3,
        'invariants':{'outside_domain_pixels':outside,'unfilled_inside_pixels':unfilled,
                      'outer_contours':len(cv2.findContours(domain.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0])},
        'lithologies':names}
