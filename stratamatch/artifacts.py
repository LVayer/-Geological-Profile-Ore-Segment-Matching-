"""Conservative removal of drawing artefacts before stratum components are built.

Only pixels supported by a detected grid/text mask are considered.  A masked
pixel is restored when the nearest classified pixels on opposite sides agree.
This avoids painting across a geological boundary merely to make a region look
cleaner; unresolved pixels remain unclassified for review.
"""
import cv2
import numpy as np


def _shape_family(shape):
    shape=str(shape or 'none')
    if 'dash' in shape:return 'dashed'
    if 'curve' in shape or 'polyline' in shape:return 'curve'
    if 'straight' in shape:return 'straight'
    return 'unknown'


def _catalogue_line_prototypes(catalogue,scale=1.):
    """Convert reviewed legend rows to line prototypes in working pixels."""
    result=[]
    for item in (catalogue or {}).get('items',[]):
        line=item.get('line') or {}
        if not line.get('present'):continue
        category=item.get('category');subtype=item.get('subtype')
        if category!='geological_boundary' and not(category=='other' and subtype in ('line','label_line')):continue
        rgb=line.get('dominant_rgb')
        if not isinstance(rgb,list) or len(rgb)!=3:continue
        lab=cv2.cvtColor(np.float32([[np.asarray(rgb)/255.]]),cv2.COLOR_RGB2LAB)[0,0]
        result.append({'id':item.get('id'),'name':item.get('name'),'category':category,
            'subtype':subtype,'rgb':[int(v) for v in rgb],'lab':lab,
            'width':max(.75,float(line.get('mean_width_px') or 1.)*float(scale)),
            'shape':_shape_family(line.get('shape')),'confidence':float(item.get('confidence',0))})
    return result


def classify_catalogue_strokes(rgb,classified,catalogue,scale=1.,colour_tolerance=14.,max_probe=12):
    """Find and classify drawn lines from legend evidence.

    Decisions deliberately follow a lexicographic order: colour nominates a
    small prototype set, width resolves same-colour symbols, and shape is the
    final tie breaker. Geological topology is only a conflict check. It cannot
    turn a red fault prototype into a black grid or vice versa.
    """
    prototypes=_catalogue_line_prototypes(catalogue,scale)
    h,w=rgb.shape[:2];empty=np.zeros((h,w),bool)
    if not prototypes:return empty.copy(),empty.copy(),empty.copy(),{'enabled':False,'reason':'no_line_prototypes'}
    # Work in bounded row blocks to avoid another full H×W×N colour cube.
    lab=np.empty((h,w,3),np.float32)
    for row in range(0,h,192):
        lab[row:min(h,row+192)]=cv2.cvtColor(rgb[row:min(h,row+192)].astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    raw_union=np.zeros((h,w),bool);nearest=np.full((h,w),-1,np.int16);best=np.full((h,w),np.inf,np.float32)
    for index,prototype in enumerate(prototypes):
        distance=np.linalg.norm(lab-prototype['lab'],axis=2);better=distance<best
        best[better]=distance[better];nearest[better]=index
    # Measure competition from actual lithology colours too. A magenta line
    # must be closer to the magenta line prototype than to a magenta stratum;
    # otherwise colour alone cannot authorize deletion and style is required.
    lithology_labs=[]
    for item in (catalogue or {}).get('items',[]):
        if item.get('category')!='stratum_swatch':continue
        colour=(item.get('appearance') or {}).get('fill_rgb')
        if isinstance(colour,list) and len(colour)==3:
            lithology_labs.append(cv2.cvtColor(np.float32([[np.asarray(colour)/255.]]),cv2.COLOR_RGB2LAB)[0,0])
    lithology_best=np.full((h,w),np.inf,np.float32)
    for prototype in lithology_labs:lithology_best=np.minimum(lithology_best,np.linalg.norm(lab-prototype,axis=2))
    raw_union=best<=float(colour_tolerance)
    # A line candidate must also possess extended stroke support. This prevents
    # black text glyphs from being treated as contacts merely because black is
    # present in the geological-boundary legend.
    stroke=np.zeros((h,w),np.uint8);minimum=max(18,round(min(h,w)*.012))
    lines=cv2.HoughLinesP((raw_union.astype(np.uint8)*255),1,np.pi/360,
        threshold=max(14,minimum//2),minLineLength=minimum,maxLineGap=max(4,round(minimum*.35)))
    segment_count=0
    if lines is not None:
        for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
            cv2.line(stroke,(int(x1),int(y1)),(int(x2),int(y2)),1,max(3,round(max(p['width'] for p in prototypes)*2+3)))
            segment_count+=1
    candidate=raw_union&(stroke>0)
    geological=np.zeros((h,w),bool);technical=np.zeros((h,w),bool);uncertain=np.zeros((h,w),bool)
    decisions=[];distance_inside=cv2.distanceTransform(raw_union.astype(np.uint8),cv2.DIST_L2,5)
    def decide_component(pixels,box):
        area=int(pixels.sum())
        if area<max(8,minimum//2):return
        ys,xs=np.where(pixels);span=float(np.hypot(xs.max()-xs.min()+1,ys.max()-ys.min()+1))
        if span<minimum:return
        colours=lab[pixels];median_lab=np.median(colours,axis=0)
        local_widths=2*distance_inside[pixels]
        width=float(max(.75,np.median(local_widths)))
        width_p90=float(np.percentile(local_widths,90))
        # Curvature is estimated from local Hough directions in this component.
        x,y,bw,bh=box;local=(pixels[y:y+bh,x:x+bw].astype(np.uint8)*255)
        local_lines=cv2.HoughLinesP(local,1,np.pi/360,threshold=max(7,minimum//3),
            minLineLength=max(8,minimum//2),maxLineGap=max(3,minimum//4))
        angles=[]
        if local_lines is not None:
            for x1,y1,x2,y2 in np.asarray(local_lines).reshape(-1,4):
                angles.append(float(np.degrees(np.arctan2(int(y2)-int(y1),int(x2)-int(x1)))%180))
        spread=float(np.std(angles)) if len(angles)>1 else 0.
        # Dash periodicity is measured along the component's principal axis.
        points=np.column_stack((xs,ys)).astype(np.float32);centred=points-points.mean(axis=0)
        _,_,vt=np.linalg.svd(centred,full_matrices=False);projection=centred@vt[0]
        bins=np.zeros(max(1,int(np.ceil(projection.max()-projection.min()))+1),bool)
        bins[np.clip(np.rint(projection-projection.min()).astype(int),0,len(bins)-1)]=True
        transitions=np.diff(np.pad(bins.astype(np.int8),(1,1)))
        dash_runs=int(np.count_nonzero(transitions==1));occupancy=float(bins.mean())
        symbolic=bool(width_p90>max(width*1.55,width+1.5) and span/max(1,min(bw,bh))>=3)
        if spread>24:component_shape='curve'
        elif dash_runs>=3 and occupancy<.78:component_shape='dashed'
        else:component_shape='straight'
        colour_scores=[float(np.linalg.norm(median_lab-p['lab'])) for p in prototypes]
        best_colour=min(colour_scores);colour_pool=[i for i,v in enumerate(colour_scores) if v<=best_colour+2.5]
        width_scores=[abs(np.log(max(width,.25)/max(prototypes[i]['width'],.25))) for i in colour_pool]
        best_width=min(width_scores);width_pool=[i for i,v in zip(colour_pool,width_scores) if v<=best_width+.28]
        def shape_penalty(i):
            expected=prototypes[i]['shape']
            if expected=='unknown':return .25
            if expected=='dashed':return 0. if component_shape=='dashed' else .45
            if component_shape=='dashed':return .5
            return 0. if expected==component_shape else .35
        # Arrow heads and repeated attached symbols favour a technical legend
        # prototype when colour and width otherwise tie.
        selected=min(width_pool,key=lambda i:shape_penalty(i)-(.18 if symbolic and prototypes[i]['category']=='other' else 0))
        prototype=prototypes[selected]
        alternatives=[i for i in width_pool if prototypes[i]['category']!=prototype['category']]
        ambiguous=bool(alternatives and abs(shape_penalty(alternatives[0])-shape_penalty(selected))<.2)
        colour_margin=float(np.median(lithology_best[pixels]-best[pixels])) if lithology_labs else float('inf')
        style_evidence=component_shape=='dashed' or symbolic
        reliable=best_colour<=float(colour_tolerance) and best_width<=1.1 and prototype['confidence']>=.75 \
                 and (colour_margin>=2 or style_evidence or prototype['category']=='geological_boundary')
        if not reliable or ambiguous:decision='uncertain';uncertain[pixels]=True
        elif prototype['category']=='geological_boundary':decision='geological_boundary';geological[pixels]=True
        else:decision='technical';technical[pixels]=True
        decisions.append({'prototype_id':prototype['id'],'prototype_name':prototype['name'],'decision':decision,
            'pixels':area,'span_px':round(span,2),'colour_delta_e':round(best_colour,2),
            'line_over_lithology_colour_margin':None if not np.isfinite(colour_margin) else round(colour_margin,2),
            'measured_width_px':round(width,2),'expected_width_px':round(prototype['width'],2),
            'shape':component_shape,'dash_runs':dash_runs,'axis_occupancy':round(occupancy,3),
            'attached_symbol_evidence':symbolic,'angle_spread':round(spread,2)})
    # Different colours are segmented independently, so a magenta section line
    # crossing a red fault does not fuse them into one path. Same-colour legend
    # prototypes stay together and are resolved by width then shape below.
    colour_groups=[]
    extended_pixels=0
    for index,prototype in enumerate(prototypes):
        placed=False
        for group in colour_groups:
            if np.linalg.norm(prototype['lab']-prototypes[group[0]]['lab'])<=3:
                group.append(index);placed=True;break
        if not placed:colour_groups.append([index])
    for group in colour_groups:
        group_candidate=candidate&np.isin(nearest,np.asarray(group,np.int16))
        joined=cv2.dilate(group_candidate.astype(np.uint8),np.ones((3,3),np.uint8))
        count,components,stats,_=cv2.connectedComponentsWithStats(joined,8)
        for component in range(1,count):
            pixels=(components==component)&group_candidate
            decide_component(pixels,tuple(map(int,stats[component,:4])))
    # Continue high-chroma catalogue strokes through small Hough gaps. Colour
    # makes this safe for the red fault and magenta/blue drafting lines; black
    # is intentionally excluded because it is shared by text and contacts.
    for index,prototype in enumerate(prototypes):
        colour=np.asarray(prototype['rgb'],float)
        if np.linalg.norm(colour-colour.mean())<=70:continue
        region=raw_union&(nearest==index)
        count,cc,stats,_=cv2.connectedComponentsWithStats(region.astype(np.uint8),8)
        seed=geological if prototype['category']=='geological_boundary' else technical
        for component in range(1,count):
            pixels=cc==component;area=int(stats[component,cv2.CC_STAT_AREA])
            _,_,bw,bh=map(int,stats[component,:4]);span=max(bw,bh);minor=max(1,min(bw,bh))
            if not np.any(seed&pixels):continue
            line_like=(span/minor>=3) or area/max(1,span)<=max(12,prototype['width']*6)
            if line_like:seed|=pixels
        # Directional closing bridges the same-colour path across an arrow,
        # crossing line or text label. Only elongated components already
        # touched by an accepted seed survive, so similarly coloured strata do
        # not become technical paths.
        accepted=seed&region
        if not np.any(accepted):continue
        gap=max(4,round(16*float(scale)));size=2*gap+1;centre=gap
        tracked=region.astype(np.uint8)
        for angle in range(0,180,15):
            radians=np.deg2rad(angle);dx=round(np.cos(radians)*gap);dy=round(np.sin(radians)*gap)
            kernel=np.zeros((size,size),np.uint8);cv2.line(kernel,(centre-dx,centre-dy),(centre+dx,centre+dy),1,1)
            tracked|=cv2.morphologyEx(region.astype(np.uint8),cv2.MORPH_CLOSE,kernel)
        count,cc,stats,_=cv2.connectedComponentsWithStats(tracked,8)
        for component in np.unique(cc[accepted]):
            if component==0:continue
            x,y,bw,bh,area=map(int,stats[component]);span=max(bw,bh);minor=max(1,min(bw,bh))
            if span/minor<3 and area/max(1,span)>max(14,prototype['width']*7):continue
            addition=(cc==component)&~seed
            seed|=cc==component;extended_pixels+=int(addition.sum())
    # For black/same-colour technical paths, contrary side evidence means the
    # legend alone is insufficient. Move these paths to review instead of
    # erasing a possible geological contact.
    topo_artifact,topo_boundary,topo_uncertain,topo_audit=classify_line_paths(classified,technical,max_probe)
    vivid=np.zeros((h,w),bool)
    for i,p in enumerate(prototypes):
        if p['category']=='other' and np.linalg.norm(np.asarray(p['rgb'],float)-np.mean(p['rgb']))>70:
            vivid|=technical&(nearest==i)
    conflict=(technical&topo_boundary)&~vivid
    technical&=~conflict;uncertain|=conflict
    del lab
    return geological,technical,uncertain,{'enabled':True,'prototype_count':len(prototypes),
        'hough_segments':segment_count,'candidate_pixels':int(candidate.sum()),
        'directional_extension_pixels':extended_pixels,
        'geological_pixels':int(geological.sum()),'technical_pixels':int(technical.sum()),
        'uncertain_pixels':int(uncertain.sum()),'topology_cross_check':topo_audit,'components':decisions}

def _clusters(values,tolerance=3):
    groups=[]
    for value in sorted(values):
        if not groups or value-groups[-1][-1]>tolerance:groups.append([value])
        else:groups[-1].append(value)
    return [int(round(np.median(group))) for group in groups]

def _regular_positions(values,limit):
    """Retain positions supported by an approximately periodic grid."""
    values=_clusters(values)
    if len(values)<3:return []
    best=[]
    for i,a in enumerate(values):
        for b in values[i+1:]:
            distance=b-a
            for divisions in range(1,9):
                pitch=distance/divisions
                if pitch<max(12,limit*.018):continue
                support=[v for v in values if abs((v-a)/pitch-round((v-a)/pitch))<=.08]
                if len(support)>len(best):best=support
    return best if len(best)>=3 else []

def _axis_grid(rgb):
    """Find repeated horizontal/vertical drafting grids across coloured fills."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);edges=cv2.Canny(gray,35,110,L2gradient=True)
    h,w=gray.shape
    # Small fixtures and thumbnails rarely contain coordinate grids; on them,
    # repeated bed edges can mimic a grid. Reserve Hough inference for drawings.
    if min(h,w)<600:return np.zeros((h,w),bool)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(35,round(min(h,w)*.025)),
                          minLineLength=max(25,round(min(h,w)*.16)),maxLineGap=max(8,round(min(h,w)*.025)))
    if lines is None:return np.zeros((h,w),bool)
    ys=[];xs=[]
    # OpenCV 4 returns N×1×4 while OpenCV 5 may return N×4.
    for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
        dx=abs(int(x2)-int(x1));dy=abs(int(y2)-int(y1))
        if dx>=w*.16 and dy<=max(2,dx*.008):ys.append(round((int(y1)+int(y2))/2))
        if dy>=h*.16 and dx<=max(2,dy*.008):xs.append(round((int(x1)+int(x2))/2))
    ys=_regular_positions(ys,h);xs=_regular_positions(xs,w)
    mask=np.zeros((h,w),np.uint8)
    for y in ys:mask[max(0,y-1):min(h,y+2),:]=1
    for x in xs:mask[:,max(0,x-1):min(w,x+2)]=1
    return mask.astype(bool)

def _technical_lines(neutral,max_half_width=None):
    """Detect long pale drafting/drill traces at arbitrary angles."""
    h,w=neutral.shape;lines=cv2.HoughLinesP((neutral.astype(np.uint8)*255),1,np.pi/360,
        threshold=max(25,round(min(h,w)*.018)),minLineLength=max(30,round(min(h,w)*.11)),
        maxLineGap=max(6,round(min(h,w)*.018)))
    mask=np.zeros((h,w),np.uint8)
    if lines is not None:
        for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
            length=float(np.hypot(int(x2)-int(x1),int(y2)-int(y1)))
            if length>=min(h,w)*.11:cv2.line(mask,(int(x1),int(y1)),(int(x2),int(y2)),1,3)
    candidate=mask.astype(bool)&neutral
    if max_half_width is not None:
        # A technical trace is narrow. Preserve the interior of broad white
        # geological bodies even when Hough sees their straight boundaries.
        half_width=cv2.distanceTransform(neutral.astype(np.uint8),cv2.DIST_L2,5)
        candidate&=half_width<=float(max_half_width)
    return candidate

def _grouped_small_ink(rgb):
    """Find text-like groups when platform OCR misses small drawing labels.

    Detection is done on a bounded thumbnail. Components must be small and
    have a nearby component on the same text line. Long structural strokes and
    isolated geological edges therefore do not qualify.
    """
    h,w=rgb.shape[:2];scale=min(1.,1800/max(h,w))
    small=cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else rgb
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,np.ones((9,9),np.uint8))
    ink=(blackhat>18).astype(np.uint8)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(ink,8)
    candidates=[]
    for i in range(1,count):
        x,y,bw,bh,area=map(int,stats[i])
        if 2<=area<=180 and 1<=bw<=28 and 2<=bh<=22 and bw/max(bh,1)<=3.2:
            candidates.append(i)
    selected=[]
    for i in candidates:
        x,y,bw,bh,_=stats[i];cx,cy=centroids[i]
        for j in candidates:
            if i==j:continue
            xx,yy,ww,hh,_=stats[j];ccx,ccy=centroids[j]
            gap=max(0,max(x,xx)-min(x+bw,xx+ww))
            if abs(cy-ccy)<=max(bh,hh)*.7 and gap<=max(8,max(bh,hh)*4):
                selected.append(i);break
    if not selected:return np.zeros((h,w),bool)
    chosen=np.isin(labels,np.asarray(selected,np.int32)).astype(np.uint8)
    if scale<1:chosen=cv2.resize(chosen,(w,h),interpolation=cv2.INTER_NEAREST)
    # Keep only locally darker strokes after mapping back; boxes are never filled.
    full_gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    full_blackhat=cv2.morphologyEx(full_gray,cv2.MORPH_BLACKHAT,np.ones((11,11),np.uint8))
    return (chosen>0)&(full_blackhat>8)


def _rect_mask(shape, boxes, sx, sy, origin):
    mask=np.zeros(shape,np.uint8);ox,oy=origin
    for x,y,w,h in boxes:
        x0=max(0,round(x*sx)-ox);y0=max(0,round(y*sy)-oy)
        x1=min(shape[1],round((x+w)*sx)-ox);y1=min(shape[0],round((y+h)*sy)-oy)
        if x1>x0 and y1>y0:mask[y0:y1,x0:x1]=1
    return mask.astype(bool)


def detect_masks(rgb, text_boxes=(), sx=1., sy=1., origin=(0,0), content=None, detect_pale_lines=False):
    """Detect light neutral ruler/grid lines and dark OCR strokes.

    Long-line morphology is intentionally restricted to neutral mid-light
    pixels. Dark geological contacts are therefore not classified as grids.
    OCR rectangles only nominate an area; within them only dark/neutral ink is
    removed, not the complete rectangle.
    """
    lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    lightness=lab[:,:,0];chroma=np.linalg.norm(lab[:,:,1:],axis=2)
    neutral=(chroma<5.5)&(lightness>=38)&(lightness<=94)
    # Drill traces are frequently exported as pure white and therefore fall
    # outside the mid-light neutral range. Long-line evidence nominates them as
    # artefacts; the later opposite-side agreement rule decides whether they
    # may be repainted, so a broad white geological unit remains white.
    pale=(chroma<5)&(lightness>94)
    h,w=neutral.shape
    hk=np.ones((1,max(15,round(w*.035))),np.uint8)
    vk=np.ones((max(15,round(h*.035)),1),np.uint8)
    horizontal=cv2.morphologyEx(neutral.astype(np.uint8),cv2.MORPH_OPEN,hk)>0
    vertical=cv2.morphologyEx(neutral.astype(np.uint8),cv2.MORPH_OPEN,vk)>0
    # Restore the original thin line extent after opening, without growing it.
    technical=_technical_lines(neutral)
    grid=((horizontal|vertical)&neutral)|_axis_grid(rgb)|technical
    # When white is a valid legend class, only pale pixels close to an already
    # detected neutral/grey technical centreline are included. Length alone is
    # insufficient: real Fe3 bodies in the user's drawing are also long, white
    # and narrow, but do not contain such a centreline.
    if detect_pale_lines:
        near_technical=cv2.dilate(technical.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)
        grid|=near_technical&pale
    boxes=_rect_mask((h,w),text_boxes,sx,sy,origin)
    text=(boxes&((lightness<72)|((lightness<88)&(chroma<9))))|_grouped_small_ink(rgb)
    # Hough/black-hat responses follow the stroke centre. Include anti-aliased
    # fringes as candidates; the restoration agreement rule remains the guard.
    grid=cv2.dilate(grid.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    text=cv2.dilate(text.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    if content is not None:
        grid&=content;text&=content
    return grid,text

def classify_line_paths(classified,candidate,max_probe=12):
    """Classify complete candidate paths using labels on both sides.

    Technical paths usually have the same geological class on both sides for
    most of their length, even when crossing several contacts. Geological
    interfaces have different classes along most of their length. Decisions
    are made per connected path rather than independently per pixel.
    """
    candidate=candidate.astype(bool);probe=classified.copy();probe[candidate]=-1
    same=np.zeros(probe.shape,np.uint8);different=np.zeros(probe.shape,np.uint8)
    for dy,dx in ((1,0),(0,1),(1,1),(1,-1)):
        negative=np.full(probe.shape,-1,np.int16);positive=np.full(probe.shape,-1,np.int16)
        for distance in range(1,max(2,int(max_probe))+1):
            if (dy,dx)==(1,0):
                negative[distance:]=np.where((negative[distance:]<0)&(probe[:-distance]>=0),probe[:-distance],negative[distance:])
                positive[:-distance]=np.where((positive[:-distance]<0)&(probe[distance:]>=0),probe[distance:],positive[:-distance])
            elif (dy,dx)==(0,1):
                negative[:,distance:]=np.where((negative[:,distance:]<0)&(probe[:,:-distance]>=0),probe[:,:-distance],negative[:,distance:])
                positive[:,:-distance]=np.where((positive[:,:-distance]<0)&(probe[:,distance:]>=0),probe[:,distance:],positive[:,:-distance])
            elif dx==1:
                negative[distance:,distance:]=np.where((negative[distance:,distance:]<0)&(probe[:-distance,:-distance]>=0),probe[:-distance,:-distance],negative[distance:,distance:])
                positive[:-distance,:-distance]=np.where((positive[:-distance,:-distance]<0)&(probe[distance:,distance:]>=0),probe[distance:,distance:],positive[:-distance,:-distance])
            else:
                negative[distance:,:-distance]=np.where((negative[distance:,:-distance]<0)&(probe[:-distance,distance:]>=0),probe[:-distance,distance:],negative[distance:,:-distance])
                positive[:-distance,distance:]=np.where((positive[:-distance,distance:]<0)&(probe[distance:,:-distance]>=0),probe[distance:,:-distance],positive[:-distance,distance:])
        valid=(negative>=0)&(positive>=0)&candidate
        same+=(valid&(negative==positive)).astype(np.uint8)
        different+=(valid&(negative!=positive)).astype(np.uint8)
    joined=cv2.dilate(candidate.astype(np.uint8),np.ones((3,3),np.uint8))
    count,paths,stats,_=cv2.connectedComponentsWithStats(joined,8)
    artifact=np.zeros(candidate.shape,bool);boundary=np.zeros(candidate.shape,bool);uncertain=np.zeros(candidate.shape,bool)
    decisions={'artifact_paths':0,'boundary_paths':0,'uncertain_paths':0,'candidate_paths':max(0,count-1)}
    for index in range(1,count):
        pixels=(paths==index)&candidate
        if not np.any(pixels):continue
        sv=int(same[pixels].sum());dv=int(different[pixels].sum());evidence=sv+dv
        if evidence and sv>=max(2,dv*2):artifact[pixels]=True;decisions['artifact_paths']+=1
        elif evidence and dv>=max(2,sv):boundary[pixels]=True;decisions['boundary_paths']+=1
        else:uncertain[pixels]=True;decisions['uncertain_paths']+=1
    decisions.update(artifact_pixels=int(artifact.sum()),boundary_pixels=int(boundary.sum()),uncertain_pixels=int(uncertain.sum()))
    return artifact,boundary,uncertain,decisions


def expand_confirmed_corridors(rgb,confirmed,max_half_width=14):
    """Expand a confirmed centreline across the full pale technical corridor.

    Boreholes are often exported as a wide white/grey strip with only their
    centre or edges detected by Hough. Expansion is distance bounded and
    limited to pale, low-chroma pixels, so it cannot flood through arbitrary
    coloured strata or the whole white page.
    """
    confirmed=confirmed.astype(bool);radius=max(0,int(max_half_width))
    if not np.any(confirmed) or radius==0:return confirmed.copy(),{'added_pixels':0,'max_half_width':radius}
    lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    pale_neutral=(lab[:,:,0]>=62)&(np.linalg.norm(lab[:,:,1:],axis=2)<10)
    distance=cv2.distanceTransform((~confirmed).astype(np.uint8),cv2.DIST_L2,5)
    expanded=confirmed|(pale_neutral&(distance<=radius))
    return expanded,{'added_pixels':int(np.count_nonzero(expanded&~confirmed)),'max_half_width':radius}


def inpaint_artifact_labels(classified,artifact_mask,valid_class_ids=None):
    """Fill every accepted artefact pixel from reliable legend-class regions.

    A nearest-seed partition continues geological interfaces through a removed
    label, grid or borehole instead of leaving a blank separator. The operation
    is restricted to the accepted artefact mask; ordinary blank regions and
    uncertain paths remain untouched. Returned seam pixels record where two
    reconstructed classes meet and therefore deserve boundary review.
    """
    out=classified.copy();artifact=artifact_mask.astype(bool);target=artifact&(out<0)
    if valid_class_ids is None:known=out>=0
    else:known=np.isin(out,np.asarray(list(valid_class_ids),dtype=out.dtype))
    if not np.any(target) or not np.any(known):
        return out,np.zeros(out.shape,bool),{'candidate_pixels':int(target.sum()),'filled_pixels':0,
            'unresolved_pixels':int(target.sum()),'seam_pixels':0,'method':'nearest_verified_legend_partition'}
    source=(~known).astype(np.uint8)
    _,nearest=cv2.distanceTransformWithLabels(source,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    lookup=np.full(int(nearest.max())+1,-1,np.int16);lookup[nearest[known]]=out[known]
    proposal=lookup[nearest];fill=target&(proposal>=0);out[fill]=proposal[fill]
    # A seam is a filled pixel adjacent to a different reconstructed class.
    seam=np.zeros(out.shape,bool)
    for dy,dx in ((1,0),(0,1)):
        a=out[dy:,dx:];b=out[:-dy or None,:-dx or None]
        adjacent=(a>=0)&(b>=0)&(a!=b)
        local_a=fill[dy:,dx:];local_b=fill[:-dy or None,:-dx or None]
        seam[dy:,dx:]|=adjacent&(local_a|local_b)
        seam[:-dy or None,:-dx or None]|=adjacent&(local_a|local_b)
    return out,seam,{'candidate_pixels':int(target.sum()),'filled_pixels':int(fill.sum()),
        'unresolved_pixels':int(np.count_nonzero(artifact&(out<0))),'seam_pixels':int(seam.sum()),
        'method':'nearest_verified_legend_partition'}


def restore(classified, artifact_mask, max_gap=7, passes=3, extended=False):
    """Fill artifact pixels only when opposite-side labels agree.

    Four direction pairs are inspected. The diagonal pairs are essential for
    inclined boreholes and grids; accepting a proposal still requires every
    available direction to agree on the same class.
    """
    out=classified.copy();candidate=artifact_mask&(out<0);recovered=0
    max_gap=max(1,int(max_gap))
    for _ in range(max(1,int(passes))):
        pending=candidate&(out<0)
        if not np.any(pending):break
        agreed=np.full(out.shape,-1,np.int32);conflict=np.zeros(out.shape,bool)
        directions=[(1,0),(0,1),(1,1),(1,-1)]
        if extended:directions += [(1,2),(2,1),(1,-2),(2,-1)]
        for dy,dx in directions:
            negative=np.full(out.shape,-1,np.int32);positive=np.full(out.shape,-1,np.int32)
            for distance in range(1,max_gap+1):
                if (dy,dx)==(1,0):
                    negative[distance:]=np.where((negative[distance:]<0)&(out[:-distance]>=0),out[:-distance],negative[distance:])
                    positive[:-distance]=np.where((positive[:-distance]<0)&(out[distance:]>=0),out[distance:],positive[:-distance])
                elif (dy,dx)==(0,1):
                    negative[:,distance:]=np.where((negative[:,distance:]<0)&(out[:,:-distance]>=0),out[:,:-distance],negative[:,distance:])
                    positive[:,:-distance]=np.where((positive[:,:-distance]<0)&(out[:,distance:]>=0),out[:,distance:],positive[:,:-distance])
                elif dx>0:
                    yy=dy*distance;xx=dx*distance
                    negative[yy:,xx:]=np.where((negative[yy:,xx:]<0)&(out[:-yy,:-xx]>=0),out[:-yy,:-xx],negative[yy:,xx:])
                    positive[:-yy,:-xx]=np.where((positive[:-yy,:-xx]<0)&(out[yy:,xx:]>=0),out[yy:,xx:],positive[:-yy,:-xx])
                else:
                    yy=dy*distance;xx=-dx*distance
                    negative[yy:,:-xx]=np.where((negative[yy:,:-xx]<0)&(out[:-yy,xx:]>=0),out[:-yy,xx:],negative[yy:,:-xx])
                    positive[:-yy,xx:]=np.where((positive[:-yy,xx:]<0)&(out[yy:,:-xx]>=0),out[yy:,:-xx],positive[:-yy,xx:])
            suggestion=np.where((negative==positive)&(negative>=0),negative,-1)
            conflict|=(agreed>=0)&(suggestion>=0)&(agreed!=suggestion)
            agreed=np.where((agreed<0)&(suggestion>=0),suggestion,agreed)
        agreed[conflict]=-1
        fill=pending&(agreed>=0)
        count=int(fill.sum())
        if not count:break
        out[fill]=agreed[fill];recovered+=count
    return out,{'candidate_pixels':int(candidate.sum()),'recovered_pixels':recovered,
                'unresolved_pixels':int(np.sum(candidate&(out<0)))}

def restore_text_oriented(classified,text_mask,max_gap=18):
    """Complete oblique thin beds through a text mask using local orientations.

    Line-shaped closings propose continuations at 15-degree intervals. If a
    broad host and a thin bed both propose a pixel, the locally smaller class
    wins only inside its narrow continuation; the host fills the remainder.
    """
    out=classified.copy();pending=text_mask.astype(bool)&(out<0)
    count,cc,stats,_=cv2.connectedComponentsWithStats(pending.astype(np.uint8),8)
    recovered=0;gap=max(3,int(max_gap));size=2*gap+1;centre=gap
    kernels=[]
    for angle in range(0,180,15):
        radians=np.deg2rad(angle);dx=round(np.cos(radians)*gap);dy=round(np.sin(radians)*gap)
        kernel=np.zeros((size,size),np.uint8);cv2.line(kernel,(centre-dx,centre-dy),(centre+dx,centre+dy),1,1)
        kernels.append(kernel)
    for index in range(1,count):
        x,y,w,h=map(int,stats[index,:4]);x0=max(0,x-gap-2);y0=max(0,y-gap-2)
        x1=min(out.shape[1],x+w+gap+2);y1=min(out.shape[0],y+h+gap+2)
        local=out[y0:y1,x0:x1];target=(cc[y0:y1,x0:x1]==index)&(local<0)
        classes=np.unique(local[local>=0]);best_size=np.full(local.shape,np.iinfo(np.int32).max,np.int32);best=np.full(local.shape,-1,np.int16)
        for class_id in classes:
            seed=(local==class_id).astype(np.uint8);proposal=np.zeros(local.shape,bool)
            for kernel in kernels:proposal|=cv2.morphologyEx(seed,cv2.MORPH_CLOSE,kernel)>0
            proposal&=target
            local_size=int(seed.sum());choose=proposal&(local_size<best_size)
            best[choose]=int(class_id);best_size[choose]=local_size
        fill=target&(best>=0);local[fill]=best[fill];out[y0:y1,x0:x1]=local;recovered+=int(fill.sum())
    return out,{'candidate_pixels':int(pending.sum()),'recovered_pixels':recovered,
                'unresolved_pixels':int(np.count_nonzero(text_mask.astype(bool)&(out<0)))}

def reconnect_artifact_corridors(classified,artifact_mask,max_gap=32,angle_step=15):
    """Reconnect a colour class only through a confirmed artefact corridor.

    Pixel-wise opposite-side tests become indecisive where a grid or borehole
    crosses a geological contact. Each lithology therefore proposes oriented
    continuations. A pixel is restored only when exactly one lithology proposes
    it, and only inside the accepted artefact mask. Competing proposals remain
    unclassified so the geological contact is preserved for review.
    """
    out=classified.copy();corridor=artifact_mask.astype(bool)&(out<0)
    if not np.any(corridor):
        return out,{'candidate_pixels':0,'recovered_pixels':0,'conflict_pixels':0,'unresolved_pixels':0,
                    'method':'unique_class_oriented_closing'}
    gap=max(3,int(max_gap));step=max(5,min(45,int(angle_step)))
    size=2*gap+1;centre=gap;kernels=[]
    for angle in range(0,180,step):
        radians=np.deg2rad(angle);dx=round(np.cos(radians)*gap);dy=round(np.sin(radians)*gap)
        kernel=np.zeros((size,size),np.uint8)
        cv2.line(kernel,(centre-dx,centre-dy),(centre+dx,centre+dy),1,1)
        kernels.append(kernel)
    proposal_count=np.zeros(out.shape,np.uint8);proposal_label=np.full(out.shape,-1,np.int16)
    for class_id in np.unique(out[out>=0]):
        seed=(out==class_id).astype(np.uint8);proposal=np.zeros(out.shape,bool)
        # The confirmed corridor itself must touch two disconnected pieces of
        # this class. This prevents a closing kernel from jumping across an
        # ordinary blank interval merely because an artefact lies somewhere
        # in the middle of that interval.
        seed_count,seed_cc=cv2.connectedComponents(seed,8)
        joined_count,joined=cv2.connectedComponents((seed.astype(bool)|corridor).astype(np.uint8),8)
        if seed_count<=2:
            eligible_joined=np.zeros(joined_count,bool)
            if seed_count==2:
                touched=np.unique(joined[seed_cc==1]);eligible_joined[touched[touched>0]]=False
        else:
            base=seed_count
            known=seed_cc>0
            pairs=np.unique(joined[known].astype(np.int64)*base+seed_cc[known].astype(np.int64))
            pair_groups=(pairs//base).astype(np.int32)
            touches=np.bincount(pair_groups,minlength=joined_count)
            eligible_joined=touches>=2
        eligible=corridor&eligible_joined[joined]
        if not np.any(eligible):continue
        for kernel in kernels:
            proposal|=(cv2.morphologyEx(seed,cv2.MORPH_CLOSE,kernel)>0)&eligible
        proposal_count[proposal]=np.minimum(255,proposal_count[proposal].astype(np.uint16)+1).astype(np.uint8)
        proposal_label[proposal]=int(class_id)
    unique=corridor&(proposal_count==1)
    out[unique]=proposal_label[unique]
    return out,{'candidate_pixels':int(corridor.sum()),'recovered_pixels':int(unique.sum()),
                'conflict_pixels':int(np.count_nonzero(corridor&(proposal_count>1))),
                'unresolved_pixels':int(np.count_nonzero(artifact_mask.astype(bool)&(out<0))),
                'method':'unique_class_oriented_closing','angle_step_degrees':step,'max_gap_pixels':gap}

def fill_enclosed_gaps(classified,content,max_fraction=.08):
    """Partition enclosed unclassified holes by nearest surrounding class.

    Components touching the content boundary remain background.  Multiple
    surrounding strata divide a hole using nearest classified pixels, so a
    white annotation gap does not arbitrarily become a single stratum.
    """
    out=classified.copy();inside=np.ones(out.shape,bool) if content is None else content.astype(bool)
    gaps=(out<0)&inside
    count,cc,stats,_=cv2.connectedComponentsWithStats(gaps.astype(np.uint8),8)
    edge=inside&~cv2.erode(inside.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    filled=0;components=0;limit=max(1,round(out.size*float(max_fraction)))
    for index in range(1,count):
        area=int(stats[index,cv2.CC_STAT_AREA])
        if area>limit:continue
        x,y,w,h=map(int,stats[index,:4]);component=cc[y:y+h,x:x+w]==index
        if np.any(component&edge[y:y+h,x:x+w]):continue
        pad=2;x0=max(0,x-pad);y0=max(0,y-pad);x1=min(out.shape[1],x+w+pad);y1=min(out.shape[0],y+h+pad)
        local=out[y0:y1,x0:x1];target=cc[y0:y1,x0:x1]==index;known=(local>=0)&~target
        if not np.any(known):continue
        source=(~known).astype(np.uint8)
        _,nearest=cv2.distanceTransformWithLabels(source,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        lookup=np.full(int(nearest.max())+1,-1,np.int16);lookup[nearest[known]]=local[known]
        proposal=lookup[nearest]
        valid=target&(proposal>=0)
        if not np.any(valid):continue
        local[valid]=proposal[valid];out[y0:y1,x0:x1]=local;filled+=int(valid.sum());components+=1
    return out,{'filled_pixels':filled,'filled_components':components,'max_fraction':float(max_fraction)}
