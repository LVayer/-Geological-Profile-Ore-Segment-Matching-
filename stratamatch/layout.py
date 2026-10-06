"""Conservative layout proposals for color software exports, not geological labels.

Large connected color-filled bodies suggest plot extents; detached rectangular,
uniform swatches must repeat plot colors and form an aligned legend group.
No detection output silently authorizes an unknown lithology or georeferencing.
"""
from copy import deepcopy
import re
import numpy as np
import cv2
from PIL import Image
from .io import local

def intersection(a,b):
    x,y,w,h=a;u,v,p,q=b
    return max(0,min(x+w,u+p)-max(x,u))*max(0,min(y+h,v+q)-max(y,v))

def detect(image,page=0):
    with Image.open(local(image)) as im:
        im.seek(page);rgb=np.array(im.convert('RGB'))
    oh,ow=rgb.shape[:2];scale=min(1.,1800/max(oh,ow))
    if scale<1:rgb=cv2.resize(rgb,(round(ow*scale),round(oh*scale)),interpolation=cv2.INTER_AREA)
    h,w=rgb.shape[:2];lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    chroma=np.linalg.norm(lab[:,:,1:],axis=2)
    # Engineering sections often use very pale pink/yellow fills. A lower
    # chroma threshold keeps them while still excluding white paper and gray grids.
    color=(chroma>6.5)&(lab[:,:,0]>18)&(lab[:,:,0]<98)
    k=max(3,min(11,round(min(w,h)*.012)|1))
    joined=cv2.morphologyEx(color.astype('uint8'),cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(joined,8)
    plots=[]
    for index in range(1,count):
        x,y,bw,bh,area=map(int,stats[index]);density=float(color[y:y+bh,x:x+bw].mean())
        if area>=w*h*.075 and bw>=w*.3 and bh>=h*.18 and density>=.22:
            plots.append({'box':[x,y,bw,bh],'area':area,'density':density,'label':index})
    plots.sort(key=lambda p:p['area'],reverse=True)
    result={'method':'color_layout_v2_full_image_legend','status':'manual_review','roi':None,'legend':[],
        'roi_candidates':[],'content_polygons':[],'content_rings':[],'warnings':[],'lithology_authorized':False,
        'legend_search_scope':'full_image_including_roi'}
    def original(box):
        x,y,bw,bh=box;xx,yy=max(0,round(x/scale)),max(0,round(y/scale))
        return [xx,yy,min(ow-xx,max(1,round(bw/scale))),min(oh-yy,max(1,round(bh/scale)))]
    result['roi_candidates']=[original(p['box']) for p in plots[:5]]
    if not plots:
        result['warnings'].append('未发现可靠彩色主图；灰度/纹理图或分散地层请手动框选')
        return result
    if len(plots)>1 and plots[1]['area']>plots[0]['area']*.45:
        result['warnings'].append('存在多个相近规模主图候选，未自动选择；请手动框选')
        return result
    roi=plots[0]['box'];result['roi']=original(roi)
    # Preserve a curved content footprint as well as the coordinate rectangle.
    # The footprint prevents inner blank areas and legends becoming strata.
    body=(labels==plots[0]['label']).astype('uint8')
    contours,hierarchy=cv2.findContours(body,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    hierarchy=hierarchy[0] if hierarchy is not None else []
    for index,contour in enumerate(contours):
        perimeter=cv2.arcLength(contour,True)
        simplified=cv2.approxPolyDP(contour,max(1.,perimeter*.0015),True)[:,0,:]
        if len(simplified)>=3:
            points=[[round(float(px)/scale),round(float(py)/scale)] for px,py in simplified]
            hole=bool(len(hierarchy) and hierarchy[index][3]>=0)
            result['content_rings'].append({'points':points,'hole':hole})
            if not hole:result['content_polygons'].append(points)
    x,y,bw,bh=roi;plot_lab=lab[y:y+bh,x:x+bw][color[y:y+bh,x:x+bw]]
    # Quantized color prototypes are sampled from the body, not from chart text.
    quantized=np.round(plot_lab/5)*5
    palette,counts=np.unique(quantized,axis=0,return_counts=True)
    # Frequent body colors are stable geological prototypes; arbitrary sampling
    # of sorted unique colors can omit exactly the colors used by the legend.
    palette=palette[np.argsort(counts)[::-1][:256]]
    # Search the complete image, including blank space inside the rectangular
    # ROI. Only the connected main-content footprint is removed.
    swatch_space=color & ~body.astype(bool)
    count,cc,stats,_=cv2.connectedComponentsWithStats(swatch_space.astype('uint8'),8)
    swatches=[]
    for index in range(1,count):
        sx,sy,sw,sh,area=map(int,stats[index]);box=[sx,sy,sw,sh]
        if min(sw,sh)<max(4,round(min(w,h)*.004)) or area>plots[0]['area']*.05 or sw>w*.2 or sh>h*.13:continue
        if not .35<=sw/sh<=7 or area/(sw*sh)<.83:continue
        pixels=lab[sy:sy+sh,sx:sx+sw];median=np.median(pixels.reshape(-1,3),axis=0)
        uniform=float(np.mean(np.linalg.norm(pixels-median,axis=2)<8))
        if uniform<.72 or np.min(np.linalg.norm(palette-median,axis=1))>12:continue
        swatches.append({'box':box,'uniformity':uniform,'rgb':np.median(rgb[sy:sy+sh,sx:sx+sw].reshape(-1,3),axis=0).astype(int).tolist()})
    # Isolated squares can be map symbols: require repeated size and aligned layout.
    supported=[]
    for a in swatches:
        ax,ay,aw,ah=a['box'];peers=[]
        for b in swatches:
            if a is b:continue
            bx,by,bww,bhh=b['box']
            same_size=.45<=aw/bww<=2.2 and .45<=ah/bhh<=2.2
            aligned=abs(ax-bx)<=max(aw,bww)*.6 or abs(ay-by)<=max(ah,bhh)*.6
            if same_size and aligned:peers.append(b)
        if peers:supported.append(a)
    # A white/unfilled legend cell has no chroma and therefore cannot be found by
    # the colour mask. Recover only a single interior gap in an otherwise regular,
    # aligned swatch column. Requiring coloured rows on both sides avoids turning
    # arbitrary empty boxes or line-symbol legends into lithology candidates.
    ordered=sorted(supported,key=lambda a:(a['box'][0],a['box'][1]))
    inferred=[]
    if len(ordered)>=4:
        widths=np.array([a['box'][2] for a in ordered],float);heights=np.array([a['box'][3] for a in ordered],float)
        median_w,median_h=float(np.median(widths)),float(np.median(heights))
        aligned=[a for a in ordered if abs(a['box'][0]-np.median([b['box'][0] for b in ordered]))<=median_w*.65]
        aligned.sort(key=lambda a:a['box'][1])
        steps=np.diff([a['box'][1] for a in aligned])
        regular=steps[(steps>=median_h*1.15)&(steps<=median_h*2.15)]
        pitch=float(np.median(regular)) if len(regular) else 0
        if pitch:
            for first,second in zip(aligned,aligned[1:]):
                ax,ay,aw,ah=first['box'];bx,by,bww,bhh=second['box'];gap=by-ay
                if 1.65*pitch<=gap<=2.35*pitch:
                    iy=round((ay+by)/2);ix=round((ax+bx)/2);iw=round((aw+bww)/2);ih=round((ah+bhh)/2)
                    # The inferred cell must have a visible coloured/dark frame
                    # and a substantially pale interior at the expected position.
                    patch=lab[max(0,iy):min(h,iy+ih),max(0,ix):min(w,ix+iw)]
                    if patch.shape[0]<5 or patch.shape[1]<5:continue
                    edge=np.concatenate((patch[0].reshape(-1,3),patch[-1].reshape(-1,3),patch[:,0],patch[:,-1]))
                    inner=patch[max(1,ih//5):max(2,ih-ih//5),max(1,iw//5):max(2,iw-iw//5)]
                    edge_signal=float(np.mean((np.linalg.norm(edge[:,1:],axis=1)>6.5)|(edge[:,0]<75)))
                    dark_mark=float(np.mean(patch[:,:,0]<75))
                    pale=float(np.mean((inner[:,:,0]>88)&(np.linalg.norm(inner[:,:,1:],axis=2)<8))) if inner.size else 0
                    if (edge_signal>=.16 or dark_mark>=.015) and pale>=.55:
                        inferred.append({'box':[ix,iy,iw,ih],'uniformity':pale,
                            'rgb':np.median(rgb[iy:iy+ih,ix:ix+iw].reshape(-1,3),axis=0).astype(int).tolist(),
                            'inferred_blank':True})
    supported=sorted(supported+inferred,key=lambda a:(a['box'][1],a['box'][0]))
    for swatch in supported:
        sx,sy,sw,sh=swatch['box'];inset=max(1,round(min(sw,sh)*.12))
        result['legend'].append({'lithology':'UNKNOWN','swatch':original([sx+inset,sy+inset,sw-2*inset,sh-2*inset]),
            'detected_box':original(swatch['box']),'rgb':swatch['rgb'],'verified':False,
            'source':'auto_layout_blank_gap' if swatch.get('inferred_blank') else 'auto_layout',
            'inferred_blank':bool(swatch.get('inferred_blank')),'uniformity':swatch['uniformity']})
    result['status']='proposed' if result['legend'] else 'manual_review'
    result['warnings'].append('矩形 ROI 仅用于坐标范围；内容曲线用于剔除内部空白与图例，物理范围必须重新核对')
    if not result['legend']:result['warnings'].append('未找到经过重复色彩和排列交叉检查的图例；可手动框选')
    else:result['warnings'].append('色块位置已建议；岩性名称需要 OCR 或人工填写并确认')
    return result

def apply_proposal(section,proposal):
    """Preserve reviewed work; always retain the new detection for inspection."""
    s=deepcopy(section);s['auto_layout']=deepcopy(proposal)
    if not s.get('layout_reviewed'):
        if proposal['roi']:s['roi']=proposal['roi']
        if proposal.get('content_polygons'):s['plot_content_polygons']=deepcopy(proposal['content_polygons'])
        if proposal.get('content_rings'):s['plot_content_rings']=deepcopy(proposal['content_rings'])
        s['layout_reviewed']=False
    # Re-running detection must refresh stale automatic proposals, while entries
    # explicitly confirmed or manually drawn by the user remain authoritative.
    existing=s.get('legend',[])
    reviewed=[e for e in existing if e.get('verified') or not str(e.get('source','')).startswith('auto_layout')]
    if s.get('layout_reviewed') and existing:
        s['legend']=existing
    elif reviewed:
        s['legend']=reviewed+[
            e for e in deepcopy(proposal['legend'])
            if not any(intersection(e['detected_box'],r.get('detected_box',r.get('swatch',[0,0,0,0])))>
                       .5*e['detected_box'][2]*e['detected_box'][3] for r in reviewed)
        ]
    else:s['legend']=deepcopy(proposal['legend'])
    return s

def attach_text(legend,lines):
    """Associate right-hand OCR words to swatch rows, rejecting shared/ambiguous rows."""
    result=deepcopy(legend);claims={}
    for i,e in enumerate(result):
        if e.get('verified'):continue
        x,y,w,h=e.get('detected_box',e['swatch']);choices=[]
        for j,line in enumerate(lines):
            words=[]
            for word in line.get('words',[]):
                bx,by,bw,bh=word['box']
                if bx>=x+w-2 and bx-x-w<=max(400,w*12) and abs(by+bh/2-y-h/2)<=max(h,bh)*.7:
                    # Do not consume another column's legend label.
                    blocking=[v for k,v in enumerate(result) if k!=i and v['swatch'][0]>x+w and v['swatch'][0]<bx and abs(v['swatch'][1]-y)<h]
                    if not blocking:words.append(word)
            if words:
                distance=min(word['box'][0]-x-w for word in words)
                choices.append((distance,j,' '.join(word['text'] for word in sorted(words,key=lambda a:a['box'][0]))))
        choices.sort()
        if choices and (len(choices)==1 or choices[1][0]-choices[0][0]>w):
            _,line_id,text=choices[0];claims.setdefault(line_id,[]).append((i,text))
    for rows in claims.values():
        if len(rows)!=1:continue
        i,text=rows[0]
        if text.strip():
            result[i]['ocr_text']=text.strip()
            result[i]['lithology']=re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])','',text.strip())
            result[i]['verified']=False
    return result

def legend_region(legend,size):
    """Crop legend rows and their right-side labels; large plots can suppress OCR."""
    boxes=[e.get('detected_box',e.get('swatch')) for e in legend if e.get('swatch')]
    if not boxes:return None
    w,h=size;pad=max(b[3] for b in boxes)
    left=max(0,min(b[0] for b in boxes)-4);top=max(0,min(b[1] for b in boxes)-pad)
    bottom=min(h,max(b[1]+b[3] for b in boxes)+pad)
    return [left,top,w-left,bottom-top]

def annotation_boxes(lines,roi,legend=()):
    """Return OCR word boxes inside the plot, excluding legend text rows.

    These boxes are proposals for pixel-level ink removal; they never authorize
    a lithology or erase the complete rectangle.
    """
    rx,ry,rw,rh=roi;legend_boxes=[e.get('detected_box',e.get('swatch')) for e in legend]
    result=[]
    for line in lines:
        for word in line.get('words',[]):
            x,y,w,h=word.get('box',[0,0,0,0]);cx,cy=x+w/2,y+h/2
            if not(rx<=cx<=rx+rw and ry<=cy<=ry+rh):continue
            if any(lx-4<=cx<=lx+lw+max(400,lw*12) and ly-h<=cy<=ly+lh+h for lx,ly,lw,lh in legend_boxes):continue
            pad=max(1,min(w,h)*.18)
            result.append([max(rx,x-pad),max(ry,y-pad),min(rx+rw,x+w+pad)-max(rx,x-pad),min(ry+rh,y+h+pad)-max(ry,y-pad)])
    return result

def prepare_image(image,page=0):
    """Create an editable import proposal including local OCR when supported."""
    from .ocr import recognize_text
    with Image.open(local(image)) as im:
        if getattr(im,'n_frames',1)>1:raise ValueError('多页 TIFF 请通过 manifest 指定 page 后载入')
        w,h=im.size
    proposal=detect(image,page)
    s=apply_proposal({'image':image,'roi':[0,0,w,h],'frame':'project_depth_datum','units':'m','bounds':[0,0,w,h],
        'registration_verified':False,'complete':False,'legend':[]},proposal)
    if s['legend']:
        ocr=recognize_text(image,page,legend_region(s['legend'],(w,h)));s['auto_layout']['ocr']={k:ocr[k] for k in ['status','language','reason'] if k in ocr}
        if ocr['status']=='available':s['legend']=attach_text(s['legend'],ocr['lines'])
        # Build the complete catalogue before recognizing the section. Unlike
        # the legacy colour list, this also reads boundary and technical-symbol
        # cells with line colour, width, style, shape and embedded labels.
        from .legend_features import analyze_legend
        s['legend_catalog']=analyze_legend(image,s['legend'],page)
        for item in s['legend_catalog']['items']:
            if item['category']!='stratum_swatch' or not item.get('linked_swatch'):continue
            for entry in s['legend']:
                if intersection(item['linked_swatch'],entry.get('swatch',[0,0,0,0]))>.5*entry['swatch'][2]*entry['swatch'][3]:
                    entry['visual_features']=item['appearance']
                    if item.get('ocr_description'):
                        entry['ocr_text']=item['ocr_description']
                        if entry.get('lithology') in ['',None,'UNKNOWN']:entry['lithology']=item['ocr_description']
                    break
    # A second OCR pass over the plot identifies labels, scale values and notes.
    # It is kept separate from legend OCR because the latter uses a tighter crop
    # and is usually more accurate for lithology names.
    if s.get('roi'):
        plot_ocr=recognize_text(image,page,s['roi'])
        s['auto_layout']['plot_ocr']={k:plot_ocr[k] for k in ['status','language','reason'] if k in plot_ocr}
        if plot_ocr['status']=='available':
            s['auto_text_boxes']=annotation_boxes(plot_ocr['lines'],s['roi'],s.get('legend',[]))
    return s
