"""Deterministic legend catalogue built before section recognition.

The catalogue separates lithology swatches, geological boundaries and other
drawing symbols.  It records visual evidence rather than trusting OCR names:
cell geometry, colour distributions, line width/style/orientation and text
presence are all retained for later artifact and boundary decisions.
"""
from __future__ import annotations

import re
import cv2
import numpy as np
from PIL import Image

from .io import local


GEOLOGICAL_TERMS=re.compile(r'断层|断裂|地质.{0,3}(界|边)|地层.{0,3}(界|边)|岩性.{0,3}(界|边)|接触.{0,3}(界|边)|不整合|构造.{0,3}(界|边)')
LINE_TERMS=re.compile(r'线|范围|边界|境界')
LABEL_TERMS=re.compile(r'编号|标高|孔深|厚度|品位|方位角|位置')


def _normalise_text(text):
    return re.sub(r'\s+','',text or '').strip()


def _intersection(a,b):
    x,y,w,h=a;u,v,p,q=b
    return max(0,min(x+w,u+p)-max(x,u))*max(0,min(y+h,v+q)-max(y,v))


def _deduplicate(boxes):
    result=[]
    for box in sorted(boxes,key=lambda b:b[2]*b[3],reverse=True):
        if any(_intersection(box,other)>=.75*min(box[2]*box[3],other[2]*other[3]) for other in result):continue
        result.append(box)
    return sorted(result,key=lambda b:(b[1],b[0]))


def _palette(pixels,max_colours=4):
    if not len(pixels):return []
    quant=(pixels//16)*16+8;colours,counts=np.unique(quant,axis=0,return_counts=True)
    order=np.argsort(counts)[::-1][:max_colours];total=max(1,int(counts.sum()))
    return [{'rgb':[int(v) for v in colours[i]],'fraction':round(float(counts[i]/total),4)} for i in order]


def _skeleton(binary):
    work=binary.astype(np.uint8).copy();skeleton=np.zeros_like(work);kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # Explicit zero borders are required: OpenCV's erosion-neutral default can
    # keep an all-one tiny glyph unchanged forever.
    for _ in range(max(work.shape)+2):
        if not np.any(work):break
        eroded=cv2.erode(work,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        opened=cv2.dilate(eroded,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        skeleton|=work&~opened
        work=eroded
    return skeleton


def _symbol_features(patch,is_swatch=False):
    h,w=patch.shape[:2];margin=max(3,round(min(h,w)*.08));inner=patch[margin:h-margin,margin:w-margin]
    lab=cv2.cvtColor(inner.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    light=lab[:,:,0];chroma=np.linalg.norm(lab[:,:,1:],axis=2)
    nonpaper=(light<96)|(chroma>4.5)
    dark=light<58
    coloured=chroma>7
    dominant_pixels=inner[nonpaper]
    result={'dominant_rgb':[int(v) for v in np.median(dominant_pixels,axis=0)] if len(dominant_pixels) else [255,255,255],
            'palette':_palette(dominant_pixels),'ink_fraction':round(float(nonpaper.mean()),4),
            'dark_fraction':round(float(dark.mean()),4),'colour_fraction':round(float(coloured.mean()),4),
            'uniformity':round(float(1-np.mean(np.std(inner.astype(np.float32),axis=(0,1)))/128),4)}
    if is_swatch:
        # Dark glyphs inside a coloured swatch are label evidence, not texture.
        fill=inner[~dark]
        result['fill_rgb']=[int(v) for v in np.median(fill,axis=0)] if len(fill) else result['dominant_rgb']
        result['embedded_label_fraction']=round(float(dark.mean()),4)
        return result,{'present':False}

    gray=cv2.cvtColor(inner,cv2.COLOR_RGB2GRAY);ink=((gray<215)|(coloured)).astype(np.uint8)
    # Remove residual cell frame after the crop and use Hough only for strokes
    # appreciably longer than ordinary Chinese/Latin character segments.
    line_input=(ink*255).astype(np.uint8)
    minimum=max(8,round(max(inner.shape)*.18))
    lines=cv2.HoughLinesP(line_input,1,np.pi/360,threshold=max(8,minimum//3),
                          minLineLength=minimum,maxLineGap=max(2,round(min(inner.shape)*.08)))
    segments=[]
    if lines is not None:
        for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
            length=float(np.hypot(int(x2)-int(x1),int(y2)-int(y1)))
            angle=float(np.degrees(np.arctan2(int(y2)-int(y1),int(x2)-int(x1)))%180)
            segments.append((length,angle,(int(x1),int(y1),int(x2),int(y2))))
    segments.sort(reverse=True);longest=segments[0][0] if segments else 0.
    component_count,component_labels,component_stats,_=cv2.connectedComponentsWithStats(ink,8)
    component_span=0.;component_index=0
    for index in range(1,component_count):
        _,_,cw,ch,area=component_stats[index];span=float(np.hypot(cw,ch))
        if area>=4 and span>component_span:component_span=span;component_index=index
    present=bool(longest>=max(inner.shape)*.25 or component_span>=max(inner.shape)*.48)
    line_mask=np.zeros(ink.shape,np.uint8)
    if present:
        for _,_,(x1,y1,x2,y2) in segments[:12]:cv2.line(line_mask,(x1,y1),(x2,y2),1,3)
        if component_index and component_span>=max(inner.shape)*.48:line_mask|=(component_labels==component_index).astype(np.uint8)
        line_mask&=ink
    skeleton=_skeleton(line_mask) if np.any(line_mask) else line_mask
    if np.any(line_mask):
        padded=np.pad(line_mask.astype(np.uint8),1,constant_values=0)
        distance=cv2.distanceTransform(padded,cv2.DIST_L2,5)[1:-1,1:-1]
        width=float(2*np.median(distance[skeleton>0])) if np.any(skeleton) else 0.
    else:width=0.
    line_pixels=inner[line_mask>0]
    line_rgb=[int(v) for v in np.median(line_pixels,axis=0)] if len(line_pixels) else [0,0,0]
    orientations=[s[1] for s in segments[:8]]
    main_angle=float(segments[0][1]) if segments else None
    angle_spread=float(np.std(orientations)) if orientations else 0.
    occupancy=longest/max(1,float(np.hypot(w,h)))
    if not present:shape='none'
    elif component_span>=max(inner.shape)*.48 and (not segments or angle_spread>28):shape='polyline_or_curve'
    elif len(segments)>=3 and angle_spread>22:shape='polyline_or_curve'
    elif occupancy<.55 and len(segments)>=3:shape='dashed'
    elif main_angle is not None and (main_angle<12 or main_angle>168):shape='horizontal_straight'
    elif main_angle is not None and 78<main_angle<102:shape='vertical_straight'
    else:shape='inclined_straight'
    return result,{'present':present,'dominant_rgb':line_rgb,'mean_width_px':round(width,2),
        'orientation_degrees':None if main_angle is None else round(main_angle,2),'shape':shape,
        'segment_count':len(segments),'longest_segment_px':round(longest,2),'connected_span_px':round(component_span,2),
        'orientation_spread':round(angle_spread,2)}


def _panel_from_swatches(size,swatches):
    width,height=size;boxes=[e.get('detected_box',e.get('swatch')) for e in swatches if e.get('swatch')]
    if not boxes:return [0,0,width,height]
    median_h=float(np.median([b[3] for b in boxes]));median_w=float(np.median([b[2] for b in boxes]))
    steps=np.diff(sorted(b[1] for b in boxes));steps=steps[steps>median_h*.8]
    pitch=float(np.median(steps)) if len(steps) else median_h*1.6
    x0=max(0,round(min(b[0] for b in boxes)-median_w*1.5));y0=max(0,round(min(b[1] for b in boxes)-pitch*3.2))
    x1=min(width,round(x0+max(median_w*15,width-x0-80)));y1=min(height,round(max(b[1]+b[3] for b in boxes)+pitch*8))
    return [x0,y0,x1-x0,y1-y0]


def _find_cells(rgb,panel,known_swatches):
    px,py,pw,ph=panel;crop=rgb[py:py+ph,px:px+pw];gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,35,125);contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    known=[e.get('detected_box',e.get('swatch')) for e in known_swatches if e.get('swatch')]
    if known:
        mw=float(np.median([b[2] for b in known]));mh=float(np.median([b[3] for b in known]))
    else:mw,mh=max(20,pw*.07),max(15,ph*.025)
    candidates=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour);area=float(cv2.contourArea(contour))
        if not (.55*mw<=w<=1.8*mw and .55*mh<=h<=1.8*mh):continue
        if not .65<=w/max(h,1)<=2.7 or area<.58*w*h:continue
        candidates.append([x+px,y+py,w,h])
    cells=_deduplicate(candidates)
    # Retain aligned columns with repeated cells; isolated title-box fragments
    # and drawing frame coordinates do not form a legend column.
    supported=[]
    for cell in cells:
        peers=[other for other in cells if abs((other[0]+other[2]/2)-(cell[0]+cell[2]/2))<=mw*.55]
        if len(peers)>=3:supported.append(cell)
    centres=sorted(b[0]+b[2]/2 for b in supported);column_groups=[]
    for centre in centres:
        if not column_groups or centre-column_groups[-1][-1]>mw*.7:column_groups.append([centre])
        else:column_groups[-1].append(centre)
    columns=[float(np.median(group)) for group in column_groups]
    return sorted(supported,key=lambda b:(min(range(len(columns)),key=lambda i:abs(columns[i]-(b[0]+b[2]/2))),b[1]))


def _words_by_cell(ocr,cells,panel):
    words=[]
    for line in ocr.get('lines',[]):
        for word in line.get('words',[]):
            if _normalise_text(word.get('text')):words.append(word)
    centres=sorted(set(round(c[0]+c[2]/2) for c in cells));groups=[]
    for centre in centres:
        if not groups or centre-groups[-1][-1]>max(30,np.median([c[2] for c in cells])*.7):groups.append([centre])
        else:groups[-1].append(centre)
    columns=[float(np.median(group)) for group in groups]
    result=[];px,py,pw,ph=panel
    for cell in cells:
        x,y,w,h=cell;cx=x+w/2;column=min(range(len(columns)),key=lambda i:abs(columns[i]-cx))
        next_left=(columns[column+1]-w*.65) if column+1<len(columns) else px+pw
        symbol=[];description=[]
        for word in words:
            bx,by,bw,bh=word['box'];wx,wy=bx+bw/2,by+bh/2
            if y-h*.22<=wy<=y+h*1.22:
                if x<=wx<=x+w:symbol.append(word)
                elif x+w-2<=wx<=next_left:description.append(word)
        symbol_text=_normalise_text(''.join(str(v['text']) for v in sorted(symbol,key=lambda z:z['box'][0])))
        description_text=_normalise_text(''.join(str(v['text']) for v in sorted(description,key=lambda z:z['box'][0])))
        result.append((symbol_text,description_text,column))
    return result


def analyze_legend(image,known_swatches=(),page=0,ocr_result=None):
    """Return a complete, auditable catalogue for a two/multi-column legend."""
    with Image.open(local(image)) as source:
        source.seek(page);rgb=np.asarray(source.convert('RGB'));size=source.size
    panel=_panel_from_swatches(size,known_swatches);cells=_find_cells(rgb,panel,known_swatches)
    if ocr_result is None:
        if cells:
            median_h=float(np.median([cell[3] for cell in cells]));top=max(panel[1],round(min(c[1] for c in cells)-median_h*1.5))
            bottom=min(panel[1]+panel[3],round(max(c[1]+c[3] for c in cells)+median_h*1.5))
            ocr_panel=[panel[0],top,panel[2],max(1,bottom-top)]
        else:ocr_panel=panel
        # Resolve through the module at call time so test and deployment OCR
        # adapters can be replaced without leaving a stale imported function.
        from . import ocr
        ocr_result=ocr.recognize_text(image,page,ocr_panel)
    texts=_words_by_cell(ocr_result,cells,panel)
    items=[]
    for number,(cell,(symbol_text,description,column)) in enumerate(zip(cells,texts),1):
        x,y,w,h=cell;overlaps=[(_intersection(cell,e.get('detected_box',e.get('swatch'))),e) for e in known_swatches]
        overlap,known=max(overlaps,key=lambda pair:pair[0],default=(0,None));is_swatch=bool(known and overlap>=.35*w*h)
        appearance,line=_symbol_features(rgb[y:y+h,x:x+w],is_swatch)
        geological=bool(line['present'] and GEOLOGICAL_TERMS.search(description))
        if is_swatch:
            category='stratum_swatch';subtype='colour_block';confidence=.98 if known.get('verified') else .9
            name=description or known.get('lithology') or 'UNKNOWN'
        elif geological:
            category='geological_boundary';subtype='label_line' if symbol_text else 'line';confidence=.92
            name=description or symbol_text or f'geological_boundary_{number}'
        else:
            category='other'
            if line['present'] and symbol_text:subtype='label_line'
            elif line['present']:subtype='line'
            else:subtype='label'
            # Semantic text corrects glyphs such as the single "I" mine-body
            # number which Hough can otherwise mistake for a technical line.
            if LABEL_TERMS.search(description) and not LINE_TERMS.search(description) and line['segment_count']<=2:subtype='label'
            confidence=.86 if description else .58;name=description or symbol_text or f'other_{number}'
        flags=[]
        if not description:flags.append('name_requires_review')
        if category=='other' and line['present'] and GEOLOGICAL_TERMS.search(symbol_text):flags.append('possible_geological_boundary')
        item={'id':f'LEG-{number:02d}','name':name,'ocr_description':description,'symbol_text':symbol_text,
              'category':category,'subtype':subtype,'confidence':round(confidence,2),'box':cell,'column':column+1,
              'appearance':appearance,'line':line,'review_flags':flags}
        if is_swatch:
            item['linked_lithology']=known.get('lithology','UNKNOWN');item['linked_swatch']=known.get('swatch')
        items.append(item)
    counts={}
    for item in items:
        key=item['category'] if item['category']!='other' else f"other.{item['subtype']}"
        counts[key]=counts.get(key,0)+1
    return {'method':'legend_catalogue_geometry_colour_ocr_v1','status':'manual_review' if any(i['review_flags'] for i in items) else 'deterministic',
            'panel':panel,'cell_count':len(cells),'counts':counts,'ocr_status':ocr_result.get('status','unknown'),
            'items':items,'limitations':['OCR names are suggestions','Geological-boundary status requires line evidence plus geological semantics','Unknown names remain review flags']}
