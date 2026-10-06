"""从剖面原图的红色 F 线标注提取候选切断边界。"""
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def _normalised(segment):
    p,q=np.asarray(segment[:2],float),np.asarray(segment[2:],float)
    d=q-p;length=float(np.linalg.norm(d))
    if length<1:return None
    d/=length
    if d[1]<0 or (abs(d[1])<1e-6 and d[0]<0):d=-d;p,q=q,p
    n=np.array([-d[1],d[0]])
    return p,q,d,n,length


def _merge_segments(segments):
    """合并同一红线的平行重复检测，保留不同走向和错开的独立线。"""
    groups=[]
    for segment in sorted(segments,key=lambda v:-np.linalg.norm(np.asarray(v[2:])-v[:2])):
        item=_normalised(segment)
        if item is None:continue
        p,q,d,n,length=item;centre=(p+q)/2
        found=None
        for group in groups:
            gd=group['direction'];gn=np.array([-gd[1],gd[0]])
            if abs(float(gd@d))<.995:continue
            if abs(float((centre-group['centre'])@gn))>7:continue
            a,b=sorted([float(p@gd),float(q@gd)])
            lo,hi=group['interval']
            if max(lo,a)-min(hi,b)>18:continue
            found=group;break
        if found is None:
            groups.append({'direction':d,'centre':centre,'interval':sorted([float(p@d),float(q@d)]),
                           'offset':float(centre@n),'length':length})
        else:
            gd=found['direction'];a,b=sorted([float(p@gd),float(q@gd)])
            found['interval']=[min(found['interval'][0],a),max(found['interval'][1],b)]
            if length>found['length']:
                found['offset']=float(centre@np.array([-gd[1],gd[0]]))
                found['length']=length
    result=[]
    for group in groups:
        d=group['direction'];n=np.array([-d[1],d[0]])
        lo,hi=group['interval'];offset=group['offset']
        result.append((d*lo+n*offset,d*hi+n*offset))
    return result


def detect_annotated_fault_lines(image_path,roi,work_shape):
    """颜色与直线一致性定位红色断层注记；不声称自动识别未标注断层。"""
    x,y,w,h=map(int,roi)
    rgb=np.asarray(Image.open(image_path).convert('RGB').crop((x,y,x+w,y+h)))
    red,green,blue=(rgb[:,:,i].astype(np.int16) for i in range(3))
    mask=((red>170)&(green<85)&(blue<85)&(red>2*green)&(red>2*blue)).astype(np.uint8)*255
    work=cv2.resize(mask,(work_shape[1],work_shape[0]),interpolation=cv2.INTER_AREA)
    raw=cv2.HoughLinesP(work,1,np.pi/180,threshold=20,
                        minLineLength=max(28,int(min(work_shape)*.045)),maxLineGap=9)
    segments=[] if raw is None else [s.astype(float) for s in raw.reshape(-1,4)]
    merged=_merge_segments(segments)
    merged=[(p,q) for p,q in merged if np.linalg.norm(q-p)>=max(45,min(work_shape)*.065)]
    return merged,{'red_annotation_pixels':int(np.count_nonzero(work)),
                   'raw_line_segments':len(segments),'merged_candidate_lines':len(merged)}


def fault_line_candidates(lines,instances,descriptors):
    """沿候选线两侧采样地层实例，生成待整套位移验证的同岩性关系。"""
    by_id={d['id']:d for d in descriptors};index_by_id={d['id']:i for i,d in enumerate(descriptors)}
    h,w=instances.shape
    candidates=[];virtual=[];line_audit=[]
    for number,(p,q) in enumerate(lines):
        delta=q-p;length=float(np.linalg.norm(delta))
        if length<1:continue
        axis=delta/length;normal=np.array([-axis[1],axis[0]])
        sides={-1:defaultdict(list),1:defaultdict(list)}
        for t in np.linspace(0,1,max(2,int(length/3))):
            point=p+t*delta
            for sign in (-1,1):
                for radius in range(2,31,2):
                    sample=np.rint(point+normal*sign*radius).astype(int)
                    if not (0<=sample[0]<w and 0<=sample[1]<h):break
                    part=int(instances[sample[1],sample[0]])
                    if part in by_id:
                        sides[sign][part].append(sample.astype(float));break
        # 采样到的线两侧部件必须有实际不同的实例；红字和边框无法满足。
        left={pid:np.median(points,axis=0) for pid,points in sides[-1].items() if len(points)>=2}
        right={pid:np.median(points,axis=0) for pid,points in sides[1].items() if len(points)>=2}
        cut_id=-(number+1)
        virtual.append({'id':cut_id,'axis':axis,'centre':(p+q)/2,'lith':-1,
                        'centreline':np.stack([p,q]),'tangents':np.stack([axis,axis]),
                        'width_profile':np.array([2.,2.])})
        line_audit.append({'fault_candidate':number,'endpoints_work_pixels':[p.tolist(),q.tolist()],
                           'left_parts':len(left),'right_parts':len(right)})
        if not left or not right:continue
        for u,pu in left.items():
            for v,pv in right.items():
                if u==v or by_id[u]['lith']!=by_id[v]['lith']:continue
                a,b=by_id[u],by_id[v]
                ai=np.argmin(np.sum((a['centreline']-pu)**2,axis=1))
                bi=np.argmin(np.sum((b['centreline']-pv)**2,axis=1))
                ta,tb=a['tangents'][ai],b['tangents'][bi]
                angle=np.degrees(np.arccos(np.clip(abs(float(ta@tb)),0,1)))/90
                link=pv-pu;unit=link/max(float(np.linalg.norm(link)),1.)
                direction=.5*angle+.25*(1-abs(float(unit@ta)))+.25*(1-abs(float(unit@tb)))
                na=np.array([-ta[1],ta[0]])
                aw=float(a['width_profile'][min(ai,len(a['width_profile'])-1)])
                bw=float(b['width_profile'][min(bi,len(b['width_profile'])-1)])
                position=min(1.,abs(float(link@na))/max(aw+bw,2.))
                score=.48*direction+.12*position+.12
                ports=[]
                for d,point in ((a,pu),(b,pv)):
                    tangent=d['axis'];n=np.array([-tangent[1],tangent[0]])
                    v0=point-d['centre']
                    if abs(float(v0@tangent))/max(d['length']*.5,1)>=abs(float(v0@n))/max(d['width']*.5,1):
                        ports.append(1 if v0@tangent>=0 else 0)
                    else:ports.append(3 if v0@n>=0 else 2)
                candidates.append({'left_index':index_by_id[u],'right_index':index_by_id[v],
                    'left_part':u,'right_part':v,'left_tip':ports[0],'right_tip':ports[1],
                    'score':round(float(score),5),'decision':'candidate','evidence_mode':'annotated_fault_line',
                    'contact_evidence_sufficient':False,'cut_part_ids':[cut_id],
                    'cut_layer_count':1,'total_cut_thickness_pixels':0.,
                    'curve_work_pixels':[pu.tolist(),pv.tolist()],
                    'weights':{'direction':.48,'position':.12},
                    'costs':{'direction':float(direction),'position':float(position)},
                    'fault_candidate':number})
    return candidates,virtual,line_audit
