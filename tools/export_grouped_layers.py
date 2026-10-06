"""Reuse reviewed balloon domains; classify and label disconnected layer groups."""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.layer_groups import recognize_layer_groups
from stratamatch.colour_refill import refill_colours
from experimental_fullres_colour_membrane import classify_full_resolution
from export_numbered_layers import _anchor,_fits,_box


def run(config,balloon,output,codes):
    cfg=json.loads(Path(config).read_text(encoding='utf-8'));name=Path(config).stem
    folder=Path(output)/name;folder.mkdir(parents=True,exist_ok=True)
    domain=np.asarray(Image.open(Path(balloon)/name/'07_总体轮廓掩膜.png'))>0
    cache=folder/'classification.npz'
    cache_reused=cache.exists()
    # Cache contains only reusable colour evidence, never previous grouping decisions.
    signature=json.dumps(cfg,sort_keys=True)
    if cache.exists():
        data=np.load(cache)
        if str(data['signature'])!=signature:raise ValueError('配置已变化，请指定新的输出目录以重新分类')
        labels=data['labels'];colours=data['colours'];white=data['white']
    else:
        labels,colours,_,_,_,pale,audit=classify_full_resolution(cfg)
        if labels.shape!=domain.shape:raise ValueError('气球范围与识别网格不一致')
        white=cv2.resize(pale.astype(np.float32),(labels.shape[1],labels.shape[0]),interpolation=cv2.INTER_AREA)>.65
        np.savez_compressed(cache,labels=labels,colours=colours,white=white,signature=signature)
        (folder/'颜色识别审计.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    labels=labels.copy();labels[~domain]=-1
    cleared_labels=labels.copy()
    white_ids=[i for i,e in enumerate(cfg['legend']) if e.get('background_ambiguous')]
    if white_ids:
        # Only broad observed white patches inside the accepted domain become white candidates.
        # Opening breaks annotation-width white bridges; no neighbouring lithology is repainted.
        candidate=white&domain&(labels<0)
        opened=cv2.morphologyEx(candidate.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        count,cc,stats,_=cv2.connectedComponentsWithStats(opened,8)
        for i in range(1,count):
            x,y,w,h,area=map(int,stats[i]);local=cc[y:y+h,x:x+w]==i
            # 白色文字/钻孔残迹不能先被认作地层而获得“禁止回填”的保护。
            # 用补零局部距离检查真实宽核心，避免紧包围矩形的无背景距离溢出。
            radius=float(cv2.distanceTransform(np.pad(local.astype(np.uint8),1),cv2.DIST_L2,5).max())
            if area>=80 and radius>=4:labels[cc==i]=white_ids[0]
    # 顺序固定：清杂 -> 已有气球外轮廓 -> 颜色回填 -> 实例识别 -> 地层分组。
    before_refill=labels.copy()
    labels,refilled,refill_audit=refill_colours(labels,domain,cfg)
    from stratamatch.fault_lines import detect_annotated_fault_lines
    source_image=Path(config).resolve().parents[2]/cfg['image']
    fault_lines,fault_detection=detect_annotated_fault_lines(source_image,cfg['roi'],labels.shape)
    grouping_cfg={**cfg,'fragment_connection_radius':0,
        'annotated_fault_lines':[[p.tolist(),q.tolist()] for p,q in fault_lines],
        'annotated_fault_detection':fault_detection,
        'annotated_fault_image':str(source_image),
        'lithology_codes':[codes[e['lithology']] for e in cfg['legend']]}
    instances,records,audit=recognize_layer_groups(labels,domain,grouping_cfg)
    audit['colour_refill']=refill_audit
    audit['pipeline_order']=['import','colour_cleanup','balloon_domain','colour_refill','layer_instances','layer_groups']
    audit['outer_domain_source']=str(Path(balloon)/name/'07_总体轮廓掩膜.png')
    audit['classification_cache_reused']=cache_reused
    for filename,stage in [('01_清杂颜色.png',cleared_labels),('02_白色候选.png',before_refill),('03_颜色回填.png',labels)]:
        rendered=np.full((*labels.shape,3),255,np.uint8)
        for index,c in enumerate(colours):rendered[stage==index]=c
        Image.fromarray(rendered).save(folder/filename)
    Image.fromarray(refilled.astype(np.uint8)*255).save(folder/'回填位置.png')
    np.savez_compressed(folder/'回填前后标签.npz',before=before_refill,after=labels,domain=domain)
    lookup={p:g for g in audit['groups'] for p in g['parts']}
    base=np.full((*labels.shape,3),255,np.uint8)
    for i,c in enumerate(colours):base[labels==i]=c
    # Magnify for readable X.M labels and preserve the region geometry by nearest-neighbour.
    zoom=2;image=cv2.resize(base,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_NEAREST)
    contours,_=cv2.findContours(domain.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,[c*zoom for c in contours],-1,(30,70,180),3,cv2.LINE_AA)
    occupied=np.zeros(image.shape[:2],bool);rows=[]
    group_map=np.full(instances.shape,-1,np.int32)
    group_ids={g['label']:i for i,g in enumerate(audit['groups'])}
    for record in sorted(records,key=lambda r:-r['area_pixels']):
        mask=instances==record['instance'];ax,ay,_=_anchor(mask);ax*=zoom;ay*=zoom
        group=lookup[record['instance']];text=group['label'];scale=.85
        (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
        origin=(ax-tw//2,ay+th//2);chosen=None
        for radius in (0,25,45,70,100,140,190):
            for angle in np.linspace(0,2*np.pi,16,endpoint=False):
                p=(int(origin[0]+radius*np.cos(angle)),int(origin[1]+radius*np.sin(angle)))
                if _fits(_box(p,(tw,th)),image.shape[:2],occupied):chosen=p;break
            if chosen is not None:break
        if chosen is None:chosen=origin
        x0,y0,x1,y1=_box(chosen,(tw,th));occupied[max(0,y0):y1+1,max(0,x0):x1+1]=True
        if chosen!=origin:cv2.line(image,(ax,ay),(chosen[0]+tw//2,chosen[1]-th//2),(40,40,40),1,cv2.LINE_AA)
        cv2.putText(image,text,chosen,cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),6,cv2.LINE_AA)
        cv2.putText(image,text,chosen,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),2,cv2.LINE_AA)
        group_map[mask]=group_ids[text]
        rows.append({**record,'label':text,'anchor_work_pixels':[ax/zoom,ay/zoom],
                     'group_parts':group['parts'],'status':group['status']})
    Image.fromarray(image).save(folder/'地层分组编号.png')
    Image.fromarray(base).save(folder/'岩性分类.png')
    # 审图时单独显示被全局搜索采用的跨界连接，避免编号文字遮住错误路径。
    diagnostics=cv2.resize(base,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_NEAREST)
    selected=0
    for edge in audit['pairs']:
        if edge.get('decision')!='grouped' or len(edge.get('curve_work_pixels',[]))<2:continue
        points=np.rint(np.asarray(edge['curve_work_pixels'])*zoom).astype(np.int32)
        colour=(0,145,25) if edge.get('cut_part_ids') else (20,90,210)
        cv2.polylines(diagnostics,[points],False,(255,255,255),6,cv2.LINE_AA)
        cv2.polylines(diagnostics,[points],False,colour,3,cv2.LINE_AA)
        selected+=1
    cv2.rectangle(diagnostics,(12,12),(1000,93),(255,255,255),-1)
    cv2.putText(diagnostics,f'GLOBAL GROUP LINKS: {selected}',(28,48),
                cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(diagnostics,'GREEN: cut-band  BLUE: direct',(28,80),
                cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),2,cv2.LINE_AA)
    Image.fromarray(diagnostics).save(folder/'跨切断带连接诊断.png')
    # 红色 F 线只是断层候选；与岩性标签分开显示，便于复核误检。
    if audit.get('fault_annotation',{}).get('lines'):
        faults=cv2.resize(base,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_NEAREST)
        for line in audit['fault_annotation']['lines']:
            p,q=[tuple(np.rint(np.asarray(v)*zoom).astype(int)) for v in line['endpoints_work_pixels']]
            cv2.line(faults,p,q,(220,20,20),3,cv2.LINE_AA)
        Image.fromarray(faults).save(folder/'红色断层线候选.png')
    np.save(folder/'区域实例.npy',instances);np.save(folder/'地层组.npy',group_map)
    legend=Image.new('RGB',(1100,80+len(colours)*75),'white');draw=ImageDraw.Draw(legend)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27)
    draw.text((20,15),f'{name}：X 为岩性；M 仅在本剖面内编号',font=font,fill='black')
    for i,(entry,colour) in enumerate(zip(cfg['legend'],colours)):
        y=80+i*75;draw.rectangle((20,y,100,y+45),fill=tuple(map(int,colour)),outline='black')
        count=sum(g['lithology_index']==i for g in audit['groups'])
        draw.text((125,y),f"{codes[entry['lithology']]}   {entry.get('lithology','颜色类')}    {count} 组"+('（白色候选，需复核）' if entry.get('background_ambiguous') else ''),font=font,fill='black')
    legend.save(folder/'岩性编号对照.png')
    audit['parts']=rows;audit['legend']=cfg['legend'];audit['roi']=cfg['roi']
    audit['white_regions_require_review']=bool(white_ids)
    (folder/'分组依据.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    with (folder/'编号对照.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['label']);writer.writeheader();writer.writerows(rows)
    print(name,{k:audit[k] for k in ('part_count','group_count','multipart_group_count')},flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',nargs='+',required=True)
    parser.add_argument('--balloon',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    # 同一批图共用岩性 X 编码，M 在每张剖面内独立编号。
    codes={}
    for path in args.config:
        for entry in json.loads(Path(path).read_text(encoding='utf-8'))['legend']:
            codes.setdefault(entry['lithology'],len(codes)+1)
    for cfg in args.config:run(cfg,args.balloon,args.output,codes)


if __name__=='__main__':main()
