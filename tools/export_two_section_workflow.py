"""Export the current two-section recognition and conservative match review.

All recognition comes from ``layer_partition``.  The matching review uses only
relationships proposed identically by DTW, graph and Markov; registration is
still unverified, so these are visual review suggestions rather than accepted
geological truth.
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.engine import compare
from stratamatch.geometry import extract,topology
from stratamatch.io import ROOT,local
from stratamatch.layer_partition import partition_layers
from stratamatch.legend_features import analyze_legend


def _font(size):
    for path in (Path(r'C:\Windows\Fonts\msyh.ttc'),Path(r'C:\Windows\Fonts\simhei.ttf')):
        if path.is_file():return ImageFont.truetype(str(path),size)
    return ImageFont.load_default()


def _write_image(path,array):
    path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(array).save(path)


def _numbered(image,instances,records):
    """Number recognized layers without drawing black geological interfaces."""
    result=image.copy();items=[]
    for record in records:
        mask=instances==record['instance'];yy,xx=np.where(mask)
        if not len(xx):continue
        distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
        ay,ax=np.unravel_index(int(np.argmax(distance)),distance.shape)
        items.append((float(yy.mean()),float(xx.mean()),record,int(ax),int(ay),mask))
    items.sort(key=lambda item:(item[0],item[1]));font=cv2.FONT_HERSHEY_SIMPLEX
    scale=max(.48,min(.68,min(result.shape[:2])/3800.));thickness=1
    number_map=np.zeros(instances.shape,np.int32);rows=[]
    for number,(_,_,record,ax,ay,mask) in enumerate(items,1):
        number_map[mask]=number;text=str(number)
        (tw,th),_=cv2.getTextSize(text,font,scale,thickness)
        origin=(max(1,ax-tw//2),max(th+1,ay+th//2))
        cv2.putText(result,text,origin,font,scale,(255,255,255),thickness+3,cv2.LINE_AA)
        cv2.putText(result,text,origin,font,scale,(0,0,0),thickness,cv2.LINE_AA)
        rows.append({'number':number,'instance':record['instance'],'lithology_index':record['lithology_index'],
                     'area_pixels':record['area_pixels'],'bbox':record['bbox'],'anchor':[ax,ay]})
    return result,number_map,rows


def _section_from_partition(cfg,result,section_id,station):
    """Build the shared geological feature schema from final instance pixels."""
    entries=[]
    for record in result['records']:
        mask=result['instances']==record['instance'];yy,xx=np.where(mask)
        if len(xx):entries.append((float(yy.mean()),float(xx.mean()),record,mask))
    entries.sort(key=lambda item:(item[0],item[1]));labels=np.full(result['instances'].shape,-1,np.int32)
    layers=[];base_colours={};display_names={}
    for index,(_,_,record,mask) in enumerate(entries):
        legend=cfg['legend'][record['lithology_index']];rgb=tuple(int(v) for v in legend['rgb'])
        lithology='RGB_%03d_%03d_%03d'%rgb;base_colours[lithology]=rgb
        display_names[lithology]=legend.get('lithology') or lithology
        layer=extract(mask,f'{section_id}_L{index+1}',lithology,cfg['bounds'],.65,cfg.get('boundary_samples',64))
        layer['legend_verified']=bool(legend.get('verified',False));layer['display_lithology']=display_names[lithology]
        yy,xx=np.where(mask);layer['_pixel_bbox']=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]
        layer['lithology_rgb']=list(rgb);layers.append(layer);labels[mask]=index
    topology(layers,labels,max(1,int(cfg.get('contact_gap_pixels',3))))
    section={'id':section_id,'frame':cfg.get('frame','project_depth_datum'),'units':cfg.get('units','m'),
             'station':float(station),'bounds':cfg['bounds'],'layers':layers,'registration_verified':False,
             'complete':False,'complex_structure':bool(np.any(result['fault_mask'])),
             'recognition':{'mode':'final_colour_fault_partition','status':'manual_review',
                            'label_id_mapping':{str(i):layer['id'] for i,layer in enumerate(layers)}}}
    return section,labels,base_colours


class _Union:
    def __init__(self,items):self.parent={item:item for item in items}
    def find(self,item):
        while self.parent[item]!=item:
            self.parent[item]=self.parent[self.parent[item]];item=self.parent[item]
        return item
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a!=b:self.parent[b]=a


def _consensus(report,sa,sb):
    """Return only exact three-method proposals; every relation remains review-only."""
    methods=list(report['methods']);by_method={}
    for method in methods:
        lookup={}
        for row in report['methods'][method]['results']:
            for source in row.get('source_layers',[]):
                lookup[source]=(tuple(sorted(row.get('target_layers',[]))),row.get('proposed_relation'))
        by_method[method]=lookup
    nodes=[('A',layer['id']) for layer in sa['layers']]+[('B',layer['id']) for layer in sb['layers']]
    union=_Union(nodes);agreed=[];conflicted=[]
    for layer in sa['layers']:
        proposals=[by_method[m].get(layer['id'],((), 'missing')) for m in methods]
        if len(set(proposals))==1 and proposals[0][0] and proposals[0][1] in {'continuous','split','merge'}:
            targets=list(proposals[0][0]);agreed.append({'source':layer['id'],'targets':targets,
                'relation':proposals[0][1],'status':'manual_review_registration_unverified'})
            for target in targets:union.union(('A',layer['id']),('B',target))
        elif len(set(proposals))>1:
            conflicted.append({'source':layer['id'],'proposals':{m:{'targets':list(p[0]),'relation':p[1]}
                               for m,p in zip(methods,proposals)}})
    return union,agreed,conflicted


def _shade(rgb,ordinal):
    """Keep the legend hue while making same-lithology instances distinguishable."""
    r,g,b=[v/255 for v in rgb];h,l,s=colorsys.rgb_to_hls(r,g,b)
    light_offsets=[0,-.18,.16,-.09,.09,-.24,.22,-.14,.13,-.05,.05,-.21,.19,-.11,.11,-.03,.03]
    saturation_scale=[1.,.82,1.16,.7,1.28]
    dl=light_offsets[ordinal%len(light_offsets)];scale=saturation_scale[(ordinal//len(light_offsets))%len(saturation_scale)]
    # Very pale/white lithologies need darker neutral shades; hue is unchanged.
    l=max(.32,min(.96,l+dl));s=max(0.,min(1.,s*scale))
    return tuple(int(round(v*255)) for v in colorsys.hls_to_rgb(h,l,s))


def _palettes(sa,sb,union,bases):
    layers={'A':{l['id']:l for l in sa['layers']},'B':{l['id']:l for l in sb['layers']}}
    groups={}
    for side,items in layers.items():
        for layer_id,layer in items.items():
            root=union.find((side,layer_id));groups.setdefault((layer['lithology'],root),[]).append((side,layer_id))
    palette={'A':{},'B':{}}
    by_lith={}
    for lith,root in groups:by_lith.setdefault(lith,[]).append(root)
    for lith,roots in by_lith.items():
        roots=sorted(set(roots),key=str)
        for ordinal,root in enumerate(roots):
            colour=_shade(bases[lith],ordinal)
            for side,layer_id in groups[(lith,root)]:palette[side][layer_id]=colour
    return palette


def _render_match_section(section,labels,palette,fault_mask):
    image=np.full((*labels.shape,3),255,np.uint8)
    mapping={int(k):v for k,v in section['recognition']['label_id_mapping'].items()}
    for index,layer_id in mapping.items():image[labels==index]=palette[layer_id]
    image[fault_mask]=[237,23,23]
    numbered,_,_=_numbered(image,labels,[{'instance':i,'lithology_index':0,'area_pixels':int(np.sum(labels==i)),
        'bbox':[0,0,0,0]} for i in range(len(mapping))])
    return numbered


def _compose(left,right,path,agreed_count,conflict_count):
    target_h=1050
    def resize(array):
        scale=target_h/array.shape[0];return Image.fromarray(array).resize((round(array.shape[1]*scale),target_h),Image.Resampling.LANCZOS)
    a,b=resize(left),resize(right);gap=70;top=95
    canvas=Image.new('RGB',(a.width+b.width+gap,top+target_h+60),'white');canvas.paste(a,(0,top));canvas.paste(b,(a.width+gap,top))
    draw=ImageDraw.Draw(canvas);title=_font(28);note=_font(20)
    draw.text((20,12),'P11（基准剖面）',fill=(20,20,20),font=title)
    draw.text((a.width+gap+20,12),'P101（目标剖面）',fill=(20,20,20),font=title)
    draw.text((20,52),f'完全同色 = DTW、图匹配、Markov三种方法一致建议；仍需人工复核。 一致 {agreed_count}，冲突 {conflict_count}',
              fill=(90,45,20),font=note)
    canvas.save(path)


def main():
    root=ROOT;source_run=root/'输出数据'/'实验'/'run_20260924_145419_f74470'/'request.json'
    request=json.loads(source_run.read_text(encoding='utf-8-sig'));configs=request['sections']
    output=root/'输出数据'/'最终流程_P11_P101';(output/'识别').mkdir(parents=True,exist_ok=True);(output/'配置').mkdir(exist_ok=True)
    prepared=[]
    for cfg,name in zip(configs,['P11','P101']):
        cfg=dict(cfg);cfg['boundary_smoothing_strength']=2.5;cfg['boundary_smoothing_passes']=8
        if not cfg.get('legend_catalog'):
            cfg['legend_catalog']=analyze_legend(cfg['image'],cfg.get('legend',[]),cfg.get('page',0))
        config_path=output/'配置'/f'{name}_最终配置.json';config_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        result=partition_layers(cfg);prepared.append((cfg,result))
        overall=result['plain_image'].copy();contours,_=cv2.findContours(result['domain'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overall,contours,-1,(0,0,0),5,cv2.LINE_AA)
        _write_image(output/'识别'/f'{name}_01_总体轮廓.png',overall)
        numbered,number_map,rows=_numbered(result['plain_image'],result['instances'],result['records'])
        _write_image(output/'识别'/f'{name}_02_地层编号.png',numbered);np.save(output/'识别'/f'{name}_编号矩阵.npy',number_map)
        (output/'识别'/f'{name}_识别审计.json').write_text(json.dumps({'layer_count':len(rows),'rows':rows,'audit':result['audit']},ensure_ascii=False,indent=2),encoding='utf-8')
    (cfg_a,res_a),(cfg_b,res_b)=prepared
    sa,labels_a,bases_a=_section_from_partition(cfg_a,res_a,'P11',0.)
    sb,labels_b,bases_b=_section_from_partition(cfg_b,res_b,'P101',1.)
    match_cfg=dict(request.get('config',{}));match_cfg.update(max_candidates=2500,large_candidate_neighbors=3)
    report=compare(sa,sb,match_cfg,methods=['dtw','graph','markov'])
    union,agreed,conflicted=_consensus(report,sa,sb);bases={**bases_a,**bases_b}
    palettes=_palettes(sa,sb,union,bases)
    match_a=_render_match_section(sa,labels_a,palettes['A'],res_a['fault_mask'])
    match_b=_render_match_section(sb,labels_b,palettes['B'],res_b['fault_mask'])
    match_dir=output/'匹配';match_dir.mkdir(exist_ok=True)
    _write_image(match_dir/'P11_匹配配色.png',match_a);_write_image(match_dir/'P101_匹配配色.png',match_b)
    _compose(match_a,match_b,match_dir/'P11_P101_地层匹配对照.png',len(agreed),len(conflicted))
    (match_dir/'匹配结果与配色.json').write_text(json.dumps({'warning':'自动建议，空间配准未验证，所有关系需人工复核',
        'methods':['dtw','graph','markov'],'exact_consensus':agreed,'conflicts':conflicted,
        'palette':{side:{key:list(value) for key,value in values.items()} for side,values in palettes.items()},
        'report':report},ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'output':str(output),'P11_layers':len(sa['layers']),'P101_layers':len(sb['layers']),
             'exact_consensus_suggestions':len(agreed),'conflicting_sources':len(conflicted)}
    (output/'流程摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
