"""JSON, UTF-8 CSV and self-contained offline visual review page."""
import csv
import colorsys
import hashlib
import json
from pathlib import Path
import numpy as np
from .io import local,write

def export_report(report,sa,sb,out):
    out=local(out); out.mkdir(parents=True,exist_ok=True)
    write(out/'matches.json',report)
    columns=['source_section','source_layers','target_section','target_layers','relation_type','proposed_relation','confidence','method','status','reasons']
    with (out/'matches.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader()
        for run in report['methods'].values():
            for row in run['results']:
                w.writerow({k:json.dumps(row[k],ensure_ascii=False) if isinstance(row.get(k),list) else row.get(k,'') for k in columns})
    payload=json.dumps({'report':report,'a':sa,'b':sb},ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>剖面地层匹配 · 实验审查</title>
<style>body{font:15px system-ui;background:#101926;color:#dce8f5;margin:25px}h1{font-size:25px}select,button{padding:8px;background:#23354c;color:white;border:1px solid #567}svg{width:100%;background:#172337;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #33465e;text-align:left} .note{color:#b4c7dc}pre{white-space:pre-wrap} .warn{color:#ffd08a}</style>
<h1>剖面地层自动识别与匹配 · 实验审查</h1><p class="note">边界保留 64 个采样点。置信度为未校准证据分数，不代表地质正确概率。虚线表示需复核，点击表行查看特征。</p>
<label>算法 <select id="method"></select></label> <span id="mode"></span><p class="warn" id="conflicts"></p><svg id="plot" viewBox="0 0 1100 420"></svg>
<table><thead><tr><th>来源</th><th>目标</th><th>最终关系</th><th>建议关系</th><th>置信度</th><th>状态</th></tr></thead><tbody id="rows"></tbody></table><pre id="info"></pre>
<script>const D=PAYLOAD; const NS='http://www.w3.org/2000/svg'; const colors={continuous:'#5fdbab',split:'#79b8ff',merge:'#c8a0ff','pinch-out':'#ffbc65',uncertain:'#ff9090'};
function element(name,attrs,parent){let e=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.appendChild(e);return e}
function xy(s,p,side){let b=s.bounds;return [(side?650:40)+(p[0]-b[0])/(b[2]-b[0])*400,40+(p[1]-b[1])/(b[3]-b[1])*330]}
function render(){let run=D.report.methods[method.value];mode.textContent='模式：'+run.provenance.mode;let svg=document.getElementById('plot');svg.replaceChildren();
for(const[s,side]of [[D.a,0],[D.b,1]]){element('text',{x:side?650:40,y:25,fill:'white'},svg).textContent=s.id+' / '+s.units;for(const l of s.layers){let pts=l.outline.map(p=>xy(s,p,side).join(',')).join(' ');let poly=element('polygon',{points:pts,fill:l.lithology==='sandstone'?'#aa8748':l.lithology==='shale'?'#537b96':'#628971',stroke:'#dbe5ef','stroke-width':.6,opacity:.85},svg);element('title',{},poly).textContent=l.id+' '+l.lithology;let p=xy(s,l.centroid,side);element('text',{x:p[0],y:p[1],fill:'white','font-size':11},svg).textContent=l.id;}}
let body=document.getElementById('rows');body.replaceChildren();for(const r of run.results){let col=colors[r.relation_type]||'#ff9090';for(const si of r.source_layers){let a=D.a.layers.find(l=>l.id===si);let p=xy(D.a,a.centroid,0);if(!r.target_layers.length)element('circle',{cx:p[0],cy:p[1],r:7,fill:'none',stroke:col,'stroke-width':3},svg);for(const ti of r.target_layers){let b=D.b.layers.find(l=>l.id===ti),q=xy(D.b,b.centroid,1);element('line',{x1:p[0],y1:p[1],x2:q[0],y2:q[1],stroke:col,'stroke-width':2,'stroke-dasharray':r.status==='accepted'?'':'6 5'},svg);}}
let tr=document.createElement('tr');for(const v of [r.source_layers.join(' + ')||'∅',r.target_layers.join(' + ')||'∅',r.relation_type,r.proposed_relation,r.confidence.toFixed(3),r.status]){let td=document.createElement('td');td.textContent=v;tr.appendChild(td)}tr.onclick=()=>info.textContent=JSON.stringify(r,null,2);body.appendChild(tr)}info.textContent=JSON.stringify(run.provenance,null,2);}
let method=document.getElementById('method'),mode=document.getElementById('mode'),info=document.getElementById('info');for(const m of Object.keys(D.report.methods)){let o=document.createElement('option');o.textContent=m;method.appendChild(o)}method.onchange=render;document.getElementById('conflicts').textContent='算法建议冲突：'+D.report.conflicts.length+'（完整记录保存在 matches.json；不以多数票覆盖）';render();</script></html>'''
    (out/'review.html').write_text(page.replace('PAYLOAD',payload),encoding='utf-8')
    method='graph' if 'graph' in report['methods'] else next(iter(report['methods']))
    preview(report,sa,sb,out/f'matching_{method}.png',method)
    export_review_artifacts(report,sa,sb,out)

def _instance_palette(section):
    """Give each lithology a distant hue and each instance a small variation."""
    lithologies=sorted({layer['lithology'] for layer in section['layers']})
    # Deliberately separated categorical hues; instance variation stays within
    # ±8% brightness so same-lithology beds remain visually related.
    # Bit-reversal order spreads every newly assigned lithology between existing
    # hues; the first 32 lithologies therefore remain distinct without repeats.
    hues=[0.,.5,.25,.75,.125,.625,.375,.875,.0625,.5625,.3125,.8125,.1875,.6875,.4375,.9375,
        .03125,.53125,.28125,.78125,.15625,.65625,.40625,.90625,.09375,.59375,.34375,.84375,.21875,.71875,.46875,.96875]
    bases={name:hues[i%len(hues)] for i,name in enumerate(lithologies)}
    palette={}
    for layer in section['layers']:
        digest=hashlib.sha256(layer['id'].encode('utf-8')).digest()
        variation=(digest[0]/255-.5)*.16
        saturation=.72+(digest[1]/255-.5)*.08;value=.83+variation
        palette[layer['id']]=tuple(round(v*255) for v in colorsys.hsv_to_rgb(bases[layer['lithology']],saturation,max(.62,min(.95,value))))
    return palette

def _target_palette(sa,sb,rows,source_colors):
    target={};mapping={}
    for row in rows:
        sources=row.get('source_layers',[]);targets=row.get('target_layers',[])
        if not targets:continue
        colors=[source_colors[s] for s in sources if s in source_colors]
        for target_id in targets:
            if colors:target[target_id]=colors
            mapping[target_id]=sources
    for layer in sb['layers']:
        shade=125+hashlib.sha256(layer['id'].encode('utf-8')).digest()[0]%45
        target.setdefault(layer['id'],[(shade,shade,shade+4)])
    return target,mapping

def _render_section(section,color_map,path):
    """Recolour recognized pixels over the source image; fall back to geometry."""
    from PIL import Image,ImageDraw
    label_path=section.get('recognition',{}).get('label_map')
    mapping={int(k):v for k,v in section.get('recognition',{}).get('label_id_mapping',{}).items()}
    if label_path and local(label_path).is_file() and section.get('image'):
        labels=np.load(local(label_path),allow_pickle=False)
        with Image.open(local(section['image'])) as image:
            x,y,w,h=section['roi'];base=image.convert('RGB').crop((x,y,x+w,y+h)).resize((labels.shape[1],labels.shape[0]),Image.Resampling.LANCZOS)
        array=np.asarray(base).copy();overlay=array.copy();valid=np.zeros(labels.shape,bool)
        yy,xx=np.indices(labels.shape)
        for index,layer_id in mapping.items():
            mask=labels==index
            colors=color_map.get(layer_id,[(145,145,150)])
            colors=colors if isinstance(colors,list) else [colors]
            if len(colors)==1:overlay[mask]=colors[0]
            else:
                stripe=((xx+yy)//8)%len(colors)
                for j,color in enumerate(colors):overlay[mask&(stripe==j)]=color
            valid|=mask
        array[valid]=(array[valid]*.22+overlay[valid]*.78).astype(np.uint8)
        boundary=np.zeros(labels.shape,bool)
        boundary[1:]|=(labels[1:]!=labels[:-1])&(labels[1:]>=0)
        boundary[:,1:]|=(labels[:,1:]!=labels[:,:-1])&(labels[:,1:]>=0)
        array[boundary]=(25,25,28);result=Image.fromarray(array)
    else:
        result=Image.new('RGB',(1400,850),'white');draw=ImageDraw.Draw(result)
        x0,y0,x1,y1=section['bounds'];sx=1320/max(x1-x0,1e-9);sy=770/max(y1-y0,1e-9)
        for layer in section['layers']:
            colors=color_map.get(layer['id'],[(145,145,150)]);color=colors[0] if isinstance(colors,list) else colors
            points=[(40+(p[0]-x0)*sx,40+(p[1]-y0)*sy) for p in layer['outline']]
            if len(points)>=3:draw.polygon(points,fill=color,outline=(25,25,28))
    scale=min(1.,1800/result.width,1100/result.height)
    if scale<1:result=result.resize((round(result.width*scale),round(result.height*scale)),Image.Resampling.LANCZOS)
    result.save(local(path))

def export_review_artifacts(report,sa,sb,out):
    """Create human-review visuals, while keeping modeling input separate."""
    out=local(out);source_colors=_instance_palette(sa);overlays={}
    source_path=out/'review_source.png';_render_section(sa,{k:[v] for k,v in source_colors.items()},source_path)
    for method,run in report['methods'].items():
        target_colors,_=_target_palette(sa,sb,run['results'],source_colors)
        target_path=out/f'review_{method}_target.png';_render_section(sb,target_colors,target_path)
        overlays[method]={'source':source_path.name,'target':target_path.name}
    manifest={'schema':'stratamatch-review-v1','source_section':sa['id'],'target_section':sb['id'],
        'matches_file':'matches.json','sections_file':'../sections.json','overlays':overlays,
        'colors':{layer_id:'#%02x%02x%02x'%color for layer_id,color in source_colors.items()},
        'instructions':'同岩性使用同色系；目标对应层复用基准层颜色；条纹表示 merge；灰色表示目标侧未对应。人工确认另存，不直接改写算法结果。'}
    write(out/'review_manifest.json',manifest)
    write(out/'modeling_input_schema.json',{
        'schema':'stratamatch-modeling-v1','purpose':'人工确认后供三维矿区建模读取',
        'required':['coordinate_frame','units','sections','correspondences','review'],
        'correspondence':{'source_section':'string','source_layers':['id'],'target_section':'string','target_layers':['id'],
            'relation_type':'continuous|split|merge','status':'human_confirmed'},
        'rule':'只有 human_confirmed 且两侧均有地层的关系可进入建模；uncertain/manual_review/rejected 不得自动导出。'})

def preview(report,sa,sb,path,method='graph'):
    from PIL import Image,ImageDraw
    im=Image.new('RGB',(1100,470),'#101926');draw=ImageDraw.Draw(im)
    def point(s,p,side):
        b=s['bounds'];return ((650 if side else 40)+(p[0]-b[0])/(b[2]-b[0])*400,55+(p[1]-b[1])/(b[3]-b[1])*330)
    draw.text((35,12),method.upper()+' / Green: continuous / Blue: split / Purple: merge / Red: review / Orange: pinch-out',fill='white')
    colors={'shale':'#537b96','sandstone':'#aa8748','limestone':'#628971'}
    for s,side in [(sa,0),(sb,1)]:
        draw.text((650 if side else 40,35),s['id'],fill='white')
        for l in s['layers']:
            draw.polygon([point(s,p,side) for p in l['outline']],fill=colors.get(l['lithology'],'#73777d'),outline='#dce8f5')
            draw.text(point(s,l['centroid'],side),l['id'],fill='white')
    for r in report['methods'][method]['results']:
        color={'continuous':'#5fdbab','split':'#79b8ff','merge':'#c8a0ff','pinch-out':'#ffbc65'}.get(r['relation_type'],'#ff9090')
        for sid in r['source_layers']:
            p=point(sa,next(a for a in sa['layers'] if a['id']==sid)['centroid'],0)
            if not r['target_layers']:draw.ellipse([p[0]-7,p[1]-7,p[0]+7,p[1]+7],outline=color,width=3)
            for tid in r['target_layers']:
                q=point(sb,next(b for b in sb['layers'] if b['id']==tid)['centroid'],1)
                draw.line([p,q],fill=color,width=2)
    draw.text((35,425),f'Experimental output. Evidence scores are uncalibrated. Open review.html for {len(report["methods"])} selected methods.',fill='#b4c7dc')
    im.save(local(path))

def metric_csv(metrics,path):
    p=local(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['method',*next(iter(metrics['summary'].values())).keys()]);w.writeheader()
        for m,r in metrics['summary'].items():w.writerow({'method':m,**r})
