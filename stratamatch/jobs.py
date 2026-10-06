"""GUI-independent job service: validate first, snapshot inputs, never overwrite runs."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import hashlib
import shutil
import math
import uuid
import numpy as np
from PIL import Image
from .io import ROOT,local,write,validate_section
from .engine import METHODS,compare
from .recognition import recognize
from .export import export_report

class Cancelled(Exception):
    pass

def import_image(path):
    """Read external originals only; the project copy becomes the runtime input."""
    src=Path(path).resolve()
    if src.suffix.lower() not in ['.png','.jpg','.jpeg','.tif','.tiff']:raise ValueError('请选择 PNG/JPEG/TIFF 图片')
    if src.is_relative_to(ROOT):return str(src.relative_to(ROOT))
    dest=local('data/imported');dest.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    dest=dest/(digest+src.suffix.lower())
    if not dest.exists():shutil.copyfile(src,dest)
    return str(dest.relative_to(ROOT))

def validate_request(sections,methods,config,output):
    local(output)
    if len(sections)<2:raise ValueError('至少需要两张按空间顺序排列的剖面')
    if not methods or any(m not in METHODS for m in methods):raise ValueError('至少勾选一种有效方法')
    ids=[s.get('id') for s in sections]
    if any(not v for v in ids) or len(set(ids))!=len(ids):raise ValueError('剖面编号不能为空或重复')
    if len({(s.get('frame'),s.get('units')) for s in sections})!=1:raise ValueError('所有剖面必须使用相同坐标基准和单位')
    last=-math.inf
    for s in sections:
        station=float(s['station'])
        if not math.isfinite(station) or station<=last:raise ValueError('列表顺序对应的 station 必须严格递增；请编辑或调整顺序')
        last=station
        if not s.get('frame') or not s.get('units'):raise ValueError('坐标基准、单位不能为空')
        b=s['bounds']
        if len(b)!=4 or not all(math.isfinite(float(v)) for v in b) or b[2]<=b[0] or b[3]<=b[1]:raise ValueError('物理范围必须为有效的 xmin,ymin,xmax,ymax')
        if 'layers' in s:validate_section(s);continue
        if 'instance_mask' in s:
            if not local(s['instance_mask']).is_file():raise ValueError('实例掩膜不存在')
        else:
            if not local(s['image']).is_file():raise ValueError('图片不存在')
            if not s.get('legend'):raise ValueError(f"{s['id']} 尚未设置图例，请双击剖面并框选图例色块")
            if not any(e.get('verified') and e.get('lithology') not in ['',None,'UNKNOWN'] for e in s['legend']):
                raise ValueError(f"{s['id']} 没有人工确认的岩性图例；请双击剖面，逐行核对并确认后再匹配")
            if len(s.get('roi',[]))!=4 or min(s['roi'][2:])<8:raise ValueError('绘图区 ROI 无效')
    if not 1<=config['max_group']<=4:raise ValueError('最大组大小为 1–4')
    if not 100<=config.get('max_candidates',1200)<=5000 or not 1<=config.get('large_candidate_neighbors',2)<=8:
        raise ValueError('候选预算须为 100–5000；大实例近邻数须为 1–8')
    for key in ['max_accept_cost','min_global_margin']:
        if not math.isfinite(config[key]) or config[key]<=0:raise ValueError('匹配参数必须是正的有限数')
    if 'gnn' in methods and config.get('gnn_model') and not local(config['gnn_model']).is_file():raise ValueError('GNN 模型文件不存在')

def run_job(sections,methods,config,output,emit=lambda text:None,cancel=None):
    sections=deepcopy(sections);config=deepcopy(config)
    validate_request(sections,methods,config,output)
    def check():
        if cancel is not None and cancel.is_set():raise Cancelled('已取消；完成的阶段保存在任务目录')
    run=local(output)/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    run.mkdir(parents=True,exist_ok=False)
    write(run/'request.json',{'sections':sections,'methods':methods,'config':config})
    def log(text):
        with (run/'run.log').open('a',encoding='utf-8') as f:f.write(text+'\n')
        emit(text)
    try:
        log(f'本次任务目录：{run}')
        resolved=[];pages=[]
        for i,s in enumerate(sections):
            check();log(f'识别/检查剖面 {i+1}/{len(sections)}：{s["id"]}')
            if 'layers' in s:resolved.append(s);continue
            recognition_keys=['lab_tolerance','lab_margin','pale_lab_tolerance','min_area_pixels','min_component_fraction','max_recognition_pixels',
                'boundary_samples','contact_gap_pixels','artifact_preprocessing','artifact_max_gap_pixels','artifact_corridor_half_width','text_max_gap_pixels','artifact_fill_passes','bridge_unclassified_gap_pixels','internal_gap_max_fraction','expected_layer_tolerance_fraction']
            recognition_cfg={**s,**{key:config[key] for key in recognition_keys if key in config}}
            diagnostics={};result,labels=recognize(recognition_cfg,diagnostics);resolved.append(result)
            folder=run/'recognition';folder.mkdir(exist_ok=True)
            label_path=folder/f'{i:03d}_labels.npy';np.save(label_path,labels)
            result.setdefault('recognition',{})['label_map']=str(label_path.relative_to(ROOT))
            colors=np.random.default_rng(8).integers(40,240,(max(1,len(result['layers'])),3),dtype='uint8')
            rgb=np.full((*labels.shape,3),255,dtype='uint8');valid=labels>=0;rgb[valid]=colors[labels[valid]]
            Image.fromarray(rgb).save(folder/f'{i:03d}_labels.png')
            if diagnostics:
                artifact=np.zeros((*labels.shape,3),dtype='uint8')
                artifact[diagnostics['grid_mask']]=[40,170,255];artifact[diagnostics['text_mask']]=[255,150,40]
                Image.fromarray(artifact).save(folder/f'{i:03d}_artifacts.png')
                if 'uncertainty_map' in diagnostics:
                    uncertainty=np.zeros((*labels.shape,3),dtype='uint8')
                    uncertainty[diagnostics['uncertainty_map']==1]=[90,140,255]
                    uncertainty[diagnostics['uncertainty_map']==2]=[255,170,40]
                    uncertainty[diagnostics['uncertainty_map']==3]=[230,60,170]
                    Image.fromarray(uncertainty).save(folder/f'{i:03d}_uncertainty.png')
            log(f"  提取 {len(result['layers'])} 个地层；未确认信息会保留复核")
        write(run/'sections.json',{'sections':resolved});contexts={}
        for i,(a,b) in enumerate(zip(resolved,resolved[1:])):
            check();log(f'匹配 {i+1}/{len(resolved)-1}：{a["id"]} → {b["id"]}；方法 {", ".join(methods)}')
            report=compare(a,b,config,contexts,methods=methods)
            check();folder=run/f'pair_{i:03d}';export_report(report,a,b,folder)
            pages.append(str(folder/'review.html'));contexts={m:r['results'] for m,r in report['methods'].items()}
            log(f'  导出完成，方法间建议冲突 {len(report["conflicts"])} 项')
        result={'status':'complete','directory':str(run),'pages':pages,
            'reviews':[str(Path(page).with_name('review_manifest.json')) for page in pages]}
        write(run/'status.json',result);log('任务完成');return result
    except Exception as exc:
        write(run/'status.json',{'status':'cancelled' if isinstance(exc,Cancelled) else 'failed','message':str(exc)})
        log(str(exc));raise
