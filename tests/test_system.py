"""Safety invariants and independent numeric/solver checks, beyond demo assertions."""
from copy import deepcopy
import json
import numpy as np
import pytest
from PIL import Image
from stratamatch.io import ROOT,SOFTWARE_ROOT,local,validate_section,read
from stratamatch.synthetic import make_case,CASES,render_case
from stratamatch.geometry import resample,extract
from stratamatch.candidates import build
from stratamatch.engine import match,compare,METHODS
from stratamatch.solvers import sequence,graph,GAP
from stratamatch.gnn import gradients,forward,tensors,predict
from stratamatch.jev import parse,decide,payload
from stratamatch import laya
from stratamatch.decisions import canonical
from stratamatch.recognition import recognize
from stratamatch.export import _instance_palette,_target_palette
from stratamatch.artifacts import detect_masks,restore,fill_enclosed_gaps

@pytest.mark.parametrize('method',METHODS)
def test_lithology_is_absolute_gate(method):
    c,_=make_case('lithology_mismatch');r=match(c['source'],c['target'],method)
    assert not r['candidates'];assert all(x['status']!='accepted' for x in r['results'])
    assert r['rejected'][0]['reason']=='lithology_hard_gate'

@pytest.mark.parametrize('kind',['split','merge'])
@pytest.mark.parametrize('method',['dtw','graph','markov'])
def test_group_truth(kind,method):
    c,_=make_case(kind);run=match(c['source'],c['target'],method)
    rows=[r for r in run['results'] if r['relation_type']==kind]
    assert len(rows)==1 and rows[0]['status']=='accepted'
    assert len(rows[0]['target_layers'] if kind=='split' else rows[0]['source_layers'])==2

@pytest.mark.parametrize('kind',['ambiguous','neighbor_change','local_anomaly','unverified_registration','near_different_shape','similar_wrong_position'])
def test_required_abstentions(kind):
    c,_=make_case(kind)
    for m in METHODS:
        r=match(c['source'],c['target'],m)
        for a in c['must_abstain']:
            assert all(x['status']!='accepted' for x in r['results'] if a in x['source_layers'])

def test_pinch_requires_independent_evidence():
    c,_=make_case('pinch_out');r=match(c['source'],c['target'])
    assert any(x['relation_type']=='pinch-out' for x in r['results'])
    c['target']['complete']=False
    assert not any(x['relation_type']=='pinch-out' for x in match(c['source'],c['target'])['results'])

def test_unknown_never_matches():
    c,_=make_case('continuous')
    for s in ['source','target']:
        for a in c[s]['layers']:a['lithology']='UNKNOWN'
    assert not build(c['source'],c['target'])[0]

def test_geometry_is_soft_evidence_not_a_candidate_gate():
    c,_=make_case('continuous');a=c['source']['layers'][1];b=c['target']['layers'][1]
    b['centroid']=[b['centroid'][0]+10000,b['centroid'][1]+10000]
    edges,rejected=build(c['source'],c['target'])
    assert any(a['id'] in [c['source']['layers'][i]['id'] for i in e['s']] and b['id'] in [c['target']['layers'][i]['id'] for i in e['t']] for e in edges)
    assert not any(r.get('reason')=='geometry_gate' for r in rejected)

def test_artifact_healing_preserves_disagreeing_geological_boundary():
    rgb=np.full((90,180,3),(236,193,96),np.uint8)
    rgb[20,:]=(175,175,175);rgb[45,:]=(175,175,175);rgb[:,70]=(175,175,175);rgb[42:49,95:125]=(20,20,20)
    classified=np.zeros((90,180),np.int16);classified[46:]=1
    # Simulate coloured grid/annotation pixels that were initially classified
    # as geology; recognition must revoke them before conservative restoration.
    classified[45,:]=0;classified[42:49,95:125]=0
    classified[20,:]=-1;classified[:,70]=-1
    grid,text=detect_masks(rgb,[[93,40,35,12]])
    artifact=grid|text;classified[artifact]=-1
    healed,stats=restore(classified,artifact,max_gap=8,passes=3)
    assert stats['recovered_pixels']>0 and healed[20,20]==0 and healed[20,120]==0
    # The vertical grid is healed within either layer, while the true contact
    # has different classes above/below and remains unresolved.
    assert healed[30,70]==0 and healed[65,70]==1
    assert healed[45,10]==-1

def test_diagonal_artifact_path_is_repaired_without_crossing_contact():
    labels=np.zeros((90,90),np.int16);labels[55:]=1
    artifact=np.zeros(labels.shape,bool)
    for offset in range(-2,3):
        y=np.arange(10,80);x=y+offset;valid=(x>=0)&(x<90)
        artifact[y[valid],x[valid]]=True
    labels[artifact]=-1
    healed,audit=restore(labels,artifact,max_gap=8,passes=3)
    assert audit['recovered_pixels']>0
    assert healed[25,25]==0 and healed[70,70]==1
    # The crossing is reconstructed on each side without moving the real
    # horizontal geological contact.
    assert np.all(healed[54,53:58]==0) and np.all(healed[55,53:58]==1)

def test_enclosed_gaps_fill_but_exterior_background_stays_empty():
    labels=np.full((60,80),-1,np.int16);content=np.zeros(labels.shape,bool);content[5:55,5:75]=True
    labels[8:52,8:72]=0;labels[20:40,38:72]=1
    labels[24:34,30:48]=-1
    filled,audit=fill_enclosed_gaps(labels,content,.1)
    assert audit['filled_pixels']>0 and np.all(filled[24:34,30:48]>=0)
    assert np.all(filled[:5]<0) and set(np.unique(filled[24:34,30:48]))<={0,1}

def test_georeferencing_and_input_integrity():
    c,_=make_case('continuous');a,b=c['source'],c['target']
    b['frame']='other'
    with pytest.raises(ValueError):match(a,b)
    b['frame']=a['frame'];b['station']=-1
    with pytest.raises(ValueError):match(a,b)
    a['layers'][0]['area']=float('nan')
    with pytest.raises(ValueError):validate_section(a)

def test_path_escape_is_rejected():
    with pytest.raises(ValueError):local('../outside.json')
    assert local('outputs/test.json').is_relative_to(ROOT)

def test_three_folder_layout_and_legacy_absolute_path_translation():
    assert {p.name for p in ROOT.iterdir()}=={'软件','输入数据','输出数据'}
    old=ROOT/'data'/'实验'/'11-P11.JPG'
    assert local(old)==ROOT/'输入数据'/'实验'/'11-P11.JPG'
    assert SOFTWARE_ROOT==ROOT/'软件'

def test_resampling_and_anisotropic_units():
    p=resample([[0,0],[0,0],[3,0],[3,4]],8)
    assert np.allclose(p[0],[0,0]) and np.allclose(p[-1],[3,4])
    mask=np.zeros((20,20),bool);mask[5:10,2:18]=True
    a=extract(mask,'a','rock',[0,0,40,60])
    assert a['area']==480 and a['mean_thickness']==15
    assert len(a['outline'])==64 and abs(a['dip'])<1e-8

def small_edges():
    return [{'id':0,'s':(0,),'t':(0,),'cost':.2,'features':[0.]*11},
            {'id':1,'s':(1,),'t':(1,),'cost':.3,'features':[0.]*11}]

def test_dp_and_milp_independent_known_optimum():
    e=small_edges()
    for solver in [lambda:sequence(e,2,2),lambda:graph(e,2,2)]:
        ids,cost,_=solver();assert set(ids)=={0,1};assert cost==pytest.approx(.5)
    assert graph(e,2,2,forbid=0)[1]==pytest.approx(2*GAP+.3)

def test_markov_partition_by_explicit_path_enumeration():
    e=small_edges();_,_,post=sequence(e,2,2,'markov');paths=[]
    def visit(i,j,c,ids):
        if (i,j)==(2,2):paths.append((np.exp(-c/.22),ids));return
        if i<2:visit(i+1,j,c+GAP,ids)
        if j<2:visit(i,j+1,c+GAP,ids)
        for edge in e:
            if edge['s'][0]==i and edge['t'][0]==j:visit(i+1,j+1,c+edge['cost'],ids+[edge['id']])
    visit(0,0,0,[]);z=sum(w for w,_ in paths)
    for eid in post:assert post[eid]==pytest.approx(sum(w for w,ids in paths if eid in ids)/z)

def test_crossing_is_not_accepted_by_graph():
    e=small_edges();e[0]['t']=(1,);e[1]['t']=(0,)
    ids,_,_=graph(e,2,2);assert len(ids)<=1

def test_gcn_backprop_finite_difference():
    rng=np.random.default_rng(5);x=rng.normal(size=(3,14));a=np.array([[.5,.5,0],[.5,0,.5],[0,.5,.5]])
    w1=rng.normal(0,.1,(14,16));w2=rng.normal(0,.1,30);y=np.array([1,0,1])
    _,d1,d2=gradients(x,a,y,w1,w2)
    for w,index,expected in [(w1,(3,7),d1[3,7]),(w2,(4,),d2[4]),(w2,(22,),d2[22])]:
        old=w[index];eps=1e-5;w[index]=old+eps;lp=gradients(x,a,y,w1,w2)[0]
        w[index]=old-eps;lm=gradients(x,a,y,w1,w2)[0];w[index]=old
        assert (lp-lm)/(2*eps)==pytest.approx(expected,abs=1e-6)

def test_jev_schema_and_mock_never_use_transport():
    e=small_edges();assert parse({'answers':{'edge_0':.2,'edge_1':{'probability':.9}}},e)==[.2,.9]
    for bad in [{},{'answers':{'edge_0':True,'edge_1':.4}},{'answers':{'edge_0':2,'edge_1':.4}}]:
        with pytest.raises(ValueError):parse(bad,e)
    c,_=make_case('continuous')
    p,meta=decide(c['source'],c['target'],e,{'mode':'mock'},transport=lambda:pytest.fail('Network called'))
    assert p is None and meta['mode']=='mock'

def test_jev_live_contract_and_failure(monkeypatch):
    monkeypatch.setenv('TEST_JEV_KEY','test-only')
    c,_=make_case('continuous');e=small_edges();cfg={'mode':'live','key_env':'TEST_JEV_KEY'}
    p,m=decide(c['source'],c['target'],e,cfg,transport=lambda:{'answers':{'edge_0':.9,'edge_1':.8}})
    assert p==[.9,.8] and m['mode']=='live'
    p,m=decide(c['source'],c['target'],e,cfg,transport=lambda:{'invalid':True})
    assert p is None and m['mode']=='fallback'

def test_laya_and_jev_share_identical_decision_content(monkeypatch):
    c,_=make_case('continuous');e,_=build(c['source'],c['target'])
    jp=payload(c['source'],c['target'],e,'jev-latest')
    from stratamatch.decisions import payload as typed_payload
    lp=typed_payload(c['source'],c['target'],e,'typed-decisions')
    assert jp['state']==lp['state'] and jp['questions']==lp['questions']
    data={'model':'laya-test','answers':{f"edge_{x['id']}":{'noul':.8} for x in e}}
    p,meta=laya.decide(c['source'],c['target'],e,{'mode':'http','endpoint':'http://127.0.0.1:8791/v1/systemone'},transport=lambda:data)
    assert p==[.8]*len(e) and meta['input_contract']=='shared_with_jev'

def test_laya_fails_closed_and_rejects_insecure_remote_http():
    c,_=make_case('continuous');e,_=build(c['source'],c['target'])
    p,meta=laya.decide(c['source'],c['target'],e,{'mode':'http','endpoint':'http://example.com/v1/systemone'})
    assert p is None and meta['mode']=='fallback'

@pytest.mark.parametrize('suffix',['png','jpg','tiff'])
def test_image_formats_legend_and_boundary_extraction(tmp_path,suffix):
    c,spec=make_case('continuous');render_case(c,spec,tmp_path)
    cfg=read(tmp_path/'manifest.json')['sections'][0]
    im=Image.open(local(cfg['image']));p=tmp_path/('format.'+suffix);im.save(p);cfg['image']=str(p)
    s,labels=recognize(cfg)
    assert {x['lithology'] for x in s['layers']}=={'shale','sandstone','limestone'}
    assert len(s['layers'])==3 and len(s['layers'][0]['outline'])==64
    assert (labels[:5]<0).all()
    for _,lith,expected,_ in spec[0]:
        layer=next(l for l in s['layers'] if l['lithology']==lith)
        index=next(int(i) for i,lid in s['recognition']['label_id_mapping'].items() if lid==layer['id'])
        got=labels==index
        iou=np.count_nonzero(got&expected)/np.count_nonzero(got|expected)
        if suffix=='jpg':
            # Lossy input must preserve label precision and expose incomplete borders.
            assert iou>.85 and np.count_nonzero(got&expected)/np.count_nonzero(got)>.99
            assert layer['quality']<=.75 and s['recognition']['lossy_boundary_review']
        else: assert iou>.99

def test_identical_color_legend_does_not_guess(tmp_path):
    c,spec=make_case('continuous');render_case(c,spec,tmp_path);cfg=read(tmp_path/'manifest.json')['sections'][0]
    cfg['legend']=[{'rgb':[236,193,96],'lithology':'sandstone','verified':True},{'rgb':[236,193,96],'lithology':'siltstone','verified':True}]
    s,_=recognize(cfg);assert not s['layers']

def test_missing_legend_is_an_error(tmp_path):
    c,spec=make_case('continuous');render_case(c,spec,tmp_path);cfg=read(tmp_path/'manifest.json')['sections'][0];cfg.pop('legend')
    with pytest.raises(ValueError):recognize(cfg)

def test_explicit_human_count_estimate_is_audit_only(tmp_path):
    c,spec=make_case('continuous');render_case(c,spec,tmp_path);cfg=read(tmp_path/'manifest.json')['sections'][0]
    # Add a small isolated same-colour fragment. A rough count must report the
    # discrepancy without deleting the fragment or changing the area floor.
    with Image.open(local(cfg['image'])) as source:
        arr=np.asarray(source.convert('RGB')).copy()
    arr[104:108,130:134]=[236,193,96];p=tmp_path/'count-prior.png';Image.fromarray(arr).save(p);cfg['image']=str(p)
    cfg['expected_layer_count']=3;cfg['min_area_pixels']=1;cfg['min_component_fraction']=0
    section,_=recognize(cfg)
    assert len(section['layers'])==4
    audit=section['recognition']['count_calibration']
    assert audit['source']=='rough_human_count_estimate' and audit['requested']==3
    assert audit['actual']==4 and not audit['changed_segmentation']
    assert section['recognition']['status']=='manual_review_count_estimate'

def test_instance_mask_import(tmp_path):
    p=tmp_path/'instances.png';arr=np.zeros((40,40),np.uint16);arr[2:18,2:38]=1;arr[20:38,2:38]=2;Image.fromarray(arr).save(p)
    cfg={'id':'A','frame':'x','units':'m','station':0,'bounds':[0,0,40,40],'instance_mask':str(p),
        'instances':[{'value':1,'id':'a','lithology':'shale','verified':True},{'value':2,'id':'b','lithology':'shale','verified':True}]}
    s,_=recognize(cfg);assert len(s['layers'])==2 and all(l['lithology']=='shale' for l in s['layers'])

def test_image_to_comparison_end_to_end(tmp_path):
    c,spec=make_case('split');render_case(c,spec,tmp_path);cfg=read(tmp_path/'manifest.json')['sections']
    a,_=recognize(cfg[0]);b,_=recognize(cfg[1]);r=compare(a,b)
    assert any(row['relation_type']=='split' for row in r['methods']['graph']['results'])

def test_gnn_checkpoint_is_synthetic_and_review_only():
    c,_=make_case('continuous');e,_=build(c['source'],c['target'])
    p=predict(e,'models/gnn_synthetic.npz');assert len(p)==len(e) and np.all((p>=0)&(p<=1))
    r=match(c['source'],c['target'],'gnn',{'gnn_model':'models/gnn_synthetic.npz'})
    assert all(x['status']!='accepted' for x in r['results'])

def test_fault_metadata_forces_review():
    c,_=make_case('continuous');c['source']['complex_structure']=True
    assert all(x['status']!='accepted' for x in match(c['source'],c['target'])['results'])

def test_empty_sections_are_explicit():
    c,_=make_case('continuous');c['source']['layers']=[]
    r=match(c['source'],c['target']);assert r['section_warnings']
    assert len(r['results'])==len(c['target']['layers'])

def test_more_than_80_instances_uses_bounded_review_mode():
    c,_=make_case('continuous');base_a=c['source']['layers'][1];base_b=c['target']['layers'][1]
    def many(base,prefix,count):
        rows=[]
        for i in range(count):
            layer=deepcopy(base);layer['id']=f'{prefix}{i:03d}';layer['order']=i
            layer['centroid']=[base['centroid'][0],5+i*1.3]
            layer['extent']=[base['extent'][0],4+i*1.3,base['extent'][2],5+i*1.3]
            layer['upper_neighbors']=[];layer['lower_neighbors']=[];layer['other_contact_neighbors']=[]
            rows.append(layer)
        return rows
    c['source']['layers']=many(base_a,'A',81);c['target']['layers']=many(base_b,'B',83)
    run=match(c['source'],c['target'],'graph',{'max_group':1,'max_candidates':120,'large_candidate_neighbors':1})
    assert any('large_instance_bounded' in warning for warning in run['section_warnings'])
    assert len(run['candidates'])<=120
    assert all(row['status']!='accepted' for row in run['results'])
    assert {v for row in run['results'] for v in row['source_layers']}=={x['id'] for x in c['source']['layers']}
    assert {v for row in run['results'] for v in row['target_layers']}=={x['id'] for x in c['target']['layers']}

def test_review_colours_preserve_instance_and_match_semantics():
    c,_=make_case('split');source=_instance_palette(c['source']);run=match(c['source'],c['target'],'dtw')
    target,_=_target_palette(c['source'],c['target'],run['results'],source)
    assert target['B_M1']==[source['A_M']] and target['B_M2']==[source['A_M']]
    merge,_=make_case('merge');palette=_instance_palette(merge['source'])
    same=np.linalg.norm(np.asarray(palette['A_M1'])-palette['A_M2'])
    different=np.linalg.norm(np.asarray(palette['A_M1'])-palette['A_U'])
    assert 0<same<different

def test_unmatched_target_uses_neutral_review_colour():
    c,_=make_case('lithology_mismatch');source=_instance_palette(c['source']);run=match(c['source'],c['target'],'dtw')
    target,_=_target_palette(c['source'],c['target'],run['results'],source)
    color=target['B_X'][0]
    assert max(color)-min(color)<=4 and color!=source['A_M']

def test_real_partitions_reject_geographic_leakage(tmp_path):
    from stratamatch.io import write
    from stratamatch.annotations import load_partition
    c,_=make_case('continuous');c['survey_group']='same_site';c['exhaustive_annotation']=True
    p=tmp_path/'case.json';write(p,c);idx=tmp_path/'index.json'
    write(idx,{'train':[str(p)],'validation':[str(p)],'test':[str(p)]})
    with pytest.raises(ValueError,match='leakage'):load_partition(idx)

@pytest.mark.parametrize('kind',CASES)
def test_no_layer_is_silently_lost_or_reused(kind):
    c,_=make_case(kind)
    for m in METHODS:
        run=match(c['source'],c['target'],m)
        for key,side in [('source_layers','source'),('target_layers','target')]:
            ids=[v for r in run['results'] for v in r[key]]
            assert len(ids)==len(set(ids))
            assert set(ids)=={x['id'] for x in c[side]['layers']}
