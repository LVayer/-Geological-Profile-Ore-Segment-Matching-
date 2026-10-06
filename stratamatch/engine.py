"""Shared validation, abstention, global alternative margin and conflict retention."""
from copy import deepcopy
import numpy as np
from .io import validate_section
from .candidates import build,FEATURES,feature_weights
from .solvers import graph,sequence
from . import gnn,jev,laya

METHODS=['dtw','graph','markov','gnn','jev','laya']

def match(sa,sb,method='graph',config=None,context=None):
    config=config or {}
    validate_section(sa); validate_section(sb)
    if method not in METHODS: raise ValueError('Unknown method')
    if sa['frame']!=sb['frame'] or sa['units']!=sb['units']: raise ValueError('Coordinate frame/units mismatch; register sections first')
    if sb['station']<=sa['station']: raise ValueError('Sections must follow increasing stations')
    aa,bb=sa['layers'],sb['layers']; weights=feature_weights(config)
    edges,rejected=build(sa,sb,config.get('max_group',3),config.get('max_candidates',1200),config.get('large_candidate_neighbors',2),weights)
    bounded_large_mode=any(r.get('reason')=='large_instance_candidate_pruning' for r in rejected)
    provenance={'mode':'deterministic','calibrated':False}; review_model=False
    if context:
        # Prior results are only bounded penalties; they cannot bypass current evidence.
        expected={r['target_layers'][0]:r.get('displacement',[0,0]) for r in context if r['status']=='accepted' and r['confidence']>=.85 and len(r['target_layers'])==1}
        scale=sa['bounds'][3]-sa['bounds'][1]
        for e in edges:
            ds=[expected.get(aa[i]['id']) for i in e['s'] if aa[i]['id'] in expected]
            if ds:
                now=np.mean([bb[j]['centroid'] for j in e['t']],0)-np.mean([aa[i]['centroid'] for i in e['s']],0)
                e['context_penalty']=float(min(.08,np.linalg.norm(now-np.mean(ds,0))/scale*.1)); e['cost']+=e['context_penalty']
    if method=='gnn':
        review_model=True; path=config.get('gnn_model')
        if path:
            p=gnn.predict(edges,path)
            from .io import local
            with np.load(local(path),allow_pickle=False) as checkpoint: domain=str(checkpoint['domain'])
            provenance={'mode':'gnn_checkpoint','training_domain':domain,'calibrated':False}
            for e,prob in zip(edges,p): e['model_probability']=float(prob); e['model_cost']=.7*e['cost']+.3*float(-np.log(max(prob,1e-6)))
        else: provenance={'mode':'untrained_fallback','calibrated':False}
    if method=='jev':
        p,provenance=jev.decide(sa,sb,edges,config.get('jev',{})); review_model=True
        if p is not None:
            for e,prob in zip(edges,p): e['model_probability']=prob; e['model_cost']=.7*e['cost']+.3*float(-np.log(max(prob,1e-6)))
    if method=='laya':
        p,provenance=laya.decide(sa,sb,edges,config.get('laya',{})); review_model=True
        if p is not None:
            for e,prob in zip(edges,p): e['model_probability']=prob; e['model_cost']=.7*e['cost']+.3*float(-np.log(max(prob,1e-6)))
    solver=(lambda forbidden=None: sequence(edges,len(aa),len(bb),method,forbidden)) if method in ['dtw','markov'] else (lambda forbidden=None:graph(edges,len(aa),len(bb),forbidden))
    failure=None
    try: selected,objective,posterior=solver()
    except RuntimeError as exc: selected=[]; objective=None; posterior={}; failure=str(exc)
    results=[]; used_s=set(); used_t=set()
    for eid in selected:
        e=edges[eid]; used_s.update(e['s']); used_t.update(e['t'])
        if bounded_large_mode:margin=0.
        else:
            try: _,alt,_=solver(eid); margin=max(0.,alt-objective)
            except RuntimeError: margin=0.
        relation='split' if len(e['t'])>1 else ('merge' if len(e['s'])>1 else 'continuous')
        f=e['features']; confidence=float(min(e['quality'],np.exp(-e['cost']*.45),1-np.exp(-margin/.35)))
        if method=='markov': confidence=min(confidence,posterior.get(eid,0.))
        reasons=list(e.get('flags',[]))
        if e['quality']<.8: reasons.append('low_recognition_quality')
        if e['cost']>config.get('max_accept_cost',1.05): reasons.append('geological_cost_high')
        if margin<config.get('min_global_margin',.28): reasons.append('ambiguous_global_alternative')
        if f[3]>.09 or f[5]>.35: reasons.append('shape_or_dip_discontinuity')
        if f[7]+f[8]>=1.5: reasons.append('stratigraphic_neighbors_changed')
        if not(sa.get('registration_verified') and sb.get('registration_verified')): reasons.append('registration_unverified')
        if sa.get('complex_structure') or sb.get('complex_structure'): reasons.append('fault_or_complex_structure_requires_review')
        if review_model: reasons.append('model_not_validated_on_real_annotations')
        if bounded_large_mode:reasons.append('large_instance_bounded_candidate_pruning')
        status='accepted' if not reasons else ('manual_review' if review_model else 'uncertain')
        results.append({'source_section':sa['id'],'source_layers':[aa[i]['id'] for i in e['s']],
            'target_section':sb['id'],'target_layers':[bb[j]['id'] for j in e['t']],
            'relation_type':relation if status=='accepted' else 'uncertain','proposed_relation':relation,
            'confidence':confidence,'confidence_kind':'uncalibrated_evidence_score','method':method,'status':status,
            'features':dict(zip(FEATURES,f)),'cost':e['cost'],'global_margin':margin,'markov_posterior':posterior.get(eid),
            'displacement':(np.mean([bb[j]['centroid'] for j in e['t']],0)-np.mean([aa[i]['centroid'] for i in e['s']],0)).tolist(),
            'reasons':reasons})
    # Cross-check actual mapped contact edges, not only counts or lithology sets.
    by_a={a['id']:a for a in aa}; by_b={b['id']:b for b in bb}
    def contacts(layer):
        return set(layer['upper_neighbors']+layer['lower_neighbors']+layer['other_contact_neighbors'])
    bad=set()
    for i,r in enumerate(results):
        if r['status']!='accepted': continue
        for j,q in enumerate(results[:i]):
            if q['status']!='accepted': continue
            source_contact=any(v in contacts(by_a[u]) for u in r['source_layers'] for v in q['source_layers'])
            target_contact=any(v in contacts(by_b[u]) for u in r['target_layers'] for v in q['target_layers'])
            if source_contact and not target_contact: bad.update([i,j])
    for i in bad:
        results[i]['status']='uncertain';results[i]['relation_type']='uncertain'
        results[i]['reasons'].append('mapped_contact_topology_conflict')
    for i,a in enumerate(aa):
        if i in used_s: continue
        # Absence alone is not proof of pinch-out: require taper + complete target + enclosing anchors.
        anchors={s:t for r in results if r['status']=='accepted' for s in r['source_layers'] for t in r['target_layers']}
        ups=[anchors[x] for x in a['upper_neighbors'] if x in anchors]; downs=[anchors[x] for x in a['lower_neighbors'] if x in anchors]
        byid={b['id']:b for b in bb}
        enclosed=any(d in byid[u]['lower_neighbors'] for u in ups for d in downs)
        no_candidate=not any(i in e['s'] for e in edges)
        pinch=bool(no_candidate and a.get('taper_ratio',1)<.25 and sb.get('complete') and enclosed and a['quality']>=.9 and sa.get('registration_verified') and sb.get('registration_verified') and not review_model and not bounded_large_mode)
        results.append({'source_section':sa['id'],'source_layers':[a['id']],'target_section':sb['id'],'target_layers':[],
            'relation_type':'pinch-out' if pinch else 'uncertain','proposed_relation':'pinch-out' if pinch else 'disappearance',
            'confidence':.8 if pinch else .0,'confidence_kind':'uncalibrated_evidence_score','method':method,'status':'accepted' if pinch else 'manual_review',
            'reasons':['taper_complete_target_enclosing_anchors'] if pinch else ['no_reliable_correspondence; absence is not proof of pinch-out']})
    # Preserve target-only layers; missing/new units must not silently vanish from outputs.
    for j,b in enumerate(bb):
        if j not in used_t:
            results.append({'source_section':sa['id'],'source_layers':[],'target_section':sb['id'],'target_layers':[b['id']],
                'relation_type':'uncertain','proposed_relation':'appearance','confidence':0.,'confidence_kind':'uncalibrated_evidence_score','method':method,'status':'manual_review','reasons':['unmatched_target']})
    return {'method':method,'provenance':provenance,'objective':objective,'solver_failure':failure,
        'matching_policy':{'hard_gate':'exact_known_lithology','soft_feature_weights':dict(zip(FEATURES,weights.tolist())),
            'weight_status':'conservative_prior_not_calibrated_on_real_annotations'},
        'section_warnings':((['empty_section_requires_manual_review'] if not aa or not bb else [])+
            (['large_instance_bounded_candidate_pruning; all relations require manual review'] if bounded_large_mode else [])),
        'results':results,'candidates':edges,'rejected':rejected}

def compare(sa,sb,config=None,contexts=None,methods=None):
    selected=METHODS if methods is None else list(dict.fromkeys(methods))
    if not selected or any(m not in METHODS for m in selected):
        raise ValueError('Select at least one valid matching method')
    runs={m:match(sa,sb,m,config,(contexts or {}).get(m)) for m in selected}
    conflicts=[]
    for a in sa['layers']:
        votes={}
        for m,run in runs.items():
            r=next(r for r in run['results'] if a['id'] in r['source_layers'])
            votes[m]={'targets':r['target_layers'],'proposed_relation':r['proposed_relation'],'status':r['status']}
        if len({(tuple(sorted(v['targets'])),v['proposed_relation']) for v in votes.values()})>1:
            conflicts.append({'source_layer':a['id'],'status':'manual_review','votes':votes})
    # No majority vote overrides an individual model or a geological conflict.
    return {'source_section':sa['id'],'target_section':sb['id'],'methods':runs,'conflicts':conflicts,'consensus_policy':'preserve_all_methods_no_automatic_overwrite'}
