"""Explicit denominators, abstention/coverage, group exactness; no inflated accuracy."""
from collections import defaultdict
from .engine import compare,METHODS
from .candidates import build
from .gnn import train
from .synthetic import CASES,make_case
from .io import write

def edge_set(relations,accepted=False):
    return {(s,t) for r in relations if not accepted or r.get('status')=='accepted' for s in r.get('sources',r.get('source_layers',[])) for t in r.get('targets',r.get('target_layers',[]))}

def group_key(r,proposal=False):
    return (tuple(sorted(r.get('sources',r.get('source_layers',[])))),tuple(sorted(r.get('targets',r.get('target_layers',[])))),r.get('relation',r.get('proposed_relation' if proposal else 'relation_type')))

def train_synthetic(path='models/gnn_synthetic.npz'):
    graphs=[]
    for seed in range(20,28):
        for name in CASES:
            c,_=make_case(name,seed); e,_=build(c['source'],c['target'])
            truth={group_key(r)[:2] for r in c['truth'] if r['targets']}
            y=[int((tuple(sorted(c['source']['layers'][i]['id'] for i in a['s'])),tuple(sorted(c['target']['layers'][j]['id'] for j in a['t']))) in truth) for a in e]
            if e: graphs.append((e,y))
    history=train(graphs,path)
    write('models/training.json',{'training_seeds':list(range(20,28)),'held_out_test_seeds':[1001,1002,1003],
        'limitation':'Same generator families; seed holdout is NOT independent real-domain validation',
        'graphs':len(graphs),'history':history,'model':path})
    return history

def evaluate(cases,config=None,out='outputs/benchmark'):
    totals={m:defaultdict(int) for m in METHODS}; detail=[]; reports=[]
    for case in cases:
        report=compare(case['source'],case['target'],config); reports.append((case,report))
        truth=edge_set(case['truth']); groups={group_key(r) for r in case['truth']}
        row={'case':case['name'],'seed':case['seed'],'conflicts':len(report['conflicts']),'methods':{}}
        for m,run in report['methods'].items():
            rows=run['results']; pred=edge_set(rows,True); proposed=edge_set(rows)
            d=defaultdict(int); d['true_edges']=len(truth); d['accepted_edges']=len(pred)
            d['correct_edges']=len(pred&truth); d['wrong_edges']=len(pred-truth); d['missed_edges']=len(truth-pred)
            d['proposal_correct_edges']=len(proposed&truth); d['proposal_wrong_edges']=len(proposed-truth)
            d['sources']=len(case['source']['layers']); d['accepted_sources']=sum(len(r['source_layers']) for r in rows if r['status']=='accepted')
            d['uncertain_sources']=sum(len(r['source_layers']) for r in rows if r['status']!='accepted')
            abstained={s for r in rows if r['status']!='accepted' for s in r['source_layers']}
            d['must_abstain']=len(case['must_abstain']); d['correct_abstentions']=len(abstained&set(case['must_abstain']))
            for relation in ['split','merge','pinch-out']:
                expected={g for g in groups if g[2]==relation}
                actual={group_key(r) for r in rows if r['status']=='accepted' and r['relation_type']==relation}
                d[relation+'_truth']=len(expected); d[relation+'_correct']=len(actual&expected); d[relation+'_false']=len(actual-expected)
            for k,v in d.items(): totals[m][k]+=v
            row['methods'][m]=dict(d)
        detail.append(row)
    summary={}
    for m,t in totals.items():
        def ratio(a,b): return t[a]/t[b] if t[b] else None
        summary[m]={**t,'accepted_precision':ratio('correct_edges','accepted_edges'),'correct_match_rate_recall':ratio('correct_edges','true_edges'),
            'wrong_match_rate':ratio('wrong_edges','accepted_edges'),'source_coverage':ratio('accepted_sources','sources'),
            'required_abstention_recall':ratio('correct_abstentions','must_abstain')}
        for r in ['split','merge','pinch-out']: summary[m][r+'_recall']=ratio(r+'_correct',r+'_truth')
    result={'dataset':'synthetic held-out seeds, shared generator families','case_count':len(cases),
        'warning':'No real-data accuracy claim. Null means denominator zero. Proposal scores are separate from accepted decisions.',
        'summary':summary,'details':detail}
    write(out+'/metrics.json',result)
    for c,r in reports: write(f'{out}/{c["name"]}_{c["seed"]}.json',r)
    return result,reports
