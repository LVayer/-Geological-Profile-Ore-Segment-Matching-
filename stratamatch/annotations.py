"""Validated real annotation partitions for GNN training and comparison."""
from .io import read,write,validate_section
from .candidates import build
from .gnn import train

def load_partition(index):
    data=read(index)
    if set(data)<={'cases'}: raise ValueError('Training requires explicit train/validation/test partitions')
    seen=set(); partitions={}
    for partition in ['train','validation','test']:
        groups=set(); cases=[]
        for path in data[partition]:
            c=read(path)
            if not c.get('survey_group'): raise ValueError('Real cases require survey_group to prevent geographic leakage')
            groups.add(c['survey_group']);validate_section(c['source']);validate_section(c['target'])
            source={l['id']:l for l in c['source']['layers']};target={l['id']:l for l in c['target']['layers']}
            used_s=set();used_t=set()
            for r in c['truth']:
                if not set(r['sources'])<=set(source) or not set(r['targets'])<=set(target): raise ValueError('Truth references missing layers')
                if set(r['sources'])&used_s or set(r['targets'])&used_t: raise ValueError('Group repeated; encode merge/split in one truth record')
                used_s.update(r['sources']);used_t.update(r['targets'])
                lith={source[s]['lithology'] for s in r['sources']}|{target[t]['lithology'] for t in r['targets']}
                if r['targets'] and (len(lith)!=1 or 'UNKNOWN' in lith): raise ValueError('Truth violates lithology rule')
            cases.append(c)
        if seen&groups: raise ValueError('survey_group leakage across train/validation/test')
        seen.update(groups);partitions[partition]=cases
    if any(not v for v in partitions.values()): raise ValueError('All partitions must be nonempty')
    return partitions

def train_annotations(index,path='models/gnn_annotations.npz'):
    partitions=load_partition(index);graphs=[]
    for c in partitions['train']:
        e,_=build(c['source'],c['target']);valid={ (tuple(sorted(r['sources'])),tuple(sorted(r['targets']))) for r in c['truth'] if r['targets'] }
        y=[int((tuple(sorted(c['source']['layers'][i]['id'] for i in r['s'])),tuple(sorted(c['target']['layers'][i]['id'] for i in r['t']))) in valid) for r in e]
        # Sparse labels cannot safely make every unlisted candidate a negative.
        if not c.get('exhaustive_annotation'): raise ValueError('Training needs exhaustive_annotation=true; incomplete labels are not negatives')
        if e:graphs.append((e,y))
    if not graphs: raise ValueError('No trainable candidates')
    history=train(graphs,path,domain='unvalidated_annotations')
    write('models/annotation_training.json',{'history':history,'model':path,'counts':{k:len(v) for k,v in partitions.items()},
        'status':'Requires held-out validation and calibration; inference stays manual_review'})
    return history
