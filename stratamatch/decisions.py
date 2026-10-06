"""Canonical typed-decision evidence shared byte-for-byte by Jev and Laya."""
import json

TASK_RULES=(
    'Geological layer correlation. Unequal or UNKNOWN lithology is forbidden. '
    'Evaluate the complete proposed continuous/split/merge membership. Use position, '
    'thickness, boundary shape, dip, order, neighbors, contacts and topology together. '
    'Shape or proximity alone is insufficient. Ambiguous evidence requires rejection.'
)

def canonical(sa,sb,edges):
    state={'task':TASK_RULES,'source':sa,'target':sb,'candidates':edges}
    questions={f"edge_{e['id']}":{
        'type':'noul',
        'instructions':f"Is candidate edge_{e['id']} a geologically justified correspondence including its complete split/merge membership?",
        'criteria':{
            'true':'The complete correspondence is jointly supported by lithology, registered position, morphology, stratigraphic order and topology.',
            'false':'It is unrelated, has incomplete group membership, conflicts with geology, or evidence is insufficient.'
        }
    } for e in edges}
    # A deterministic string keeps provider inputs identical even when one SDK only
    # accepts text state. Provider adapters may change model/auth, never evidence.
    return {'state':json.dumps(state,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False),'questions':questions}

def payload(sa,sb,edges,model):
    return {'model':model,**canonical(sa,sb,edges)}

def parse_noul(data,edges):
    answers=data.get('answers')
    if not isinstance(answers,dict):raise ValueError('Missing answers; check provider schema')
    out=[]
    for edge in edges:
        value=answers.get(f"edge_{edge['id']}")
        if isinstance(value,dict):
            # TypeSafe-compatible servers use `noul`; older adapters used
            # `probability`. Accept either, but never infer from confidence.
            value=value.get('noul',value.get('probability'))
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not 0<=value<=1:
            raise ValueError('Invalid/missing noul probability')
        out.append(float(value))
    return out
