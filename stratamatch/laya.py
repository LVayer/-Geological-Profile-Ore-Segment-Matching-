"""Laya adapter using the same typed-decision evidence as Jev.

Modes: mock (default), http (TypeSafe-compatible local service), python
(optional `laya` package/open weights). All modes fail closed.
"""
import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse
from .decisions import canonical,payload,parse_noul
from .jev import NoRedirect

_AGENTS={}

def parse(data,edges):return parse_noul(data,edges)

def _safe_endpoint(endpoint):
    parsed=urlparse(endpoint)
    if parsed.scheme=='https':return
    if parsed.scheme=='http' and parsed.hostname in {'127.0.0.1','localhost','::1'}:return
    raise ValueError('Laya HTTP must use HTTPS or a loopback localhost endpoint')

def decide(sa,sb,edges,config,transport=None):
    mode=config.get('mode','mock');model=config.get('model','typed-decisions')
    request=payload(sa,sb,edges,model)
    if mode=='mock':return None,{'mode':'mock','provider':'laya','reason':'Laya was not called; geological fallback only','input_contract':'shared_with_jev'}
    try:
        if mode=='python':
            if transport:
                data=transport()
            else:
                import laya
                checkpoint=config.get('checkpoint','convaiinnovations/laya');subfolder=config.get('subfolder','typed-decisions')
                key=(checkpoint,subfolder)
                if key not in _AGENTS:_AGENTS[key]=laya.load(checkpoint,subfolder=subfolder)
                common=canonical(sa,sb,edges)
                state=json.loads(common['state'])
                data=_AGENTS[key].predict(state,common['questions'])
        elif mode=='http':
            endpoint=config.get('endpoint','http://127.0.0.1:8791/v1/systemone');_safe_endpoint(endpoint)
            body=json.dumps(request,ensure_ascii=False,allow_nan=False).encode()
            if len(body)>config.get('max_request_bytes',180000):raise ValueError('Request too large')
            if transport:data=transport()
            else:
                headers={'Content-Type':'application/json'};key=os.getenv(config.get('key_env','LAYA_API_KEY'))
                if key:headers['Authorization']='Bearer '+key
                req=urllib.request.Request(endpoint,data=body,headers=headers,method='POST')
                with urllib.request.build_opener(NoRedirect()).open(req,timeout=config.get('timeout_seconds',30)) as response:
                    raw=response.read(2000001)
                    if len(raw)>2000000:raise ValueError('Response exceeds bound')
                    data=json.loads(raw)
        else:raise ValueError('Laya mode must be mock, http, or python')
        return parse(data,edges),{'mode':mode,'provider':'laya','model':data.get('model',model),'response_validated':True,'input_contract':'shared_with_jev'}
    except (ImportError,ValueError,TimeoutError,OSError,urllib.error.URLError) as exc:
        return None,{'mode':'fallback','provider':'laya','reason':type(exc).__name__,'input_contract':'shared_with_jev'}
