"""Venice Decisions API adapter; fail closed, never turn a mock into real evidence."""
import json
import os
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse
from .decisions import payload as shared_payload,parse_noul

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('API redirects are disabled to protect credentials')

def payload(sa,sb,edges,model='jev-latest'):
    return shared_payload(sa,sb,edges,model)

def parse(data,edges):
    return parse_noul(data,edges)

def decide(sa,sb,edges,config,transport=None):
    mode=config.get('mode','mock')
    if mode!='live':
        return None,{'mode':'mock','reason':'No external model called; geological fallback only'}
    endpoint=config.get('endpoint','https://api.venice.ai/api/v1/decisions')
    if urlparse(endpoint).scheme!='https': raise ValueError('Jev endpoint requires HTTPS')
    key=os.getenv(config.get('key_env','VENICE_API_KEY'))
    if not key: return None,{'mode':'fallback','reason':'API key missing'}
    body=json.dumps(payload(sa,sb,edges,config.get('model','jev-latest')),ensure_ascii=False,allow_nan=False).encode()
    if len(body)>config.get('max_request_bytes',180000):
        return None,{'mode':'fallback','reason':'Request too large; reduce section or candidate count'}
    opener=urllib.request.build_opener(NoRedirect())
    def send():
        request=urllib.request.Request(endpoint,data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        with opener.open(request,timeout=config.get('timeout_seconds',20)) as response:
            raw=response.read(2000001)
            if len(raw)>2000000: raise ValueError('Response exceeds bound')
            return json.loads(raw)
    # POST retries can incur a second charge; default zero retries, opt-in only.
    for attempt in range(min(config.get('retries',0),2)+1):
        try:
            result=(transport or send)()
            return parse(result,edges),{'mode':'live','response_validated':True}
        except (ValueError,TimeoutError,OSError,urllib.error.URLError) as exc:
            reason=type(exc).__name__ # never persist tokens or raw error bodies
            if attempt<min(config.get('retries',0),2): time.sleep(.5*2**attempt)
    return None,{'mode':'fallback','reason':reason}
