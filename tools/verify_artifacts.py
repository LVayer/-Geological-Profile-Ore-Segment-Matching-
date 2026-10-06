"""Independent output audit and reproducible implementation report, all local."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from stratamatch.io import ROOT,SOFTWARE_ROOT,read,write,local
from stratamatch.synthetic import make_case,render_case
from stratamatch.recognition import recognize

def main():
    checked=0;accepted=0
    for p in read('data/synthetic/index.json')['cases']:
        c=read(p);run=read(f"outputs/benchmark/{c['name']}_{c['seed']}.json")
        truth={(tuple(sorted(t['sources'])),tuple(sorted(t['targets'])),t['relation']) for t in c['truth']}
        for method,r in run['methods'].items():
            source=[s for item in r['results'] for s in item['source_layers']]
            target=[s for item in r['results'] for s in item['target_layers']]
            assert len(source)==len(set(source)) and set(source)=={l['id'] for l in c['source']['layers']}
            assert len(target)==len(set(target)) and set(target)=={l['id'] for l in c['target']['layers']}
            for item in r['results']:
                if item['status']=='accepted':
                    accepted+=1
                    key=(tuple(sorted(item['source_layers'])),tuple(sorted(item['target_layers'])),item['relation_type'])
                    assert key in truth,(c['name'],method,key)
            checked+=1
    c,spec=make_case('continuous');render_case(c,spec,'data/recognition_validation')
    base=read('data/recognition_validation/manifest.json')['sections'][0];pixel=[]
    for suffix in ['png','jpg','tiff']:
        path=local(f'data/recognition_validation/input.{suffix}')
        Image.open(local(base['image'])).save(path)
        cfg={**base,'image':str(path.relative_to(ROOT))};section,labels=recognize(cfg)
        for _,lith,expected,_ in spec[0]:
            layer=next(l for l in section['layers'] if l['lithology']==lith)
            idx=next(int(i) for i,lid in section['recognition']['label_id_mapping'].items() if lid==layer['id'])
            got=labels==idx;intersection=np.sum(got&expected)
            pixel.append({'format':suffix,'lithology':lith,'iou':float(intersection/np.sum(got|expected)),
                'assigned_pixel_precision':float(intersection/np.sum(got)),'quality':layer['quality']})
    write('outputs/recognition-metrics.json',{'dataset':'one controlled three-layer figure, no real-data claim','rows':pixel})
    tests=ET.parse(local('outputs/test-results.xml')).getroot()
    suites=list(tests.iter('testsuite'));count=sum(int(s.get('tests','0')) for s in suites)
    failures=sum(int(s.get('failures','0'))+int(s.get('errors','0')) for s in suites)
    assert failures==0
    files=[*SOFTWARE_ROOT.glob('stratamatch/*.py'),*SOFTWARE_ROOT.glob('tests/*.py'),SOFTWARE_ROOT/'config/default.json',SOFTWARE_ROOT/'requirements-lock.txt']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write('outputs/verification.json',{'tests':count,'failures':failures,'method_runs_audited':checked,'accepted_events_audited':accepted,
        'all_accepted_events_match_complete_truth':True,'png_jpeg_tiff_pixel_metrics':'outputs/recognition-metrics.json','sha256':hashes})
    print(json.dumps({'tests':count,'failures':failures,'method_runs_audited':checked,'accepted_events_audited':accepted}))

if __name__=='__main__':main()
