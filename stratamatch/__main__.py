import argparse
import logging
from .io import read,write,local
from .engine import compare
from .synthetic import generate
from .evaluation import train_synthetic,evaluate
from .export import export_report,metric_csv

def main():
    parser=argparse.ArgumentParser(description='Conservative geological section correlation experiment')
    parser.add_argument('command',choices=['demo','generate','train','benchmark','recognize','match','pipeline'])
    parser.add_argument('--input');parser.add_argument('--output',default='outputs/run');parser.add_argument('--config',default='config/default.json')
    args=parser.parse_args(); config=read(args.config)
    local('logs').mkdir(exist_ok=True)
    logging.basicConfig(filename=local('logs/run.log'),level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',encoding='utf-8')
    logging.info('command=%s',args.command)
    if args.command in ['demo','generate']: cases=generate()
    if args.command in ['demo','train']:
        if args.command=='train' and args.input:
            from .annotations import train_annotations
            history=train_annotations(args.input)
        else: history=train_synthetic()
        print('GNN training:',history[-1])
    if args.command in ['demo','benchmark']:
        if args.command=='benchmark': cases=[read(p) for p in read(args.input or 'data/synthetic/index.json')['cases']]
        metrics,reports=evaluate(cases,config)
        metric_csv(metrics,'outputs/benchmark/metrics.csv')
        for c,r in reports:
            if c['seed']==1001: export_report(r,c['source'],c['target'],f'outputs/review/{c["name"]}')
        for m,r in metrics['summary'].items(): print(m,{k:r[k] for k in ['accepted_precision','correct_match_rate_recall','wrong_edges','missed_edges','required_abstention_recall']})
    if args.command in ['recognize','pipeline']:
        if not args.input: parser.error('--input manifest required')
        from .recognition import recognize
        from PIL import Image
        import numpy as np
        sections=[]
        for idx,cfg in enumerate(read(args.input)['sections']):
            s,labels=recognize(cfg);sections.append(s)
            # Persist label IDs so recognition can be independently audited.
            p=local(f'{args.output}/recognition/{idx}_labels.npy');p.parent.mkdir(parents=True,exist_ok=True);np.save(p,labels)
            rng=np.random.default_rng(8);colors=rng.integers(40,240,(max(1,len(s['layers'])),3),dtype='uint8')
            overlay=np.full((*labels.shape,3),255,dtype='uint8'); valid=labels>=0;overlay[valid]=colors[labels[valid]]
            Image.fromarray(overlay).save(local(f'{args.output}/recognition/{idx}_labels.png'))
        write(f'{args.output}/sections.json',{'sections':sections})
    if args.command in ['pipeline','match']:
        if args.command=='match':
            if not args.input: parser.error('--input sections.json required')
            sections=read(args.input)['sections']
        if len(sections)<2: raise ValueError('Need >=2 sections')
        ids=[s['id'] for s in sections]
        if len(ids)!=len(set(ids)): raise ValueError('Section IDs must be unique')
        contexts={}
        for i,(sa,sb) in enumerate(zip(sections,sections[1:])):
            report=compare(sa,sb,config,contexts)
            export_report(report,sa,sb,f'{args.output}/pair_{i:03d}')
            contexts={m:run['results'] for m,run in report['methods'].items()}
    print('Completed. All outputs stay inside project.')

if __name__=='__main__': main()
