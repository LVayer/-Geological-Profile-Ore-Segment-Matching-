"""Export the four reviewable stages of colour-first reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.staged_reconstruction import reconstruct_stages


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(local(args.config).read_text(encoding='utf-8-sig'))
    output=local(args.output);output.mkdir(parents=True,exist_ok=True)
    result=reconstruct_stages(cfg)
    files={'stage1':'背景图1_仅地层颜色.png','stage2':'背景图2_总体轮廓.png',
           'stage3':'背景图3_线状空白修复.png','final':'最终地层识别结果.png'}
    for key,name in files.items():Image.fromarray(result[key]).save(output/name)
    np.save(output/'最终岩性标签.npy',result['labels'])
    Image.fromarray((result['domain'].astype(np.uint8)*255)).save(output/'最终总体轮廓掩膜.png')
    audit=result['audit'];audit['files']=list(files.values())+['最终岩性标签.npy','最终总体轮廓掩膜.png']
    (output/'分阶段重建审计.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'stage1':audit['stage1'],'stage2':audit['stage2'],
        'stage3':{k:v for k,v in audit['stage3'].items() if k!='passes_detail'},'stage4':audit['stage4'],
        'final_invariants':audit['final_invariants']},ensure_ascii=False))


if __name__=='__main__':main()
