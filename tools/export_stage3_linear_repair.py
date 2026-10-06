"""Export background image 3 without running the later region-repair stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.staged_reconstruction import reconstruct_stage3


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg=json.loads(local(args.config).read_text(encoding='utf-8-sig'))
    image,_,audit=reconstruct_stage3(cfg)
    output=local(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(image).save(output)
    audit_file=output.with_name(output.stem+'_审计.json')
    audit_file.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    stage3={k:v for k,v in audit['stage3'].items() if k!='passes_detail'}
    print(json.dumps({'file':str(output),'stage3':stage3,'invariants':audit['invariants']},ensure_ascii=False))


if __name__=='__main__':
    main()
