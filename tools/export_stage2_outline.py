"""Export only the generalized outer-outline review image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.staged_reconstruction import reconstruct_stage2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(local(args.config).read_text(encoding='utf-8-sig'))
    image,domain,audit=reconstruct_stage2(cfg);output=local(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(image).save(output)
    audit_path=output.with_name('背景图2_总体轮廓审计.json')
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'file':str(output),'domain_pixels':int(domain.sum()),'audit':audit['stage2'],
                      'invariants':audit['invariants']},ensure_ascii=False))


if __name__=='__main__':main()
