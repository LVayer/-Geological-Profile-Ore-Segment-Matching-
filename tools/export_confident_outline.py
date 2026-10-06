"""Export the directly observed, high-confidence outer-boundary arcs only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.staged_reconstruction import reconstruct_confident_outline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg=json.loads(local(args.config).read_text(encoding='utf-8-sig'))
    image,arcs,audit=reconstruct_confident_outline(cfg)
    output=local(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(image).save(output)
    print(json.dumps({'file':str(output),'trusted_arc_pixels':int(arcs.sum()),
        'audit':audit['confident_outline']},ensure_ascii=False))


if __name__=='__main__':main()
