"""Export the first colour/fault-based layer partition for manual review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.layer_partition import partition_layers


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(local(args.config).read_text(encoding='utf-8-sig'))
    result=partition_layers(cfg);output=local(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(result['image']).save(output)
    np.save(output.with_name(output.stem+'_地层编号.npy'),result['instances'])
    report={'image':str(output),'layer_count':len(result['records']),
            'layers':result['records'],'audit':result['audit']}
    report_file=output.with_name(output.stem+'_统计.json')
    report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'image':str(output),'report':str(report_file),
                      'layer_count':len(result['records']),'audit':result['audit']},ensure_ascii=False))


if __name__=='__main__':main()
