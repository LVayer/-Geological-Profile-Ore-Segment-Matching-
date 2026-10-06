"""Export colour-only layer boundaries and outline-plus-fault review images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.layer_partition import partition_layers


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(local(args.config).read_text(encoding='utf-8-sig'))
    result=partition_layers(cfg);folder=local(args.output);folder.mkdir(parents=True,exist_ok=True)
    first=folder/'步骤1_按颜色分层_黑色边界.png'
    second=folder/'步骤2_总体轮廓与断层线.png'
    Image.fromarray(result['colour_only_image']).save(first)
    Image.fromarray(result['outline_fault_image']).save(second)
    summary={'step1_image':str(first),'step2_image':str(second),
             'step1_colour_layers':result['audit']['colour_components_before_fault'],
             'unassigned_classification':result['audit']['unassigned_classification'],
             'local_boundary_validation':result['audit']['local_boundary_validation'],
             'fault_pixels':result['audit']['fault_barrier_pixels'],
             'fault_detection':result['audit']['fault']}
    report=folder/'分层两步检查_统计.json'
    report.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({**summary,'report':str(report)},ensure_ascii=False))


if __name__=='__main__':main()
