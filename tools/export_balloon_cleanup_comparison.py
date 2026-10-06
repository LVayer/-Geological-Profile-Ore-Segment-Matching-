"""在相同 ROI、图例和分辨率下比较颜色清理，避免混入气球形状差异。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experimental_fullres_colour_membrane import classify_full_resolution
from stratamatch.io import local


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for path in args.config:
        cfg=json.loads(Path(path).read_text(encoding='utf-8'))
        x,y,w,h=cfg['roi']
        with Image.open(local(cfg['image'])) as source:
            original=np.asarray(source.convert('RGB').crop((x,y,x+w,y+h)))
        panels=[];counts=[];audits=[]
        for enabled in (False,True):
            result=classify_full_resolution({**cfg,'adaptive_colour_cleanup':enabled})
            keep=result[4]>=0;counts.append(int(keep.sum()));audits.append(result[-1])
            cleaned=np.where(keep[:,:,None],original,255).astype(np.uint8)
            panel=Image.fromarray(cleaned);panel.thumbnail((1000,900));panels.append(panel)
            del result,cleaned,keep
        height=max(p.height for p in panels)+40
        sheet=Image.new('RGB',(2000,height),'white');draw=ImageDraw.Draw(sheet)
        for index,panel in enumerate(panels):
            sheet.paste(panel,(1000*index,40))
            draw.text((1000*index+15,12),['BEFORE: fixed colour tolerance','AFTER: adaptive swatch + background competition'][index],fill='black')
        name=Path(path).stem
        sheet.save(args.output/(name+'_清杂对照.png'))
        (args.output/(name+'_清杂统计.json')).write_text(json.dumps({
            'retained_before':counts[0],'retained_after':counts[1],
            'removed_extra':counts[0]-counts[1], 'accuracy_measured':False,
            'audit_before':audits[0],'audit_after':audits[1]},ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,counts,flush=True)


if __name__=='__main__':main()
