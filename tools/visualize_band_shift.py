"""把分组审计中的切断带共同表观位移画在实际岩性图上。"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def render(folder):
    folder=Path(folder)
    audit=json.loads((folder/'分组依据.json').read_text(encoding='utf-8'))
    image=cv2.cvtColor(np.asarray(Image.open(folder/'岩性分类.png').convert('RGB')),cv2.COLOR_RGB2BGR)
    scale=image.shape[1]/np.load(folder/'区域实例.npy',mmap_mode='r').shape[1]
    bands=audit.get('band_registration',{}).get('bands',[])
    bands=sorted((b for b in bands if b['matched_layers']>=2 and
                  b.get('joint_evidence_sufficient') and b['shift_spread_pixels'] is not None),
                 key=lambda b:(-b['matched_layers'],b['shift_spread_pixels']))
    shown=[]
    for band in bands:
        centre=np.asarray(band['cut_centre'],float)
        if any(np.linalg.norm(centre-other)<50 for other in shown):continue
        shift=float(band['apparent_shift_along_cut_pixels'])
        if abs(shift)<5:continue
        axis=np.asarray(band['cut_axis'],float)
        start=np.rint(centre*scale).astype(int)
        end=np.rint((centre+shift*axis)*scale).astype(int)
        cv2.arrowedLine(image,tuple(start),tuple(end),(255,70,0),5,cv2.LINE_AA,tipLength=.17)
        label=f"{band['matched_layers']} pairs | {abs(shift):.0f} px"
        cv2.putText(image,label,tuple(start+np.array([8,-9])),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),5,cv2.LINE_AA)
        cv2.putText(image,label,tuple(start+np.array([8,-9])),cv2.FONT_HERSHEY_SIMPLEX,.75,(20,20,20),2,cv2.LINE_AA)
        shown.append(centre)
        if len(shown)>=16:break
    Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).save(folder/'切断带共同位移示意.png')
    preview=cv2.resize(image,(1800,round(image.shape[0]*1800/image.shape[1])),interpolation=cv2.INTER_AREA)
    Image.fromarray(cv2.cvtColor(preview,cv2.COLOR_BGR2RGB)).save(folder/'位移预览.png')
    return len(shown)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folders',nargs='+')
    args=parser.parse_args()
    for folder in args.folders:print(folder,'arrows',render(folder))
