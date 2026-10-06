"""Export a numbered final layer map and a machine-readable number mapping."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.layer_partition import partition_layers


def _anchor(mask):
    """Choose the pixel farthest from a layer edge for a stable label anchor."""
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    y,x=np.unravel_index(int(np.argmax(distance)),distance.shape)
    return int(x),int(y),float(distance[y,x])


def _box(origin,size,padding=3):
    x,y=origin;w,h=size
    return x-padding,y-h-padding,x+w+padding,y+padding


def _fits(box,shape,occupied):
    x0,y0,x1,y1=box;h,w=shape
    if x0<1 or y0<1 or x1>=w-1 or y1>=h-1:return False
    return not np.any(occupied[y0:y1+1,x0:x1+1])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--no-layer-outlines',action='store_true',
                        help='Use the filled colour result without black layer-interface outlines.')
    args=parser.parse_args();cfg=json.loads(local(args.config).read_text(encoding='utf-8-sig'))
    result=partition_layers(cfg)
    image=(result['plain_image'] if args.no_layer_outlines else result['image']).copy()
    instances=result['instances']
    output=local(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    base_stem=output.stem[:-3] if output.stem.endswith('_编号') else output.stem
    fault_image=np.full_like(image,255)
    fault_image[result['fault_mask']]=np.asarray([237,23,23],np.uint8)
    fault_file=output.with_name(base_stem+'_断层识别.png')
    Image.fromarray(fault_image).save(fault_file)

    # Spatial reading order is deterministic and easier to inspect than the
    # internal colour-grouped component order: top-to-bottom, then left-to-right.
    items=[]
    for record in result['records']:
        mask=instances==record['instance'];yy,xx=np.where(mask)
        if not len(xx):continue
        item=dict(record);item['centroid_pixels']=[round(float(xx.mean()),2),round(float(yy.mean()),2)]
        item['_mask']=mask;items.append(item)
    items.sort(key=lambda item:(item['centroid_pixels'][1],item['centroid_pixels'][0]))

    font=cv2.FONT_HERSHEY_SIMPLEX;scale=max(.48,min(.68,min(image.shape[:2])/3800.))
    thickness=1;occupied=np.zeros(instances.shape,bool);number_map=np.zeros(instances.shape,np.int32)
    rows=[]
    for number,item in enumerate(items,1):
        mask=item.pop('_mask');number_map[mask]=number
        ax,ay,radius=_anchor(mask);text=str(number)
        (tw,th),baseline=cv2.getTextSize(text,font,scale,thickness)
        preferred=(int(round(ax-tw/2)),int(round(ay+th/2)))
        chosen=None;placement='inside'
        preferred_box=_box(preferred,(tw,th))
        # A label stays inside only when the local inscribed radius can contain
        # it and it does not overlap an earlier number.
        if radius>=max(th*.65,tw*.38) and _fits(preferred_box,instances.shape,occupied):
            chosen=preferred
        else:
            placement='callout'
            # Search a short spiral around the actual layer. Small/thin units
            # receive a leader line rather than losing their unique number.
            for distance in range(18,181,12):
                for angle in np.linspace(0,2*np.pi,16,endpoint=False):
                    cx=int(round(ax+np.cos(angle)*distance));cy=int(round(ay+np.sin(angle)*distance))
                    candidate=(int(round(cx-tw/2)),int(round(cy+th/2)))
                    if _fits(_box(candidate,(tw,th)),instances.shape,occupied):
                        chosen=candidate;break
                if chosen is not None:break
        if chosen is None:
            chosen=preferred;placement='overlap_fallback'
        box=_box(chosen,(tw,th));x0,y0,x1,y1=box
        x0=max(0,x0);y0=max(0,y0);x1=min(occupied.shape[1]-1,x1);y1=min(occupied.shape[0]-1,y1)
        occupied[y0:y1+1,x0:x1+1]=True
        centre=(chosen[0]+tw//2,chosen[1]-th//2)
        if placement!='inside':
            cv2.line(image,(ax,ay),centre,(255,255,255),3,cv2.LINE_AA)
            cv2.line(image,(ax,ay),centre,(0,0,0),1,cv2.LINE_AA)
        # White halo plus black glyph remains readable on every lithology colour.
        cv2.putText(image,text,chosen,font,scale,(255,255,255),thickness+3,cv2.LINE_AA)
        cv2.putText(image,text,chosen,font,scale,(0,0,0),thickness,cv2.LINE_AA)
        rows.append({'number':number,'instance_index':item['instance'],
            'lithology_index':item['lithology_index'],'lithology':item['lithology'],
            'area_pixels':item['area_pixels'],'part_count':int(item.get('part_count',1)),'bbox':item['bbox'],
            'centroid_pixels':item['centroid_pixels'],'anchor_pixels':[ax,ay],
            'label_position_pixels':[int(chosen[0]),int(chosen[1])],'placement':placement})

    Image.fromarray(image).save(output)
    # Avoid duplicated “编号_编号” when the requested image name already ends
    # with that suffix.
    np.save(output.with_name(base_stem+'_编号矩阵.npy'),number_map)
    json_file=output.with_name(base_stem+'_编号对照.json')
    json_file.write_text(json.dumps({'layer_count':len(rows),'numbering':'top_to_bottom_then_left_to_right',
        'audit':result['audit'],'layers':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    csv_file=output.with_name(base_stem+'_编号对照.csv')
    with csv_file.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['number','instance_index','lithology_index','lithology',
            'area_pixels','part_count','bbox','centroid_pixels','anchor_pixels','label_position_pixels','placement'])
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({'image':str(output),'fault_image':str(fault_file),'layer_count':len(rows),'json':str(json_file),
        'csv':str(csv_file),'inside_labels':sum(row['placement']=='inside' for row in rows),
        'callouts':sum(row['placement']=='callout' for row in rows),
        'overlap_fallbacks':sum(row['placement']=='overlap_fallback' for row in rows),
        'layer_outlines':not args.no_layer_outlines},ensure_ascii=False))


if __name__=='__main__':main()
