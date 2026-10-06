"""Regenerate the P101 layer-count review artifacts from an audited config.

This is intentionally a reusable exporter: the structured label array is the
source of truth, while all PNGs and tables are derived from the same array.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.recognition import recognize


def palette(count: int) -> np.ndarray:
    """Create deterministic colours with strong adjacent-instance contrast."""
    colours=[]
    for index in range(count):
        hue=(index*.61803398875)%1.0
        saturation=.68+.24*((index%3)/2)
        value=.76+.21*(((index//3)%2))
        hsv=np.uint8([[[round(hue*179),round(saturation*255),round(value*255)]]])
        colours.append(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)[0,0])
    return np.asarray(colours,np.uint8)


def blank_holes(labels: np.ndarray) -> tuple[int,int]:
    """Count unassigned pixels enclosed by an individual layer contour."""
    holes=0;pixels=0
    for index in range(int(labels.max())+1):
        mask=(labels==index).astype(np.uint8)
        contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:continue
        for contour,node in zip(contours,hierarchy[0]):
            if node[3]<0:continue
            inside=np.zeros(labels.shape,np.uint8);cv2.drawContours(inside,[contour],-1,1,-1)
            blank=int(np.count_nonzero((inside>0)&(labels<0)))
            if blank:holes+=1;pixels+=blank
    return holes,pixels


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--count',type=int,default=112)
    parser.add_argument('--min-component-fraction',type=float,default=.00005)
    parser.add_argument('--min-area-pixels',type=int,default=60)
    args=parser.parse_args()
    config_path=local(args.config);output=local(args.output);output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(config_path.read_text(encoding='utf-8-sig'))
    cfg['expected_layer_count']=args.count
    # The rough count is audit-only. These scale-aware general defaults control
    # noise independently and are suitable for images of different sizes.
    cfg['min_component_fraction']=args.min_component_fraction
    cfg['min_area_pixels']=args.min_area_pixels
    cfg.setdefault('artifact_max_gap_pixels',32);cfg.setdefault('artifact_corridor_half_width',14);cfg.setdefault('artifact_fill_passes',4)
    cfg.setdefault('bridge_unclassified_gap_pixels',48);cfg.setdefault('internal_gap_max_fraction',.08)
    cfg.setdefault('pale_lab_tolerance',7)
    diagnostics={};section,labels=recognize(cfg,diagnostics)
    colours=palette(len(section['layers']))
    flat=np.full((*labels.shape,3),255,np.uint8);valid=labels>=0;flat[valid]=colours[labels[valid]]
    boundary=valid&~cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    flat[boundary]=0
    Image.fromarray(flat).save(output/'地层划分_独立颜色.png')

    x,y,w,h=cfg['roi']
    with Image.open(local(cfg['image'])) as source:
        base=np.asarray(source.convert('RGB').crop((x,y,x+w,y+h)).resize((labels.shape[1],labels.shape[0]),Image.Resampling.LANCZOS)).copy()
    overlay=base.copy();overlay[valid]=(base[valid]*.23+flat[valid]*.77).astype(np.uint8);overlay[boundary]=0
    Image.fromarray(overlay).save(output/'地层划分_叠加原图.png')

    # Two diagnostic views keep line-like artifacts, text, and unexplained
    # regions separate so reviewers can detect both missed and excessive repair.
    artifact_view=base.copy()
    line=diagnostics.get('line_mask',np.zeros(labels.shape,bool));text=diagnostics.get('text_mask',np.zeros(labels.shape,bool))
    artifact_view[line]=(artifact_view[line]*.25+np.array([30,210,255])*.75).astype(np.uint8)
    artifact_view[text]=(artifact_view[text]*.25+np.array([255,145,35])*.75).astype(np.uint8)
    protected=diagnostics.get('protected_boundary_mask',np.zeros(labels.shape,bool))
    undecided=diagnostics.get('uncertain_line_mask',np.zeros(labels.shape,bool))
    artifact_view[protected]=(artifact_view[protected]*.2+np.array([40,220,90])*.8).astype(np.uint8)
    artifact_view[undecided]=(artifact_view[undecided]*.2+np.array([235,45,210])*.8).astype(np.uint8)
    content=diagnostics.get('content_mask')
    if content is not None:
        edge=content&~cv2.erode(content.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        artifact_view[edge]=[255,240,20]
    Image.fromarray(artifact_view).save(output/'主图轮廓与杂项分类.png')
    uncertain_view=base.copy();uncertain=diagnostics.get('uncertainty_map',np.zeros(labels.shape,np.uint8))
    for value,colour in ((1,[70,135,255]),(2,[255,155,30]),(3,[230,45,170]),(4,[255,40,40])):
        mask=uncertain==value;uncertain_view[mask]=(uncertain_view[mask]*.2+np.asarray(colour)*.8).astype(np.uint8)
    Image.fromarray(uncertain_view).save(output/'未分类区域诊断.png')

    # A deliberately simple review plate for the current step: show exactly
    # which pixels the real pipeline regards as removable drawing artefacts,
    # then redraw only stratum-to-stratum interfaces and the final domain edge.
    # The outer edge is twice the nominal two-pixel internal interface width.
    interface_review=base.copy();accepted_artifact=diagnostics.get('artifact_mask',np.zeros(labels.shape,bool))
    interface_review[accepted_artifact]=255
    internal=np.zeros(labels.shape,np.uint8)
    for index in range(int(labels.max())+1):
        layer_contours,_=cv2.findContours((labels==index).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(internal,layer_contours,-1,1,2,lineType=cv2.LINE_8)
    interface_review[internal>0]=0
    if content is not None:
        contours,_=cv2.findContours(content.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(interface_review,contours,-1,(0,0,0),4,lineType=cv2.LINE_8)
    Image.fromarray(interface_review).save(output/'杂项清除与界面认定.png')

    numbered=Image.fromarray(flat.copy());draw=ImageDraw.Draw(numbered);font=ImageFont.load_default()
    mapping=section['recognition']['label_id_mapping'];id_to_layer={layer['id']:layer for layer in section['layers']}
    rows=[]
    for index in range(len(mapping)):
        layer=id_to_layer[mapping[str(index)]];ys,xs=np.where(labels==index)
        px=int(np.median(xs));py=int(np.median(ys));display=f'L{index+1:03d}'
        box=draw.textbbox((px,py),display,font=font,anchor='mm');draw.rectangle(box,fill='white',outline='black')
        draw.text((px,py),display,fill='black',font=font,anchor='mm')
        rows.append({'display_id':display,'layer_id':layer['id'],'test_legend_class':layer['lithology'],
                     'working_pixels':int(len(xs)),'area':layer['area'],'centroid_x':layer['centroid'][0],
                     'centroid_y':layer['centroid'][1],'quality':layer['quality'],
                     'color_rgb':[int(v) for v in colours[index]]})
    numbered.save(output/'地层划分_编号图.png')
    np.save(output/'地层标签.npy',labels)
    with (output/'地层清单.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (output/'地层清单.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    holes,hole_pixels=blank_holes(labels)
    audit={'source_image':cfg['image'],'human_estimate':args.count,
           'result_count':len(section['layers']),'enclosed_blank_holes':holes,'enclosed_blank_pixels':hole_pixels,
           'recognition':section['recognition'],
           'legend_note':'TEST_LITH labels are colour-class placeholders, not verified geological names',
           'colour_policy':'deterministic high-contrast RGB per instance; black instance boundaries',
           'diagnostic_legend':{'主图轮廓与杂项分类.png':{'yellow':'content boundary','cyan':'accepted grid/borehole/linear artifact','orange':'text artifact','green':'protected geological boundary','magenta':'uncertain line path'},
                                '未分类区域诊断.png':{'blue':'content-boundary uncertainty','orange':'unresolved artifact neighborhood','magenta':'interior uncertainty','red':'reconstructed geological seam through removed artifact'}},
           'interface_review':{'file':'杂项清除与界面认定.png','white':'accepted text/grid/borehole/technical artifacts',
                               'black_internal':'boundary between two retained stratum instances (2-pixel shared edge)',
                               'black_outer':'final content-domain outline (4 pixels)'},
           'files':['地层划分_独立颜色.png','地层划分_叠加原图.png','地层划分_编号图.png','主图轮廓与杂项分类.png','未分类区域诊断.png','杂项清除与界面认定.png','地层清单.csv','地层清单.json','地层标签.npy']}
    (output/'参数审计.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'layers':len(section['layers']),'holes':holes,'hole_pixels':hole_pixels,
                      'post_filter_fill':section['recognition']['post_filter_gap_fill']},ensure_ascii=False))


if __name__=='__main__':main()
