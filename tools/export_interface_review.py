"""Export the current artefact-removal and geological-interface review plate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.recognition import recognize


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--audit')
    args=parser.parse_args()
    cfg=json.loads(local(args.config).read_text(encoding='utf-8-sig'))
    # Keep this diagnostic bit-for-bit aligned with the standard P101 review
    # exporter. The rough human count remains audit-only and is not needed here.
    cfg['min_component_fraction']=.00005;cfg['min_area_pixels']=60
    cfg.setdefault('artifact_max_gap_pixels',32);cfg.setdefault('artifact_corridor_half_width',14)
    cfg.setdefault('artifact_fill_passes',4);cfg.setdefault('bridge_unclassified_gap_pixels',48)
    cfg.setdefault('internal_gap_max_fraction',.08);cfg.setdefault('pale_lab_tolerance',7)
    diagnostics={};section,labels=recognize(cfg,diagnostics)
    x,y,w,h=cfg['roi']
    with Image.open(local(cfg['image'])) as source:
        base=np.asarray(source.convert('RGB').crop((x,y,x+w,y+h)).resize(
            (labels.shape[1],labels.shape[0]),Image.Resampling.LANCZOS)).copy()

    # Start from verified stratum-instance pixels only. Unclassified printing,
    # labels and page furniture therefore stay white even when OCR missed an
    # individual glyph. Accepted artifacts are then explicitly cleared too.
    review=np.full_like(base,255);valid=labels>=0;review[valid]=base[valid]
    artifact=diagnostics.get('artifact_mask',np.zeros(labels.shape,bool))
    review[artifact]=255
    # Remove original near-black ink before drawing accepted contacts. This is
    # a visualization-safe operation: geological contacts are restored below
    # from instance contours and the protected legend-boundary mask, while
    # missed black labels cannot survive merely because inpainting assigned
    # their pixels to a surrounding stratum.
    base_lab=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    original_black=(base_lab[:,:,0]<38)&(np.linalg.norm(base_lab[:,:,1:],axis=2)<18)
    review[original_black]=255

    # Draw the complete contour of every retained instance. This also covers a
    # geological interface whose original stroke occupies pixels between the
    # two colour regions, so coloured source strokes cannot remain half-visible.
    internal=np.zeros(labels.shape,np.uint8)
    for index in range(int(labels.max())+1):
        # Text holes inside an instance are not geological contacts. Drawing
        # only the external component contour prevents a removed label from
        # reappearing as black outlined characters in the review plate.
        contours,_=cv2.findContours((labels==index).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(internal,contours,-1,1,2,lineType=cv2.LINE_8)
    review[internal>0]=0
    # Only catalogue-confirmed geological strokes are redrawn here. Generic
    # topology-only "protected" paths can still contain unresolved label ink;
    # they remain in the audit mask instead of being presented as accepted.
    protected=diagnostics.get('catalogue_boundary_mask',np.zeros(labels.shape,bool))
    protected=cv2.dilate(protected.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    review[protected]=0

    # The final inferred content-domain outline uses four pixels, exactly twice
    # the nominal internal interface width requested for this review step.
    content=diagnostics.get('content_mask',labels>=0)
    # The review plate is about the geological drawing. Page annotations and
    # the inset legend outside the inferred content outline are blanked so they
    # cannot be mistaken for retained strata or boundaries.
    review[~content]=255
    contours,_=cv2.findContours(content.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(review,contours,-1,(0,0,0),4,lineType=cv2.LINE_8)
    output=local(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(review).save(output)
    summary={'file':str(output),'layers':len(section['layers']),
                      'white_artifact_pixels':int(artifact.sum()),
                      'internal_boundary_pixels':int(np.count_nonzero(internal)),
                      'outer_contours':len(contours),
                      'legend_line_classification':section.get('recognition',{}).get('artifact_preprocessing',{}).get('legend_line_classification',{}),
                      'final_domain_hard_clip':section.get('recognition',{}).get('artifact_preprocessing',{}).get('final_domain_hard_clip',{})}
    if args.audit:
        audit=local(args.audit);audit.parent.mkdir(parents=True,exist_ok=True)
        audit.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
