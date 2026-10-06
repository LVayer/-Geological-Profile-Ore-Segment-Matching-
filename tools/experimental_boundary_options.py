"""Render isolated outer-domain experiments without changing the production pipeline.

These previews address a special ambiguity: a boundary lithology can have the
same colour as the page background.  They deliberately consume the already
exported production number map as a conservative seed, then try four different
ways to extend only the missing right-hand domain.  No result is written back
to a project config or to ``stratamatch``.
"""
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


def _solid_outer(mask: np.ndarray) -> np.ndarray:
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    out=np.zeros(mask.shape,np.uint8)
    if contours:cv2.fillPoly(out,[max(contours,key=cv2.contourArea)],1)
    return out.astype(bool)


def _domain_from_outline(path: Path,extended_width: int) -> np.ndarray:
    """Recover the closed production domain without using fault-split labels."""
    review=np.asarray(Image.open(path).convert('RGB'))
    dark=np.max(review,axis=2)<55
    barrier=cv2.dilate(dark.astype(np.uint8),np.ones((3,3),np.uint8))>0
    passable=(~barrier).astype(np.uint8)
    flood=passable.copy();mask=np.zeros((flood.shape[0]+2,flood.shape[1]+2),np.uint8)
    cv2.floodFill(flood,mask,(0,0),2)
    enclosed=flood!=2
    contours,_=cv2.findContours(enclosed.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    current=np.zeros(enclosed.shape,np.uint8)
    if contours:cv2.fillPoly(current,[max(contours,key=cv2.contourArea)],1)
    expanded=np.zeros((current.shape[0],extended_width),bool);expanded[:,:current.shape[1]]=current>0
    return expanded


def _smooth(values: np.ndarray, valid: np.ndarray, width: int=41) -> np.ndarray:
    x=np.arange(len(values));result=values.astype(float).copy()
    if valid.sum()>=2:result[~valid]=np.interp(x[~valid],x[valid],result[valid])
    k=max(5,width|1);return cv2.GaussianBlur(result.reshape(1,-1),(k,1),0).ravel()


def _edges(mask: np.ndarray):
    valid=mask.any(axis=0);top=np.zeros(mask.shape[1],float);bottom=np.zeros_like(top)
    for x in np.where(valid)[0]:
        ys=np.where(mask[:,x])[0];top[x]=ys[0];bottom[x]=ys[-1]
    return valid,_smooth(top,valid),_smooth(bottom,valid)


def _colour_support(rgb: np.ndarray,legend: list[dict]) -> np.ndarray:
    lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    best=np.full(rgb.shape[:2],np.inf,np.float32)
    for entry in legend:
        colour=np.asarray(entry.get('rgb',[255,255,255]),np.float32)
        proto=cv2.cvtColor(np.float32([[colour/255]]),cv2.COLOR_RGB2LAB)[0,0]
        # Pure white remains ambiguous and is handled by geometry, not colour.
        if proto[0]>=96 and np.linalg.norm(proto[1:])<5:continue
        best=np.minimum(best,np.linalg.norm(lab-proto,axis=2))
    # JPEG page white can fall within a loose tolerance of a pale-pink unit.
    # Require the lithology prototype to beat the measured page background.
    border=np.concatenate((rgb[:20].reshape(-1,3),rgb[-20:].reshape(-1,3),
                           rgb[:,-20:].reshape(-1,3)),axis=0)
    background=np.median(border,axis=0).astype(np.float32)
    bg_proto=cv2.cvtColor(np.float32([[background/255]]),cv2.COLOR_RGB2LAB)[0,0]
    bg_distance=np.linalg.norm(lab-bg_proto,axis=2)
    return (best<13)&(best+2.5<bg_distance)&(bg_distance>4)


def _render(rgb: np.ndarray,mask: np.ndarray,path: Path,uncertain: np.ndarray|None=None):
    image=np.full_like(rgb,255);image[mask]=rgb[mask]
    if uncertain is not None:
        blend=image[uncertain].astype(np.float32)*.45+np.asarray([255,210,0],np.float32)*.55
        image[uncertain]=np.uint8(np.clip(blend,0,255))
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image,contours,-1,(0,0,0),max(2,round(min(mask.shape)*.0015)),cv2.LINE_AA)
    Image.fromarray(image).save(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--number-map',required=True)
    parser.add_argument('--outline-image',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(local(args.config).read_text(encoding='utf-8-sig'))
    output=local(args.output);output.mkdir(parents=True,exist_ok=True)
    source=Image.open(local(cfg['image'])).convert('RGB');page_w,page_h=source.size
    rx,ry,rw,rh=cfg['roi'];numbers=np.load(local(args.number_map),mmap_mode='r')
    scale=numbers.shape[0]/rh;extended_w=round((page_w-rx)*scale)
    crop=source.crop((rx,ry,page_w,min(page_h,ry+rh))).resize((extended_w,numbers.shape[0]),Image.Resampling.LANCZOS)
    rgb=np.asarray(crop);h,w=rgb.shape[:2]
    # The number map is split at every confirmed fault and therefore cannot be
    # used to recover a single domain.  Read the already closed production
    # outline instead; the map is retained only to preserve the exact scale.
    seed=_domain_from_outline(local(args.outline_image),w)
    support=_colour_support(rgb,cfg['legend'])
    # A detected legend may lie inside the rectangular plot ROI.  It is page
    # furniture, so exclude its reviewed rectangle from all experiments.
    exclusion=np.zeros((h,w),bool)
    if cfg.get('legend_exclusion_box'):
        x,y,bw,bh=cfg['legend_exclusion_box'];x0=round((x-rx)*scale);y0=round((y-ry)*scale)
        exclusion[max(0,y0):min(h,round(y0+bh*scale)),max(0,x0):min(w,round(x0+bw*scale))]=True
        support[exclusion]=False
    support=cv2.morphologyEx(support.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))>0
    valid,base_top,base_bottom=_edges(seed);base_end=int(np.where(valid)[0].max())
    # Form a decaying-tangent corridor from the last reliable boundary.  Only
    # colour evidence inside this corridor may extend the domain, which keeps
    # legends, frames and title blocks out of the search.
    recent=np.where(valid & (np.arange(w)>=max(0,base_end-360)))[0]
    sample=np.where(valid & (np.arange(w)>=max(0,base_end-150)))[0]
    top_slope=float(np.median(np.diff(base_top[sample]))) if len(sample)>3 else 0.
    bottom_slope=float(np.median(np.diff(base_bottom[sample]))) if len(sample)>3 else 0.
    predicted_top=np.full(w,base_top[base_end],float);predicted_bottom=np.full(w,base_bottom[base_end],float)
    for x in range(base_end+1,w):
        decay=np.exp(-(x-base_end)/260.)
        predicted_top[x]=predicted_top[x-1]+np.clip(top_slope,-2.5,2.5)*decay
        predicted_bottom[x]=predicted_bottom[x-1]+np.clip(bottom_slope,-3.5,3.5)*decay
    evidence=np.zeros(w,bool)
    for x in range(base_end+1,w):
        lo=max(0,round(predicted_top[x])-70);hi=min(h,round(predicted_bottom[x])+90)
        evidence[x]=hi>lo and np.count_nonzero(support[lo:hi,x])>=max(3,round((hi-lo)*.008))
    # Bridge background-coloured beds, but stop before page furniture after a
    # sustained run without geological colour evidence.
    last=base_end;gap=0;max_gap=max(45,round(h*.035))
    for x in range(base_end+1,w-8):
        if evidence[x]:last=x;gap=0
        else:gap+=1
        if gap>max_gap:break
    target=last
    target=max(base_end,min(w-2,target))

    # Option 1: use observed non-white lithology pixels as a vertical envelope.
    option1=seed.copy();top1=base_top.copy();bottom1=base_bottom.copy()
    thickness=float(np.median(base_bottom[recent]-base_top[recent])) if len(recent) else h*.2
    for x in range(max(0,base_end-15),target+1):
        x0=max(0,x-5);x1=min(w,x+6)
        lo=max(0,round(predicted_top[x])-70);hi=min(h,round(predicted_bottom[x])+90)
        ys=np.where(support[lo:hi,x0:x1])[0]+lo
        if len(ys):top1[x]=np.percentile(ys,2);bottom1[x]=np.percentile(ys,98)
        elif x>base_end:
            top1[x]=top1[x-1];bottom1[x]=bottom1[x-1]
        if bottom1[x]-top1[x]<thickness*.22:bottom1[x]=top1[x]+thickness
    rng=np.arange(max(0,base_end-15),target+1);top1[rng]=_smooth(top1[rng],np.ones(len(rng),bool),31)
    bottom1[rng]=_smooth(bottom1[rng],np.ones(len(rng),bool),31)
    for x in range(base_end+1,target+1):option1[max(0,round(top1[x])):min(h,round(bottom1[x])+1),x]=True
    option1=_solid_outer(option1)

    # Option 2: extrapolate the reliable top/bottom curves, then snap them to
    # strong local image gradients so a background-coloured unit can be kept.
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gx=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0));gy=np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1))
    # Cross-section outer surfaces are predominantly horizontal/oblique.
    # Downweight vertical drill/grid strokes whose response is mostly gx.
    gradient=gy+.28*gx
    gradient[exclusion]=0
    option2=seed.copy();last_t=int(base_top[base_end]);last_b=int(base_bottom[base_end])
    for x in range(base_end+1,target+1):
        # The colour envelope is a soft geometric prior, while the local
        # gradient supplies sub-pixel boundary placement.
        pred_t=float(.45*predicted_top[x]+.55*top1[x])
        pred_b=float(.45*predicted_bottom[x]+.55*bottom1[x])
        pred_t=np.clip(pred_t,last_t-5,last_t+5);pred_b=np.clip(pred_b,last_b-5,last_b+5)
        def snap(pred,last):
            lo=max(0,round(pred)-18);hi=min(h,round(pred)+19);scores=gradient[lo:hi,x]
            if len(scores) and scores.max()>18:
                yy=np.arange(lo,hi);rank=scores-1.4*np.abs(yy-pred)-.7*np.abs(yy-last)
                candidate=lo+int(np.argmax(rank));return int(np.clip(candidate,last-4,last+4))
            return int(round(pred))
        last_t=snap(pred_t,last_t);last_b=snap(pred_b,last_b)
        if last_b-last_t<thickness*.25:last_b=round(last_t+thickness)
        option2[max(0,last_t):min(h,last_b+1),x]=True
    option2=_solid_outer(option2)

    # Option 3: water grows only inside the broad union corridor; black,
    # magenta and red line evidence acts as a barrier.  Open leaks therefore do
    # not authorize growth across the page.
    corridor=cv2.dilate((option1|option2).astype(np.uint8),np.ones((11,11),np.uint8))>0
    dark=np.max(rgb,axis=2)<85
    red=(rgb[:,:,0]>170)&(rgb[:,:,1]<105)&(rgb[:,:,2]<130)
    magenta=(rgb[:,:,0]>180)&(rgb[:,:,2]>160)&(rgb[:,:,1]<150)
    barrier=cv2.dilate((dark|red|magenta).astype(np.uint8),np.ones((3,3),np.uint8))>0
    traversable=corridor&~barrier&~exclusion
    n,cc=cv2.connectedComponents(traversable.astype(np.uint8),8)
    seed_core=cv2.erode(seed.astype(np.uint8),np.ones((9,9),np.uint8))>0
    keep=np.zeros(n,bool)
    ids=np.unique(cc[seed_core]);keep[ids]=True;keep[0]=False
    option3=_solid_outer(keep[cc]|seed)

    votes=option1.astype(np.uint8)+option2.astype(np.uint8)+option3.astype(np.uint8)
    option4=_solid_outer(votes>=2);uncertain=(votes>0)&(votes<3)&~seed
    _render(rgb,option1,output/'方案1_ROI扩展与颜色包络.png')
    _render(rgb,option2,output/'方案2_曲线延拓与梯度吸附.png')
    _render(rgb,option3,output/'方案3_受约束注水.png')
    _render(rgb,option4,output/'方案4_混合共识与不确定区.png',uncertain)
    report={'image':Path(cfg['image']).name,'working_shape':[h,w],'production_width':int(numbers.shape[1]),
            'production_domain_right':base_end,'evidence_target_right':target,
            'added_pixels':{'option1':int((option1&~seed).sum()),'option2':int((option2&~seed).sum()),
                            'option3':int((option3&~seed).sum()),'consensus':int((option4&~seed).sum())},
            'uncertain_pixels':int(uncertain.sum()),'production_unchanged':True}
    (output/'方案审计.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
