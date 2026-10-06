"""Geological content-domain inference after high-confidence artifact removal."""
import cv2
import numpy as np


def enforce_single_outer_domain(domain,prior=None):
    """Return one closed outer geological domain and reject remote islands."""
    allowed=np.ones(domain.shape,bool) if prior is None else prior.astype(bool)
    candidate=domain.astype(bool)&allowed
    count,cc,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
    if count<=1:return candidate,{'components_before':0,'discarded_pixels':0,'outer_contours':0}
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));main=(cc==largest).astype(np.uint8)
    contours,_=cv2.findContours(main,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    strict=np.zeros(domain.shape,np.uint8)
    if contours:cv2.drawContours(strict,[max(contours,key=cv2.contourArea)],-1,1,-1)
    strict=strict.astype(bool)&allowed
    return strict,{'components_before':count-1,'discarded_pixels':int(np.count_nonzero(candidate&~strict)),
                   'outer_contours':1 if np.any(strict) else 0}

def infer_domain(classified,prior=None,pale_indices=(),artifact_mask=None,boundary_support=None):
    """Build a conservative content domain from cleaned geological support.

    The supplied curve is a maximum permitted domain, not unquestioned truth.
    Non-pale, legend-classified regions provide foreground seeds. Their cleaned
    exterior contours define the domain; enclosed pale/background-coloured
    units are retained without allowing page background to grow indefinitely.
    """
    h,w=classified.shape;allowed=np.ones((h,w),bool) if prior is None else prior.astype(bool)
    artifacts=np.zeros((h,w),bool) if artifact_mask is None else artifact_mask.astype(bool)
    support=(classified>=0)&allowed&~artifacts
    if pale_indices:support&=~np.isin(classified,np.asarray(pale_indices,np.int16))
    # Legend-confirmed geological contacts may coincide with the outer edge or
    # replace colour pixels along a thin bed. Admit only contact pixels near
    # geological colour support; remote legend/table strokes cannot enlarge the
    # drawing domain.
    boundary=np.zeros((h,w),bool) if boundary_support is None else boundary_support.astype(bool)&allowed
    if np.any(boundary):
        near=cv2.dilate(support.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
        support|=boundary&near
    # Remove isolated coloured lettering and line fragments before constructing
    # the domain. Thin true beds remain connected to broader geological support.
    count,cc,stats,_=cv2.connectedComponentsWithStats(support.astype(np.uint8),8)
    core=np.zeros((h,w),np.uint8);area_floor=max(12,round(h*w*.00002))
    for index in range(1,count):
        x,y,bw,bh,area=map(int,stats[index]);ratio=max(bw/max(1,bh),bh/max(1,bw))
        if area>=area_floor and not(ratio>45 and min(bw,bh)<=max(3,round(min(h,w)*.0015))):core[cc==index]=1
    k=max(3,min(17,(round(min(h,w)*.004)|1)))
    core=cv2.morphologyEx(core,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
    # A small support halo joins anti-aliased contacts. Filling external
    # contours then recovers enclosed white units without accepting remote page.
    radius=max(3,min(18,round(min(h,w)*.008)))
    distance=cv2.distanceTransform((core==0).astype(np.uint8),cv2.DIST_L2,5)
    halo=(distance<=radius)&allowed
    contours,_=cv2.findContours(halo.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    domain=np.zeros((h,w),np.uint8)
    if contours:cv2.drawContours(domain,contours,-1,1,-1)
    domain=domain.astype(bool)&allowed
    if np.count_nonzero(domain)<np.count_nonzero(core):domain=core.astype(bool)&allowed
    audit={'method':'cleaned_support_envelope_v2_legend_boundary','prior_used':prior is not None,'support_pixels':int(support.sum()),
           'core_pixels':int(core.sum()),'domain_pixels':int(domain.sum()),'support_halo_pixels':radius,
           'accepted_boundary_support_pixels':int(np.count_nonzero(boundary&support)),
           'excluded_prior_pixels':int(np.count_nonzero(allowed&~domain))}
    return domain,audit
