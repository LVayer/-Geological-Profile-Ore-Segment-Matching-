"""Mask-derived physical geometry; vertical thickness is not true bed thickness."""
import cv2
import numpy as np

def resample(points, n=64, closed=False):
    p = np.asarray(points, float)
    if closed:
        p = np.vstack([p, p[0]])
    d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(d) > 1e-12]
    p, d = p[keep], d[keep]
    if len(p) < 2:
        return np.repeat(p[:1], n, axis=0).tolist()
    t = np.linspace(0, d[-1], n, endpoint=not closed)
    return np.column_stack([np.interp(t, d, p[:, k]) for k in range(2)]).tolist()

def extract(mask, layer_id, lithology, bounds, quality=1., n=64):
    h, w = mask.shape
    x0,y0,x1,y1 = bounds
    sx,sy = (x1-x0)/w,(y1-y0)/h
    yy,xx = np.where(mask)
    if not len(xx):
        raise ValueError('Empty layer mask')
    def world(p):
        p = np.asarray(p,float)
        return np.column_stack([x0+(p[:,0]+.5)*sx,y0+(p[:,1]+.5)*sy])
    cols = np.unique(xx)
    upper = np.array([[c,np.where(mask[:,c])[0][0]] for c in cols])
    lower = np.array([[c,np.where(mask[:,c])[0][-1]] for c in cols])
    thick = (lower[:,1]-upper[:,1]+1)*sy
    # Multiple intervals in one column expose overhangs/folds: envelope is ambiguous.
    multi = sum(np.count_nonzero(np.diff(np.r_[0,mask[:,c],0].astype(int)) == 1)>1 for c in cols)/len(cols)
    contours,hierarchy = cv2.findContours(mask.astype('uint8'),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    rings = [world(c[:,0,:]) for c in contours if len(c)>=3]
    if not rings:
        raise ValueError('Layer too small for an outline')
    perimeter = sum(np.linalg.norm(np.diff(np.vstack([r,r[0]]),axis=0),axis=1).sum() for r in rings)
    main = max(rings,key=len)
    mid = (world(upper)+world(lower))/2
    slope = np.polyfit(mid[:,0],mid[:,1],1)[0] if len(cols)>1 else 0.
    edge = max(1,len(thick)//10)
    taper = float(min(np.mean(thick[:edge]),np.mean(thick[-edge:]))/max(np.mean(thick),1e-8))
    return dict(id=layer_id,lithology=lithology,order=0,centroid=world([[xx.mean(),yy.mean()]])[0].tolist(),
        area=float(len(xx)*sx*sy),perimeter=float(perimeter),extent=[float(x0+xx.min()*sx),float(y0+yy.min()*sy),float(x0+(xx.max()+1)*sx),float(y0+(yy.max()+1)*sy)],
        width=float(len(cols)*sx),mean_thickness=float(thick.mean()),thickness_variation=float(thick.std()/thick.mean()),
        thickness_profile=np.column_stack([world(upper)[:,0],thick]).tolist(),dip=float(np.degrees(np.arctan(slope))),
        upper_boundary=resample(world(upper),n),lower_boundary=resample(world(lower),n),outline=resample(main,n,True),
        outline_rings=[resample(r,n,True) for r in rings],taper_ratio=taper,
        quality=float(min(quality,.55 if multi>.05 else 1.)),geometry_flags=['multi_interval_columns'] if multi>.05 else [],
        upper_neighbors=[],lower_neighbors=[],other_contact_neighbors=[],local_topology={})

def topology(layers,masks,gap=3):
    # A small gap tolerates exported boundary strokes; large faults remain barriers.
    kernel=np.ones((2*gap+1,2*gap+1),np.uint8)
    label_map=masks if isinstance(masks,np.ndarray) and masks.ndim==2 else None
    # A label map avoids retaining one full-image Boolean array per layer. This
    # matters for large exported sections containing dozens of disconnected beds.
    dilated=None if label_map is not None else [cv2.dilate(m.astype('uint8'),kernel).astype(bool) for m in masks]
    sizes=np.bincount(label_map[label_map>=0],minlength=len(layers)) if label_map is not None else None
    for i,a in enumerate(layers):
        if label_map is not None:
            # Restrict morphology to the component bounding box plus its contact
            # margin. Scanning the complete multi-million-pixel map once per
            # layer made fragmented real drawings unreasonably slow.
            bx,by,bw,bh=a.pop('_pixel_bbox')
            x0=max(0,bx-gap);y0=max(0,by-gap);x1=min(label_map.shape[1],bx+bw+gap);y1=min(label_map.shape[0],by+bh+gap)
            window=label_map[y0:y1,x0:x1]
            expanded=cv2.dilate((window==i).astype('uint8'),kernel).astype(bool)
            ids,counts=np.unique(window[expanded & (window>=0)],return_counts=True)
            contacts={int(j):int(n) for j,n in zip(ids,counts) if int(j)!=i}
        else:contacts={j:int(np.count_nonzero(dilated[i]&masks[j])) for j in range(len(layers)) if j!=i}
        for j,contact in contacts.items():
            b=layers[j]
            target_size=int(sizes[j]) if label_map is not None else int(np.count_nonzero(masks[j]))
            if contact < max(3,.002*target_size): continue
            dy=b['centroid'][1]-a['centroid'][1]
            dx=abs(b['centroid'][0]-a['centroid'][0])
            key='other_contact_neighbors' if dx>abs(dy)*2 else ('lower_neighbors' if dy>0 else 'upper_neighbors')
            a[key].append(b['id'])
        a['local_topology']={k:len(a[k]) for k in ['upper_neighbors','lower_neighbors','other_contact_neighbors']}
    for order,a in enumerate(sorted(layers,key=lambda x:(x['centroid'][1],x['centroid'][0]))):
        a['order']=order
