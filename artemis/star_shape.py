"""Conservative local correction of measured point-source elongation."""
from concurrent.futures import CancelledError

import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree

from .color import smoothstep


def apply(master, points, limb, affine, *, viewport=None, cfg=None, cancel=None):
    """Circularize reliable isolated sources without synthesizing new structure.

    The measured local residual is geometrically transformed with an area-
    preserving matrix; its integrated luminance is renormalized. Resampling is
    inherently not bit-exact, so ambiguous, crowded and limb-adjacent sources
    are deliberately left untouched. Coordinates refer to the full master.
    """
    cfg=cfg or {}
    size=int(cfg.get('radius_px', 10))
    min_ratio=float(cfg.get('minimum_axis_ratio', 1.12))
    max_ratio=float(cfg.get('maximum_axis_ratio', 2.5))
    min_snr=float(cfg.get('minimum_snr', 4.5))
    if size<5 or not 1<=min_ratio<max_ratio:
        raise ValueError('Invalid star correction settings')
    h,w=master.shape[:2]
    origin=np.array(viewport[:2] if viewport else (0,0))
    points=np.asarray(points,dtype=float).reshape(-1,2)-origin
    valid=np.isfinite(points).all(axis=1)
    rounded=np.rint(points)
    valid &= (rounded[:,0]>=size)&(rounded[:,0]<w-size)&(rounded[:,1]>=size)&(rounded[:,1]<h-size)
    points=points[valid]
    inverse=cv.invertAffineTransform(affine)
    yy,xx=np.mgrid[-size:size+1,-size:size+1].astype(np.float32)
    rr=np.hypot(xx,yy)
    annulus=rr>=size-2
    window=1-smoothstep((rr-(size-3))/3)
    aperture=np.exp(-.5*(rr/4.2)**2)
    crowded=np.zeros(len(points),bool)
    if len(points)>1:
        pairs=cKDTree(points).query_pairs(2*size+1,output_type='ndarray')
        if len(pairs):
            crowded[pairs.ravel()]=True
    counts={'enabled':True,'examined':int(len(points)),'corrected':0,
            'already_round':0,'ambiguous':0,'crowded':int(crowded.sum()),'parameters':dict(cfg)}
    for i,(x,y) in enumerate(points):
        if cancel is not None and i%64==0 and cancel.is_set():
            raise CancelledError()
        if crowded[i]:
            continue
        full=np.array([x+origin[0],y+origin[1],1.])
        native=inverse@full
        if limb.radius(*native)<1+(size+8)/max(limb.axes):
            counts['ambiguous']+=1
            continue
        ix,iy=round(x),round(y)
        tile=master[iy-size:iy+size+1,ix-size:ix+size+1]
        image=tile.astype(np.float32)
        if np.any(image>=65530):
            counts['ambiguous']+=1
            continue
        gray=image[...,0]*.2126+image[...,1]*.7152+image[...,2]*.0722
        background=np.median(image[annulus],axis=0)
        bg_luma=float(background@np.array([.2126,.7152,.0722],np.float32))
        sigma=1.4826*float(np.median(np.abs(gray[annulus]-np.median(gray[annulus]))))+1
        weight=np.maximum(gray-bg_luma-1.2*sigma,0)*aperture
        total=float(weight.sum())
        peak=float(np.max(gray-bg_luma))
        if peak<min_snr*sigma or total<=0:
            counts['ambiguous']+=1
            continue
        cx=float((weight*xx).sum()/total)
        cy=float((weight*yy).sum()/total)
        if np.hypot(cx,cy)>3.5:
            counts['ambiguous']+=1
            continue
        dx,dy=xx-cx,yy-cy
        covariance=np.array([[(weight*dx*dx).sum(),(weight*dx*dy).sum()],
                             [(weight*dx*dy).sum(),(weight*dy*dy).sum()]])/total
        values,axes=np.linalg.eigh(covariance)
        if values[0]<.18:
            counts['ambiguous']+=1
            continue
        ratio=float(np.sqrt(values[1]/values[0]))
        if ratio<min_ratio:
            counts['already_round']+=1
            continue
        if ratio>max_ratio:
            counts['ambiguous']+=1
            continue
        # A has determinant one: contract the major axis and expand the minor.
        factor=ratio**.5
        A=axes@np.diag([factor,1/factor])@axes.T
        center=np.array([size+cx,size+cy])
        transform=np.column_stack((A,center-A@center)).astype(np.float32)
        signal=(image-background)*window[...,None]
        corrected=cv.warpAffine(signal,transform,(2*size+1,2*size+1),flags=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_CONSTANT,borderValue=0)
        weights=np.array([.2126,.7152,.0722],np.float32)
        before=float(signal.sum(axis=(0,1))@weights)
        after=float(corrected.sum(axis=(0,1))@weights)
        if before<=0 or after<=0 or not .8<before/after<1.25:
            counts['ambiguous']+=1
            continue
        corrected*=np.float32(before/after)
        tile[:]=np.rint(np.clip(image-signal+corrected,0,65535)).astype(np.uint16)
        counts['corrected']+=1
    return counts
