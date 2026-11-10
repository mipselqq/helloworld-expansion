"""Optional black-sky finish that protects the Earth, atmospheric rim and sources."""
from concurrent.futures import CancelledError

import cv2 as cv
import numpy as np

from .color import smoothstep, srgb_decode, srgb_encode


def apply(master, limb, affine, points, venus, *, catalog_points=(),
          viewport=None, sample=1, cfg=None, cancel=None):
    """Apply after grain/sharpening; retain the planet and visible source light.

    Source positions are in full master coordinates. Recover catalog emission
    in linear light, matching how it was originally added to the synthetic sky.
    """
    cfg = cfg or {}
    start, end = cfg.get('limb_start', 1.025), cfg.get('limb_end', 1.075)
    core, fade = cfg.get('star_core_px', 3.5), cfg.get('star_fade_px', 5.)
    vcore, vfade = cfg.get('venus_core_px', 20.), cfg.get('venus_fade_px', 28.)
    if not (0 < start < end and core >= 0 and fade > 0 and vcore >= 0 and vfade > 0):
        raise ValueError('Invalid AMOLED mask parameters')
    h, w = master.shape[:2]
    x0, y0 = viewport[:2] if viewport else (0, 0)
    origin = np.array([x0, y0], dtype=float) + (sample-1)/2
    positions = (np.asarray(points, dtype=float).reshape(-1, 2)-origin)/sample
    catalog = (np.asarray(catalog_points, dtype=float).reshape(-1, 2)-origin)/sample
    venus = (np.asarray(venus, dtype=float)-origin)/sample
    core, fade, vcore, vfade = (v/sample for v in (core, fade, vcore, vfade))
    inverse = cv.invertAffineTransform(affine)
    grain_sigma=float(cfg.get('grain_highpass_px', 3.0))/sample
    if grain_sigma <= 0:
        raise ValueError('Grain high-pass scale must be positive')
    # Sample all backgrounds before modifying any tiles, including at tile edges.
    sources=[]
    apertures = ([(p,core,fade,False) for p in positions]+[(venus,vcore,vfade,False)]+
                 [(p,core,fade,True) for p in catalog])
    for point,inner,outer,is_catalog in apertures:
        px,py=point
        if not np.isfinite(px) or not np.isfinite(py):
            continue
        radius=inner+outer
        if px < -radius or px >= w+radius or py < -radius or py >= h+radius:
            continue
        ring_outer=max(3., radius+5/sample)
        left,right=max(0,int(np.floor(px-ring_outer))),min(w,int(np.ceil(px+ring_outer))+1)
        top,bottom=max(0,int(np.floor(py-ring_outer))),min(h,int(np.ceil(py+ring_outer))+1)
        if left>=right or top>=bottom:
            continue
        xx=np.arange(left,right)[None,:]-px
        yy=np.arange(top,bottom)[:,None]-py
        distance=np.hypot(xx,yy)
        annulus=(distance>=max(radius,ring_outer-4/sample)) & (distance<=ring_outer)
        if not annulus.any():
            annulus=np.ones(distance.shape,bool)
        patch=master[top:bottom,left:right].astype(np.float32)
        if is_catalog:
            background=srgb_decode(np.median(patch[annulus],axis=0)/65535)
            noise=0.
        else:
            gray=patch[...,0]*.2126+patch[...,1]*.7152+patch[...,2]*.0722
            background=float(np.median(gray[annulus]))
            noise=.5*1.4826*float(np.median(np.abs(gray[annulus]-background)))
        sources.append((px,py,inner,outer,is_catalog,background,noise))
    # Keep an untouched copy so Gaussian support crosses tile boundaries and
    # the original small-scale texture remains continuous everywhere.
    source=master.copy()
    pad=max(2,int(np.ceil(4.5*grain_sigma)))
    x = np.arange(w, dtype=np.float32)[None,:]*sample+origin[0]
    total_black = 0
    for y0_tile in range(0, h, 96):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        y1_tile = min(h, y0_tile+96)
        y = np.arange(y0_tile,y1_tile,dtype=np.float32)[:,None]*sample+origin[1]
        nx = inverse[0,0]*x+inverse[0,1]*y+inverse[0,2]
        ny = inverse[1,0]*x+inverse[1,1]*y+inverse[1,2]
        keep = 1-smoothstep((limb.radius(nx,ny)-start)/(end-start))
        lo,hi=max(0,y0_tile-pad),min(h,y1_tile+pad)
        work=source[lo:hi].astype(np.float32)
        tile=work[y0_tile-lo:y1_tile-lo]
        blurred=cv.GaussianBlur(work,(0,0),grain_sigma)[y0_tile-lo:y1_tile-lo]
        # The zero baseline cannot represent negative deviations; keep the
        # positive part of the measured residual without adding a gray floor.
        grain=np.maximum(tile-blurred,0)
        result=tile*keep[...,None]+grain*(1-keep[...,None])
        for px,py,inner,outer,is_catalog,background,noise in sources:
            limit=inner+outer
            if py < y0_tile-limit or py >= y1_tile+limit:
                continue
            left,right=max(0,int(np.floor(px-limit))),min(w,int(np.ceil(px+limit))+1)
            top,bottom=max(y0_tile,int(np.floor(py-limit))),min(y1_tile,int(np.ceil(py+limit))+1)
            if left>=right or top>=bottom:
                continue
            xx=np.arange(left,right,dtype=np.float32)[None,:]-px
            yy=np.arange(top,bottom,dtype=np.float32)[:,None]-py
            fade_mask=1-smoothstep((np.hypot(xx,yy)-inner)/outer)
            patch=tile[top-y0_tile:bottom-y0_tile,left:right]
            if is_catalog:
                light=np.maximum(srgb_decode(patch/65535)-background,0)
                emission=srgb_encode(light)*(65535*fade_mask[...,None])
            else:
                gray=patch[...,0]*.2126+patch[...,1]*.7152+patch[...,2]*.0722
                strength=np.clip((gray-background-noise)/np.maximum(gray,1),0,1)*fade_mask
                emission=patch*strength[...,None]
            section=result[top-y0_tile:bottom-y0_tile,left:right]
            np.maximum(section,emission,out=section)
        output=master[y0_tile:y1_tile]
        output[:]=np.rint(np.clip(result,0,65535)).astype(np.uint16)
        total_black+=int(np.count_nonzero(np.all(output==0,axis=-1)))
    return {'enabled': True, 'black_pixels': total_black, 'total_pixels': h*w,
            'candidate_sources': len(sources), 'parameters': dict(cfg)}
