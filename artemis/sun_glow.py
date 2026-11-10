"""Optional, conservative artistic reduction of the broad sunward sky glow.

NASA identifies zodiacal light in this area. The correction is intentionally
not presented as a physical separation of sunlight, dust and window scatter.
"""
import cv2 as cv
import numpy as np

from .color import luma, smoothstep


def apply(image, limb, venus, points, cfg):
    """Correct broad background and excess noise while protecting limb and stars."""
    if not cfg.get('enabled', True):
        return {'enabled': False}
    h, w = image.shape[:2]
    scale = int(cfg.get('analysis_scale', 8))
    sw, sh = max(1, w//scale), max(1, h//scale)
    small = cv.resize(image, (sw, sh), interpolation=cv.INTER_AREA)
    yy, xx = np.indices((sh, sw), dtype=np.float32)
    x = (xx+.5)*w/sw-.5
    y = (yy+.5)*h/sh-.5
    dx, dy = x-limb.center[0], y-limb.center[1]
    direction = np.asarray(venus, dtype=float)-limb.center
    direction /= np.linalg.norm(direction)
    cosine = (dx*direction[0]+dy*direction[1])/np.maximum(np.hypot(dx,dy), 1)
    sky = (limb.radius(x,y)>1.095).astype(np.float32)
    support = cv.GaussianBlur(sky, (0,0), 24)
    field = cv.GaussianBlur(small*sky[...,None], (0,0), 24)/np.maximum(support[...,None], .001)
    quiet = (sky>0) & (cosine<.05) & (support>.7)
    if quiet.sum()<100:
        raise ValueError('Insufficient sky to estimate glow baseline')
    floor = np.percentile(field[quiet], 50, axis=0).astype(np.float32)
    # Ignore isolated stars in the noise correction without hiding them from
    # the broad sky subtraction; no new point sources are synthesized.
    protect = np.zeros((h,w),np.uint8)
    for px, py in np.asarray(points).reshape(-1,2):
        if np.isfinite(px) and np.isfinite(py) and -20<px<w+20 and -20<py<h+20:
            cv.circle(protect,(int(round(px)),int(round(py))),7,255,-1)
    cv.circle(protect,tuple(np.rint(venus).astype(int)),34,255,-1)
    protect=cv.GaussianBlur(protect,(0,0),3).astype(np.float32)/255
    source=image.copy()
    grid_x=np.arange(w,dtype=np.float32)[None,:]
    peak=0.
    affected=0
    for start in range(0,h,96):
        end=min(h,start+96)
        lo,hi=max(0,start-14),min(h,end+14)
        tile=source[lo:hi]
        rows=slice(start-lo,end-lo)
        gy=np.arange(start,end,dtype=np.float32)[:,None]
        mx=np.broadcast_to((grid_x+.5)*sw/w-.5,(end-start,w)).astype(np.float32)
        my=np.broadcast_to((gy+.5)*sh/h-.5,(end-start,w)).astype(np.float32)
        local=cv.remap(field,mx,my,cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE)
        vx,vy=grid_x-limb.center[0],gy-limb.center[1]
        radial=limb.radius(grid_x,gy)
        cos=(vx*direction[0]+vy*direction[1])/np.maximum(np.hypot(vx,vy),1)
        angular=smoothstep((cos-cfg.get('outer_cos',.35))/(cfg.get('inner_cos',.72)-cfg.get('outer_cos',.35)))
        rim=smoothstep((radial-cfg.get('rim_start',1.025))/(cfg.get('rim_end',1.10)-cfg.get('rim_start',1.025)))
        mask=(angular*rim).astype(np.float32)
        excess=np.maximum(local-floor,0)*np.float32(cfg.get('strength',.9))
        low=excess*mask[...,None]
        smooth=cv.GaussianBlur(tile,(0,0),2.4)[rows]
        fine=tile[rows]-smooth
        noise=(np.float32(cfg.get('noise_strength',.55))*mask*(1-protect[start:end]))[...,None]*fine
        image[start:end]=np.clip(tile[rows]-low-noise,0,1)
        peak=max(peak,float(luma(low).max())*255)
        affected+=int(np.count_nonzero(mask>.5))
    return {'enabled': True, 'description': 'Artistic sky glow suppression; zodiacal light may be reduced',
            'baseline_rgb_8bit':(floor*255).tolist(),'peak_background_reduction_8bit':peak,
            'pixels_over_half_weight':affected,'parameters':dict(cfg)}
