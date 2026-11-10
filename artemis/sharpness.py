"""Apply the approved frequency response on the native-scale wallpaper."""
from dataclasses import dataclass
from concurrent.futures import CancelledError

import cv2 as cv
import numpy as np

from .color import luma, smoothstep


def blur(image, sigma):
    return cv.GaussianBlur(image, (0,0), sigma)


def band(image, low, high):
    return (blur(image, low) if low else image)-blur(image, high)


def apply(master, limb, affine, stars, cfg, cancel=None):
    source=master.copy()
    inverse=cv.invertAffineTransform(affine)
    h,w=source.shape[:2]
    xx=np.arange(w,dtype=np.float32)[None,:]
    sigmas=cfg["sigmas_px"]
    pad=int(np.ceil(4.5*max(sigmas)))
    clouds,limb_gains,sky=np.float32([cfg[k] for k in ("clouds","limb","sky")])
    for start in range(0,h,96):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        end=min(h,start+96)
        lo,hi=max(0,start-pad),min(h,end+pad)
        work=source[lo:hi].astype(np.float32)/65535
        use=slice(start-lo,end-lo)
        yy=np.arange(start,end,dtype=np.float32)[:,None]
        nx=inverse[0,0]*xx+inverse[0,1]*yy+inverse[0,2]
        ny=inverse[1,0]*xx+inverse[1,1]*yy+inverse[1,2]
        r=limb.radius(nx,ny)
        near=smoothstep((r-.955)/.034)[...,None]
        space=smoothstep((r-1.014)/.031)[...,None]
        gains=clouds+(limb_gains-clouds)*near
        gains+=space*(sky-gains)
        delta=np.zeros_like(work[use])
        previous=work
        for i,sigma in enumerate(sigmas):
            filtered=blur(work,sigma)
            detail=(previous-filtered)[use]
            delta+=gains[...,i,None]*(detail*space+luma(detail)[...,None]*(1-space))
            previous=filtered
        # Preserve fitted point sources rather than broadening their measured PSFs.
        for x,y in stars[(stars[:,1]>=start-7)&(stars[:,1]<end+7)&
                         (stars[:,0]>=-7)&(stars[:,0]<w+7)]:
            u0,u1=max(0,int(x)-7),min(w,int(x)+9)
            v0,v1=max(start,int(y)-7),min(end,int(y)+9)
            dist=np.hypot(np.arange(u0,u1)[None,:]-x,np.arange(v0,v1)[:,None]-y)
            delta[v0-start:v1-start,u0:u1]*=smoothstep((dist-2.6)/4)[...,None]
        master[start:end]=np.rint(np.clip(work[use]+delta,0,1)*65535).astype(np.uint16)


@dataclass
class Probe:
    rows: slice
    cols: slice
    before: np.ndarray
    reference: np.ndarray
    radius: np.ndarray
    valid: np.ndarray
    offset: tuple
    correlation: float


def aurora_probe(master, reference, limb, affine, registration, cfg):
    w,h=cfg["aurora_crop"]
    center=affine@np.r_[cfg["aurora_center_native"],1.]
    x,y=np.rint(center-[w/2,h/2]).astype(int)
    rows,cols=slice(y,y+h),slice(x,x+w)
    yy,xx=np.indices((h,w),dtype=np.float32)
    xx+=x
    yy+=y
    inverse=cv.invertAffineTransform(affine)
    transform=registration@np.vstack((inverse,[0,0,1]))
    u=np.float32(transform[0,0]*xx+transform[0,1]*yy+transform[0,2])
    v=np.float32(transform[1,0]*xx+transform[1,1]*yy+transform[1,2])
    before=master[rows,cols].astype(np.float32)/257
    matched=cv.remap(reference,u,v,cv.INTER_LANCZOS4).astype(np.float32)/257
    # Refine translation from coherent structure rather than from fine grain.
    win=cv.createHanningWindow((w,h),cv.CV_32F)
    shift,response=cv.phaseCorrelate(band(luma(before),2,6),band(luma(matched),2,6),win)
    if response>.12 and np.linalg.norm(shift)<2.8:
        dx,dy=transform[:,:2]@np.array(shift)
        u+=dx
        v+=dy
        matched=cv.remap(reference,u,v,cv.INTER_LANCZOS4).astype(np.float32)/257
    else:
        shift=(0.,0.)
    r=limb.radius(inverse[0,0]*xx+inverse[0,1]*yy+inverse[0,2],
                  inverse[1,0]*xx+inverse[1,1]*yy+inverse[1,2])
    valid=(v<reference.shape[0]*.970)&(v>75)
    valid[:20]=False
    valid[-20:]=False
    valid[:,:20]=False
    valid[:,-20:]=False
    return Probe(rows,cols,before,matched,r,valid,shift,float(response))


def evaluate(master, probe, path, cfg):
    import matplotlib.pyplot as plt

    after=master[probe.rows,probe.cols].astype(np.float32)/257
    target=probe.reference
    base=probe.before
    variants={"before":base}
    for radius,amount in ((.75,.25),(1.5,.28),(2.6,.20)):
        variants[f"unsharp_{radius:g}"]=np.clip(base+amount*band(luma(base),0,radius)[...,None],0,255)
    variants.update(final=after,reference=target)
    r=probe.radius
    yy,xx=np.indices(r.shape)
    holdout=((xx//104+2*(yy//104))%5<2)&probe.valid
    regions={"clouds":holdout&(r<.978),"aurora":holdout&(r>=.982)&(r<=1.028),
             "sky":holdout&(r>1.04)}
    bands=((0,.72),(.72,1.45),(1.45,2.9),(2.9,5.8))
    target_bands=[band(luma(target),*b) for b in bands]
    metrics={}
    def rms(z):
        return float(np.sqrt(np.mean(z*z)))
    for name,image in variants.items():
        current=[band(luma(image),*b) for b in bands]
        metrics[name]={}
        for region,mask in regions.items():
            fine=current[0]
            peaks=(fine==cv.dilate(fine,np.ones((3,3),np.uint8)))&mask
            metrics[name][region]={"band_rms_8bit":[rms(f[mask]) for f in current],
                "error_rms_8bit":[rms((f-t)[mask]) for f,t in zip(current,target_bands)],
                "fine_peaks_per_mp":{str(k):float(1e6*np.sum(peaks&(fine>k))/mask.sum()) for k in (2,3,4,6)}}
    fig,axes=plt.subplots(2,2,figsize=(14,8.1),layout="constrained")
    high=int(np.flatnonzero(probe.valid.any(axis=1))[-1])+1
    for ax,name,title in zip(axes.ravel(),("reference","before","unsharp_1.5","final"),
            ("Registered 8K reference","Previous master","Conventional sharpening","Approved multiscale render")):
        ax.imshow(np.clip(variants[name][:high]/255,0,1),interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path,dpi=175)
    plt.close(fig)
    return {"enabled":True,"settings":cfg,"crop_xy":[int(probe.cols.start),int(probe.rows.start)],
            "phase_offset_px":list(probe.offset),"phase_response":probe.correlation,
            "bands_sigma_px":bands,"validation_pixels":{k:int(v.sum()) for k,v in regions.items()},
            "metrics":metrics}


def conventional(master, sigma=1.5, amount=.28, cancel=None):
    """Apply the conventional unsharp mask used in the aurora comparison."""
    source=master.copy()
    height=len(master)
    pad=int(np.ceil(4.5*sigma))
    for start in range(0,height,96):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        end=min(height,start+96)
        lo,hi=max(0,start-pad),min(height,end+pad)
        tile=source[lo:hi].astype(np.float32)/65535
        rows=slice(start-lo,end-lo)
        detail=amount*band(luma(tile),0,sigma)[rows]
        master[start:end]=np.rint(np.clip(tile[rows]+detail[...,None],0,1)*65535).astype(np.uint16)
