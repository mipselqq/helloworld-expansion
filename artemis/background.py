"""Continue the measured background and its grain beyond the rotated footprint."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .color import luma


@dataclass
class Background:
    field: np.ndarray
    wide_field: np.ndarray
    floor: np.ndarray
    covariance: np.ndarray
    neighbor_corr: float
    scale: int
    std: np.ndarray
    noise_field: np.ndarray


def estimate(graded, limb):
    h,w=graded.shape[:2]
    scale=8
    lo=cv.resize(graded,(w//scale,h//scale),interpolation=cv.INTER_AREA)
    yy,xx=np.indices(lo.shape[:2],dtype=np.float32)
    xx,yy=xx*scale+3.5,yy*scale+3.5
    sky=limb.radius(xx,yy)>1.075
    coverage=cv.GaussianBlur(sky.astype(np.float32),(0,0),6)
    field=cv.GaussianBlur(lo*sky[...,None],(0,0),6)/np.maximum(coverage[...,None],1e-5)
    edge=((xx<230)|(xx>w-230)|(yy<115)|(yy>h-115))&sky
    floor=np.percentile(field[edge],40,axis=0)
    coarse_coverage=cv.GaussianBlur(sky.astype(np.float32),(0,0),27)
    wide_field=cv.GaussianBlur(lo*sky[...,None],(0,0),27)/np.maximum(coarse_coverage[...,None],1e-5)
    wide_field[coarse_coverage<.05]=floor
    field[coverage<.05]=floor
    samples=[]
    pairs=[]
    for y in range(50,h-115,203):
        for x in range(55,w-125,211):
            if limb.radius(x+40,y+40)<1.12:
                continue
            patch=graded[y:y+80,x:x+80]
            noise=patch-cv.GaussianBlur(patch,(0,0),4)
            med=np.median(noise,axis=(0,1))
            mad=1.4826*np.median(np.abs(noise-med),axis=(0,1))+.0002
            noise=np.clip(noise-med,-3.2*mad,3.2*mad)
            samples.append(noise.reshape(-1,3))
            a,b=luma(noise[:,:-1]),luma(noise[:,1:])
            pairs.append(np.corrcoef(a.ravel(),b.ravel())[0,1])
    covariance=np.cov(np.concatenate(samples),rowvar=False)
    corr=float(np.median(pairs))
    std=np.sqrt(np.diag(covariance))
    # Dark patches around the window can carry a much lower noise level than
    # the illuminated zodiacal cone. Estimate the local noise scale separately.
    residual=graded-cv.GaussianBlur(graded,(0,0),4)
    power=cv.resize(np.minimum(residual**2,np.float32((4*std)**2)),
                    (w//scale,h//scale),interpolation=cv.INTER_AREA)
    variance=cv.GaussianBlur(power*sky[...,None],(0,0),7)
    support=cv.GaussianBlur(sky.astype(np.float32),(0,0),7)
    std_grid=np.sqrt(variance/np.maximum(support[...,None],1e-5))
    std_grid[support<.1]=std
    noise_field=np.float32(np.clip(std_grid/std,.16,2.5))
    return Background(field,wide_field,floor,covariance,corr,scale,std,noise_field)


def sky_at(background, x, y, w, h):
    u=np.clip(x,0,w-1)
    v=np.clip(y,0,h-1)
    distance=np.hypot(x-u,y-v)
    bx=(u+.5)/background.scale-.5
    by=(v+.5)/background.scale-.5
    sampled=cv.remap(background.field,bx,by,cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE)
    wide=cv.remap(background.wide_field,bx,by,cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE)
    # Continue fine edge structure only in the immediate feather. Progressively
    # use the broad field so a feature touching an edge cannot turn into a ray
    # perpendicular to that edge in the extrapolated sky.
    step=np.divide(32,distance,out=np.zeros_like(distance),where=distance>0)
    p=(np.clip(u-(x-u)*step,0,w-1)+.5)/background.scale-.5
    q=(np.clip(v-(y-v)*step,0,h-1)+.5)/background.scale-.5
    inner=cv.remap(background.field,p,q,cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE)
    distance=distance[...,None]
    slope=np.clip((sampled-inner)/32,-.00004,.00004)
    fine=np.exp(-(distance/200)**2)
    value=wide+fine*(sampled-wide+slope*(105*np.tanh(distance/105)))
    weight=np.exp(-(distance/820)**2)
    sky=np.clip(background.floor+weight*(value-background.floor),.002,.2)
    ratio=cv.remap(background.noise_field,bx,by,cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE)
    return sky,np.float32(.92*ratio*background.std),distance
