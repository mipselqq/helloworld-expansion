"""Fit the registered SDR grade and measure held-out spatial fidelity."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
from skimage.metrics import structural_similarity


def srgb_decode(x):
    return np.where(x <= .04045, x/12.92, ((np.maximum(x,0)+.055)/1.055)**2.4)


def srgb_encode(x):
    return np.where(x <= .0031308, x*12.92, 1.055*np.maximum(x,0)**(1/2.4)-.055)


def smoothstep(x):
    x = np.clip(x,0,1)
    return x*x*(3-2*x)


def luma(rgb):
    return rgb @ np.array([.2126,.7152,.0722],dtype=np.float32)


def comparison_inputs(rgb, reference, affine):
    # Align in NASA geometry at 1/4 native resolution. Antialias both images in linear display light.
    h,w = rgb.shape[:2]
    rh,rw = reference.shape[:2]
    size = (w//4,h//4)
    s = cv.resize(srgb_decode(rgb.astype(np.float32)/255),size,interpolation=cv.INTER_AREA)
    ref = np.empty((rh//4,rw//4,3),np.float32)
    for c in range(3):
        ref[...,c] = cv.resize(srgb_decode(reference[...,c].astype(np.float32)/65535),
                               (rw//4,rh//4),interpolation=cv.INTER_AREA)
    matrix = affine.copy()
    matrix[:,2] = (matrix[:,2]+1.5*(matrix[:,:2].sum(1)-1))/4
    flags = cv.INTER_CUBIC | cv.WARP_INVERSE_MAP
    t = cv.warpAffine(ref,matrix,size,flags=flags)
    yy,xx = np.indices(size[::-1],dtype=np.float32)
    mapped = matrix[:,:2] @ np.stack((xx.ravel(),yy.ravel())) + matrix[:,2:3]
    u,v = mapped.reshape(2,*yy.shape)
    # Stay inside the shared active photograph, away from the moving rounded video matte.
    valid = (u>rw/4*.115)&(u<rw/4*.885)&(v>17)&(v<rh/4-20)&(t.max(axis=2)>.0007)
    return srgb_encode(s), srgb_encode(np.clip(t,0,1)), valid, xx*4+1.5, yy*4+1.5


def features(rgb, x, y, radius, center, axes, mode):
    r,g,b = np.moveaxis(rgb,-1,0)
    lum = luma(rgb)
    rg = r-g
    bg = b-g
    # Smooth display-referred opponent-color basis, with luminance splines.
    hue = np.clip(rg,-.4,.4)
    blue = np.clip(bg,-.4,.4)
    f = [np.ones_like(lum),lum,lum**2,lum**3,
         hue,blue,3*hue*hue,3*blue*blue,3*hue*blue,2*lum*hue,2*lum*blue]
    for knot in (.035,.09,.19,.35,.55,.75):
        f.append(np.maximum(lum-knot,0)**2)
    if mode in ("regional", "adaptive", "lab"):
        u = np.clip((x-center[0])/axes[0],-2.4,2.4)
        v = np.clip((y-center[1])/axes[1],-1.7,1.7)
        disk = 1-smoothstep((radius-.992)/.031)
        sky = 1-disk
        f.extend([disk,disk*lum,disk*lum**2,disk*hue,disk*blue])
        for field in (u,v,u*u,v*v,u*v):
            f.extend([sky*field,sky*field*lum,disk*field,disk*field*lum])
    if mode in ("adaptive", "lab"):
        # Saturated land and the warm window reflection are sparsely represented
        # in the global gamut. Add a smoothly gated warm-hue response.
        warm = smoothstep((r-b-.028)/.13) * disk
        warm_hi = smoothstep((r-g-.016)/.12) * disk
        f.extend([warm,warm*lum,warm*lum**2,warm_hi,warm_hi*lum,
                  warm*hue,warm*blue,warm*u,warm*v,warm*u*lum,warm*v*lum,
                  5*hue**3,5*blue**3,5*hue*hue*blue,5*hue*blue*blue,
                  3*lum*hue*hue,3*lum*blue*blue,3*lum*hue*blue,
                  3*lum*lum*hue,3*lum*lum*blue])
        for cx,cy in ((-.60,.45),(-.44,.55),(-.27,.44),(-.54,.30)):
            patch = warm*np.exp(-((u-cx)**2+(v-cy)**2)/.11)
            f.extend((patch,patch*lum))
    return np.column_stack(f).astype(np.float32)


@dataclass
class Grade:
    mode: str
    params: np.ndarray
    limb: object

    def apply(self,rgb,x,y):
        if self.mode=="monotone":
            y0=luma(rgb)
            target=PchipInterpolator(self.params[0],self.params[1])(np.clip(y0,0,1))
            result=target[...,None]+(rgb-y0[...,None])*self.params[2,:3]+self.params[3,:3]
        else:
            shape=rgb.shape
            xf,yf=np.broadcast_arrays(x,y)
            radius=self.limb.radius(xf,yf).ravel()
            f=features(rgb.reshape(-1,3),xf.ravel(),yf.ravel(),radius,self.limb.center,
                       self.limb.axes,self.mode)
            correction=(f@self.params).reshape(shape)
            if self.mode=="lab":
                result=lab2rgb(rgb2lab(rgb)+100*correction)
            else:
                result=rgb+correction
        return np.clip(result,0,1).astype(np.float32)


def fit(source,target,valid,x,y,limb,seed):
    radius=limb.radius(x,y)
    tile_x,tile_y=(x//192).astype(int),(y//192).astype(int)
    # Entire 192 px NASA tiles are assigned to the holdout before subsampling.
    fold = (3*tile_x+7*tile_y+tile_x*tile_y)%11
    training=valid&(fold>=3)
    held=valid&~training
    rng=np.random.default_rng(seed)
    ids=np.flatnonzero(training)
    # Balance space, surface, and the thin rim for the regression.
    rr=radius.ravel()[ids]
    samples=[]
    for region in (rr<.98,(rr>=.98)&(rr<=1.04),rr>1.04):
        subset=ids[region]
        samples.append(rng.choice(subset,min(55000,len(subset)),replace=False))
    ids=np.concatenate(samples)
    a,b=source.reshape(-1,3)[ids],target.reshape(-1,3)[ids]
    xx,yy,rr=x.ravel()[ids],y.ravel()[ids],radius.ravel()[ids]
    lum=luma(a)
    knots=np.linspace(0,1,25)
    tone=[]
    for point in knots:
        neighborhood=np.abs(lum-point)<.029
        tone.append(np.median(luma(b[neighborhood])) if neighborhood.sum()>8 else np.nan)
    tone=np.array(tone)
    mask=np.isfinite(tone)
    tone=np.interp(knots,knots[mask],tone[mask])
    tone=isotonic_regression(tone).x
    chroma_a=a-luma(a)[:,None]
    chroma_b=b-luma(b)[:,None]
    chroma_slope=(chroma_a*chroma_b).sum(0)/np.maximum((chroma_a**2).sum(0),1e-6)
    params=np.vstack((knots,tone,np.resize(chroma_slope,len(knots)),np.resize(np.median(chroma_b-chroma_a*chroma_slope,axis=0),len(knots))))
    models=[Grade("monotone",params,limb)]
    for name in ("global","regional","adaptive","lab"):
        f=features(a,xx,yy,rr,limb.center,limb.axes,name).astype(np.float64)
        penalty=np.full(f.shape[1],.22)
        penalty[:11]=.032
        penalty[17:]=.85
        penalty[42:]=.12
        weight=np.ones(len(f))
        d=(rgb2lab(b)-rgb2lab(a))/100 if name=="lab" else b-a
        for _ in range(5):
            matrix=f.T*weight
            c=np.linalg.solve(matrix@f+np.diag(penalty),matrix@d)
            weight=np.minimum(1,.036/np.maximum(np.linalg.norm(f@c-d,axis=1),1e-7))
        models.append(Grade(name,c.astype(np.float32),limb))
    return models, held


def tiled_apply(grade, rgb, x=None, y=None, tile=40):
    h,w=rgb.shape[:2]
    if x is None:
        x=np.arange(w,dtype=np.float32)[None,:]
        y=np.arange(h,dtype=np.float32)[:,None]
    out=np.empty_like(rgb,dtype=np.float32)
    for start in range(0,h,tile):
        sl=slice(start,start+tile)
        rows=rgb[sl]
        if rows.dtype==np.uint8:
            rows=rows.astype(np.float32)/255
        out[sl]=grade.apply(rows,x if x.shape[0]==1 else x[sl],y[sl])
    return out


def metrics(predicted,target,valid,holdout,x,y,limb,original):
    r=limb.radius(x,y)
    bright=luma(original)
    neutral=np.ptp(original,axis=-1)<.19
    masks={"earth":r<.985,"clouds":(r<.965)&(bright>.49)&neutral,
           "atmosphere":(r>=.985)&(r<1.023),"background":r>=1.023}
    lab=rgb2lab(predicted)
    ref=rgb2lab(target)
    de=deltaE_ciede2000(lab,ref)
    _, ssim=structural_similarity(luma(predicted),luma(target),data_range=1,full=True)
    # A valid 7×7 support region is required for local SSIM.
    ssim_valid=cv.erode(valid.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    result={}
    for key,mask in masks.items():
        m=mask&valid&holdout
        value=de[m]
        diff=predicted[m]-target[m]
        result[key]={"pixels":int(m.sum()),"mean_de":float(value.mean()),"median_de":float(np.median(value)),
                     "p95_de":float(np.percentile(value,95)),"lt2_pct":float(100*np.mean(value<2)),
                     "lt5_pct":float(100*np.mean(value<5)),"ssim":float(ssim[m&ssim_valid].mean()),
                     "psnr_db":float(-10*np.log10(np.mean(diff**2))),
                     "mae_rgb_255":float(255*np.mean(np.abs(diff)))}
    return result,de
