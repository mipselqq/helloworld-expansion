"""Detect and subtract broad optical ridges while retaining photographic detail."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter
from scipy.optimize import least_squares

from .color import luma, smoothstep, srgb_encode


@dataclass
class Ridge:
    name: str
    axis: str
    along: np.ndarray
    center: np.ndarray
    sigma: np.ndarray
    rgb: np.ndarray
    stats: dict



@dataclass
class Halo:
    ridge: Ridge
    along: np.ndarray
    offsets: np.ndarray
    field: np.ndarray

    def pixels(self, along, across):
        center = np.interp(along,self.ridge.along,self.ridge.center)
        offset = across-center
        v = np.interp(along,self.along,np.arange(len(self.along)))
        mx = np.float32((offset-self.offsets[0])/(self.offsets[1]-self.offsets[0]))
        my = np.float32(np.broadcast_to(v,mx.shape))
        field = cv.remap(self.field,mx,my,cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT)
        support = (along>=self.along[0])&(along<=self.along[-1])
        return field*support[...,None]

def native_grid(shape, maximum=4):
    h,w = shape[:2]
    return w//maximum,h//maximum


def sample_profiles(image, size, along, across, axis):
    # OpenCV samples at pixel centers in the decimated image.
    w,h = size
    mx,my = (across,along) if axis=="y" else (along,across)
    mx = np.asarray((mx+.5)*image.shape[1]/w-.5,np.float32)
    my = np.asarray((my+.5)*image.shape[0]/h-.5,np.float32)
    return cv.remap(image,mx,my,cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE)


def profiles(image, shape, cfg):
    h,w = shape[:2]
    axis = cfg["axis"]
    guide = np.array(cfg["guide"],dtype=float)
    along = np.arange(guide[0,0],guide[-1,0]+1,cfg["step"])
    along = np.r_[along,guide[-1,0]] if along[-1]<guide[-1,0] else along
    guess = PchipInterpolator(*guide.T)(along)
    off = np.arange(-216,217,4)
    across = guess[:,None]+off
    center = np.broadcast_to(along[:,None],across.shape)
    samples = [sample_profiles(image,(w,h),center+dy,across,axis) for dy in (-10,-5,0,5,10)]
    rgb = np.median(samples,axis=0)
    keep = (across>=2)&(across<(w if axis=="y" else h)-2)
    return along,guess,off,rgb,keep


def fit_row(off, colors, valid, search, bounds):
    x,y = off[valid],luma(colors[valid])
    if len(x)<34:
        return None
    basis = np.column_stack((np.ones(len(x)),x/140,(x/140)**2))
    low,high = bounds
    initial_baseline = np.median(y[np.abs(x)>75])
    start = [initial_baseline,0,0,max(.4, np.percentile(y,80)-initial_baseline),0,np.sqrt(low*high)]

    def residual(p):
        bg = basis@p[:3]
        core = np.exp(-.5*((x-p[4])/p[5])**2)
        return np.r_[bg+p[3]*core-y,p[2]*.12,(p[5]-np.sqrt(low*high))*.005]

    fit = least_squares(residual,start,
                       bounds=([-np.inf,-np.inf,-np.inf,0,-search,low],
                               [np.inf,np.inf,np.inf,np.inf,search,high]),
                       loss="soft_l1",f_scale=.26,max_nfev=75)
    p = fit.x
    deviation = float(np.median(np.abs(residual(p)[:-2])))
    core = np.exp(-.5*((x-p[4])/p[5])**2)
    matrix = np.column_stack((basis,core))
    weight = np.ones(len(x))
    for _ in range(3):
        c = np.linalg.lstsq(matrix*weight[:,None],colors[valid]*weight[:,None],rcond=None)[0]
        weight = np.minimum(1, .55/np.maximum(np.abs(luma(matrix@c-colors[valid])), .01))
    return p[4],p[5],c[-1],p[3],deviation


def smooth_parameter(along, good, value, blur):
    interpolated = np.interp(along,along[good],np.asarray(value)[good])
    return gaussian_filter1d(median_filter(interpolated,size=3,mode="nearest"),blur,mode="nearest")


def measure(image, shape, cfg):
    along,guess,off,rgb,valid = profiles(image,shape,cfg)
    result = [fit_row(off,row,keep,cfg["search_px"],cfg["sigma_px"])
              for row,keep in zip(rgb,valid)]
    shift = np.array([r[0] if r else 0 for r in result])
    width = np.array([r[1] if r else 0 for r in result])
    channels = np.array([r[2] if r else np.zeros(3) for r in result])
    amp = np.array([r[3] if r else 0 for r in result])
    error = np.array([r[4] if r else 0 for r in result])
    good = (amp>cfg["minimum_contrast"])&(amp>2.1*error)&(np.abs(shift)<.95*cfg["search_px"])
    # Drop isolated fitted peaks and retain profiles supported along the ridge.
    coherent = cv.blur(good.astype(np.float32)[None,:],(5,1)).ravel()>=.38
    good &= coherent
    if good.sum()<7:
        return None
    center = guess+smooth_parameter(along,good,shift,1.8)
    sigma = smooth_parameter(along,good,width,2.1)
    color = np.stack([smooth_parameter(along,good,c,2.4) for c in channels.T],axis=-1)
    # The short candidate fades outside the measured support rather than growing
    # along an unconstrained extrapolation in the blank margin.
    support = np.maximum(along[good][0]-along,along-along[good][-1])
    if cfg["axis"]=="x":
        color *= (1-smoothstep((support-52)/205))[:,None]
    stats={"detected_profiles":int(good.sum()),"sampled_profiles":len(good),
           "contrast_median_8bit":float(np.median(amp[good])),
           "cross_fwhm_median_px":float(np.median(width[good])*2.355),
           "cross_fit_mae_median_8bit":float(np.median(error[good])),
           "along_detected_span_px":[float(along[good][0]),float(along[good][-1])]}
    return Ridge(cfg["name"],cfg["axis"],along,center,sigma,color,stats)



def halo_residual(image, shape, limb, ridge, along, offsets, remove_core):
    """Estimate a local baseline from the flanks of the already detected arc."""
    h,w = shape[:2]
    center = np.interp(along,ridge.along,ridge.center)
    across = center[:,None]+offsets
    p = np.broadcast_to(along[:,None],across.shape)
    colors = np.median([sample_profiles(image,(w,h),p+i,across,ridge.axis)
                        for i in (-8,0,8)],axis=0)
    if remove_core:
        sigma = np.interp(along,ridge.along,ridge.sigma)
        rgb = np.stack([np.interp(along,ridge.along,c) for c in ridge.rgb.T],axis=-1)
        q = np.abs(offsets/sigma[:,None])
        colors -= (np.exp(-.5*q*q)*smoothstep((3.6-q)/.75))[...,None]*rgb[:,None,:]
    x,y = (across,p) if ridge.axis=="y" else (p,across)
    valid = (limb.radius(x,y)>1.10)&(x>=0)&(x<w)&(y>=0)&(y<h)
    side = (np.abs(offsets)>=280)&(np.abs(offsets)<=388)
    t = offsets/350
    basis = np.column_stack((np.ones(len(t)),t,t*t))
    residual = np.zeros_like(colors)
    usable = np.zeros(len(along),bool)
    for i,(row,keep) in enumerate(zip(colors,valid)):
        select = keep&side
        if (select&(t<0)).sum()<12 or (select&(t>0)).sum()<12:
            continue
        f = basis[select]
        data = row[select]
        weight = np.ones(len(data))
        for _ in range(3):
            weighted = f.T*weight
            fit = np.linalg.solve(weighted@f+np.diag([0,0,7.]),weighted@data)
            weight = np.minimum(1,.55/np.maximum(np.abs(luma(f@fit-data)),.025))
        residual[i] = row-basis@fit
        usable[i] = True
    return residual,usable


def measure_halo(image, shape, limb, ridge, spec):
    start,end = spec["halo_along_px"]
    step = spec["step"]
    along = np.arange(start,end+1,step,dtype=float)
    if along[-1]<end:
        along = np.r_[along,end]
    offsets = np.arange(-400,401,4,dtype=float)
    residual,usable = halo_residual(image,shape,limb,ridge,along,offsets,True)
    if usable.sum()<8:
        return None
    filled = np.column_stack([np.interp(along,along[usable],v[usable])
                              for v in residual.reshape(len(along),-1).T]).reshape(residual.shape)
    field = gaussian_filter(np.clip(filled,-3.5,3.5),sigma=(2.3,4.7,0),mode="nearest")
    # Keep the residual local to this section and interpolate smoothly to the
    # untouched sky. A signed correction can lift a small dark oversubtraction.
    width = spec["halo_half_width_px"]
    cross = smoothstep((width-np.abs(offsets))/115)
    ramp = smoothstep((along-start)/250)
    amplitude = np.abs(luma(field))
    field *= (ramp[:,None]*cross*np.minimum(1,1.6/np.maximum(amplitude,1e-5)))[...,None]
    return Halo(ridge,along,offsets,field.astype(np.float32))


def check_halo(before, after, shape, limb, halo):
    along = (halo.along[1:]+halo.along[:-1])/2
    along = along[along>=halo.along[0]+300]
    cross = np.abs(halo.offsets)<=145
    metrics = []
    for image,remove_core in ((before,True),(after,False)):
        residual,usable = halo_residual(image,shape,limb,halo.ridge,along,halo.offsets,remove_core)
        smooth = gaussian_filter1d(luma(residual[usable]),4.7,axis=1,mode="nearest")
        metrics.append(float(np.sqrt(np.mean(smooth[:,cross]**2))))
    return {"ridge":halo.ridge.name, "along_px":[float(halo.along[0]),float(halo.along[-1])],
            "max_luma_correction_8bit":float(np.abs(luma(halo.field)).max()),
            "control_cuts":len(along), "profile_rms_core_only_8bit":metrics[0],
            "profile_rms_refined_8bit":metrics[1]}

def apply(graded, limb, venus, cfg, raw=None):
    """Modify only an additive, feathered sky surface and return the QA model."""
    size = native_grid(graded.shape)
    before = cv.resize(graded,size,interpolation=cv.INTER_AREA)*255
    if not cfg["enabled"]:
        return before,before,[],{"enabled":False}
    smooth = cv.medianBlur(np.float32(before),3)
    fits = [(spec,r) for spec in cfg["ridges"] if (r:=measure(smooth,graded.shape,spec)) is not None]
    ridges = [ridge for _,ridge in fits]
    halos = [halo for spec,ridge in fits if "halo_along_px" in spec
             and (halo:=measure_halo(smooth,graded.shape,limb,ridge,spec)) is not None]
    yy,xx = np.indices(graded.shape[:2],sparse=True,dtype=np.float32)
    nlimb = cfg["protect_limb"]
    nvenus = cfg["protect_venus_px"]
    total_peak = 0.
    clipped = 0
    for start in range(0,len(graded),96):
        rows = slice(start,start+96)
        y = yy[rows]
        delta = np.zeros_like(graded[rows],np.float32)
        for ridge in ridges:
            a,b = (y,xx) if ridge.axis=="y" else (xx,y)
            center = np.interp(a,ridge.along,ridge.center)
            sigma = np.interp(a,ridge.along,ridge.sigma)
            amplitude = np.stack([np.interp(a,ridge.along,c) for c in ridge.rgb.T],axis=-1)
            dist = np.abs((b-center)/sigma)
            window = np.exp(-.5*dist**2)*smoothstep((3.6-dist)/.75)
            window *= (a>=ridge.along[0])&(a<=ridge.along[-1])
            delta += np.float32(window[...,None]*amplitude/255)
        for halo in halos:
            if halo.ridge.axis=="y" and (start+96<halo.along[0] or start>halo.along[-1]):
                continue
            a,b = (y,xx) if halo.ridge.axis=="y" else (xx,y)
            delta += halo.pixels(a,b)/255
        protect = smoothstep((limb.radius(xx,y)-nlimb[0])/(nlimb[1]-nlimb[0]))
        protect *= smoothstep((np.hypot(xx-venus[0],y-venus[1])-nvenus[0])/(nvenus[1]-nvenus[0]))
        delta *= protect[...,None]
        total_peak=max(total_peak,float(np.abs(luma(delta)).max())*255)
        corrected = graded[rows]-delta
        clipped += int(((corrected<0)|(corrected>1)).any(axis=-1).sum())
        graded[rows] = np.clip(corrected,0,1)
    after = cv.resize(graded,size,interpolation=cv.INTER_AREA)*255
    stats={"enabled":True,"max_luma_correction_8bit":total_peak,"clipped_pixels":clipped,"ridges":[]}
    raw_small = None
    if raw is not None:
        raw_small = srgb_encode(cv.resize(raw,size,interpolation=cv.INTER_AREA).astype(np.float32)/65535)*255
    for ridge in ridges:
        description={**ridge.stats,"name":ridge.name,"axis":ridge.axis,
                     "along_px":ridge.along.tolist(),"center_px":ridge.center.tolist(),
                     "sigma_px":ridge.sigma.tolist(),"rgb_contrast_8bit":ridge.rgb.tolist()}
        description.update(check_profiles(before,after,raw_small,graded.shape,ridge))
        stats["ridges"].append(description)
    stats["halos"] = [check_halo(before,after,graded.shape,limb,halo) for halo in halos]
    return before,after,ridges,stats


def check_profiles(before,after,raw,shape,ridge):
    # Fit the known transverse template to fixed, disjoint neighboring cuts.
    # These cuts are shifted by half a fit step and were not used for parameter fits.
    along = (ridge.along[1:]+ridge.along[:-1])/2
    use = (along>=ridge.stats["along_detected_span_px"][0])&(along<=ridge.stats["along_detected_span_px"][1])
    along = along[use]
    center = np.interp(along,ridge.along,ridge.center)
    sigma = np.interp(along,ridge.along,ridge.sigma)
    off = np.arange(-175,176,5)
    p = center[:,None]+off
    q = np.broadcast_to(along[:,None],p.shape)
    fitted=[]
    for image in (before,after,raw):
        if image is None:
            fitted.append(None)
            continue
        cuts = luma(sample_profiles(image,(shape[1],shape[0]),q,p,ridge.axis))
        valid = (p>=2)&(p<(shape[1] if ridge.axis=="y" else shape[0])-2)
        amplitude=[]
        for row,keep,w in zip(cuts,valid,sigma):
            core=np.exp(-.5*(off/w)**2)
            fit=np.column_stack((np.ones(len(off)),off/175,(off/175)**2,core))
            amplitude.append(np.linalg.lstsq(fit[keep],row[keep],rcond=None)[0][-1])
        fitted.append(np.asarray(amplitude))
    old,new,source_raw = fitted
    return {"holdout_profiles":len(along),"holdout_contrast_before_8bit":float(np.median(old)),
            "holdout_contrast_after_8bit":float(np.median(new)),
            "holdout_median_abs_after_8bit":float(np.median(np.abs(new))),
            "nef_same_template_median_8bit":float(np.median(source_raw)) if source_raw is not None else None}
