"""Derive the stellar selection function and render only catalog projections."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import expit

from .astrometry import Camera, radec
from .color import luma, smoothstep, srgb_decode


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def in_rectangle(xy, w, h, margin=0):
    x,y=xy.T
    return (x>=margin)&(x<w-margin)&(y>=margin)&(y<h-margin)


def fit_logistic(mag, radius, detected):
    def residual(p):
        local_limit=p[0]-p[2]*(radius**2-.5**2)
        return expit((local_limit-mag)/p[1])-detected

    return least_squares(residual,[6.2,.5,1],bounds=([3,.1,0],[9,3,6])).x


def fit_profile(gray, x, y):
    radius=9
    u,v=round(x),round(y)
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1].astype(float)
    patch=gray[v-radius:v+radius+1,u-radius:u+radius+1]
    background=np.median(patch[np.hypot(xx,yy)>7.5])
    peak=patch.max()-background

    def residual(p):
        a,b,c,sx,sy,rho,base,dx,dy=p
        px,py=(xx-b)/sx,(yy-c)/sy
        return (a*np.exp(-(px*px+py*py-2*rho*px*py)/(2*(1-rho*rho)))+
                base+dx*xx+dy*yy-patch).ravel()

    fit=least_squares(residual,[max(peak,1e-7),x-u,y-v,1.65,1.65,0,background,0,0],
                      bounds=([0,-3,-3,.55,.55,-.7,-.05,-.01,-.01],
                              [1,3,3,4.5,4.5,.7,1,.01,.01]),
                      max_nfev=60,loss="soft_l1",f_scale=max(peak*.1,2e-5))
    return fit.x


@dataclass
class Sky:
    xy: np.ndarray
    native: np.ndarray
    flux: np.ndarray
    profiles: np.ndarray
    bv: np.ndarray
    stats: dict


def prepare(solve, graded, original, limb, affine, dimensions, cfg, raw=None):
    camera=solve["camera"]
    stars,vectors=solve["stars"],solve["vectors"]
    detected=solve["detections"]
    i,j=solve["matches"]
    p=camera.pixel(vectors)
    h,w=original.shape[:2]
    eligible=(camera.in_front(vectors)&in_rectangle(p,w,h,18)&(limb.radius(*p.T)>1.045))
    labels=np.isin(np.flatnonzero(eligible),i).astype(float)
    visible_mags=stars[eligible,6]
    dist=np.linalg.norm((p-camera.p[4:6])/camera.p[3],axis=1)
    limit,width,vignetting=fit_logistic(visible_mags,dist[eligible],labels)
    # Edge sensitivity is held at the last measured radius in the virtual extension.
    local_limit=limit-vignetting*(np.minimum(dist,np.percentile(dist[eligible],98))**2-.5**2)
    bins=[]
    for mag in np.arange(2,8.5,.5):
        selected=(visible_mags>=mag)&(visible_mags<mag+.5)
        bins.append([float(mag),int(selected.sum()),int(labels[selected].sum())])
    ok=(detected.snr[j]>=8)&(stars[i,6]>2)&in_rectangle(detected.xy[j],w,h,15)
    bright_i,bright_j=i[ok],j[ok]
    gray=luma(srgb_decode(graded))
    raw_gray=None if raw is None else cv.cvtColor(raw,cv.COLOR_RGB2GRAY).astype(np.float32)/65535
    profiles=[]
    raw_fwhm=[]
    for x,y in detected.xy[bright_j]:
        fit=fit_profile(gray,x,y)
        profiles.append(fit)
        if raw_gray is not None:
            rf=fit_profile(raw_gray,x,y)
            raw_fwhm.append(2.355*np.sqrt(rf[3]*rf[4]))
    profiles=np.array(profiles)
    good=(profiles[:,0]>1e-5)&(profiles[:,3:5].max(1)<4.45)
    profiles,bright_i,bright_j=profiles[good],bright_i[good],bright_j[good]
    aperture_flux=2*np.pi*profiles[:,0]*profiles[:,3]*profiles[:,4]*np.sqrt(1-profiles[:,5]**2)
    bv=np.nan_to_num(stars[:,7],nan=.6).clip(-.3,2)
    # Fit on actual image radii, then freeze attenuation at the last sampled radius.
    model=np.column_stack((np.ones(len(stars)),stars[:,6]-5,dist**2,bv-.6))
    c=least_squares(lambda c:model[bright_i]@c-np.log10(aperture_flux),[-2,-.4,-.1,0],
                    loss="soft_l1",f_scale=.15).x
    fit_error=rms(model[bright_i]@c-np.log10(aperture_flux))
    calibrated_radius=np.percentile(dist[bright_i],98)
    model[:,2]=np.minimum(dist,calibrated_radius)**2
    canvas=p@affine[:,:2].T+affine[:,2]
    # Keep the faint tail down to the measured selection function's low-probability
    # boundary. Their empirical flux then determines whether they rise above the grain.
    pmin=cfg["faint_relative_detection"]
    faint_cut=np.minimum(cfg["max_catalog_magnitude"],local_limit+width*np.log((1-pmin)/pmin))
    selected=np.flatnonzero(camera.in_front(vectors)&in_rectangle(canvas,*dimensions,9)&
                            ~in_rectangle(p,w,h,0)&(stars[:,6]<=faint_cut))
    samples=np.array([Camera(params).pixel(vectors[selected]) for params in solve["bootstrap"]])
    scatter=np.percentile(np.linalg.norm(samples-p[selected],axis=2),95,axis=0)
    alt=np.linalg.norm(solve["alternative"].pixel(vectors[selected])-p[selected],axis=1)
    allowed=scatter<=cfg["max_extension_uncertainty_px"]
    ids=selected[allowed]
    fit_near=cKDTree(detected.xy[bright_j]).query(p[ids],k=min(12,len(bright_j)))[1]
    shapes=np.median(profiles[fit_near][:,:,3:6],axis=1)
    flux=10**(model[ids]@c)
    gain=float(cfg.get("extension_flux_gain",1.0))
    fade=float(cfg.get("extension_gain_fade_px",180.0))
    if not np.isfinite(gain) or gain<=0 or not np.isfinite(fade) or fade<=0:
        raise ValueError("Extension star brightness and fade must be positive and finite")
    outside=np.linalg.norm(p[ids]-np.clip(p[ids],[0,0],[w-1,h-1]),axis=1)
    flux*=1+(gain-1)*smoothstep(outside/fade)
    stats={"magnitude_50":float(limit),"completeness_width":float(width),"offaxis_completeness_coefficient":float(vignetting),
           "extension_limit_v":np.percentile(local_limit[selected],[0,50,100]).tolist(), "bins":bins,
           "psf_stars":int(len(profiles)),"fwhm_median":float(np.median(2.355*np.sqrt(profiles[:,3]*profiles[:,4]))),
           "fwhm_raw_median":float(np.median(raw_fwhm)) if raw_fwhm else None,
           "fwhm_p10_p90":np.percentile(2.355*np.sqrt(profiles[:,3]*profiles[:,4]),[10,90]).tolist(),
           "log10_flux_model":c.tolist(),"log10_flux_rms":fit_error,
           "extension_flux_gain":gain,"extension_gain_fade_px":fade,
           "eligible_outside":int(len(selected)),"rendered":int(len(ids)),
           "below_half_completeness":int((stars[ids,6]>local_limit[ids]).sum()),
           "rendered_max_v":float(stars[ids,6].max()), "min_detection_probability":pmin,
           "calibrated_flux_radius":float(calibrated_radius),
           "projection_scatter_px":np.percentile(scatter[allowed],[50,95,100]).tolist() if allowed.any() else [],
           "model_sensitivity_px":np.percentile(alt[allowed],[50,95,100]).tolist() if allowed.any() else []}
    report_rows=[]
    ra_dec=radec(vectors[ids])
    for index,s,delta,radec_i,position in zip(ids,scatter[allowed],alt[allowed],ra_dec,canvas[ids]):
        report_rows.append([int(stars[index,0]),*radec_i,stars[index,6],*position,s,delta])
    return Sky(canvas[ids],p[ids],flux,shapes,bv[ids],stats),report_rows


def render(tile_linear, x0, y0, sky, rotation):
    h,w=tile_linear.shape[:2]
    candidates=np.flatnonzero((sky.xy[:,0]>x0-12)&(sky.xy[:,0]<x0+w+12)&
                              (sky.xy[:,1]>y0-12)&(sky.xy[:,1]<y0+h+12))
    for i in candidates:
        x,y=sky.xy[i]-[x0,y0]
        xa,xb=max(0,int(x)-12),min(w,int(x)+13)
        ya,yb=max(0,int(y)-12),min(h,int(y)+13)
        xx,yy=np.meshgrid(np.arange(xa,xb)-x,np.arange(ya,yb)-y)
        sx,sy,rho=sky.profiles[i]
        a,b=(rotation[0,0]*xx+rotation[1,0]*yy)/sx,(rotation[0,1]*xx+rotation[1,1]*yy)/sy
        core=np.exp(-(a*a+b*b-2*rho*a*b)/(2*(1-rho*rho)))
        wing=np.exp(-(a*a+b*b)/7.5)
        psf=(.94*core+.06*wing)/(2*np.pi*sx*sy*(.94*np.sqrt(1-rho*rho)+.06*3.75))
        bv=sky.bv[i]
        color=np.array([1+.28*(bv-.45),1,1-.34*(bv-.45)])
        color/=luma(color)
        tile_linear[ya:yb,xa:xb]+=sky.flux[i]*psf[...,None]*color
