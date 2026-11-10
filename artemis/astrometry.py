"""Fit an apparent-J2000 pinhole with radial distortion and validate on Hipparcos."""
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import spiceypy as sp

from .catalog import load_catalog, unit
from .photometry import aperture, nearest_unique, point_candidates


@dataclass
class Camera:
    p: np.ndarray

    @property
    def rotation(self):
        return Rotation.from_rotvec(self.p[:3]).as_matrix()

    def distort(self, q):
        r = np.linalg.norm(q, axis=-1, keepdims=True)
        # Extend the calibrated radial law by its tangent beyond the sensor corner.
        # This defines a monotone virtual projection instead of a divergent polynomial.
        knee = .97
        rc = np.minimum(r, knee)
        k1,k2 = self.p[6:8]
        rd = rc*(1+k1*rc**2+k2*rc**4)
        rd += np.maximum(r-knee,0)*(1+3*k1*knee**2+5*k2*knee**4)
        result = q*np.divide(rd, r, out=np.ones_like(r), where=r!=0)
        if len(self.p)>8:
            p1,p2,stretch,shear = self.p[8:12]
            capped = rc + .14*np.tanh(np.maximum(r-knee,0)/.14)
            x,y = np.moveaxis(q*np.divide(capped,r,out=np.ones_like(r),where=r!=0),-1,0)
            tx=2*p1*x*y+p2*(3*x*x+y*y)
            ty=p1*(x*x+3*y*y)+2*p2*x*y
            result = result + np.stack((tx+shear*q[...,1],ty+stretch*q[...,1]),axis=-1)
        return result

    def pixel(self, sky):
        v = np.asarray(sky) @ self.rotation.T
        q = v[..., :2] / np.maximum(v[..., 2:3], 1e-6)
        return self.p[4:6] + self.p[3]*self.distort(q)

    def sky(self, pixels):
        target = (np.asarray(pixels)-self.p[4:6])/self.p[3]
        q = target.copy()
        for _ in range(13):
            q += target-self.distort(q)
        return unit(np.concatenate((q, np.ones_like(q[..., :1])), axis=-1)) @ self.rotation

    def in_front(self, sky):
        return np.asarray(sky) @ self.rotation[2] > .2


def mission_state(folder, timestamp):
    sp.kclear()
    for name in ["naif0012.tls", "de440s.bsp", "pck00011.tpc", "earth_000101_261115_260819.bpc",
                 "art002_rec_260402_260410_jsc_bet_revf_gapfill.bsp", "art002_rec_260402_260410_jsc_bet_revf.bsp"]:
        sp.furnsh(str(folder / name))
    et = sp.str2et(timestamp)
    observer, _ = sp.spkgeo(-24, et, "J2000", 399)
    bary, _ = sp.spkgeo(-24, et, "J2000", 0)
    earth, lt = sp.spkpos("EARTH", et, "J2000", "LT+S", "-24")
    venus, _ = sp.spkpos("VENUS", et, "J2000", "LT+S", "-24")
    sun, _ = sp.spkpos("SUN", et, "J2000", "LT+S", "-24")
    return dict(et=et, observer=observer, bary=bary, earth=earth, venus=venus, sun=sun,
                itrf=sp.pxform("J2000", "ITRF93", et-lt))


def basis(a, b):
    x = unit(a)
    y = unit(b - x*np.dot(x,b))
    return np.column_stack((x, y, np.cross(x,y)))


def preliminary(shape, limb, venus_xy, state, cfg):
    h, w = shape[:2]
    center = (np.array([w,h])-1)/2
    focal = cfg["focal_mm"]/cfg["pitch_mm"]
    v_earth, v_venus = unit(state["earth"]), unit(state["venus"])
    ray_earth = unit(np.r_[(limb.center-center)/focal, 1])
    ray_venus = unit(np.r_[(venus_xy-center)/focal, 1])
    r = basis(ray_earth, ray_venus) @ basis(v_earth, v_venus).T
    initial = np.r_[Rotation.from_matrix(r).as_rotvec(), focal, center, 0, 0]
    points = limb.points[::4]
    # WGS84 line–ellipsoid tangency, expressed in near-equivalent pixel units.
    radius = np.array([6378.137, 6378.137, 6356.752])+18
    observer = (-state["earth"] @ state["itrf"].T)/radius

    def residual(fit):
        p = initial.copy()
        p[[0,1,2,3,6]] = fit
        cam = Camera(p)
        directions = (cam.sky(points) @ state["itrf"].T)/radius
        distance = np.linalg.norm(np.cross(observer, directions), axis=1) / np.linalg.norm(directions, axis=1)
        return np.r_[(distance-1)*1400, (cam.pixel(v_venus)-venus_xy)*2, (p[3]-focal)/40]

    fit = least_squares(residual, initial[[0,1,2,3,6]], loss="soft_l1", f_scale=2)
    initial[[0,1,2,3,6]] = fit.x
    return Camera(initial)


def match(camera, apparent, catalog, detections, limb, shape, max_mag, radius, min_snr):
    h,w = shape[:2]
    projection = camera.pixel(apparent)
    x,y = projection.T
    ok = (catalog[:,6]<=max_mag) & camera.in_front(apparent) & (x>12) & (x<w-13) & (y>12) & (y<h-13)
    ok &= limb.radius(x,y)>1.035
    si = np.flatnonzero(ok)
    di = np.flatnonzero(detections.snr>=min_snr)
    i,j = nearest_unique(projection[si], detections.xy[di], radius)
    return si[i], di[j]


def solve(rgb, limb, cache, cfg, timestamp, seed):
    state = mission_state(cache / "kernels", timestamp)
    stars, vectors = load_catalog(cache / "catalog" / "hipparcos.tsv",
                                  2000+state["et"]/(365.25*86400)-1991.25,
                                  state["bary"][:3]/sp.convrt(1,"AU","KM"),
                                  state["bary"][3:]/sp.clight())
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    sx, sy = cfg["initial_venus_xy"]
    roi = gray[sy-32:sy+33,sx-32:sx+33]
    _, _, _, (vx,vy) = cv.minMaxLoc(cv.GaussianBlur(roi,(0,0),2))
    venus_xy, *_ = aperture(gray, sx-32+vx, sy-32+vy)
    venus = unit(state["venus"])
    detections = point_candidates(rgb, limb)
    initial = preliminary(rgb.shape, limb, venus_xy, state, cfg)
    print("Preliminary camera",initial.p,"candidate peaks",len(detections.xy),flush=True)
    cam = initial

    def fit(cam, i, j, order=2, use_venus=True, asymmetric=True):
        target = detections.xy[j]
        valid = [0,1,2,3,4,5,6,7] if order==2 else [0,1,2,3,4,5,6]
        starting = np.pad(cam.p,(0,12-len(cam.p)))
        if asymmetric:
            valid += [8,9,10,11]
        else:
            starting[8:]=0
        if order==1:
            starting[7] = 0

        def residual(p):
            model = starting.copy()
            model[valid] = p
            current = Camera(model)
            residuals = (current.pixel(vectors[i])-target).ravel()
            if use_venus:
                residuals = np.r_[residuals, (current.pixel(venus)-venus_xy)*.55]
            if asymmetric:
                residuals = np.r_[residuals, model[8:10]*35, model[10:12]*120]
            return residuals

        result = least_squares(residual, starting[valid], x_scale="jac", loss="soft_l1", f_scale=1.1,
                               max_nfev=400)
        starting[valid] = result.x
        return Camera(starting)

    for magnitude, radius, snr in [(4.6,65,7), (5.2,18,6.5), (6.2,9,6.5), (6.8,6,5.5),
                                   (7.5,3.8,5.0), (cfg["max_catalog_magnitude"],2.8,cfg["minimum_snr"])]:
        i,j = match(cam, vectors, stars, detections, limb, rgb.shape, magnitude, radius, snr)
        print("Matches", magnitude,radius,len(i),np.round(stars[i[:15],6],1),flush=True)
        if len(i) >= 8:
            cam = fit(cam,i,j,asymmetric=radius<=9)
    if len(i)<cfg["minimum_projection_inliers"]:
        raise RuntimeError("The star field failed the minimum astrometric support check")
    pred = cam.pixel(vectors[i])
    resid = np.linalg.norm(pred-detections.xy[j],axis=1)
    good = resid < min(2.8, max(1.5, np.median(resid)*3.5))
    i,j = i[good],j[good]
    rng = np.random.default_rng(seed)
    test = rng.permutation(len(i))[:max(5,len(i)//5)]
    train = np.setdiff1d(np.arange(len(i)),test)
    held = fit(cam, i[train], j[train], use_venus=False)
    held_error = np.linalg.norm(held.pixel(vectors[i[test]])-detections.xy[j[test]],axis=1)
    cam = fit(cam,i,j,use_venus=False)
    alternative = fit(cam,i,j,order=1,use_venus=False)
    residual = np.linalg.norm(cam.pixel(vectors[i])-detections.xy[j],axis=1)
    samples = []
    for _ in range(cfg["bootstrap_trials"]):
        sample = rng.integers(0,len(i),len(i))
        samples.append(fit(cam,i[sample],j[sample],use_venus=False).p)
    return dict(camera=cam, alternative=alternative, state=state, stars=stars, vectors=vectors,
                detections=detections, matches=(i,j), errors=residual, held=held_error,
                venus_xy=venus_xy, venus_prediction=cam.pixel(venus), bootstrap=np.array(samples))


def radec(directions):
    x,y,z = np.moveaxis(np.asarray(directions),-1,0)
    return np.column_stack((np.degrees(np.arctan2(y,x))%360, np.degrees(np.arctan2(z,np.hypot(x,y)))))
