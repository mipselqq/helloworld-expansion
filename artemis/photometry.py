"""Point detection and local empirical aperture photometry."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Detections:
    xy: np.ndarray
    snr: np.ndarray
    flux: np.ndarray
    peak: np.ndarray
    fwhm: np.ndarray


def point_candidates(rgb, limb, sigma=1.55, minimum=4.6):
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY).astype(np.float32)
    low = cv.GaussianBlur(gray, (0,0), sigma)
    wide = cv.GaussianBlur(gray, (0,0), sigma*3.0)
    response = low-wide
    # Estimate local filter noise from a winsorized variance.
    power = response**2
    baseline = cv.blur(power, (37,37))
    noise = np.sqrt(cv.blur(np.minimum(power, 5*baseline+.04), (53,53)) + .055)
    snr = response/noise
    peak = cv.dilate(response, np.ones((7,7), np.uint8))
    yy, xx = np.nonzero((snr > minimum) & (response == peak))
    keep = (limb.radius(xx,yy)>1.035) & (xx>12) & (yy>12) & (xx<gray.shape[1]-13) & (yy<gray.shape[0]-13)
    yy, xx = yy[keep], xx[keep]
    points = []
    for x, y in zip(xx, yy):
        position, flux, amplitude, fwhm = aperture(gray, x, y)
        points.append([*position, snr[y,x], flux, amplitude, fwhm])
    points = np.asarray(points)
    order = np.argsort(-points[:,2])
    points = points[order]
    return Detections(points[:,:2], points[:,2], points[:,3], points[:,4], points[:,5])


def aperture(gray, x, y):
    x, y = round(x), round(y)
    yy, xx = np.mgrid[-10:11,-10:11]
    rad = np.hypot(xx, yy)
    patch = gray[y-10:y+11, x-10:x+11].astype(float)
    ring = rad >= 7.5
    background = np.linalg.lstsq(np.column_stack((np.ones(ring.sum()), xx[ring], yy[ring])),
                                patch[ring], rcond=None)[0]
    signal = patch-background[0]-background[1]*xx-background[2]*yy
    weight = np.maximum(signal,0) * np.exp(-rad**2/16) * (rad<6)
    sum_w = max(weight.sum(), 1e-8)
    dx, dy = (weight*xx).sum()/sum_w, (weight*yy).sum()/sum_w
    rr = np.hypot(xx-dx, yy-dy)
    core = (rr < 5.5)
    flux = signal[core].sum()
    positive = np.maximum(signal, 0) * (rr<4.5)
    fwhm = 2.355*np.sqrt((positive*rr**2).sum()/max(2*positive.sum(),1e-8))
    return np.array([x+dx, y+dy]), flux, float(signal.max()), fwhm


def nearest_unique(xy, candidates, radius):
    distance, idx = cKDTree(candidates).query(xy, distance_upper_bound=radius)
    used = set()
    left, right = [], []
    for index in np.argsort(distance):
        if not np.isfinite(distance[index]):
            break
        target = int(idx[index])
        if target not in used:
            left.append(index)
            right.append(target)
            used.add(target)
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)
