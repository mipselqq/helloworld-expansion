"""Resample real photographic grain with point sources masked from donor windows."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .color import luma, smoothstep, srgb_encode


@dataclass
class Donors:
    centers: np.ndarray
    blocked: np.ndarray
    patch: int
    overlap: int
    decay: float
    seed: int


def find_donors(graded, background, limb, exclude, rotation, cfg):
    patch = cfg["grain_patch"]
    overlap = cfg["grain_overlap"]
    if not 0 < overlap < patch // 2:
        raise ValueError("Grain overlap must be smaller than half a patch")
    h,w = graded.shape[:2]
    blocked = np.zeros((h,w), np.uint8)
    for point in np.rint(exclude).astype(int):
        if 0 <= point[0] < w and 0 <= point[1] < h:
            cv.circle(blocked, tuple(point), 7, 1, cv.FILLED)
    integral = cv.integral(blocked)
    # The bound includes subpixel jitter, rotation and the Lanczos support.
    margin = int(np.ceil(patch/2*np.max(np.abs(rotation).sum(axis=0)))) + 13
    xx, yy = np.meshgrid(np.arange(200, w-200, 19), np.arange(170, h-170, 19))
    occupancy = (integral[yy+margin,xx+margin] - integral[yy-margin,xx+margin]
                 - integral[yy+margin,xx-margin] + integral[yy-margin,xx-margin])
    clear = occupancy < .095*(2*margin)**2
    clear &= limb.radius(xx,yy) > 1.12
    x,y = xx[clear],yy[clear]
    sample = background.field[y//background.scale, x//background.scale]
    # Match donor illumination to the actual blue-black field, avoiding the limb
    # and the noisier illuminated zodiacal cone. Match luma, including shadow chroma.
    delta = np.abs(luma(sample)-luma(background.floor))
    limit = max(2.2/255, np.percentile(delta, 55))
    keep = delta <= limit
    centers = np.column_stack((x[keep], y[keep]))
    if len(centers) < 30:
        raise RuntimeError("Too few uncontaminated sky donors")
    return Donors(centers, blocked, patch, overlap, cfg["grain_decay_px"], cfg["seed"])


class Grain:
    """Overlap real RGB residuals without mirroring the astronomical image."""
    def __init__(self, linear, rotation, donors, width):
        self.linear = linear
        self.donors = donors
        self.width = width
        self.rng = np.random.default_rng(donors.seed)
        patch, overlap = donors.patch, donors.overlap
        self.stride = patch-overlap
        coord = np.arange(patch,dtype=np.float32)-(patch-1)/2
        yy,xx = np.meshgrid(coord,coord,indexing="ij")
        self.map_x = np.float32(rotation[0,0]*xx+rotation[1,0]*yy)
        self.map_y = np.float32(rotation[0,1]*xx+rotation[1,1]*yy)
        window = np.ones(patch,np.float32)
        edge = smoothstep((np.arange(overlap,dtype=np.float32)+.5)/overlap)
        window[:overlap] = np.sqrt(edge)
        window[-overlap:] = np.sqrt(1-edge)
        self.weight = (window[:,None]*window)[...,None]
        self.carry = np.zeros((overlap,width,3),np.float32)
        self.buffer = np.empty((0,width,3),np.float32)
        samples = np.array([self._patch() for _ in range(192)])
        self.sigma = np.float32(np.sqrt(np.mean(samples**2,axis=(0,1,2))))
        self.luma_sigma = float(np.sqrt(np.mean(luma(samples)**2)))
        self._stripe()

    def _sample(self):
        x,y = self.donors.centers[self.rng.integers(len(self.donors.centers))]
        x,y = np.array([x,y])+self.rng.uniform(-8,8,2)
        mx,my = np.float32(self.map_x+x),np.float32(self.map_y+y)
        sample = srgb_encode(np.maximum(cv.remap(self.linear,mx,my,cv.INTER_LANCZOS4),0))
        blocked = cv.remap(self.donors.blocked,mx,my,cv.INTER_NEAREST).astype(bool)
        if blocked.any():
            good = (~blocked).astype(np.float32)
            smoothed = cv.GaussianBlur(sample*good[...,None],(0,0),4.5)
            support = cv.GaussianBlur(good,(0,0),4.5)
            residual = sample-smoothed/np.maximum(support[...,None],.02)
            residual -= residual[~blocked].mean(axis=0)
        else:
            residual = sample-cv.GaussianBlur(sample,(0,0),4.5)
            residual -= residual.mean(axis=(0,1))
        return residual,blocked

    def _patch(self):
        residual,blocked = self._sample()
        for _ in range(4):
            if not blocked.any():
                break
            replacement,also_blocked = self._sample()
            valid = blocked & ~also_blocked
            residual[valid] = replacement[valid]
            blocked &= also_blocked
        residual[blocked] = 0
        return residual

    def _balanced_patch(self):
        # Exposure and color vary between donor locations. Match their RMS to the
        # pooled donor estimate so the patch grid does not modulate the sky.
        residual = self._patch()
        intensity = np.sqrt(np.mean(luma(residual)**2))
        gain = np.clip(self.luma_sigma / max(intensity, 1e-6), .5, 1.9)
        amplitude = np.sqrt(np.mean(residual**2, axis=(0,1)))*gain
        balance = np.clip(self.sigma/np.maximum(amplitude,1e-6), .85, 1.18)
        return residual*np.float32(gain*balance)

    def _stripe(self):
        patch, overlap = self.donors.patch, self.donors.overlap
        result = np.zeros((patch,self.width,3),np.float32)
        for x in range(-self.stride,self.width,self.stride):
            lo,hi = max(x,0),min(x+patch,self.width)
            source = slice(lo-x,hi-x)
            result[:,lo:hi] += self._balanced_patch()[:,source]*self.weight[:,source]
        result[:overlap] += self.carry
        self.carry = result[self.stride:].copy()
        return result[:self.stride]

    def take(self,n):
        parts = [self.buffer]
        available = len(self.buffer)
        while available < n:
            stripe = self._stripe()
            parts.append(stripe)
            available += len(stripe)
        rows = np.concatenate(parts)
        self.buffer = rows[n:]
        return rows[:n]
