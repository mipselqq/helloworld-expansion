"""Limb measurement and unit-scale native-to-wallpaper geometry."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d


@dataclass
class Limb:
    center: np.ndarray
    axes: np.ndarray
    angle: float
    points: np.ndarray
    rms: float

    def radius(self, x, y):
        c, s = np.cos(self.angle), np.sin(self.angle)
        x, y = x - self.center[0], y - self.center[1]
        return np.hypot((c*x+s*y)/self.axes[0], (-s*x+c*y)/self.axes[1])

    def boundary(self, azimuth):
        c, s = np.cos(self.angle), np.sin(self.angle)
        x, y = np.cos(azimuth)*self.axes[0], np.sin(azimuth)*self.axes[1]
        return self.center + np.column_stack((c*x-s*y, s*x+c*y))


def detect_limb(rgb):
    h, w = rgb.shape[:2]
    small = cv.resize(rgb, (w//4, h//4), interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(small, cv.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.uint8(np.clip(gray * 255, 0, 255))
    _, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)
    x0, y0, bw, bh = cv.boundingRect(contour)
    center = np.array([x0+(bw-1)/2, y0+(bh-1)/2])*4+1.5
    radius = (bw+bh)
    angles = np.linspace(0, 2*np.pi, 900, endpoint=False)
    offset = np.arange(-90, 66, dtype=float)
    x, y = [(c + func(angles)[:, None]*(radius+offset)).astype(np.float32)
            for c, func in zip(center, (np.cos, np.sin))]
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY).astype(np.float32)
    spokes = cv.remap(gray, x, y, cv.INTER_LINEAR)
    # Choose the dominant gradient of the atmospheric edge after suppressing JPEG grain.
    gradient = -gaussian_filter1d(spokes, 2.0, axis=1, order=1)
    j = np.clip(gradient.argmax(1), 1, len(offset)-2)
    i = np.arange(len(j))
    peak = gradient[i, j]
    left, right = gradient[i, j-1], gradient[i, j+1]
    sub = np.clip(0.5*(left-right)/np.minimum(left-2*peak+right, -1e-6), -.5, .5)
    edges = np.column_stack((x[i, j]+sub*np.cos(angles), y[i, j]+sub*np.sin(angles)))

    def residual(p):
        model = Limb(p[:2], p[2:4], p[4], edges, 0)
        return (model.radius(*edges.T)-1) * np.mean(p[2:4])

    bounds = (np.r_[center-80, radius-100, radius-100, -4],
              np.r_[center+80, radius+100, radius+100, 4])
    selected = peak > np.percentile(peak, 10)
    fit = least_squares(lambda p: residual(p)[selected], [*center, radius, radius-4, .1],
                        bounds=bounds, loss="soft_l1", f_scale=3)
    selected &= np.abs(residual(fit.x)) < 8
    fit = least_squares(lambda p: residual(p)[selected], fit.x, bounds=bounds,
                        loss="soft_l1", f_scale=2)
    return Limb(fit.x[:2], fit.x[2:4], fit.x[4], edges[selected],
                float(np.sqrt(np.mean(residual(fit.x)[selected]**2))))


def rotation_from_registration(affine, snap_degrees=.04):
    u, _, vt = np.linalg.svd(affine[:2, :2])
    rotation = u @ vt
    angle = np.arctan2(rotation[1, 0], rotation[0, 0])
    cardinal = round(angle / (np.pi/2))
    if abs(np.degrees(angle-cardinal*np.pi/2)) <= snap_degrees:
        rotations = [np.eye(2), np.array([[0, -1], [1, 0]]), -np.eye(2),
                     np.array([[0, 1], [-1, 0]])]
        return rotations[cardinal % 4]
    return rotation


def canvas(shape, limb, rotation, min_height):
    h, w = shape[:2]
    corners = (np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])-limb.center) @ rotation.T
    radius = np.abs(corners).max(axis=0)
    # Integer multiples of 7:3, with every original pixel center inside the canvas.
    units = int(np.ceil(max((radius[0]*2+1)/7, (radius[1]*2+1)/3, min_height/3)))
    width, height = 7*units, 3*units
    output_center = (np.array([width, height])-1)/2
    affine = np.column_stack((rotation, output_center - rotation @ limb.center))
    return (width, height), affine
