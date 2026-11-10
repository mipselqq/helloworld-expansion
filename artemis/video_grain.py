"""Measure fixed screen texture in the video and synthesize its coarse luma layer."""
from dataclasses import dataclass
from concurrent.futures import CancelledError
import json

import cv2 as cv
import numpy as np
from scipy.optimize import minimize_scalar

from .acquire import write_json
from .color import luma
from .video import decode

SIZE = (3840, 2160)
TONE_EDGES = np.array([0, 20, 38, 65, 105, 165, 210], dtype=float)
BANDS = ((1.25, 2.5), (2.5, 5.), (5., 10.))


def blur(image, sigma):
    return cv.GaussianBlur(image, (0, 0), sigma)


def band(image, bounds):
    lo, hi = bounds
    return blur(image, lo) - blur(image, hi)


def warp(image, transform):
    return cv.warpAffine(image, transform, SIZE, flags=cv.INTER_CUBIC | cv.WARP_INVERSE_MAP,
                         borderMode=cv.BORDER_REFLECT_101)


def screen_frames(clip, registration, work):
    candidates = registration["candidates"]
    indices = [row["index"] for row in candidates]
    key = {"indices": indices, "clip": registration["checkpoint_key"]["clip_sha256"],
           "size": list(SIZE), "version": 1}
    frame_path, index_path = work/"grain_screen_frames.npy", work/"grain_screen_frames.json"
    if frame_path.exists() and index_path.exists() and json.loads(index_path.read_text()) == key:
        return np.load(frame_path, mmap_mode="r")
    expr = "+".join(f"eq(n\\,{i})" for i in indices)
    spec = f"select={expr},scale=3840:2160:flags=area+accurate_rnd:in_color_matrix=bt709"
    temp = frame_path.with_suffix(".partial.npy")
    frames = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint16,
                                       shape=(len(indices), SIZE[1], SIZE[0]))
    for i,rgb in zip(range(len(indices)),decode(clip,spec,SIZE,np.uint16),strict=True):
        frames[i] = np.rint(luma(rgb.astype(np.float32))).astype(np.uint16)
    del frames
    temp.replace(frame_path)
    write_json(index_path,key)
    return np.load(frame_path,mmap_mode="r")


def residual_group(frames, transforms, indices):
    # Estimate the moving photograph in this group's coordinates. Leave each
    # analyzed frame out when predicting its scene to avoid self-correlation.
    aligned = np.stack([warp(frames[i].astype(np.float32)/257,transforms[i]) for i in indices])
    combined = aligned.sum(axis=0)
    residual = []
    for i,frame in zip(indices,aligned):
        scene = (combined-frame)/(len(indices)-1)
        predicted = warp(scene,cv.invertAffineTransform(transforms[i]))
        residual.append(np.clip(frames[i].astype(np.float32)/257-predicted,-15,15))
    return np.median(residual,axis=0).astype(np.float32),combined/len(indices)


def screen_covariances(frames, registration):
    candidates = registration["candidates"]
    affine = np.array([np.array(row["affine_1920"])*2 for row in candidates])
    affine[:,:,2] += .5
    ref = next(i for i,row in enumerate(candidates) if row["index"]==registration["best"]["index"])
    transforms = affine@np.vstack((cv.invertAffineTransform(affine[ref]),[0,0,1]))
    first,first_scene = residual_group(frames,transforms,range(0,len(frames),2))
    second,second_scene = residual_group(frames,transforms,range(1,len(frames),2))
    scene = (first_scene+second_scene)*.5
    y,x = np.indices(scene.shape,dtype=np.float32,sparse=True)
    delta = transforms[0]-transforms[-1]
    drift = np.hypot(delta[0,0]*x+delta[0,1]*y+delta[0,2],
                     delta[1,0]*x+delta[1,1]*y+delta[1,2])
    active = ((x>SIZE[0]*.132)&(x<SIZE[0]*.868)&(y>85)&(y<SIZE[1]-85)&(drift>3.2))
    tone = blur(scene,7)
    structure = np.sqrt(blur(band(scene,(1.4,4.7))**2,5).clip(0))
    samples=[]
    masks=[]
    for lo,hi in zip(TONE_EDGES[:-1],TONE_EDGES[1:]):
        mask = active&(tone>=lo)&(tone<hi)
        mask &= structure<=np.percentile(structure[mask],35)
        masks.append(mask)
        samples.append({"luma_median":float(np.median(tone[mask])),
                        "pixels":int(mask.sum()),"bands":[]})
    for bounds in BANDS:
        a,b=band(first,bounds),band(second,bounds)
        controls=(np.roll(b,14,axis=0),np.roll(b,-17,axis=1))
        for mask,row in zip(masks,samples):
            values = a[mask]
            v = values-values.mean()
            ys = [q[mask] for q in (b,*controls)]
            covariance = np.array([np.mean(v*(q-q.mean())) for q in ys])
            common = max(float(covariance[0])-float(np.mean(np.abs(covariance[1:]))),0)
            rho = float(covariance[0]/max(v.std()*ys[0].std(),1e-8))
            row["bands"].append({"coherent_rms_8bit":float(np.sqrt(common)),
                                 "screen_correlation":rho,
                                 "covariance_8bit_sq":covariance.tolist()})
    return samples


def kernel(sigma, broad):
    # Match OpenCV's discrete kernels and their normalization in the renderer.
    radius = int(np.ceil(4.3*broad))
    width = 2*radius+1
    short, long = (cv.getGaussianKernel(width,s).ravel() for s in (sigma,broad))
    return np.outer(short,short)-np.outer(long,long)


def filter_gains(sigma, broad):
    tap = kernel(sigma,broad)
    pad = np.pad(tap,50)
    norm = np.linalg.norm(tap)
    return np.array([np.linalg.norm(band(pad,pair))/norm for pair in BANDS])


def fit_noise(samples, registration):
    observed = np.array([[b["coherent_rms_8bit"] for b in row["bands"]] for row in samples])
    # The quiet dark and light regions constrain the filter shape. The middle
    # tones are retained for fitting opacity, where scene leakage is stronger.
    anchor = observed[[0,1,5]]
    weight = np.array([1.,1.,.65])
    broad = 6.5

    def levels(gain, samples):
        return (samples @ (gain*weight))/(gain @ (gain*weight))

    def loss(sigma):
        gain = filter_gains(sigma,broad)
        prediction = levels(gain,anchor)[:,None]*gain
        return float(np.mean(((prediction-anchor)/np.maximum(anchor,.01))**2))

    fit = minimize_scalar(loss,bounds=(.35,3.3),method="bounded")
    screen_sigma = float(fit.x)
    gain = filter_gains(screen_sigma,broad)
    opacity = levels(gain,observed)
    factor = np.sqrt(np.linalg.det(np.array(registration["affine_native"])[:,:2]))/2
    curve = [[0.,0.], *[[row["luma_median"],float(k)] for row,k in zip(samples,opacity)],
             [255.,0.]]
    return {"analysis_size":list(SIZE),"frame_indices":[r["index"] for r in registration["candidates"]],
            "screen_to_master_scale":float(1/factor),"screen_bands_sigma_px":BANDS,
            "samples":samples,"opacity_curve_8bit":curve,"sigma_master_px":screen_sigma/factor,
            "broad_master_px":broad/factor,"band_gains":gain.tolist(),"fit_loss":float(fit.fun)}


def calibrate(clip, registration, work):
    path = work/"video_grain_model.json"
    key = {"version":1,"reference":registration["checkpoint_key"],
           "affines":[r["affine_1920"] for r in registration["candidates"]]}
    for candidate in (path,work.parent.parent/"reference"/"video_grain_model.json"):
        if candidate.exists():
            data = json.loads(candidate.read_text())
            if data["key"]==key:
                return data["model"]
    frames = screen_frames(clip,registration,work)
    samples = screen_covariances(frames,registration)
    model = fit_noise(samples,registration)
    write_json(path,{"key":key,"model":model})
    return model


@dataclass
class Field:
    """Seeded band-limited noise with continuous convolution across tiles."""
    width: int
    sigma: float
    broad: float
    seed: int

    def __post_init__(self):
        tap = kernel(self.sigma,self.broad)
        self.pad = tap.shape[0]//2
        self.norm = float(np.linalg.norm(tap))
        self.rng = np.random.default_rng(self.seed)
        self.pending = self.rng.standard_normal((2*self.pad,self.width),dtype=np.float32)

    def take(self,n):
        raw = np.concatenate((self.pending,self.rng.standard_normal((n,self.width),dtype=np.float32)))
        self.pending = raw[-2*self.pad:].copy()
        size=(2*self.pad+1,)*2
        short = cv.GaussianBlur(raw,size,self.sigma)
        broad = cv.GaussianBlur(raw,size,self.broad)
        return (short[self.pad:self.pad+n]-broad[self.pad:self.pad+n])/self.norm


def apply(master, model, cfg, cancel=None):
    """Add the luma overlay to the assembled image, leaving scene geometry alone."""
    curve = np.asarray(model["opacity_curve_8bit"],dtype=np.float32)
    h,w = master.shape[:2]
    stream = Field(w,model["sigma_master_px"],model["broad_master_px"],cfg["seed"])
    sum_delta=0.
    power=0.
    limited=0
    for start in range(0,h,96):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        tile = master[start:start+96].astype(np.float32)/65535
        tone = luma(tile)*255
        sigma = np.interp(tone,curve[:,0],curve[:,1]).astype(np.float32)
        delta = stream.take(len(tile))*sigma*(cfg["strength"]/255)
        used = np.clip(delta,-tile.min(axis=-1),1-tile.max(axis=-1))
        limited += int(np.count_nonzero(delta!=used))
        sum_delta += float(np.sum(used,dtype=np.float64))
        power += float(np.sum(used**2,dtype=np.float64))
        master[start:start+len(tile)] = np.uint16(np.rint((tile+used[...,None])*65535))
    return {"enabled":True,"strength":cfg["strength"],"seed":cfg["seed"],
            "applied_mean_luma_8bit":sum_delta/(h*w)*255,
            "applied_rms_luma_8bit":float(np.sqrt(power/(h*w))*255),
            "headroom_limited_pixels":limited,**model}
