"""Native NEF development and measured correspondence to the public JPEG."""
import cv2 as cv
import numpy as np
import rawpy

from .color import luma, srgb_encode, smoothstep


def develop(raw_path, checkpoint):
    if not raw_path.exists():
        return None
    if checkpoint.exists():
        return np.load(checkpoint, mmap_mode="r")
    with rawpy.imread(str(raw_path)) as sensor:
        s = sensor.sizes
        rgb = sensor.postprocess(output_bps=16, use_camera_wb=True, no_auto_bright=True,
                                 gamma=(1,1), highlight_mode=rawpy.HighlightMode.Blend)
        active = rgb[s.crop_top_margin:s.crop_top_margin+s.crop_height,
                     s.crop_left_margin:s.crop_left_margin+s.crop_width]
    np.save(checkpoint, active)
    return active


def check_alignment(rgb, raw):
    h,w = rgb.shape[:2]
    jpg = cv.cvtColor(rgb,cv.COLOR_RGB2GRAY).astype(np.float32)/255
    nef = srgb_encode(luma(raw.astype(np.float32)/65535))
    results=[]
    for y in range(600, h-420, 400):
        for x in range(1500, w-1450, 440):
            p = jpg[y:y+160,x:x+160]
            q = nef[y:y+160,x:x+160]
            if p.mean()<.2:
                continue
            p = p-cv.GaussianBlur(p,(0,0),7)
            q = q-cv.GaussianBlur(q,(0,0),7)
            shift, response = cv.phaseCorrelate(p,q,cv.createHanningWindow((160,160),cv.CV_32F))
            if response>.4:
                results.append(shift)
    values=np.array(results)
    return {"patches":len(values), "median_xy_px":np.median(values,axis=0).tolist(),
            "rms_px":float(np.sqrt(np.mean(np.sum(values**2,axis=1)))),
            "near_clipped_jpeg_pixels":int((rgb.min(2)>=250).sum())}


def preserve_highlights(graded, rgb, raw, alignment, limb):
    """Restore luminance variation in JPEG near-clipped highlights using the NEF."""
    if raw is None or alignment["rms_px"]>.45:
        return 0
    h,w=rgb.shape[:2]
    changed=0
    xx=np.arange(w)[None,:]
    for y in range(0,h,96):
        lo,hi=max(0,y-16),min(h,y+112)
        source=rgb[lo:hi].astype(np.float32)/255
        jj=luma(source)
        rr=srgb_encode(luma(raw[lo:hi].astype(np.float32)/65535))
        j0=cv.GaussianBlur(jj,(0,0),2.5)
        r0=cv.GaussianBlur(rr,(0,0),2.5)
        weight=smoothstep((source.min(2)-.96)/.035)*(limb.radius(xx,np.arange(lo,hi)[:,None])<1.02)
        detail=np.clip((rr-r0)*1.15-(jj-j0),-.025,.025)*weight
        rows=slice(y-lo,min(y+96,h)-lo)
        graded[y:y+96]=np.clip(graded[y:y+96]+detail[rows,...,None],0,1)
        changed+=int(np.count_nonzero(weight[rows]))
    return changed
