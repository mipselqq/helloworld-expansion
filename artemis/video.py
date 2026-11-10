"""Compare adjacent native 8K frames and register the shared photograph."""
import hashlib
import json
import subprocess

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

from .acquire import sha256, write_json


def decode(path, vf, size, dtype, count=None):
    w, h = size
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "8", "-i", str(path),
           "-filter_threads", "3", "-vf", vf, "-an", "-fps_mode", "passthrough"]
    if count:
        cmd += ["-frames:v", str(count)]
    cmd += ["-pix_fmt", "rgb48le" if dtype == np.uint16 else "rgb24", "-f", "rawvideo", "-"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as process:
        block_size = w*h*3*np.dtype(dtype).itemsize
        while block := process.stdout.read(block_size):
            yield np.frombuffer(block, dtype=dtype).reshape(h, w, 3)
        if process.wait():
            raise RuntimeError("FFmpeg frame decode failed")


def gray(rgb):
    return cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)


def sift_alignment(source, frame):
    sift = cv.SIFT_create(nfeatures=5000, contrastThreshold=.016)
    source_small = cv.resize(source, None, fx=.30, fy=.30, interpolation=cv.INTER_AREA)
    a, da = sift.detectAndCompute(gray(source_small), None)
    b, db = sift.detectAndCompute(gray(frame), None)
    pairs = cv.BFMatcher(cv.NORM_L2).knnMatch(da, db, k=2)
    accepted = [m for m, n in pairs if m.distance < .74*n.distance]
    p = (np.array([a[m.queryIdx].pt for m in accepted])+0.5)/.30-0.5
    q = np.array([b[m.trainIdx].pt for m in accepted])
    affine, keep = cv.estimateAffine2D(p, q, method=cv.RANSAC, ransacReprojThreshold=2,
                                      maxIters=5000, confidence=.999)
    if affine is None or keep.sum() < 25:
        raise ValueError("Insufficient NASA-to-video feature matches")
    keep = keep.ravel().astype(bool)
    return affine, p[keep], q[keep]


def structure(img):
    image = img.astype(np.float32)
    mean = cv.GaussianBlur(image, (0, 0), 5)
    residual = cv.GaussianBlur(image, (0, 0), .75)-mean
    return residual / np.sqrt(cv.GaussianBlur(residual**2, (0, 0), 5)+3)


def refine(source, frame, affine):
    h, w = frame.shape[:2]
    warped = cv.warpAffine(gray(source), affine, (w, h), flags=cv.INTER_AREA)
    a, b = structure(warped), structure(gray(frame))
    pairs = []
    for y in range(48, h-48, 36):
        for x in range(48, w-48, 36):
            patch = a[y-16:y+17, x-16:x+17]
            if patch.std() < .22 or not (32 < warped[y, x] < 232):
                continue
            results = cv.matchTemplate(b[y-22:y+23, x-22:x+23], patch, cv.TM_CCOEFF_NORMED)
            _, score, _, (dx, dy) = cv.minMaxLoc(results)
            if score < .62 or not (0 < dx < 12 and 0 < dy < 12):
                continue
            v = results[dy, dx]
            l, r, u, d = results[dy, dx-1], results[dy, dx+1], results[dy-1, dx], results[dy+1, dx]
            sx = .5*(l-r)/min(l-2*v+r, -1e-6)
            sy = .5*(u-d)/min(u-2*v+d, -1e-6)
            pairs.append((x, y, x+dx+sx-6, y+dy+sy-6, score))
    pairs = np.array(pairs)
    if len(pairs) < 30:
        raise ValueError("Too few high-correlation refinement patches")
    p, q = pairs[:, :2], pairs[:, 2:4]
    # A spatial holdout of disjoint patches measures registration independently.
    test = ((p[:, 0]//108 + 2*(p[:, 1]//108)) % 5) == 0
    x = np.column_stack((p-np.array([w/2,h/2]), np.ones(len(p))))
    fit = least_squares(lambda c: (x[~test] @ c.reshape(3,2)-(q-p)[~test]).ravel(),
                        np.zeros(6), loss="soft_l1", f_scale=.35)
    delta = fit.x.reshape(3,2)
    correction = np.column_stack((np.eye(2)+delta[:2].T, delta[2]-np.array([w/2,h/2]) @ delta[:2]))
    refined = correction @ np.vstack((affine, [0,0,1]))
    error = np.linalg.norm(x @ delta+p-q, axis=1)
    # A zoom-in improves sampling, so evaluate coherent detail instead of raw edge energy.
    corr = float(np.median(pairs[:, 4]))
    return refined, {"patches": len(p), "holdout_patches": int(test.sum()),
                     "holdout_rms_px": float(np.sqrt(np.mean(error[test]**2))),
                     "holdout_p95_px": float(np.percentile(error[test], 95)),
                     "patch_correlation": corr}


def checkpoint_key(source, clip, cfg):
    return {"clip_sha256":sha256(clip),
            "photo_rgb_sha256":hashlib.sha256(np.ascontiguousarray(source)).hexdigest(),
            "settings":{key:cfg[key] for key in ("clip_start","target_second","frame_window","frame_step")}}


def select(source, clip, cache, cfg, force=False):
    frame_path, json_path = cache / "reference.npy", cache / "registration.json"
    key=checkpoint_key(source,clip,cfg)
    if not force and frame_path.exists() and json_path.exists():
        saved=json.loads(json_path.read_text())
        if saved.get("checkpoint_key")==key:
            return np.load(frame_path, mmap_mode="r"),saved
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v",
                                    "-show_streams", "-show_frames", "-show_entries",
                                    "frame=best_effort_timestamp_time:stream=width,height,color_space,color_range,color_transfer,color_primaries,r_frame_rate",
                                    "-of", "json", str(clip)])
    info = json.loads(probe)
    stream = info["streams"][0]
    frames = info["frames"]
    target = cfg["target_second"]-cfg["clip_start"]
    indices = [i for i, f in enumerate(frames) if
               abs(float(f["best_effort_timestamp_time"])-target) <= cfg["frame_window"]]
    indices = indices[::cfg["frame_step"]]
    expr = "+".join(f"eq(n\\,{i})" for i in indices)
    results = []
    for idx, frame in zip(indices, decode(clip, f"select={expr},scale=1920:1080:flags=area:in_color_matrix=bt709",
                                         (1920, 1080), np.uint8), strict=True):
        initial, p, q = sift_alignment(source, frame)
        affine, stats = refine(source, frame, initial)
        stats.update(index=idx, approx_video_seconds=cfg["clip_start"]+float(frames[idx]["best_effort_timestamp_time"]),
                     sift_inliers=len(p), affine_1920=affine.tolist())
        results.append(stats)
    # Correlation rewards matching detail and penalizes motion blur, noise and non-shared overlays.
    best = max(results, key=lambda x: x["patch_correlation"]-.04*x["holdout_rms_px"]
               -.003*abs(x["approx_video_seconds"]-cfg["target_second"]))
    w, h = stream["width"], stream["height"]
    spec = f"select=eq(n\\,{best['index']}),scale={w}:{h}:flags=accurate_rnd+full_chroma_int:in_color_matrix=bt709"
    (rgb,) = decode(clip, spec, (w,h), np.uint16, 1)
    np.save(frame_path, rgb)
    scale = w/1920
    affine = np.array(best["affine_1920"])*scale
    affine[:, 2] += (scale-1)/2
    report = {"stream": stream, "best": best, "candidates": results, "affine_native": affine.tolist(), "checkpoint_key":key}
    write_json(json_path, report)
    return rgb, report
