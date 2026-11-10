"""Non-UI render state for the wallpaper editor."""
from dataclasses import dataclass
from concurrent.futures import CancelledError
from functools import lru_cache
import json
from pathlib import Path
import tomllib
import threading

import cv2 as cv
import numpy as np
from PIL import Image

from . import astrometry, background, color, compose, flare, geometry, raw, sharpness
from . import sky_render, sun_glow, texture, video, video_grain


@dataclass(frozen=True)
class Settings:
    grade: bool = True
    stars: bool = True
    sky_grain: bool = True
    screen_grain: bool = True
    clean_flare: bool = True
    sun_glow: bool = True
    sharpness: str = "conventional"


@dataclass
class Variant:
    photo: np.ndarray
    sky_background: background.Background
    donors: texture.Donors | None = None
    stars: sky_render.Sky | None = None


class Project:
    def __init__(self, config):
        cv.setNumThreads(6)
        self.cfg=tomllib.loads(config.read_text())
        root=config.resolve().parent
        self.cache=root/self.cfg["paths"]["cache"]
        self.output=root/self.cfg["paths"]["output"]
        self.reference=root/"reference"
        self.source=self.cache/"source"
        self.work=self.cache/"work"
        self.render_cfg=self.cfg["render"]
        self.seed=self.render_cfg["seed"]
        self.work.mkdir(parents=True,exist_ok=True)
        photo=self.source/"art002e000192.jpg"
        self.clip=self.source/"melodysheep_clip.mp4"
        if not photo.is_file() or not self.clip.is_file():
            raise FileNotFoundError("Сначала загрузите исходники: python -m artemis --stage acquire")
        with Image.open(photo) as image:
            self.icc=image.info["icc_profile"]
            self.original=np.array(image.convert("RGB"))
        self.limb=geometry.detect_limb(self.original)
        self.ref,self.registration=video.select(self.original,self.clip,self.work,self.cfg["source"])
        self.registration_affine=np.array(self.registration["affine_native"])
        self.rotation=geometry.rotation_from_registration(self.registration_affine)
        self.size,self.affine=geometry.canvas(self.original.shape,self.limb,self.rotation,
                                             self.render_cfg["minimum_height"])
        self.keep=(int(2.15*max(self.limb.axes)),)*2
        self.astro=astrometry.solve(self.original,self.limb,self.cache,self.cfg["sky"],
                                    self.cfg["source"]["capture_utc"],self.seed)
        self.venus=self.astro["venus_xy"]
        _,j=self.astro["matches"]
        observed=self.astro["detections"].xy[j]
        h,w=self.original.shape[:2]
        edge=np.min(np.minimum(observed,np.array([w,h])-1-observed),axis=1)
        self.anchors=self.project(observed[edge<self.render_cfg["feather"]+12])
        self.points=self.project(np.vstack((observed,self.venus)))
        camera=self.astro["camera"]
        vectors=self.astro["vectors"]
        self.excluded=np.vstack((camera.pixel(vectors[camera.in_front(vectors)]),
                                 self.astro["detections"].xy,self.venus))
        self.nef=raw.develop(self.source/"art002e000192.NEF",self.work/"raw_active.npy")
        self.alignment=raw.check_alignment(self.original,self.nef) if self.nef is not None else None
        self.grade=None
        self.grain_model=None
        self.grain_lock=threading.Lock()

    def project(self, points):
        return np.asarray(points)@self.rotation.T+self.affine[:,2]

    def color_model(self):
        if self.grade is not None:
            return self.grade
        for report in (self.output/"measurements.json",self.reference/"measurements.json"):
            if not report.exists():
                continue
            measured=json.loads(report.read_text())
            if (measured["reference"].get("checkpoint_key")==self.registration["checkpoint_key"]
                and measured["execution"]["seed"]==self.seed):
                model=measured["color"]["chosen_model"]
                self.grade=color.Grade(model,np.array(measured["color"]["coefficients"][model],np.float32),self.limb)
                return self.grade
        original,target,valid,x,y=color.comparison_inputs(self.original,self.ref,self.registration_affine)
        models,holdout=color.fit(original,target,valid,x,y,self.limb,self.seed)
        def score(model):
            fitted=color.tiled_apply(model,original,x,y)
            metrics,_=color.metrics(fitted,target,valid,holdout,x,y,self.limb,original)
            return sum(metrics[k]["mean_de"]*weight for k,weight in
                       (("earth",.6),("clouds",.15),("atmosphere",.15),("background",.1)))
        self.grade=min(models,key=score)
        return self.grade

    @lru_cache(maxsize=2)
    def variant(self, grade, clean_flare, reduce_glow):
        if grade:
            photo=color.tiled_apply(self.color_model(),self.original)
        else:
            photo=self.original.astype(np.float32)/255
        raw.preserve_highlights(photo,self.original,self.nef,self.alignment,self.limb)
        if clean_flare:
            flare.apply(photo,self.limb,self.venus,self.cfg["flare"])
        if reduce_glow:
            sun_glow.apply(photo,self.limb,self.venus,self.excluded,
                           {**self.cfg.get("sun_glow", {}),"enabled":True})
        return Variant(photo,background.estimate(photo,self.limb))

    def render(self, options, *, viewport=None, sample=1, cancel=None):
        def check_cancel():
            if cancel is not None and cancel.is_set():
                raise CancelledError()
        check_cancel()
        variant=self.variant(options.grade,options.clean_flare,options.sun_glow)
        check_cancel()
        if options.stars and variant.stars is None:
            variant.stars,_=sky_render.prepare(self.astro,variant.photo,self.original,self.limb,
                                               self.affine,self.size,self.cfg["sky"],self.nef)
        if options.sky_grain and sample==1 and variant.donors is None:
            variant.donors=texture.find_donors(variant.photo,variant.sky_background,self.limb,
                                               self.excluded,self.rotation,self.render_cfg)
        check_cancel()
        stars=variant.stars if options.stars else None
        donors=variant.donors if options.sky_grain and sample==1 else None
        master,_=compose.compose(variant.photo,variant.sky_background,stars,self.affine,self.size,
                                 self.render_cfg["feather"],self.anchors,donors,
                                 viewport=viewport,sample=sample,cancel=cancel)
        if options.screen_grain and sample==1:
            if self.grain_model is None:
                with self.grain_lock:
                    if self.grain_model is None:
                        self.grain_model=video_grain.calibrate(self.clip,self.registration,self.work)
            check_cancel()
            video_grain.apply(master,self.grain_model,self.cfg["video_grain"],cancel=cancel)
        check_cancel()
        if options.sharpness=="conventional":
            sharpness.conventional(master,sigma=1.5/sample,cancel=cancel)
        elif options.sharpness=="multiscale" and sample==1:
            origin=np.array(viewport[:2] if viewport else (0,0))
            points=self.points if stars is None else np.vstack((self.points,stars.xy))
            affine=self.affine.copy()
            affine[:,2]-=origin
            sharpness.apply(master,self.limb,affine,points-origin,self.cfg["sharpness"],cancel=cancel)
        return master

    def preview(self, options, size, area, dimensions=(1600,800)):
        if area=="frame":
            w,h=size
            available_w,available_h=dimensions
            scale=min(1,available_w/w,available_h/h)
            preview_w,preview_h=max(1,round(w*scale)),max(1,round(h*scale))
            sample=max(1,int(min(self.size[0]/preview_w,self.size[1]/preview_h)))
            image=self.render(options,sample=sample)
            image=compose.resize_wallpaper(image,(max(1,round(w*scale)),max(1,round(h*scale))),
                                           tuple(k/sample for k in self.keep))
        else:
            if area=="aurora":
                center=self.project(self.cfg["sharpness"]["aurora_center_native"])
            else:
                # Representative extension on the left of the photographic footprint.
                center=np.array(self.size)*[.17,.52]
            w,h=min(1500,dimensions[0]),min(720,dimensions[1])
            x,y=np.rint(center-[w/2,h/2]).astype(int)
            image=self.render(options,viewport=(int(x),int(y),w,h))
        return Image.fromarray(np.uint8(np.rint(image.astype(np.float32)/257)))

    def preview_view(self, options, size, dimensions, center, zoom, cancel=None):
        """Render only the visible portion of the wallpaper in master coordinates."""
        cw,ch=dimensions
        layout=view_layout(self.size,size,self.keep,dimensions,center,zoom)
        canvas=Image.new("RGB",(cw,ch),(8,13,22))
        if layout is None:
            return canvas
        (sx0,sy0,sx1,sy1),(dx0,dy0,dx1,dy1),scale=layout
        pad=32
        x0=max(0,int(np.floor(sx0))-pad)
        y0=max(0,int(np.floor(sy0))-pad)
        x1=min(self.size[0],int(np.ceil(sx1))+pad)
        y1=min(self.size[1],int(np.ceil(sy1))+pad)
        sample=max(1,int(1/scale))
        image=self.render(options,viewport=(x0,y0,x1-x0,y1-y0),sample=sample,cancel=cancel)
        image=Image.fromarray(np.uint8(np.rint(image.astype(np.float32)/257)))
        bounds=((sx0-x0)/sample,(sy0-y0)/sample,(sx1-x0)/sample,(sy1-y0)/sample)
        part=image.resize((dx1-dx0,dy1-dy0),Image.Resampling.LANCZOS,box=bounds)
        canvas.paste(part,(dx0,dy0))
        return canvas

    def save(self, options, size, path: Path):
        master=self.render(options)
        image=compose.resize_wallpaper(master,size,self.keep)
        temporary=path.with_name(f".{path.stem}.partial{path.suffix}")
        try:
            compose.save_wallpaper(temporary,image,self.icc)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def view_layout(master, output, keep, canvas, center, zoom):
    """Map the visible output pixels to a master rectangle and screen rectangle."""
    mw,mh=master
    ow,oh=output
    cw,ch=canvas
    box,content=compose.frame_box(master,output,keep)
    left,top=(ow-content[0])/2,(oh-content[1])/2
    scale=min(cw/ow,ch/oh)*zoom
    ox=cw/2-center[0]*ow*scale
    oy=ch/2-center[1]*oh*scale
    x0=max(0,int(np.ceil(ox+left*scale)))
    y0=max(0,int(np.ceil(oy+top*scale)))
    x1=min(cw,int(np.floor(ox+(left+content[0])*scale)))
    y1=min(ch,int(np.floor(oy+(top+content[1])*scale)))
    if x1<=x0 or y1<=y0:
        return None
    ax,ay,bx,by=box
    sx0=ax+((x0-ox)/scale-left)*(bx-ax)/content[0]
    sx1=ax+((x1-ox)/scale-left)*(bx-ax)/content[0]
    sy0=ay+((y0-oy)/scale-top)*(by-ay)/content[1]
    sy1=ay+((y1-oy)/scale-top)*(by-ay)/content[1]
    # Clamp tiny floating point excursions at the source boundaries.
    source=(max(0,sx0),max(0,sy0),min(mw,sx1),min(mh,sy1))
    return source,(x0,y0,x1,y1),scale*content[0]/(bx-ax)
