"""One linear-light warp, seamless tiled composite, and lossless deliverables."""
from dataclasses import replace
from concurrent.futures import CancelledError
import struct
import zlib

import cv2 as cv
import numpy as np
from PIL import Image
import tifffile

from .background import sky_at
from .texture import Grain
from .color import srgb_decode, srgb_encode, smoothstep
from .sky_render import render as render_stars


def compose(graded, background, sky, affine, size, feather, anchors, donors, *,
            viewport=None, sample=1, cancel=None):
    x0,y0,view_width,view_height=viewport or (0,0,*size)
    width,height=((view_width+sample-1)//sample,(view_height+sample-1)//sample)
    origin=np.array([x0,y0])+(sample-1)/2
    anchors=(anchors-origin)/sample
    if sky is not None:
        if sample>1:
            profile=sky.profiles.copy()
            profile[:,:2]=np.hypot(profile[:,:2]/sample,np.sqrt(1/12))
            sky=replace(sky,xy=(sky.xy-origin)/sample,profiles=profile,flux=sky.flux/sample**2)
        elif x0 or y0:
            sky=replace(sky,xy=sky.xy-origin)
    h,w=graded.shape[:2]
    linear=srgb_decode(graded)
    if cancel is not None and cancel.is_set():
        raise CancelledError()
    output=np.empty((height,width,3),np.uint16)
    rotation=affine[:,:2]
    translation=affine[:,2]
    xx=np.arange(width,dtype=np.float32)[None,:]*sample+origin[0]-translation[0]
    grain=Grain(linear,rotation,donors,width) if donors is not None else None
    if cancel is not None and cancel.is_set():
        raise CancelledError()
    if sample>1:
        linear=cv.resize(linear,((w+sample-1)//sample,(h+sample-1)//sample),interpolation=cv.INTER_AREA)
    for start in range(0,height,84):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        end=min(start+84,height)
        yy=np.arange(start,end,dtype=np.float32)[:,None]*sample+origin[1]-translation[1]
        sx=np.float32(rotation[0,0]*xx+rotation[1,0]*yy)
        sy=np.float32(rotation[0,1]*xx+rotation[1,1]*yy)
        space,edge_sigma,distance_outside=sky_at(background,sx,sy,w,h)
        synthetic=srgb_decode(space)
        if sky is not None:
            render_stars(synthetic,0,start,sky,rotation)
        # A genuine Lanczos4 warp at a 1:1 mapping. Avoid negative lobes in near-black.
        mx,my=sx,sy
        if sample>1:
            mx=(sx+.5)*(linear.shape[1]/w)-.5
            my=(sy+.5)*(linear.shape[0]/h)-.5
        photo=cv.remap(linear,mx,my,cv.INTER_LANCZOS4,borderMode=cv.BORDER_REPLICATE)
        photo=np.maximum(photo,0)
        distance=np.minimum(np.minimum(sx,w-1-sx),np.minimum(sy,h-1-sy))
        alpha=smoothstep(distance/feather)[...,None]
        for x,y in anchors[(anchors[:,1]>=start-11)&(anchors[:,1]<end+11)&
                           (anchors[:,0]>=-11)&(anchors[:,0]<width+11)]:
            u0,u1=max(0,round(x)-11),min(width,round(x)+12)
            v0,v1=max(start,round(y)-11),min(end,round(y)+12)
            u=np.arange(u0,u1)-x
            v=np.arange(v0,v1)[:,None]-y
            peak=np.exp(-(u*u+v*v)/21)[...,None]
            window=alpha[v0-start:v1-start,u0:u1]
            window+=np.float32((1-window)*peak)
        mixed=photo*alpha+synthetic*(1-alpha)
        # Complementary power weighting preserves grain variance through the feather.
        result=srgb_encode(mixed)
        if grain is not None:
            noise_blend=np.exp(-(distance_outside/donors.decay)**2)
            noise_scale=1+noise_blend*(np.clip(edge_sigma/grain.sigma,.15,2.5)-1)
            result+=grain.take(end-start)*noise_scale*np.sqrt(1-alpha**2)
        output[start:end]=np.rint(np.clip(result,0,1)*65535).astype(np.uint16)
    return output, grain.sigma if grain is not None else None


def png_with_profile(path, rgb, icc):
    ok, encoded=cv.imencode(".png",rgb[...,::-1], [cv.IMWRITE_PNG_COMPRESSION,4])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    header=encoded[:33].tobytes()
    payload=b"sRGB IEC61966-2.1\0\0"+zlib.compress(icc)
    chunk=b"iCCP"+payload
    with path.open("wb") as out:
        out.write(header)
        out.write(struct.pack(">I",len(payload)))
        out.write(chunk)
        out.write(struct.pack(">I",zlib.crc32(chunk)))
        out.write(encoded[33:])


def frame_box(source_size, target_size, keep=(0,0)):
    """Center-crop to the target aspect without cutting the protected footprint."""
    mw,mh=source_size
    w,h=target_size
    crop_width,crop_height=min(mw,mh*w/h),min(mh,mw*h/w)
    expand=max(1,keep[0]/crop_width,keep[1]/crop_height)
    crop_width,crop_height=min(mw,crop_width*expand),min(mh,crop_height*expand)
    box=((mw-crop_width)/2,(mh-crop_height)/2,(mw+crop_width)/2,(mh+crop_height)/2)
    factor=min(w/crop_width,h/crop_height)
    content=(min(w,max(1,round(crop_width*factor))),min(h,max(1,round(crop_height*factor))))
    return box,content


def resize_wallpaper(master, size, keep=(0,0), *, black_padding=False):
    """Resample the display-referred RGB image in linear light."""
    mh,mw=master.shape[:2]
    w,h=size
    box,content=frame_box((mw,mh),size,keep)
    if (w,h)==(mw,mh):
        return master
    left,top=(w-content[0])//2,(h-content[1])//2
    rgb=np.empty((h,w,3),np.uint16)
    for c in range(3):
        channel=Image.fromarray(srgb_decode(master[...,c].astype(np.float32)/65535))
        resized=np.asarray(channel.resize(content,Image.Resampling.LANCZOS,box=box))
        codes=np.uint16(np.rint(np.clip(srgb_encode(resized),0,1)*65535))
        if content!=(w,h):
            rgb[...,c]=0 if black_padding else np.median(codes[[0,-1],:])
        rgb[top:top+content[1],left:left+content[0],c]=codes
    return rgb


def save_wallpaper(path, rgb, icc):
    if path.suffix.lower() in (".tif",".tiff"):
        tifffile.imwrite(path,rgb,photometric="rgb",compression="deflate",
                         predictor=True,iccprofile=icc,metadata=None)
    elif path.suffix.lower()==".png":
        png_with_profile(path,rgb,icc)
    else:
        raise ValueError("Выберите PNG или TIFF.")


def save_exports(master, output, cfg, icc, *, black_padding=False):
    save_wallpaper(output/"artemis_ii_21x9_master.png",master,icc)
    save_wallpaper(output/"artemis_ii_21x9_master_16bit.tif",master,icc)
    w,h=cfg["desktop_width"],cfg["desktop_height"]
    rgb=np.uint8(np.rint(resize_wallpaper(master,(w,h),black_padding=black_padding).astype(np.float32)/257))
    Image.fromarray(rgb).save(output/"artemis_ii_3440x1440.png",icc_profile=icc)
    preview=Image.fromarray(rgb).resize((1720,720),Image.Resampling.LANCZOS)
    preview.save(output/"preview.jpg",quality=94,subsampling=0,icc_profile=icc)
