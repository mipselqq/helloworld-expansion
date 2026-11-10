"""Report projections, diagnostics, and independent output checks."""
import csv
import io
import zipfile

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, Sip
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shapefile

from .astrometry import radec
from .catalog import unit
from .color import luma, srgb_encode, srgb_decode
from .sky_render import rms


def csv_write(path, fields, values):
    with path.open("w", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(fields)
        writer.writerows(values)


def angular_distance(a,b):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)), np.dot(a,b))))


def make_wcs(cam, affine, dimensions, out, capture_utc):
    # Exact local CD followed by a polynomial approximation to the inverse lens.
    w,h=dimensions
    r,t=affine[:,:2],affine[:,2]
    ref=cam.p[4:6]@r.T+t
    bore=cam.rotation[2]
    ra,dec=np.radians(radec([bore])[0])
    east=np.array([-np.sin(ra),np.cos(ra),0])
    north=np.cross(bore,east)
    stretch,shear=cam.p[10:12] if len(cam.p)>8 else (0,0)
    jacobian=np.array([[1,shear],[0,1+stretch]])
    cd=np.degrees(np.stack((east,north))@cam.rotation[:2].T @ np.linalg.inv(jacobian) @ r.T/cam.p[3])
    xx,yy=np.meshgrid(np.linspace(0,w-1,41),np.linspace(0,h-1,35))
    sample=np.column_stack((xx.ravel(),yy.ravel()))
    v=cam.sky((sample-t)@r)
    uv=np.degrees(np.column_stack((v@east,v@north))/(v@bore)[:,None])@np.linalg.inv(cd).T
    scale=cam.p[3]
    p=(sample-ref)/scale
    terms=[(i,j) for i in range(10) for j in range(10-i) if i+j>=2]
    design=np.column_stack([p[:,0]**i*p[:,1]**j for i,j in terms])
    fit=np.linalg.lstsq(design,uv-(sample-ref),rcond=None)[0]
    a,b=np.zeros((10,10)),np.zeros((10,10))
    for (i,j),coefficient in zip(terms,fit):
        a[i,j],b[i,j]=coefficient/scale**(i+j)
    wcs=WCS(naxis=2)
    wcs.wcs.crpix=ref+1
    wcs.wcs.crval=np.degrees([ra,dec])
    wcs.wcs.ctype=["RA---TAN-SIP","DEC--TAN-SIP"]
    wcs.wcs.cd=cd
    wcs.sip=Sip(a,b,None,None,ref+1)
    hdr=wcs.to_header(relax=True)
    hdr["WCSNAME"]="Artemis II apparent J2000"
    hdr["MJD-OBS"]=Time(capture_utc,scale="utc").mjd
    hdr["DATE-OBS"]=capture_utc
    hdr.add_comment("Coordinates approximate the observer-apparent J2000 vector directions.")
    hdr.add_comment("Camera model and proper motion corrections are documented in report.md.")
    fits.PrimaryHDU(np.empty((0,0),np.uint8),header=hdr).writeto(out,overwrite=True)
    offset=np.linalg.norm(wcs.all_world2pix(radec(v),0,quiet=True)-sample,axis=1)
    return {"sip_rms_px":rms(offset),"sip_max_px":float(offset.max())}


def coastlines(path, astro):
    cam,state=astro["camera"],astro["state"]
    observer=-state["earth"]@state["itrf"].T
    a,b=6378.137,6356.752
    e2=1-b*b/(a*a)
    with zipfile.ZipFile(path) as archive:
        shp=next(n for n in archive.namelist() if n.endswith(".shp"))
        reader=shapefile.Reader(shp=io.BytesIO(archive.read(shp)))
        for shape in reader.iterShapes():
            coords=np.asarray(shape.points)
            if len(coords)<8:
                continue
            lon,lat=np.radians(coords).T
            n=a/np.sqrt(1-e2*np.sin(lat)**2)
            surface=n[:,None]*np.column_stack((np.cos(lon)*np.cos(lat),np.sin(lon)*np.cos(lat),
                                               (1-e2)*np.sin(lat)))
            norm=surface/np.array([a*a,a*a,b*b])
            direction=observer-surface
            visible=(norm*direction).sum(1)>0
            xy=cam.pixel(-direction@state["itrf"])
            xy[~visible]=np.nan
            if visible.any():
                yield xy


def astronomy_plot(path,rgb,astro,limb,sky,affine,dimensions,coast):
    cam, stars, vectors = astro["camera"],astro["stars"],astro["vectors"]
    i,j=astro["matches"]
    observed=astro["detections"].xy[j]
    predicted=cam.pixel(vectors[i])
    fig,axes=plt.subplots(1,2,figsize=(16,6.2),layout="constrained")
    left,right=axes
    h,w=rgb.shape[:2]
    left.imshow(cv.resize(rgb,(w//3,h//3),interpolation=cv.INTER_AREA),extent=(-.5,w-.5,h-.5,-.5))
    left.scatter(*observed.T,s=15,facecolors="none",edgecolors="#6cfac2",linewidth=.6,label="Observed Hipparcos")
    left.scatter(*predicted.T,s=7,marker="+",c="#ffb577",linewidth=.45,label="Projected")
    for line in coastlines(coast,astro):
        left.plot(*line.T,c="#48daf2",alpha=.56,lw=.44)
    left.scatter(*astro["venus_xy"],s=108,facecolors="none",edgecolors="#ffb577",label="Venus")
    left.legend(loc="upper right",fontsize=8,framealpha=.9)
    left.set(xlim=(0,w),ylim=(h,0),title="Independent sky and coastline checks in NASA pixels",
             xlabel="Native column",ylabel="Native row")
    right.set_facecolor("#091323")
    rot,t=affine[:,:2],affine[:,2]
    boundary=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1],[0,0]]) @ rot.T+t
    right.plot(*boundary.T,color="#647ba2",label="NASA footprint")
    disk=limb.boundary(np.linspace(0,2*np.pi,361))@rot.T+t
    right.fill(*disk.T,color="#234868")
    right.scatter(*(observed@rot.T+t).T,s=3,c="#6cfac2",label="Verified source stars")
    right.scatter(*sky.xy.T,s=6,c="#ffb577",label="Extended projections")
    right.set(xlim=(0,dimensions[0]),ylim=(dimensions[1],0),aspect="equal",
              title="Extended sky in output pixels",xlabel="Output column",ylabel="Output row")
    right.legend(loc="lower right",fontsize=8)
    fig.savefig(path,dpi=170)
    plt.close(fig)


def color_plot(path,source,pred,target,de,valid):
    fig,axs=plt.subplots(2,2,figsize=(12,8.3),layout="constrained")
    for ax,data,title in zip(axs.ravel()[:3],(source,target,pred),
                             ("NASA before grading","Registered 8K video reference","NASA after measured grade")):
        ax.imshow(np.where(valid[...,None],data,0)[::-1,::-1])
        ax.set_title(title)
        ax.axis("off")
    heat=axs.ravel()[3].imshow(np.where(valid,de,np.nan)[::-1,::-1],cmap="magma",vmin=0,vmax=8)
    axs.ravel()[3].set_title("Per-pixel ΔE2000 in common overlap")
    axs.ravel()[3].axis("off")
    fig.colorbar(heat,ax=axs.ravel()[3],shrink=.79)
    fig.savefig(path,dpi=180)
    plt.close(fig)


def seam_plot(path,master,affine,original_shape,feather):
    h,w=original_shape[:2]
    a,b=affine[:,:2],affine[:,2]
    mw,mh=master.shape[1],master.shape[0]
    edge=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1],[0,0]])
    inner=np.array([[feather,feather],[w-1-feather,feather],[w-1-feather,h-1-feather],
                    [feather,h-1-feather],[feather,feather]])
    sizes=(min(mw,1700),round(mh*min(mw,1700)/mw))
    thumb=cv.resize(master,sizes,interpolation=cv.INTER_AREA).astype(np.float32)/65535
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    ax=axes[0,0]
    ax.imshow(thumb,extent=(0,mw,mh,0))
    ax.plot(*(edge@a.T+b).T,color="#ffad70",lw=.65)
    ax.plot(*(inner@a.T+b).T,color="#65ffcd",lw=.5)
    ax.set(title="Photo boundary and interior feather",xlim=(0,mw),ylim=(mh,0))
    profiles=[]
    texture_ratios=[]
    for ax,label,x,side in ((axes[0,1],"Source left edge",0,1),(axes[1,0],"Source right edge",w-1,-1)):
        depth=np.arange(-240,241)
        u=np.broadcast_to(x+side*depth,(450,len(depth)))
        v=np.arange(h//2-225,h//2+225)[:,None]
        mapx=np.float32(u*a[0,0]+v*a[0,1]+b[0])
        mapy=np.float32(u*a[1,0]+v*a[1,1]+b[1])
        image=cv.remap(master,mapx,mapy,cv.INTER_LINEAR).astype(np.float32)/65535
        ax.imshow(np.clip(image*2.9,0,1))
        ax.axvline(240,color="#ffad70",lw=.7)
        ax.axvline(240+feather,color="#65ffcd",lw=.65)
        ax.set(title=f"{label} · 100% crop, 2.9× code values")
        ax.axis("off")
        luminance=luma(image)
        profiles.append(luminance.mean(0))
        residual=luminance-cv.GaussianBlur(luminance,(0,0),4)
        outer=residual[:,100:210]
        inner=residual[:,240+feather+18:240+feather+128]
        texture_ratios.append(float(np.std(outer)/np.std(inner)))
    ax=axes[1,1]
    for p,label in zip(profiles,("Source left","Source right")):
        ax.plot(depth,p*255,label=label)
    ax.axvline(0,c="#cc6d33",lw=.8)
    ax.axvline(feather,c="#418a75",lw=.8)
    ax.set(title="Mean luma across both boundaries",xlabel="Native pixels inward",ylabel="8-bit-equivalent luma")
    ax.legend()
    fig.savefig(path,dpi=175)
    plt.close(fig)
    return {"profiles": [{"edge_jump_8bit":float(abs(p[241]-p[239])*255),
                           "luma_10px_in_out":float(abs(p[250]-p[230])*255),
                           "outside_inside_texture_rms_ratio":ratio}
                         for p,ratio in zip(profiles,texture_ratios)]}


def inspection_crops(folder, master, limb, affine, sky):
    rotation,translation=affine[:,:2],affine[:,2]
    points={"rim":limb.boundary(np.array([.77]))[0],
            "ocean_cloud":limb.center+[440,-565],
            "boundary":[35,master.shape[0]*.46]}
    canvas={k:np.asarray(v)@rotation.T+translation for k,v in points.items()}
    if len(sky.xy):
        canvas["extended_stars"]=sky.xy[len(sky.xy)//2]
    for name,p in canvas.items():
        x,y=np.round(p).astype(int)
        crop=master[max(0,y-170):y+170,max(0,x-250):x+250]
        image=Image.fromarray(np.uint8(np.rint(crop.astype(np.float32)/257)))
        image.save(folder/f"{name}_100.png")
        image.resize((image.width*2,image.height*2),Image.Resampling.NEAREST).save(folder/f"{name}_200.png")


def texture_comparison(path, master, probes, crop_size):
    """Compare the same-size sky samples at native pixel scale and in the positive tail."""
    h,w=master.shape[:2]
    cw,ch=crop_size
    labels=("Photographed sky","Left extension","Right extension")
    fig,axes=plt.subplots(2,3,figsize=(13.6,5.9),gridspec_kw={"height_ratios":[1.3,1]},
                          layout="constrained")
    stats={}
    for label,(u,v),image_axis,plot_axis in zip(labels,probes,axes[0],axes[1]):
        x,y=np.rint([u*w,v*h]).astype(int)
        rgb=master[y:y+ch,x:x+cw].astype(np.float32)/65535
        light=luma(rgb)*255
        residual=light-cv.GaussianBlur(light,(0,0),2.2)
        maximum=residual==cv.dilate(residual,np.ones((5,5),np.uint8))
        local_rms=np.array([residual[yy:yy+48,xx:xx+48].std()
                            for yy in range(0,ch-47,24) for xx in range(0,cw-47,24)])
        levels=np.linspace(0,30,121)
        tail=np.array([np.mean(residual>level) for level in levels])
        stats[label]={"origin_xy":[int(x),int(y)],"size":[cw,ch],
                      "median_luma":float(np.median(light)),
                      "residual_rms":float(residual.std()),
                      "residual_p95":float(np.percentile(residual,95)),
                      "residual_p99":float(np.percentile(residual,99)),
                      "residual_p999":float(np.percentile(residual,99.9)),
                      "local_rms_cv":float(local_rms.std()/local_rms.mean()),
                      "peaks_above_8_per_megapixel":float(np.mean(maximum&(residual>8))*1e6)}
        image_axis.imshow(rgb,interpolation="nearest")
        image_axis.set_title(label)
        image_axis.axis("off")
        plot_axis.semilogy(levels,np.maximum(tail,1e-5),c="#286c8d")
        plot_axis.set(xlim=(0,30),ylim=(1e-4,.5),xlabel="Residual luma (8-bit scale)")
        plot_axis.grid(alpha=.22)
    axes[1,0].set_ylabel("Fraction above level")
    fig.savefig(path,dpi=195)
    plt.close(fig)
    return stats

def flare_plot(path, before, after, ridges, limb, venus, shape):
    """Matched contrast views in native coordinates and the removed additive field."""
    h,w=shape[:2]
    yy,xx=np.indices(before.shape[:2],dtype=np.float32)
    xx=(xx+.5)*w/before.shape[1]-.5
    yy=(yy+.5)*h/before.shape[0]-.5
    sky=limb.radius(xx,yy)>1.047
    weight=sky.astype(np.float32)
    extent=(-.5,w-.5,h-.5,-.5)

    def bandpass(image):
        gray=luma(image)
        narrow=cv.GaussianBlur(gray*weight,(0,0),4)/np.maximum(cv.GaussianBlur(weight,(0,0),4),1e-6)
        broad=cv.GaussianBlur(gray*weight,(0,0),36)/np.maximum(cv.GaussianBlur(weight,(0,0),36),1e-6)
        return np.ma.masked_where(~sky,narrow-broad)

    fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout="constrained")
    for ax,image,title in zip(axes[0],(before,after),("Before · graded NASA sky","After · corrected NASA sky")):
        m=ax.imshow(bandpass(image),extent=extent,cmap="RdBu_r",vmin=-2.7,vmax=2.7)
        ax.set(title=title,xlabel="Native column",ylabel="Native row")
    fig.colorbar(m,ax=list(axes[0]),shrink=.68,label="Band-pass luma, 8-bit codes")
    delta=luma(before-after)
    corrected=axes[1,0].imshow(delta,extent=extent,cmap="RdBu_r",vmin=-2,vmax=5)
    for ridge in ridges:
        active=np.linalg.norm(ridge.rgb,axis=1)>.30
        x,y=(ridge.center,ridge.along) if ridge.axis=="y" else (ridge.along,ridge.center)
        axes[1,0].plot(np.where(active,x,np.nan),np.where(active,y,np.nan),
                      c="#45dbfd",lw=.7,label=ridge.name)
    axes[1,0].scatter(*venus,marker="+",s=32,c="#f4ffb5",label="Venus")
    axes[1,0].set(title="Signed luma correction and tracked centers",xlabel="Native column",ylabel="Native row")
    axes[1,0].legend(fontsize=7,loc="upper right")
    fig.colorbar(corrected,ax=axes[1,0],shrink=.67,label="Luma removed, 8-bit codes")
    for ridge,color in zip(ridges,("#8345bd","#16758b"),strict=False):
        y0,y1=ridge.stats["along_detected_span_px"]
        ok=(ridge.along>=y0)&(ridge.along<=y1)
        axes[1,1].plot(ridge.along[ok],luma(ridge.rgb[ok]),c=color,label=ridge.name)
    axes[1,1].set(title="Estimated optical ridge contrast",xlabel="Along-ridge native pixels",
                  ylabel="Peak luma, 8-bit codes")
    axes[1,1].grid(alpha=.22)
    if ridges:
        axes[1,1].legend(fontsize=8)
    fig.savefig(path,dpi=180)
    plt.close(fig)

def grain_probes(master, reference, limb, affine, registration):
    inverse=cv.invertAffineTransform(affine)
    probes=[]
    for label,offset in (("Earth", (440,-565)),("Sky near the limb",(1110,1280))):
        point=limb.center+offset
        center=point@affine[:,:2].T+affine[:,2]
        x,y=np.rint(center-[166,99]).astype(int)
        rows,cols=slice(y,y+198),slice(x,x+332)
        v,u=np.indices((198,332),dtype=np.float32)
        transform=registration@np.vstack((inverse,[0,0,1]))
        sx=transform[0,0]*(u+x)+transform[0,1]*(v+y)+transform[0,2]
        sy=transform[1,0]*(u+x)+transform[1,1]*(v+y)+transform[1,2]
        matched=cv.remap(reference,np.float32(sx),np.float32(sy),cv.INTER_LANCZOS4)
        probes.append((label,rows,cols,master[rows,cols].copy(),matched))
    return probes


def grain_plot(path, master, probes, stats):
    fig=plt.figure(figsize=(13.8,8.2),layout="constrained")
    grid=fig.add_gridspec(3,4,height_ratios=(1,1,.9))
    subtitles=("8K video, registered","Before the overlay","After the overlay","Added luma, amplified")
    for i,(label,rows,cols,before,reference) in enumerate(probes):
        after=master[rows,cols]
        for j,(title,rgb) in enumerate(zip(subtitles[:3],(reference,before,after))):
            ax=fig.add_subplot(grid[i,j])
            ax.imshow(rgb.astype(np.float32)/65535,interpolation="nearest")
            ax.set_title(f"{label} · {title}",fontsize=10)
            ax.axis("off")
        ax=fig.add_subplot(grid[i,3])
        diff=luma(after.astype(np.float32)-before.astype(np.float32))/257
        heat=ax.imshow(diff,cmap="RdBu_r",vmin=-1.4,vmax=1.4,interpolation="nearest")
        ax.set_title(f"{label} · {subtitles[3]}",fontsize=10)
        ax.axis("off")
        fig.colorbar(heat,ax=ax,shrink=.67,label="8-bit luma")
    tone,rms=np.array(stats["opacity_curve_8bit"]).T
    ax=fig.add_subplot(grid[2,:2])
    ax.plot(tone,rms*stats["strength"],"o-",c="#24637c")
    ax.set(xlabel="Display luma (0–255)",ylabel="Added grain RMS (8-bit codes)",
           title="Fitted overlay amplitude")
    ax.grid(alpha=.2)
    ax=fig.add_subplot(grid[2,2:])
    tone=np.array([q["luma_median"] for q in stats["samples"]])
    for j,pair in enumerate(stats["screen_bands_sigma_px"]):
        y=np.array([q["bands"][j]["screen_correlation"] for q in stats["samples"]])
        ax.plot(tone,y,"o-",label=f"σ {pair[0]:g}–{pair[1]:g} px")
    ax.set(xlabel="Display luma (0–255)",ylabel="Correlation between frame groups",
           title="Fixed screen texture after compensating image motion")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(path,dpi=175)
    plt.close(fig)
