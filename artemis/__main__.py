"""Run the Artemis II wallpaper pipeline."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time
import tomllib

from astropy.coordinates import SkyCoord
import cv2 as cv
import numpy as np
from PIL import Image

from . import amoled, astrometry, background, color, compose, flare, geometry, raw, sharpness, sky_render, star_shape, sun_glow, texture, validation, video, video_grain
from .acquire import acquire, sha256, write_json
from .report import make_report


def run(config, offline, refresh_video, stage):
    start=time.monotonic()
    cfg=tomllib.loads(config.read_text())
    root=config.resolve().parent
    cache,out=(root/cfg["paths"][name] for name in ("cache","output"))
    work,source=cache/"work",cache/"source"
    for folder in (work,out):
        folder.mkdir(parents=True,exist_ok=True)
    cv.setNumThreads(6)
    if not offline:
        acquire(cache,{**cfg["source"],**cfg["sky"]})
    if stage=="acquire":
        return
    photo=source/"art002e000192.jpg"
    with Image.open(photo) as image:
        icc=image.info["icc_profile"]
        rgb=np.array(image.convert("RGB"))
    limb=geometry.detect_limb(rgb)
    print("Limb",limb.center,limb.axes,limb.rms,flush=True)
    ref,reg=video.select(rgb,source/"melodysheep_clip.mp4",work,cfg["source"],refresh_video)
    native_registration=np.array(reg["affine_native"])
    rotation=geometry.rotation_from_registration(native_registration)
    dimensions,affine=geometry.canvas(rgb.shape,limb,rotation,cfg["render"]["minimum_height"])
    print("Master dimensions",dimensions,flush=True)
    astro=astrometry.solve(rgb,limb,cache,cfg["sky"],cfg["source"]["capture_utc"],cfg["render"]["seed"])
    cam=astro["camera"]
    i,j=astro["matches"]
    predicted=cam.pixel(astro["vectors"][i])
    detected=astro["detections"].xy[j]
    matches=np.column_stack((astro["stars"][i,0],astro["stars"][i,6],astro["detections"].snr[j],
                             detected,predicted,astro["errors"]))
    validation.csv_write(out/"verified_hipparcos.csv",
                         ["hip","V","matched_filter_snr","x_observed","y_observed","x_predicted","y_predicted","residual_px"],matches)
    wcs=validation.make_wcs(cam,affine,dimensions,out/"wallpaper_wcs.fits",cfg["source"]["capture_utc"])
    if stage=="calibrate":
        write_json(work/"calibration.json",{"camera":cam.p.tolist(),"affine":affine.tolist(),
                   "dimensions":dimensions,"wcs":wcs,"native_star_rms":sky_render.rms(astro["errors"])})
        return
    initial,target,valid,x,y=color.comparison_inputs(rgb,ref,native_registration)
    models,holdout=color.fit(initial,target,valid,x,y,limb,cfg["render"]["seed"])
    comparisons={}
    comparisons["original"],_=color.metrics(initial,target,valid,holdout,x,y,limb,initial)
    for model in models:
        fitted=color.tiled_apply(model,initial,x,y)
        comparisons[model.mode],_=color.metrics(fitted,target,valid,holdout,x,y,limb,initial)
    def score(model):
        metrics=comparisons[model.mode]
        return sum(metrics[key]["mean_de"]*weight for key,weight in
                   (("earth",.6),("clouds",.15),("atmosphere",.15),("background",.1)))
    grade=min(models,key=score)
    print("Selected color model",grade.mode,flush=True)
    graded=color.tiled_apply(grade,rgb)
    nef=raw.develop(source/"art002e000192.NEF",work/"raw_active.npy")
    alignment=raw.check_alignment(rgb,nef) if nef is not None else None
    restored=raw.preserve_highlights(graded,rgb,nef,alignment,limb)
    flare_before,flare_after,ridges,flare_stats=flare.apply(
        graded,limb,astro["venus_xy"],cfg["flare"],nef)
    for ridge in flare_stats.get("ridges",[]):
        along=np.array(ridge["along_px"])
        center=np.array(ridge["center_px"])
        low,high=ridge["along_detected_span_px"]
        selection=(along>=low)&(along<=high)
        coords=np.column_stack((center,along) if ridge["axis"]=="y" else (along,center))
        ra,dec=astrometry.radec(cam.sky(coords[selection])).T
        ridge["galactic_latitude_deg"]=np.percentile(SkyCoord(ra,dec,unit="deg").galactic.b.deg,
                                                       [0,50,100]).tolist()
    for halo in flare_stats.get("halos",[]):
        ridge=next(r for r in ridges if r.name==halo["ridge"])
        low,high=halo["along_px"]
        along=np.linspace(low,high,40)
        center=np.interp(along,ridge.along,ridge.center)
        q=np.concatenate([np.column_stack((center+dx,along) if ridge.axis=="y" else (along,center+dx))
                          for dx in (-200,0,200)])
        ra,dec=astrometry.radec(cam.sky(q)).T
        halo["galactic_latitude_deg"]=np.percentile(SkyCoord(ra,dec,unit="deg").galactic.b.deg,[0,50,100]).tolist()
    print("Optical ridge contrast",[(r["name"],round(r["holdout_contrast_before_8bit"],2),
          round(r["holdout_contrast_after_8bit"],2)) for r in flare_stats.get("ridges",[])],flush=True)
    validation.flare_plot(out/"flare_analysis.png",flare_before,flare_after,ridges,
                          limb,astro["venus_xy"],rgb.shape)
    stars_in_front=cam.in_front(astro["vectors"])
    excluded=np.vstack((cam.pixel(astro["vectors"][stars_in_front]),astro["detections"].xy,astro["venus_xy"]))
    glow_stats=sun_glow.apply(graded,limb,astro["venus_xy"],excluded,
                              cfg.get("sun_glow",{"enabled":False}))
    common=color.srgb_encode(cv.resize(color.srgb_decode(graded),(initial.shape[1],initial.shape[0]),
                                       interpolation=cv.INTER_AREA))
    reproduction,de=color.metrics(common,target,valid,holdout,x,y,limb,initial)
    comparisons["delivered_grade"]=reproduction
    print("Delivered color",reproduction,flush=True)
    validation.color_plot(out/"color_comparison.png",initial,common,target,de,valid)
    sky,rows=sky_render.prepare(astro,graded,rgb,limb,affine,dimensions,cfg["sky"],nef)
    validation.csv_write(out/"extended_hipparcos.csv",
        ["hip","ra_apparent_deg","dec_apparent_deg","V","x_master","y_master","bootstrap_p95_px","one_term_delta_px"],rows)
    print("Sky",sky.stats,flush=True)
    validation.astronomy_plot(out/"astronomy_validation.png",rgb,astro,limb,sky,affine,dimensions,
                              cache/"catalog"/"ne_10m_coastline.zip")
    bg=background.estimate(graded,limb)
    print("Compositing",flush=True)
    shape=np.array(rgb.shape[1::-1])
    d_edge=np.min(np.minimum(detected,shape-1-detected),axis=1)
    anchors=detected[d_edge < cfg["render"]["feather"]+12] @ rotation.T + affine[:,2]
    donors=texture.find_donors(graded,bg,limb,excluded,rotation,cfg["render"])
    print("Eligible source grain windows",len(donors.centers),flush=True)
    master,grain_sigma=compose.compose(graded,bg,sky,affine,dimensions,cfg["render"]["feather"],anchors,donors)
    star_cfg=cfg.get("correct_stars",{})
    star_stats={"enabled":False}
    if star_cfg.get("enabled",False):
        candidates=astro["detections"].xy@rotation.T+affine[:,2]
        star_stats=star_shape.apply(master,np.vstack((candidates,sky.xy)),limb,affine,cfg=star_cfg)
    grain_cfg=cfg["video_grain"]
    cinematic={"enabled":False}
    if grain_cfg["enabled"]:
        model=video_grain.calibrate(source/"melodysheep_clip.mp4",reg,work)
        probes=validation.grain_probes(master,ref,limb,affine,native_registration)
        cinematic=video_grain.apply(master,model,grain_cfg)
        print("Video overlay luma RMS",cinematic["applied_rms_luma_8bit"],flush=True)
        validation.grain_plot(out/"video_grain_comparison.png",master,probes,cinematic)
    detail={"enabled":False}
    detail_cfg=cfg.get("sharpness",{})
    if detail_cfg.get("enabled"):
        probe=sharpness.aurora_probe(master,ref,limb,affine,native_registration,detail_cfg)
        points=np.vstack((detected,astro["venus_xy"]))@rotation.T+affine[:,2]
        sharpness.apply(master,limb,affine,np.vstack((points,sky.xy)),detail_cfg)
        detail=sharpness.evaluate(master,probe,out/"sharpness_comparison.png",detail_cfg)
        print("Aurora detail check",detail["metrics"]["final"],flush=True)
    amoled_cfg=cfg.get("amoled",{})
    amoled_stats={"enabled":False}
    if amoled_cfg.get("enabled",False):
        candidates=astro["detections"].xy@rotation.T+affine[:,2]
        venus_master=astro["venus_xy"]@rotation.T+affine[:,2]
        amoled_stats=amoled.apply(master,limb,affine,candidates,venus_master,
                                  catalog_points=sky.xy,cfg=amoled_cfg)
    compose.save_exports(master,out,cfg["render"],icc,black_padding=amoled_stats["enabled"])
    texture_stats=validation.texture_comparison(out/"sky_texture_comparison.png",master,
                             cfg["render"]["texture_probe_origins"],cfg["render"]["texture_probe_size"])
    seams=validation.seam_plot(out/"seam_analysis.png",master,affine,rgb.shape,cfg["render"]["feather"])
    crops=work/"inspection"
    crops.mkdir(exist_ok=True)
    validation.inspection_crops(crops,master,limb,affine,sky)
    h,w=rgb.shape[:2]
    mw,mh=dimensions
    bore=cam.rotation[2]
    from .catalog import unit
    north=unit(np.array([0,0,1])-bore*bore[2])
    north_pixel=cam.pixel(unit(bore+.001*north))-cam.p[4:6]
    angle=float(np.degrees(np.arctan2(rotation[1,0],rotation[0,0])))
    def fov(corners):
        v=cam.sky(corners)
        return [validation.angular_distance(v[0],v[1]),validation.angular_distance(v[2],v[3])]
    native_mid=np.array([[0,(h-1)/2],[w-1,(h-1)/2],[(w-1)/2,0],[(w-1)/2,h-1]])
    final_mid=np.array([[0,(mh-1)/2],[mw-1,(mh-1)/2],[(mw-1)/2,0],[(mw-1)/2,mh-1]])
    rng=np.random.default_rng(cfg["render"]["seed"])
    px=rng.uniform([0,0],[mw-1,mh-1],(2000,2))
    px=(px-affine[:,2])@rotation
    sub=-astro["state"]["earth"]@astro["state"]["itrf"].T
    sub_lat=float(np.degrees(np.arctan2(sub[2],np.hypot(*sub[:2]))))
    sub_lon=float(np.degrees(np.arctan2(sub[1],sub[0])))
    # Offset-prediction controls estimate random associations at the same tolerance.
    from scipy.spatial import cKDTree
    kd=cKDTree(astro["detections"].xy)
    null=[int((kd.query(predicted+d)[0]<2.3).sum()) for d in ([17,0],[-19,0],[0,21],[0,-23])]
    data={
        "geometry":{"source_dimensions":[w,h],"master_dimensions":[mw,mh],"rotation_clockwise_deg":angle,
                    "affine_native_to_master":affine.tolist(),"limb_center_native":limb.center.tolist(),
                    "limb_center_master":(rotation@limb.center+affine[:,2]).tolist(),
                    "limb_axes":limb.axes.tolist(),"limb_fit_rms_px":limb.rms,"limb_samples":len(limb.points),
                    "centering_error_px":float(np.linalg.norm(rotation@limb.center+affine[:,2]-(np.array(dimensions)-1)/2)),
                    "footprint_native_to_output_scale":float(np.linalg.det(rotation)**.5),
                    "limb_min_canvas_margin_px":float(np.min(np.minimum(limb.points@rotation.T+affine[:,2],
                      np.array(dimensions)-1-(limb.points@rotation.T+affine[:,2])))),
                    "min_native_limb_to_footprint_edge_px":float(np.min(np.minimum(limb.points,np.array([w,h])-1-limb.points)))},
        "reference":reg,"raw":{"registration":alignment,"highlight_pixels_touched":restored},
        "color":{"chosen_model":grade.mode,"coefficients":{m.mode:m.params.tolist() for m in models},
                 "metrics":comparisons,"common_dimensions":[initial.shape[1],initial.shape[0]]},
        "astronomy":{"camera_parameters":cam.p.tolist(),"one_term_camera_parameters":astro["alternative"].p.tolist(),
            "bootstrap_parameters":astro["bootstrap"].tolist(),"observer_j2000_km_kms":astro["state"]["observer"].tolist(),
            "subspacecraft_lon_lat_deg":[sub_lon,sub_lat],"boresight_apparent_radec_deg":astrometry.radec([bore])[0].tolist(),
            "native_north_clockwise_from_up_deg":float(np.degrees(np.arctan2(north_pixel[0],-north_pixel[1]))),
            "earth_center_projected":cam.pixel(unit(astro["state"]["earth"])).tolist(),
            "sun_projected":cam.pixel(unit(astro["state"]["sun"])).tolist(),
            "native_fov_deg":fov(native_mid),"master_fov_deg":fov((final_mid-affine[:,2])@rotation),
            "verified_matches":len(i),"match_rms_px":sky_render.rms(astro["errors"]),
            "match_p95_px":float(np.percentile(astro["errors"],95)),"held_out_rms_px":sky_render.rms(astro["held"]),
            "held_out_count":len(astro["held"]),"null_match_counts":null,
            "venus_observed_xy":astro["venus_xy"].tolist(),"venus_predicted_xy":astro["venus_prediction"].tolist(),
            "venus_residual_px":float(np.linalg.norm(astro["venus_prediction"]-astro["venus_xy"])),
            "pixel_to_sky_roundtrip_max_px":float(np.linalg.norm(cam.pixel(cam.sky(px))-px,axis=1).max()),
            "wcs":wcs,"stellar_photometry":sky.stats},
        "flare":flare_stats,
        "sun_glow":glow_stats,
        "amoled":amoled_stats,
        "correct_stars":star_stats,
        "video_grain":cinematic,
        "sharpness":detail,
        "background":{"floor_srgb":bg.floor.tolist(),"residual_covariance":bg.covariance.tolist(),
            "native_neighbor_corr":bg.neighbor_corr,"grain_donors":len(donors.centers),
                      "grain_sigma_8bit":(grain_sigma*255).tolist(),
            "residual_sigma_8bit":(bg.std*255).tolist(),"feather_px":cfg["render"]["feather"],
                      "feather_star_anchors":len(anchors),"seams":seams,"texture_probes":texture_stats},
        "execution":{"platform":platform.platform(),"python":platform.python_version(),"opencv":cv.__version__,
                     "ffmpeg":subprocess.check_output(["ffmpeg","-version"],text=True).splitlines()[0],
                     "seed":cfg["render"]["seed"],"config_sha256":sha256(config),
                     "wall_seconds":time.monotonic()-start},
        "artifacts":{p.name:sha256(p) for p in out.glob("artemis_ii_*") if p.is_file()},
    }
    provenance=work/"provenance.json"
    if not provenance.exists():
        provenance=root/"reference"/"provenance.json"
    data["provenance"]=json.loads(provenance.read_text()) if provenance.exists() else None
    write_json(out/"measurements.json",data)
    make_report(out/"report.md",data,cfg)
    print("Completed",out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("config.toml"))
    parser.add_argument("--offline",action="store_true",help="Use already acquired inputs")
    parser.add_argument("--refresh-video",action="store_true",help="Re-decode and register candidate video frames")
    parser.add_argument("--stage",choices=("all","acquire","calibrate"),default="all")
    args=parser.parse_args()
    run(args.config,args.offline,args.refresh_video,args.stage)


if __name__=="__main__":
    main()
