"""Checks for the delivered files and the measured pipeline invariants."""
import csv
import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import cv2 as cv
import numpy as np
from PIL import Image
import pytest
import tifffile

from artemis.astrometry import Camera, radec
from artemis.geometry import detect_limb

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"


@pytest.fixture(scope="module")
def results():
    return json.loads((OUTPUT/"measurements.json").read_text())


def test_master_dimensions_and_center(results):
    geometry = results["geometry"]
    w,h = geometry["master_dimensions"]
    assert 9*w == 21*h
    assert geometry["centering_error_px"] < 1e-6
    assert geometry["limb_min_canvas_margin_px"] > 300
    assert geometry["min_native_limb_to_footprint_edge_px"] > results["background"]["feather_px"]
    m = np.array(geometry["affine_native_to_master"])
    np.testing.assert_allclose(m[:,:2] @ m[:,:2].T, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(m[:,:2] @ geometry["limb_center_native"] + m[:,2], [(w-1)/2,(h-1)/2])
    with Image.open(OUTPUT/"artemis_ii_3440x1440.png") as image:
        assert image.size == (3440,1440)
    # Detect the rendered limb independently, allowing the grading to move the
    # maximum gradient within the atmospheric band.
    png = cv.imread(str(OUTPUT/"artemis_ii_21x9_master.png"),cv.IMREAD_UNCHANGED)
    assert png.shape == (h,w,3) and png.dtype == np.uint16
    small = cv.resize(png,(w//2,h//2),interpolation=cv.INTER_AREA)
    limb = detect_limb(np.uint8(small[...,::-1]/257))
    scaled = (limb.center+.5)*np.array([w/(w//2),h/(h//2)])-.5
    np.testing.assert_allclose(scaled,geometry["limb_center_master"],atol=3.0)


def test_lossless_masters_agree():
    png_path=OUTPUT/"artemis_ii_21x9_master.png"
    with Image.open(png_path) as im:
        assert im.info["icc_profile"]
    png=cv.imread(str(png_path),cv.IMREAD_UNCHANGED)
    with tifffile.TiffFile(OUTPUT/"artemis_ii_21x9_master_16bit.tif") as tif:
        assert tif.pages[0].bitspersample == 16
        assert tif.pages[0].tags[34675].value
        t=tif.asarray()
    assert all(np.array_equal(png[y:y+64,:,::-1],t[y:y+64]) for y in range(0,len(png),64))


def test_alignment_and_heldout_color(results):
    reg=results["reference"]["best"]
    assert reg["sift_inliers"] > 1000
    assert reg["holdout_patches"] > 50
    assert reg["holdout_rms_px"] < .45
    color=results["color"]["metrics"]
    thresholds={"earth":2.35,"clouds":2.2,"atmosphere":2.65,"background":1.25}
    for name,limit in thresholds.items():
        m=color["delivered_grade"][name]
        assert m["pixels"] > 1000
        assert m["mean_de"] < limit
        assert m["lt5_pct"] > 94
        assert m["mean_de"] < color["original"][name]["mean_de"]


def test_astrometry_and_catalog_projection(results):
    a=results["astronomy"]
    assert a["verified_matches"] > 350
    assert a["held_out_rms_px"] < 1.35
    assert a["venus_residual_px"] < 6
    assert max(a["null_match_counts"]) < .05*a["verified_matches"]
    assert a["pixel_to_sky_roundtrip_max_px"] < 1e-5
    camera=Camera(np.array(a["camera_parameters"]))
    affine=np.array(results["geometry"]["affine_native_to_master"])
    w,h=results["geometry"]["master_dimensions"]
    rng=np.random.default_rng(550)
    pixels=rng.uniform([0,0],[w-1,h-1],(500,2))
    native=(pixels-affine[:,2])@affine[:,:2]
    sky=camera.sky(native)
    np.testing.assert_allclose(camera.pixel(sky),native,atol=1e-6)
    with fits.open(OUTPUT/"wallpaper_wcs.fits") as hdus:
        wcs=WCS(hdus[0].header)
    estimated=wcs.all_world2pix(radec(sky),0)
    assert np.sqrt(np.mean(np.sum((estimated-pixels)**2,axis=1))) < 1
    with (OUTPUT/"extended_hipparcos.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==a["stellar_photometry"]["rendered"] and len(rows)>30
    assert len({row["hip"] for row in rows})==len(rows)
    x,y=np.array([(float(row["x_master"]),float(row["y_master"])) for row in rows]).T
    assert ((0<x)&(x<w)&(0<y)&(y<h)).all()
    positions=(np.column_stack((x,y))-affine[:,2])@affine[:,:2]
    nw,nh=results["geometry"]["source_dimensions"]
    assert not ((positions[:,0]>=0)&(positions[:,0]<nw)&(positions[:,1]>=0)&(positions[:,1]<nh)).any()
    ra,dec=np.radians([(float(row["ra_apparent_deg"]),float(row["dec_apparent_deg"])) for row in rows]).T
    vectors=np.column_stack((np.cos(ra)*np.cos(dec),np.sin(ra)*np.cos(dec),np.sin(dec)))
    np.testing.assert_allclose(camera.pixel(vectors),positions,atol=1e-6)


def test_no_measurable_edge_step(results):
    profiles=results["background"]["seams"]["profiles"]
    assert max(x["edge_jump_8bit"] for x in profiles)<1.0
    assert max(x["luma_10px_in_out"] for x in profiles)<1.0


def test_extension_reproduces_photographic_speckle(results):
    probes = results["background"]["texture_probes"]
    source = probes["Photographed sky"]
    for name in ("Left extension", "Right extension"):
        sample = probes[name]
        assert abs(sample["median_luma"]-source["median_luma"]) < .75
        for key in ("residual_p99", "residual_p999", "peaks_above_8_per_megapixel"):
            assert .75 < sample[key]/source[key] < 1.25
        assert sample["local_rms_cv"] < source["local_rms_cv"]+.10
    photometry = results["astronomy"]["stellar_photometry"]
    assert 0 < photometry["below_half_completeness"] < photometry["rendered"]


def test_optical_ridges_are_detected_and_reduced(results):
    flare = results["flare"]
    assert flare["enabled"]
    assert len(flare["ridges"]) == 2
    for ridge in flare["ridges"]:
        before = ridge["holdout_contrast_before_8bit"]
        after = ridge["holdout_contrast_after_8bit"]
        assert ridge["detected_profiles"] > 30
        assert ridge["holdout_profiles"] > 30
        assert before > 1.0
        assert abs(after) < .30*before
        raw_contrast = ridge["nef_same_template_median_8bit"]
        assert raw_contrast is None or raw_contrast > 1.0
    halo, = flare["halos"]
    assert halo["control_cuts"] > 10
    assert halo["profile_rms_refined_8bit"] < .75*halo["profile_rms_core_only_8bit"]
    assert (OUTPUT/"flare_analysis.png").stat().st_size > 100_000


def test_video_grain_is_screen_locked_and_reproducible(results):
    from artemis.video_grain import Field

    stats = results["video_grain"]
    assert stats["enabled"]
    assert len(stats["frame_indices"]) >= 12
    assert 1 < stats["sigma_master_px"] < 3
    assert .08 < stats["applied_rms_luma_8bit"] < .65
    assert abs(stats["applied_mean_luma_8bit"]) < .004
    for sample in (stats["samples"][0],stats["samples"][-1]):
        coarse=sample["bands"][0]
        screen,*shifted=coarse["covariance_8bit_sq"]
        assert screen > 3*max(abs(x) for x in shifted)
        assert coarse["screen_correlation"] > .5
    field = Field(128,stats["sigma_master_px"],stats["broad_master_px"],stats["seed"])
    tiled=np.concatenate((field.take(17),field.take(61),field.take(33)))
    other = Field(128,stats["sigma_master_px"],stats["broad_master_px"],stats["seed"])
    np.testing.assert_allclose(tiled,other.take(111),atol=2e-6)
    assert (OUTPUT/"video_grain_comparison.png").stat().st_size > 100_000


def test_approved_detail_response_matches_aurora(results):
    detail=results["sharpness"]
    assert detail["enabled"] and detail["phase_response"]>.7
    metrics=detail["metrics"]
    for region in ("clouds","aurora"):
        current=metrics["final"][region]["error_rms_8bit"][0]
        assert current < .70*metrics["before"][region]["error_rms_8bit"][0]
        assert current < metrics["unsharp_1.5"][region]["error_rms_8bit"][0]
    peaks={name:row["sky"]["fine_peaks_per_mp"]["6"] for name,row in metrics.items()}
    assert abs(peaks["final"]-peaks["reference"]) < .25*abs(peaks["before"]-peaks["reference"])
    assert (OUTPUT/"sharpness_comparison.png").stat().st_size > 100_000
