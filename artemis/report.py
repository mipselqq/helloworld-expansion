"""Write the human-readable run report from the numeric manifest."""
import numpy as np


def table(metrics):
    lines=["| Region | N | Mean ΔE00 | Median | P95 | <2 | <5 | SSIM | PSNR dB | RGB MAE / 255 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label,m in metrics.items():
        lines.append(f"| {label.title()} | {m['pixels']:,} | {m['mean_de']:.3f} | {m['median_de']:.3f} | "
                     f"{m['p95_de']:.3f} | {m['lt2_pct']:.2f}% | {m['lt5_pct']:.2f}% | "
                     f"{m['ssim']:.4f} | {m['psnr_db']:.2f} | {m['mae_rgb_255']:.2f} |")
    return "\n".join(lines)


def make_report(path,d,cfg):
    geo,ref,ast,c,b=d["geometry"],d["reference"],d["astronomy"],d["color"],d["background"]
    src=d["provenance"]["sources"] if d["provenance"] else []
    sky=ast["stellar_photometry"]
    w,h=geo["master_dimensions"]
    ra,dec=ast["boresight_apparent_radec_deg"]
    best=ref["best"]
    frame_scales=[float(np.linalg.det(np.array(item["affine_1920"])[:,:2])**.5)*4 for item in ref["candidates"]]
    frame_angles=[float(np.degrees(np.arctan2(item["affine_1920"][1][0],item["affine_1920"][0][0])))
                  for item in ref["candidates"]]
    raw=d["raw"]["registration"]
    raw_provenance = (
        "The archive fulfilled its downloadable raw staging request and supplied the 26 MB Nikon NEF associated "
        "with 017A5565.NEF. LibRaw exposed a 5584 × 3728 raster and an integer active crop of (8, 8, 5568, 3712)."
        if raw else "This run used the full-resolution publication while the Nikon NEF was unavailable."
    )
    raw_details = (
        f"The NEF provided an independent 16-bit linear-sRGB PSF measurement and supplied luminance texture in "
        f"**{d['raw']['highlight_pixels_touched']:,}** near-white pixels. The active RAW and JPEG aligned at "
        f"**{raw['rms_px']:.3f} px RMS** over {raw['patches']} phase-correlation patches."
        if raw else "The stellar PSF and the highlight response were measured from the NASA publication."
    )
    text=f"""# Artemis II · Hello, World

## Finished files

The **{w:,} × {h:,}** master is an exact 21:9. Both lossless masters are 16-bit RGB with embedded sRGB profiles. The 3440 × 1440 export uses linear-light Lanczos and a centered vertical crop to keep the planet round at the requested ratio.

## Inputs and timing

NASA photograph art002e000192, also cataloged as ART002-E-192. The JSC archive confirms 2026-04-03 00:27:39 GMT. The published EXIF adds .26 seconds and records a Nikon D5 serial 3500017, 22 mm, f/4, ISO 51200 and 1/4 s. The UTC interpretation follows JSC's image record. The JPEG has a 5568 × 3712, 4:4:4, 8-bit sRGB grid. {raw_provenance}

The published NASA JPEG provided color and most spatial detail. {raw_details} The NASA Lightroom publication supplied the high-ISO shadow chroma.

The video is acquired with system yt-dlp and Firefox cookies. The video target is the native 7680 × 4320, 24000/1001 fps AV1 stream, YouTube format {cfg['source']['format']}, decoded from a stream-copy clip with FFmpeg. Its metadata describes limited-range BT.709. FFmpeg expands the range and decodes BT.709 Y′CbCr to display RGB codes, interpreted as sRGB display codes for desktop output. Linear sRGB is used for spatial filtering and compositing. The choice reproduces standard SDR computer playback and does not recover the creator's pre-encode working space.

Selected clip frame **{best['index']}**, approximately **{best['approx_video_seconds']:.3f} s** on YouTube, based on coherent texture and holdout error near 5:07. The stream-copy trim can shift the effective timeline by about a frame. Across {len(ref['candidates'])} sampled frames the photograph's scale ranged from {min(frame_scales):.4f} to {max(frame_scales):.4f} times NASA scale and its image-plane rotation from {min(frame_angles):.3f}° to {max(frame_angles):.3f}°. The video also uses a dark rounded matte and slow reframing. Those pixels are excluded. No video-frame pixels are composited into the wallpaper.

- [NASA image and caption](https://images.nasa.gov/details/art002e000192)
- [JSC acquisition record and raw request]({cfg['source']['archive_record']})
- [Melodysheep video]({cfg['source']['video_url']}&t=307)
- [NASA NAIF Artemis II SPICE documentation](https://naif.jpl.nasa.gov/pub/naif/pds/pds4/artemis2/artemis2_spice/document/spiceds_v001.html)
- [ESA Hipparcos Main Catalogue, CDS I/239](https://cdsarc.cds.unistra.fr/viz-bin/cat/I/239)

### SHA-256 input manifest

| Cached source | SHA-256 |
|---|---|
"""
    text+="\n".join(f"| {item['file']} | `{item['sha256']}` |" for item in src)
    text+=f"""

## Image geometry

The fit used {geo['limb_samples']} radial gradient samples, with **{geo['limb_fit_rms_px']:.3f} px RMS** relative to the ellipse. The native center in zero-indexed pixel coordinates is {np.round(geo['limb_center_native'],4).tolist()} and the semi-axes are {np.round(geo['limb_axes'],3).tolist()} px. The output center is **{geo['limb_center_master']}**, with a computed centering error of **{geo['centering_error_px']:.6g} px**. The nearest measured limb is {geo['limb_min_canvas_margin_px']:.1f} px from a canvas edge.

The SIFT/RANSAC and affine refinement retained {best['sift_inliers']:,} inliers and used {best['holdout_patches']} spatial holdout patches, with {best['holdout_rms_px']:.3f} px RMS and {best['holdout_p95_px']:.3f} px P95 at 1920 × 1080. At the native 8K video resolution the same RMS is {4*best['holdout_rms_px']:.2f} px. The orthogonal part of the affine sets the wallpaper rotation to **{geo['rotation_clockwise_deg']:.4f}°** clockwise in image coordinates. The final mapping is a rotation of unit scale and a fractional translation. One linear-light Lanczos4 pass samples the genuine photograph at that mapping. Canvas dimensions are integer multiples of 7:3 large enough to contain every rotated source corner while centering the measured limb. The affine matrix is in measurements.json.

## Color reproduction

Metrics compare only valid overlapping pixels on a {c['common_dimensions'][0]} × {c['common_dimensions'][1]} NASA-coordinate grid, with linear-light antialiasing and a spatial holdout of entire 192 px NASA tiles. Earth is the interior surface and includes the cloud subset. The cloud mask is a luminance/chroma proxy. The atmosphere mask is an annulus around the limb and therefore includes some adjoining pixels. SSIM uses display luma. PSNR and RGB MAE use full-range display RGB, with MAE expressed on a 0–255 scale.

{table(c['metrics']['delivered_grade'])}

The <2 and <5 percentages are threshold-based perceptual similarity rates. They do not establish identical source colors. The heatmap in color_comparison.png covers the shared active image, while the numbers above use only held-out tiles.

| Candidate | Earth mean ΔE00 | Cloud mean | Atmosphere mean | Background mean |
|---|---:|---:|---:|---:|
"""
    for name in ("original","monotone","global","regional","adaptive","lab","delivered_grade"):
        m=c["metrics"][name]
        text+=f"| {name} | "+" | ".join(f"{m[k]['mean_de']:.3f}" for k in ("earth","clouds","atmosphere","background"))+" |\n"
    text+=f"""
The selected grade is **{c['chosen_model']}**. The candidates include monotone PCHIP luminance transfer, a smooth 3D opponent-color and tone basis equivalent to a continuous LUT, a region-weighted version with smoothly varying corrections, a warm-hue adaptive fit to better reproduce the muted Saharan tones, and a perceptual CIELAB residual fit. The grade changes colors at native resolution. Original atmospheric and window structure stays in place, including the reflection visible over the disk. JPEG treatment, shot noise, the video encode, and motion/resampling leave measurable differences.

## Absolute sky solution

SPICE used the postflight JSC Best Estimated Trajectory for Orion (NAIF -24), JPL DE440s, and the ITRF93 Earth orientation kernel. At exposure the spacecraft geocentric J2000 state was `{np.round(ast['observer_j2000_km_kms'],5).tolist()}` in km and km/s, a geocentric distance of {np.linalg.norm(ast['observer_j2000_km_kms'][:3]):.2f} km. The sub-spacecraft longitude and latitude were {np.round(ast['subspacecraft_lon_lat_deg'],4).tolist()}°. The April 3 exposure lies outside the published reconstructed handheld-camera pointing kernel for this camera. The result derives from the photograph itself.

- Boresight in observer-apparent J2000 axes **RA {ra:.6f}°, Dec {dec:.6f}°**. Celestial north at the boresight is {ast['native_north_clockwise_from_up_deg']:.3f}° clockwise from native image up.
- Fitted effective focal length **{ast['camera_parameters'][3]:.3f} px** and principal point {np.round(ast['camera_parameters'][4:6],3).tolist()}. The projected geocenter is {np.round(ast['earth_center_projected'],3).tolist()}, separated from the optical axis and the ellipse center by projection and viewing geometry.
- Native horizontal and vertical centerline FOV **{ast['native_fov_deg'][0]:.2f}° × {ast['native_fov_deg'][1]:.2f}°**. The extended virtual plane spans **{ast['master_fov_deg'][0]:.2f}° × {ast['master_fov_deg'][1]:.2f}°** along its centerlines.
- Verified **{ast['verified_matches']} Hipparcos associations** with **{ast['match_rms_px']:.3f} px** 2D fit RMS, {ast['match_p95_px']:.3f} px P95 and **{ast['held_out_rms_px']:.3f} px** refit holdout RMS over {ast['held_out_count']} matches. Shifted-sky controls gave {ast['null_match_counts']} associations within 2.3 px for the same set of predictions. The association list includes observed and predicted pixel coordinates.
- Venus initialized roll along with the geocenter and the oblate Earth limb. It was removed from the final stellar fit. Its measured and projected centers differ by **{ast['venus_residual_px']:.2f} px**. The measured image has a saturated asymmetric core and flare. Natural Earth coastlines projected from the spacecraft offer a separate geographic consistency check in astronomy_validation.png.
- The fit uses two radial coefficients {np.round(ast['camera_parameters'][6:8],7).tolist()}, two Brown tangential terms and small fitted axis stretch and shear. The stellar holdout captures the combined impact of unmodeled lens/window effects and centroid noise. A linear radial tangent and smoothly bounded tangential terms continue the projection beyond the physical sensor radius. A one-radial-term fit with the same continuation estimates model sensitivity.
- FITS WCS approximates the calibrated projection in apparent J2000 axes. SIP inverse residual is {ast['wcs']['sip_rms_px']:.4f} px RMS and {ast['wcs']['sip_max_px']:.4f} px max on the test grid. The exact parametric pixel→sky→pixel round-trip had a max error of {ast['pixel_to_sky_roundtrip_max_px']:.3g} px. Neither number measures observational accuracy.

## Extension and visibility

The extension uses an artistic linear-flux multiplier of {sky.get('extension_flux_gain',1.0):g}, smoothly introduced over {sky.get('extension_gain_fade_px',180.0):g} native pixels beyond the photograph. A multiplier of one restores the fitted photometric brightness.

Hipparcos coordinates were propagated from epoch J1991.25 with proper motion, barycentric observer parallax and first-order stellar aberration. The detection function was fitted by comparing visible source candidates with detections on the original space background. Its half-completeness limit is **V = {sky['magnitude_50']:.2f}** at half a focal length off axis, with a logistic width of {sky['completeness_width']:.2f} mag. The original matched-filter detection threshold is {cfg['sky']['minimum_snr']:.1f} in empirical local-noise units. Position-dependent completeness gives V limits of {np.round(sky['extension_limit_v'],2).tolist()} across the eligible extension (min/median/max). The renderer continues down the measured completeness curve to a probability floor of {100*sky['min_detection_probability']:.1f}% and a catalog ceiling of V = {sky['rendered_max_v']:.2f}. Catalog flux is attenuated at the camera edge using the fitted photometry. **{sky['rendered']}** of {sky['eligible_outside']} eligible stars were rendered, including {sky['below_half_completeness']} sources below the local half-completeness level. Their individual measured-model fluxes determine whether they rise above the background. The actual image already supplies the in-footprint stars. In the extension all added point sources are catalog projections.

The fit measured {sky['psf_stars']} bright stellar profiles. Median geometric-mean FWHM is {sky['fwhm_median']:.2f} px after grading and {sky['fwhm_raw_median'] or float('nan'):.2f} px in the NEF measurement. Local anisotropic profiles, empirical V-to-flux calibration with off-axis and B–V terms, and a weak Gaussian wing determine their appearance. The robust log-flux model has {sky['log10_flux_rms']:.3f} dex RMS. The original measurement folds lens, window, exposure, and publication effects together. Relative star color is an approximate Johnson B–V rendering. The unobserved extension has no pixel-level optical ground truth.

Bootstrap 95% displacement, summarized as median / P95 / maximum across rendered stars, is {np.round(sky['projection_scatter_px'],2).tolist()} px. The analogous displacement between two-term and one-term radial models is {np.round(sky['model_sensitivity_px'],2).tolist()} px. Bootstrap scatter covers the fit's statistical stability. The model comparison exposes a separate extrapolation systematic. The WCS and extended table specify predicted positions within the adopted virtual projection. The strongest astrometric evidence applies to the original footprint.

| V interval start, 0.5 mag bins | Eligible | Matched |
|---|---:|---:|
"""
    text+="\n".join(f"| {m:.1f} | {n} | {k} |" for m,n,k in sky["bins"])
    text+=f"""

The extra sky continues the masked low-frequency illumination at the photographic boundary and gently decays to a measured blue-black floor. Observed zodiacal light and atmospheric glow stay in the original photograph. Fine structure is sampled directly from graded NASA sky residuals. Overlapping jittered windows from {b['grain_donors']} eligible donor centers preserve the empirical RGB texture and positive tail, including nonstellar speckle. Per-window contrast balancing and complementary power overlap suppress patch boundaries. The donor masks exclude projected Hipparcos positions, Venus and measured point-source candidates. This separates the photographed grain from catalog-projected stars. Local noise amplitude approaches the boundary measurement at the seam and returns to donor amplitude farther out. A {b['feather_px']} px feather operates in the sky outside the fitted Earth and locally preserves {b['feather_star_anchors']} verified stellar peaks in the transition. Measured edge jumps over the two diagnostic strips were {np.round([v['edge_jump_8bit'] for v in b['seams']['profiles']],4).tolist()} display-luma codes on a 0–255 scale. The ratios of outside to inside high-pass noise RMS are {np.round([v['outside_inside_texture_rms_ratio'] for v in b['seams']['profiles']],3).tolist()}.

### Matched sky texture crops

| Region | Luma RMS | Luma P99 | Luma P99.9 | Local peaks >8 per MP | Local RMS CV |
|---|---:|---:|---:|---:|---:|
"""
    for name,probe in b.get("texture_probes",{}).items():
        text+=f"| {name} | {probe['residual_rms']:.2f} | {probe['residual_p99']:.2f} | {probe['residual_p999']:.2f} | {probe['peaks_above_8_per_megapixel']:,.0f} | {probe['local_rms_cv']:.3f} |\n"
    text+=f"""
Values use a 2.2 px Gaussian high-pass of display luma on the same 0–255 scale. Local RMS CV measures the relative spread of RMS across 48 × 48 tiles. The comparison in sky_texture_comparison.png uses identical pixel scale and brightness rendering for the three sample regions.

## Optical ridge removal

The detector searches around two native-coordinate curve guides for a smooth positive ridge in local transverse cuts. It separates each ridge from a slowly changing baseline, fits color and width along the curve and subtracts the smooth field from graded RGB. The Earth and atmospheric rim and a core around Venus have explicit protection masks. Point-source detail and photographic texture stay in the residual. The RAW values below use the same geometric template on the independent NEF development. Their display conversion differs from the JPEG grade.

| Ridge | Fit cuts | Median width px | Verification cuts | Median luma before | Median luma after | NEF luma contrast |
|---|---:|---:|---:|---:|---:|---:|
"""
    for r in d["flare"].get("ridges",[]):
        raw_contrast=r["nef_same_template_median_8bit"]
        raw_label=f"{raw_contrast:.2f}" if raw_contrast is not None else "—"
        text+=f"| {r['name']} | {r['detected_profiles']} | {r['cross_fwhm_median_px']:.1f} | {r['holdout_profiles']} | {r['holdout_contrast_before_8bit']:.2f} | {r['holdout_contrast_after_8bit']:.2f} | {raw_label} |\n"
    text+=f"""
The table uses display-luma codes on a 0–255 scale. Verification cuts lie between the fitted cross-sections. The visual comparison and signed subtraction map are in flare_analysis.png. Beyond the photograph, the diffuse model progressively suppresses short spatial scales with distance from the edge, limiting spurious rays caused by continuing the last recorded rows or columns.

"""
    for halo in d["flare"].get("halos",[]):
        lo,median,hi=halo["galactic_latitude_deg"]
        text+=f"""A second pass fits a signed diffuse residual on the lower part of the long arc using uncontaminated sky on both sides as a local baseline. Across {halo['control_cuts']} interleaved cuts the broad-profile RMS changed from {halo['profile_rms_core_only_8bit']:.3f} to {halo['profile_rms_refined_8bit']:.3f} display-luma codes. The halo neighborhood spans Galactic latitudes {lo:.1f}° to {hi:.1f}° with median {median:.1f}°. Its high latitude and shape favor an optical explanation. NASA separately identifies the diffuse cone terminating near Venus as zodiacal light. [NASA preliminary lunar science report](https://science.nasa.gov/feature/artemis-ii-preliminary-lunar-science-report/)

"""
    glow=d.get("sun_glow",{})
    if glow.get("enabled"):
        text+=f"""## Sunward sky glow

An optional artistic correction reduces the broad sunward background and its high-frequency noise. The photographed atmospheric rim and detected point sources are protected. The estimated maximum background correction was {glow['peak_background_reduction_8bit']:.2f} display-luma codes. This alters real sky information: NASA identifies zodiacal light in this region, and the present image cannot uniquely separate it from optical scattering. [NASA description](https://science.nasa.gov/earth/earth-observatory/a-moonlit-earth-as-seen-from-artemis-ii/)

"""
    stars=d.get("correct_stars",{})
    if stars.get("enabled"):
        text+=f"""## Star shape correction

For {stars['corrected']:,} isolated sources an area-preserving transform reduces measured elongation, with integrated luminance renormalized. {stars['already_round']:,} were already round, while {stars['crowded']:,} crowded and {stars['ambiguous']:,} ambiguous detections were left unchanged. Geometric resampling changes subpixel information; this is an aesthetic correction and cannot guarantee preservation of every original detail.

"""
    black=d.get("amoled",{})
    if black.get("enabled"):
        text+=f"""## AMOLED sky

After grain and sharpening, a smoothed sky estimate is subtracted from the background. Positive fine-scale residuals retain grain on a zero baseline. The atmospheric rim is protected and detected point sources use their own local background estimates. Catalog star light is recovered in linear RGB before encoding it against black. {black['black_pixels']:,} of {black['total_pixels']:,} master pixels are fully black.

"""
    video_grain=d.get("video_grain",{})
    if video_grain.get("enabled"):
        shape=video_grain["analysis_size"]
        text+=f"""## Fixed screen grain

The fixed texture estimate uses {len(video_grain['frame_indices'])} decoded frames at {shape[0]} × {shape[1]} working resolution. Each of two disjoint groups reconstructs the moving photograph separately. Frame residuals are pooled in screen coordinates, with each frame excluded from its own predicted photograph. Cross-group covariance is compared with shifted controls. The fit uses coarse spatial bands and tonal samples with weak scene structure. The result can include static video processing and encoding structure.

The grain pass adds a seeded luma texture to the full master using a Gaussian band-pass with fine sigma {video_grain['sigma_master_px']:.2f} native px, broad sigma {video_grain['broad_master_px']:.2f} px and tone-dependent amplitude. The measured RMS actually applied over the canvas was {video_grain['applied_rms_luma_8bit']:.3f} display-luma codes on a 0–255 scale. The renderer uses the measured scale and amplitude to synthesize the field. Color comparisons above measure the pre-overlay grade. The before and after crops and the added layer are in video_grain_comparison.png.

"""
    detail=d.get("sharpness",{})
    if detail.get("enabled"):
        metrics=detail["metrics"]
        text+="""## Native-scale detail response

The aurora comparison uses the registered video frame and the same native-scale wallpaper crop. Interleaved spatial tiles excluded from the tuning pass supply the comparison. The approved settings reduce fine-scale energy and increase the contribution of the middle band. The scale of the Earth and the sampled stellar core pixels are protected. The PNG comparison includes conventional unsharp masking at the same scale.

| Render | Fine cloud RMS error | Fine aurora RMS error | Fine sky peaks above 6 per MP |
|---|---:|---:|---:|
"""
        for name,row in metrics.items():
            text+=f"| {name} | {row['clouds']['error_rms_8bit'][0]:.2f} | {row['aurora']['error_rms_8bit'][0]:.2f} | {row['sky']['fine_peaks_per_mp']['6']:,.0f} |\n"
        text+="\nFine values use the residual between the image and a 0.72 px Gaussian blur. RMS errors compare luminance to the video. The fixed threshold peak counts characterize the sparse appearance of the nearby sky. The filters are configured in the sharpness section of config.toml.\n\n"
    text+=f"""## Reproduction

See README.md in the project root. Python {d['execution']['python']} and OpenCV {d['execution']['opencv']} ran on the Arch Linux host using CPU implementations and tiled rendering in {d["execution"]["wall_seconds"]:.1f} s for this cached run. The host has an Intel Core i9-9900K, 16 GB RAM and a GeForce RTX 2070. GPU processing was unnecessary. Random seed {d['execution']['seed']}. The source hashes, fitted camera, model coefficients, frame scores and output hashes are in measurements.json. The extended and verified Hipparcos CSVs and FITS header accompany this report.
"""
    path.write_text(text)
