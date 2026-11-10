"""Download documented inputs atomically and retain their checksums."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess

from PIL import Image, ExifTags
import requests

NAIF = "https://naif.jpl.nasa.gov/pub/naif/pds/pds4/artemis2/artemis2_spice/spice_kernels"
VIZIER = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"
COAST = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"
KERNELS = {
    "naif0012.tls": NAIF + "/lsk/naif0012.tls",
    "de440s.bsp": "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440s.bsp",
    "art002_rec_260402_260410_jsc_bet_revf.bsp": NAIF + "/spk/art002_rec_260402_260410_jsc_bet_revf.bsp",
    "art002_rec_260402_260410_jsc_bet_revf_gapfill.bsp": NAIF + "/spk/art002_rec_260402_260410_jsc_bet_revf_gapfill.bsp",
    "pck00011.tpc": NAIF + "/pck/pck00011.tpc",
    "earth_000101_261115_260819.bpc": NAIF + "/pck/earth_000101_261115_260819.bpc",
    "art002_handcam_v0.ti": NAIF + "/ik/art002_handcam_v0.ti",
}


def download(url, path, params=None):
    if path.exists():
        return path
    temp = path.with_suffix(path.suffix + ".download")
    with requests.get(url, params=params, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        with temp.open("wb") as out:
            for block in response.iter_content(1024 * 1024):
                out.write(block)
    temp.replace(path)
    return path


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def metadata(path):
    with Image.open(path) as im:
        exif = im.getexif()
        tags = {**dict(exif), **dict(exif.get_ifd(34665))}
        return {"dimensions": im.size, "mode": im.mode, "icc_bytes": len(im.info.get("icc_profile", b"")),
                "exif": {ExifTags.TAGS.get(k, str(k)): str(v) for k, v in tags.items() if k < 37500 or k > 37510}}


def acquire(cache, cfg):
    source = cache / "source"
    kernels = cache / "kernels"
    catalog = cache / "catalog"
    for folder in [source, kernels, catalog, cache / "work"]:
        folder.mkdir(parents=True, exist_ok=True)
    files = [(cfg[key], source / name) for key, name in {
        "photo_url": "art002e000192.jpg", "library_url": "nasa_library.jpg",
        "archive_record": "raw_record.html", "archive_metadata": "ART002-E-192.txt",
    }.items()]
    files.extend((url, kernels / name) for name, url in KERNELS.items())
    files.append((COAST, catalog / "ne_10m_coastline.zip"))
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending = [pool.submit(download, url, path) for url, path in files]
        pending.append(pool.submit(download, VIZIER, catalog / "hipparcos.tsv", {
            "-source": "I/239/hip_main", "-out": "HIP,RAICRS,DEICRS,Plx,pmRA,pmDE,Vmag,B-V",
            "Vmag": "<" + str(cfg["max_catalog_magnitude"]), "-out.max": "unlimited",
        }))
        for job in pending:
            job.result()
    clip = source / "melodysheep_clip.mp4"
    video_origin = source / "video_origin.json"
    if not clip.exists():
        subprocess.run([cfg["yt_dlp"], "--ignore-config", "--cookies-from-browser", cfg["cookies_from_browser"],
                        "--no-playlist", "--downloader-args", "ffmpeg:-loglevel error", "-f", cfg["format"],
                        "--download-sections", f"*{cfg['clip_start']}-{cfg['clip_end']}",
                        "-o", str(clip.with_suffix(".%(ext)s")), cfg["video_url"]], check=True)
        write_json(video_origin, {"yt_dlp": cfg["yt_dlp"], "cookies_from_browser": cfg["cookies_from_browser"],
                                 "format": cfg["format"], "clip_sha256": sha256(clip)})
    raw = source / "art002e000192.NEF"
    if not raw.exists():
        try:
            download(cfg["raw_url"], raw)
        except requests.HTTPError as error:
            if error.response.status_code != 404:
                raise
    files += [(cfg["video_url"], clip), (VIZIER, catalog / "hipparcos.tsv")]
    if raw.exists():
        files.append((cfg["raw_url"], raw))
    write_json(cache / "work" / "provenance.json", {
        "sources": [{"file": str(p.relative_to(cache)), "url": url, "sha256": sha256(p), "bytes": p.stat().st_size}
                    for url, p in files],
        "nasa_jpeg": metadata(source / "art002e000192.jpg"),
        "raw_cached": raw.exists(),
        "video_acquisition": json.loads(video_origin.read_text()) if video_origin.exists() else None,
    })
