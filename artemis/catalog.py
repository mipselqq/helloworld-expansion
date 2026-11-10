"""Epoch and observer corrections for the ESA Hipparcos Main Catalogue."""
import csv

import numpy as np


def unit(vector):
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def load_catalog(path, years, observer_au, observer_v_over_c):
    with path.open() as stream:
        rows = csv.DictReader((line for line in stream if not line.startswith("#") and line.strip()),
                              delimiter="\t")
        fields = ("HIP", "RAICRS", "DEICRS", "Plx", "pmRA", "pmDE", "Vmag", "B-V")
        data = []
        for row in rows:
            if not row["HIP"].strip().isdigit() or not row["RAICRS"].strip() or not row["DEICRS"].strip():
                continue
            data.append([float(row[f]) if row[f].strip() else np.nan for f in fields])
    stars = np.asarray(data)
    stars = stars[np.isfinite(stars[:, 6])]
    ra, dec = np.radians(stars[:, 1:3]).T
    direction = np.column_stack((np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)))
    east = np.column_stack((-np.sin(ra), np.cos(ra), np.zeros(len(ra))))
    north = np.column_stack((-np.cos(ra)*np.sin(dec), -np.sin(ra)*np.sin(dec), np.cos(dec)))
    mas_to_rad = np.pi / (180*3600*1000)
    parallax, pm_ra, pm_dec = np.nan_to_num(stars[:, 3:6]).T
    direction = unit(direction + mas_to_rad*(years*(east*pm_ra[:, None] + north*pm_dec[:, None])
                                             - np.maximum(parallax, 0)[:, None]*observer_au))
    # First-order stellar aberration. O(v²/c²) contributes far below a pixel.
    apparent = unit(direction + observer_v_over_c)
    return stars, apparent
