import csv
import math
import os

import pytest

import os
import sys

sys.path.insert(0, os.getcwd())
import scripts.geom_validator_cached_polite as geom


def _ang_sep_arcsec(ra1, dec1, ra2, dec2):
    ra1r, ra2r = math.radians(ra1), math.radians(ra2)
    d1r, d2r = math.radians(dec1), math.radians(dec2)
    cosang = math.sin(d1r) * math.sin(d2r) + math.cos(d1r) * math.cos(d2r) * math.cos(ra1r - ra2r)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang)) * 3600.0


def _load_first_z22_obs(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get("site") or "").strip().upper() == "Z22":
                return row
    raise RuntimeError("No Z22 row found in obs CSV")


def test_geom_validator_paths_z22():
    if os.environ.get("NEOTUBE_RUN_HORIZONS_TESTS") != "1":
        pytest.skip("Set NEOTUBE_RUN_HORIZONS_TESTS=1 to run Horizons-based tests.")
    pytest.importorskip("astroquery")
    pytest.importorskip("astropy")

    obs = _load_first_z22_obs("runs/ceres/obs.csv")
    t_obs_iso = obs["t_utc"]

    site_catalog = geom.fetch_mpc_obs_catalog_cached()
    site_entry = site_catalog.get("Z22")
    site_ecef = geom.mpc_site_to_ecef_km(site_entry)
    assert site_ecef is not None

    t_obs = geom.Time(t_obs_iso, scale="utc")
    earth_vec = geom.horizons_vectors_cached("399", t_obs.tdb.iso, center="@ssb", refplane="frame")
    earth_bary = earth_vec[:3]
    earth_vel = earth_vec[3:]

    site_bary, site_vel = geom.site_bary_and_vel_from_mpc_site(
        "Z22", t_obs_iso, site_catalog, earth_bary, earth_vel
    )

    ra_a, dec_a, dbg = geom.predict_apparent_radec_for_obs(
        "2000001",
        t_obs_iso,
        site_bary,
        site_vel,
        earth_bary_km=earth_bary,
        earth_vel_km_s=earth_vel,
        site_ecef_km=site_ecef,
    )

    t_em = t_obs.tdb - geom.TimeDelta(float(dbg.tau_s), format="sec")
    obj_vec = geom.horizons_vectors_cached("2000001", t_em.iso, center="@ssb", refplane="frame")
    obj_bary = obj_vec[:3]
    obj_vel = obj_vec[3:]

    ra_b, dec_b = geom.astropy_apparent_radec(
        obj_bary, obj_vel, t_em, t_obs_iso, earth_bary, site_ecef
    )

    from astroquery.jplhorizons import Horizons

    hz = Horizons(id="1", location="Z22", epochs=t_obs.jd, id_type="smallbody")
    eph = hz.ephemerides()
    hz_ra = float(eph["RA"][0])
    hz_dec = float(eph["DEC"][0])

    sep_a = _ang_sep_arcsec(ra_a, dec_a, hz_ra, hz_dec)
    sep_b = _ang_sep_arcsec(ra_b, dec_b, hz_ra, hz_dec)

    assert sep_a < 0.3
    assert sep_b < 5.0
