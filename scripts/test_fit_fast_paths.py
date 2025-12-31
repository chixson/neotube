import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from neotube import fit as nf
from neotube import propagate as pr
from neotube.models import Observation


def test_body_bary_posvel_cached_matches_direct():
    t = Time("2024-01-01T00:00:00", scale="tdb")
    pos_cached, vel_cached = nf._body_bary_posvel_for_time("earth", t)
    pos_direct, vel_direct = pr._body_posvel_km_single("earth", t)
    np.testing.assert_allclose(pos_cached, pos_direct, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(vel_cached, vel_direct, rtol=0.0, atol=0.0)


def test_observation_line_of_sight_direction():
    t = Time("2024-01-01T00:00:00", scale="tdb")
    obs = Observation(
        time=t,
        ra_deg=120.0,
        dec_deg=-30.0,
        sigma_arcsec=0.5,
        site="500",
        observer_pos_km=np.zeros(3),
    )
    _, direction = nf._observation_line_of_sight(obs, allow_unknown_site=True)

    coord = SkyCoord(
        ra=obs.ra_deg * u.deg,
        dec=obs.dec_deg * u.deg,
        distance=1.0 * u.au,
        frame="icrs",
        obstime=obs.time,
    )
    expected = coord.cartesian.xyz.to(u.km).value
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(direction, expected, rtol=0.0, atol=0.0)
