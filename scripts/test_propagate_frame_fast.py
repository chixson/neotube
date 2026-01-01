import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ICRS, SkyCoord

from neotube import propagate as pr


def test_icrs_from_horizons_au_matches_skycoord():
    x, y, z = 1.2, -0.3, 0.5
    vx, vy, vz = 0.01, -0.02, 0.005
    pos, vel = pr._icrs_from_horizons_au(x, y, z, vx, vy, vz)

    rep = CartesianRepresentation(x * u.au, y * u.au, z * u.au)
    diff = CartesianDifferential(vx * u.au / u.day, vy * u.au / u.day, vz * u.au / u.day)
    coord = SkyCoord(rep.with_differentials(diff), frame=ICRS())
    pos_ref = coord.cartesian.xyz.to(u.km).value
    vel_ref = coord.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value

    np.testing.assert_allclose(pos, pos_ref, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(vel, vel_ref, rtol=0.0, atol=1e-12)
