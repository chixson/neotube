import numpy as np
import pytest

from neotube import propagate as pr


nb = pytest.importorskip("numba")


def _light_time_linear_numpy(obj_bary, obj_bary_vel, obs_bary, iters, c_km_s):
    out = obj_bary.copy()
    for _ in range(iters):
        vec = out - obs_bary
        rho = np.linalg.norm(vec, axis=1)
        dt = rho / c_km_s
        out = obj_bary - obj_bary_vel * dt[:, None]
    return out


def test_light_time_linear_numba_matches_numpy():
    if not pr._HAS_NUMBA:
        pytest.skip("numba not available")
    rng = np.random.default_rng(123)
    n = 64
    obj_bary = rng.normal(size=(n, 3)) * 1.0e6
    obj_bary_vel = rng.normal(size=(n, 3)) * 10.0
    obs_bary = rng.normal(size=(n, 3)) * 1.0e6
    iters = 2
    c_km_s = 299792.458

    numba_out = pr._light_time_linear_numba(
        obj_bary, obj_bary_vel, obs_bary, iters, c_km_s
    )
    numpy_out = _light_time_linear_numpy(
        obj_bary, obj_bary_vel, obs_bary, iters, c_km_s
    )
    np.testing.assert_allclose(numba_out, numpy_out, rtol=1e-10, atol=1e-10)


def test_radec_from_topovec_numba_matches_numpy():
    if not pr._HAS_NUMBA:
        pytest.skip("numba not available")
    rng = np.random.default_rng(321)
    n = 128
    topovec = rng.normal(size=(n, 3))

    ra_numba, dec_numba = pr._radec_from_topovec_numba(topovec)
    xy = np.hypot(topovec[:, 0], topovec[:, 1])
    ra_numpy = (np.degrees(np.arctan2(topovec[:, 1], topovec[:, 0])) + 360.0) % 360.0
    dec_numpy = np.degrees(np.arctan2(topovec[:, 2], xy))

    np.testing.assert_allclose(ra_numba, ra_numpy, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dec_numba, dec_numpy, rtol=1e-10, atol=1e-10)
