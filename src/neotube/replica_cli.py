from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.time import Time

from .fit import load_posterior, residuals_batch, residuals_parallel, sample_replicas
from .propagate import predict_radec_batch
from .fit_cli import load_observations
from .sites import get_site_ephemeris
from .ranging import (
    _ranging_reference_observation,
    add_local_spread_parallel,
    add_range_jitter,
    build_attributable,
    build_state_from_ranging,
    sample_ranged_replicas,
    score_candidate,
    build_state_from_ranging_multiobs,
)

try:
    from astroquery.jplhorizons import Horizons
except Exception:
    Horizons = None

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(cache=True)
    def _studentt_loglike_numba(residuals, sigma_vec, nu):
        out = np.empty(residuals.shape[0], dtype=np.float64)
        for i in range(residuals.shape[0]):
            acc = 0.0
            for j in range(residuals.shape[1]):
                t = (residuals[i, j] / sigma_vec[j]) ** 2
                acc += (nu + 1.0) * np.log1p(t / nu)
            out[i] = -0.5 * acc
        return out


_BUILD_CTX: dict[str, object] = {}


def _init_build_ctx(
    obs_eval,
    obs_ref,
    epoch,
    attrib,
    perturbers,
    max_step,
    use_kepler,
    max_iter,
    multiobs: bool,
) -> None:
    global _BUILD_CTX
    _BUILD_CTX = {
        "obs_eval": obs_eval,
        "obs_ref": obs_ref,
        "epoch": epoch,
        "attrib": attrib,
        "perturbers": perturbers,
        "max_step": max_step,
        "use_kepler": use_kepler,
        "max_iter": max_iter,
        "multiobs": multiobs,
    }


def _build_states_chunk(chunk: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    ctx = _BUILD_CTX
    rhos, rhodots = chunk
    out = []
    if ctx["multiobs"]:
        for r, rd in zip(rhos, rhodots):
            out.append(
                build_state_from_ranging_multiobs(
                    ctx["obs_eval"],
                    ctx["obs_ref"],
                    ctx["epoch"],
                    ctx["attrib"],
                    float(r),
                    float(rd),
                    ctx["perturbers"],
                    ctx["max_step"],
                    use_kepler=bool(ctx["use_kepler"]),
                    max_iter=int(ctx["max_iter"]),
                )
            )
    else:
        for r, rd in zip(rhos, rhodots):
            out.append(
                build_state_from_ranging(
                    ctx["obs_ref"],
                    ctx["epoch"],
                    ctx["attrib"],
                    float(r),
                    float(rd),
                )
            )
    return np.array(out, dtype=float)


def _build_states_serial(
    rhos: np.ndarray,
    rhodots: np.ndarray,
    obs_eval,
    obs_ref,
    epoch,
    attrib,
    perturbers,
    max_step: float,
    use_kepler: bool,
    max_iter: int,
    multiobs: bool,
) -> np.ndarray:
    out = []
    if multiobs:
        for r, rd in zip(rhos, rhodots):
            out.append(
                build_state_from_ranging_multiobs(
                    obs_eval,
                    obs_ref,
                    epoch,
                    attrib,
                    float(r),
                    float(rd),
                    perturbers,
                    max_step,
                    use_kepler=bool(use_kepler),
                    max_iter=int(max_iter),
                )
            )
    else:
        for r, rd in zip(rhos, rhodots):
            out.append(build_state_from_ranging(obs_ref, epoch, attrib, float(r), float(rd)))
    return np.array(out, dtype=float)


def _build_states_parallel(
    rhos: np.ndarray,
    rhodots: np.ndarray,
    obs_eval,
    obs_ref,
    epoch,
    attrib,
    perturbers,
    max_step: float,
    use_kepler: bool,
    max_iter: int,
    n_workers: int,
    chunk_size: int | None,
    multiobs: bool,
) -> np.ndarray:
    if n_workers <= 1 or len(rhos) <= 1:
        return _build_states_serial(
            rhos,
            rhodots,
            obs_eval,
            obs_ref,
            epoch,
            attrib,
            perturbers,
            max_step,
            use_kepler,
            max_iter,
            multiobs,
        )
    if chunk_size is None or chunk_size <= 0:
        chunk_size = max(32, int(len(rhos) // max(1, n_workers * 4)))
    chunks = []
    for start in range(0, len(rhos), chunk_size):
        chunks.append((rhos[start : start + chunk_size], rhodots[start : start + chunk_size]))
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(n_workers)),
        mp_context=ctx,
        initializer=_init_build_ctx,
        initargs=(
            obs_eval,
            obs_ref,
            epoch,
            attrib,
            perturbers,
            max_step,
            use_kepler,
            max_iter,
            multiobs,
        ),
    ) as executor:
        for arr in executor.map(_build_states_chunk, chunks):
            results.append(arr)
    if not results:
        return np.empty((0, 6), dtype=float)
    return np.vstack(results)


def _radec_chunk_worker(payload):
    chunk_states, epoch_str = payload
    epoch = Time(epoch_str, scale="utc")
    epochs = [epoch] * chunk_states.shape[0]
    ra, dec = predict_radec_batch(chunk_states, epochs)
    return ra, dec


_SMC_CTX: dict[str, object] = {}


def _init_smc_ctx(
    obs_eval,
    obs_ref,
    epoch,
    attrib,
    perturbers,
    max_step,
    use_kepler,
    max_iter,
    multiobs: bool,
    observations,
    allow_unknown_site: bool,
) -> None:
    global _SMC_CTX
    _SMC_CTX = {
        "obs_eval": obs_eval,
        "obs_ref": obs_ref,
        "epoch": epoch,
        "attrib": attrib,
        "perturbers": perturbers,
        "max_step": max_step,
        "use_kepler": use_kepler,
        "max_iter": max_iter,
        "multiobs": multiobs,
        "observations": observations,
        "allow_unknown_site": allow_unknown_site,
    }


def _smc_build_chunk(chunk: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    ctx = _SMC_CTX
    rhos, rhodots = chunk
    out = []
    if ctx["multiobs"]:
        for r, rd in zip(rhos, rhodots):
            out.append(
                build_state_from_ranging_multiobs(
                    ctx["obs_eval"],
                    ctx["obs_ref"],
                    ctx["epoch"],
                    ctx["attrib"],
                    float(r),
                    float(rd),
                    ctx["perturbers"],
                    ctx["max_step"],
                    use_kepler=bool(ctx["use_kepler"]),
                    max_iter=int(ctx["max_iter"]),
                )
            )
    else:
        for r, rd in zip(rhos, rhodots):
            out.append(
                build_state_from_ranging(
                    ctx["obs_ref"],
                    ctx["epoch"],
                    ctx["attrib"],
                    float(r),
                    float(rd),
                )
            )
    return np.array(out, dtype=float)


def _smc_resid_chunk(states: np.ndarray) -> np.ndarray:
    ctx = _SMC_CTX
    return residuals_batch(
        states,
        ctx["epoch"],
        ctx["observations"],
        ctx["perturbers"],
        ctx["max_step"],
        use_kepler=bool(ctx["use_kepler"]),
        allow_unknown_site=bool(ctx["allow_unknown_site"]),
    )


def _chunk_pairs(
    rhos: np.ndarray, rhodots: np.ndarray, chunk_size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    chunks = []
    for start in range(0, len(rhos), chunk_size):
        chunks.append((rhos[start : start + chunk_size], rhodots[start : start + chunk_size]))
    return chunks


def _chunk_states(states: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    return [states[i : i + chunk_size] for i in range(0, states.shape[0], chunk_size)]


def _get_smc_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("forkserver")
    except ValueError:
        return mp.get_context("spawn")

def _jpl_location_from_site(site: str | None) -> str | None:
    if not site:
        return None
    config = get_site_ephemeris(site)
    if not config:
        return None
    ephem_id = str(config.get("ephemeris_id"))
    if not ephem_id:
        return None
    if "@" in ephem_id:
        return ephem_id
    return f"500@{ephem_id}"


def _attach_photometry(
    observations,
    *,
    source: str,
    target: str | None,
    location: str | None,
    sigma_mag_default: float,
    h0_override: float | None,
    h_sigma: float,
    g_override: float | None,
):
    if source == "none":
        return None
    if source == "obs":
        if h0_override is None:
            raise SystemExit("--photometry-h is required when using --photometry-source obs")
        for ob in observations:
            if ob.sigma_mag is None and ob.mag is not None:
                ob.sigma_mag = sigma_mag_default
        return {
            "h0": float(h0_override),
            "h_sigma": float(h_sigma),
            "g": float(g_override) if g_override is not None else 0.15,
            "sigma_mag_default": float(sigma_mag_default),
        }
    if source != "jpl":
        raise SystemExit(f"Unsupported photometry source: {source}")
    if Horizons is None:
        raise SystemExit("astroquery not available; cannot use --photometry-source jpl")
    if target is None:
        raise SystemExit("--photometry-target is required when using --photometry-source jpl")
    epochs = [ob.time.jd for ob in observations]
    if location is None:
        location = _jpl_location_from_site(observations[0].site)
    if location is None:
        location = "@399"
    obj = Horizons(id=target, location=location, epochs=epochs, id_type="smallbody")
    eph = obj.ephemerides()
    h0 = h0_override if h0_override is not None else float(eph["H"][0])
    g = g_override if g_override is not None else float(eph["G"][0])
    for ob, row in zip(observations, eph):
        ob.mag = float(row["V"])
        if ob.sigma_mag is None:
            ob.sigma_mag = sigma_mag_default
    return {
        "h0": float(h0),
        "h_sigma": float(h_sigma),
        "g": float(g),
        "sigma_mag_default": float(sigma_mag_default),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample replicas from an existing orbit posterior.")
    parser.add_argument("--posterior", type=Path, required=True, help="Path to posterior .npz artifact.")
    parser.add_argument("--n", type=int, default=500, help="Number of replicas to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path("replicas.csv"), help="Output path for replica table.")
    parser.add_argument("--perturbers", nargs="+", default=["earth", "mars", "jupiter"], help="Perturbers for propagation.")
    parser.add_argument(
        "--method",
        choices=["multit", "gaussian"],
        default="multit",
        help="Sampling method ('multit'=Student-t, 'gaussian'=Gaussian).",
    )
    parser.add_argument("--nu", type=float, default=4.0, help="Degrees of freedom when using Student-t sampling.")
    parser.add_argument(
        "--smc",
        action="store_true",
        help="Use SMC sampling over the posterior likelihood instead of ranged/gaussian sampling.",
    )
    parser.add_argument(
        "--smc-param",
        choices=("cartesian", "attributable"),
        default="cartesian",
        help="Parameterization for SMC: cartesian (6D state) or attributable (rho,rhodot).",
    )
    parser.add_argument("--smc-steps", type=int, default=6, help="Number of SMC tempering steps.")
    parser.add_argument(
        "--smc-ess-frac",
        type=float,
        default=0.5,
        help="Resample when ESS < ess_frac * N.",
    )
    parser.add_argument(
        "--smc-mcmc-steps",
        type=int,
        default=1,
        help="Number of MCMC rejuvenation steps per SMC stage.",
    )
    parser.add_argument(
        "--smc-proposal-scale",
        type=float,
        default=0.5,
        help="Proposal scale for SMC rejuvenation (fraction of std for cartesian; rho-scale for attributable).",
    )
    parser.add_argument(
        "--smc-pool",
        choices=("threads", "process"),
        default="threads",
        help="Parallel pool type for SMC (threads avoids semaphore/permission issues).",
    )
    parser.add_argument(
        "--smc-prior-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to posterior covariance for SMC cartesian prior.",
    )
    parser.add_argument(
        "--smc-prior-a-min-au",
        type=float,
        default=None,
        help="Optional minimum semi-major axis (AU) for SMC cartesian prior.",
    )
    parser.add_argument(
        "--smc-prior-a-max-au",
        type=float,
        default=None,
        help="Optional maximum semi-major axis (AU) for SMC cartesian prior.",
    )
    parser.add_argument(
        "--smc-prior-bound-only",
        action="store_true",
        help="If set, require bound heliocentric orbits (E < 0) in SMC cartesian prior.",
    )
    parser.add_argument(
        "--mix-posterior-frac",
        type=float,
        default=0.0,
        help="If >0 with --ranged, draw a mixture of posterior and ranged proposals and reweight by likelihood.",
    )
    parser.add_argument(
        "--mix-oversample",
        type=float,
        default=1.0,
        help="Oversampling factor for mixture pool before reweighting (e.g. 2.0).",
    )
    parser.add_argument(
        "--mix-likelihood",
        choices=("studentt", "gaussian"),
        default="studentt",
        help="Likelihood used for mixture reweighting (default: studentt).",
    )
    parser.add_argument(
        "--range-jitter-n",
        type=int,
        default=0,
        help="Per-state range jitter count (0 disables).",
    )
    parser.add_argument(
        "--range-jitter-rho-min-au",
        type=float,
        default=1.8,
        help="Min rho (AU) for range jitter.",
    )
    parser.add_argument(
        "--range-jitter-rho-max-au",
        type=float,
        default=4.5,
        help="Max rho (AU) for range jitter.",
    )
    parser.add_argument(
        "--range-jitter-rhodot-max-kms",
        type=float,
        default=20.0,
        help="Max |rhodot| (km/s) for range jitter.",
    )
    parser.add_argument(
        "--range-jitter-reweight",
        action="store_true",
        help="Reweight range-jitter pool by likelihood before resampling.",
    )
    parser.add_argument(
        "--range-jitter-likelihood",
        choices=("studentt", "gaussian"),
        default="studentt",
        help="Likelihood used for range jitter reweighting.",
    )
    parser.add_argument(
        "--ranged",
        action="store_true",
        help="Use ranged/importance sampling to respect distance degeneracy.",
    )
    parser.add_argument("--obs", type=Path, default=None, help="Observation CSV for ranged sampling.")
    parser.add_argument("--n-proposals", type=int, default=50000, help="Number of ranged proposals.")
    parser.add_argument("--rho-min-au", type=float, default=1e-4, help="Min rho (AU).")
    parser.add_argument("--rho-max-au", type=float, default=5.0, help="Max rho (AU).")
    parser.add_argument("--rhodot-max-kms", type=float, default=50.0, help="Max |rho_dot| (km/s).")
    parser.add_argument(
        "--range-profile",
        choices=["neo", "main-belt", "jupiter-trojan", "tno", "comet", "wide", "main", "mba"],
        default=None,
        help=(
            "Preset rho/rhodot ranges for common object classes. "
            "Choices:\n"
            "  neo             (near-earth objects)      : rho=[1e-4,2.0] AU, rhodot<=100 km/s\n"
            "  main-belt (mba) (main-belt asteroids)    : rho=[1.8,4.5] AU, rhodot<=20 km/s\n"
            "  jupiter-trojan  (Trojan asteroids)       : rho=[4.5,5.7] AU, rhodot<=10 km/s\n"
            "  tno              (trans-Neptunian objects): rho=[20,100] AU, rhodot<=5 km/s\n"
            "  comet            (comets, wide eccentric) : rho=[0.01,50] AU, rhodot<=200 km/s\n"
            "  wide             (very wide exploratory)  : rho=[1e-4,100] AU, rhodot<=100 km/s\n"
            "Aliases: 'main' or 'mba' -> 'main-belt'."
        ),
    )
    parser.add_argument(
        "--rho-prior",
        choices=["volume", "log", "uniform"],
        default="log",
        help="Prior for rho proposals: volume (~rho^2), log (1/rho), or uniform.",
    )
    parser.add_argument(
        "--rho-prior-power",
        type=float,
        default=2.0,
        help="Power for rho prior (log weight term = power * log(rho)).",
    )
    parser.add_argument(
        "--ranging-multiobs",
        action="store_true",
        help="Construct ranged proposals by fitting rho/rhodot across multiple observations.",
    )
    parser.add_argument(
        "--ranging-multiobs-indices",
        type=str,
        default=None,
        help="Comma-separated observation indices to use for multi-obs construction (default: all).",
    )
    parser.add_argument(
        "--ranging-multiobs-max-iter",
        type=int,
        default=2,
        help="Max Gauss-Newton iterations for multi-obs ranging construction.",
    )
    parser.add_argument(
        "--ranging-attrib-mode",
        choices=("vector", "linear"),
        default="vector",
        help="Attributable fit for ranged proposals: vector (fit unit vectors) or linear RA/Dec.",
    )
    parser.add_argument(
        "--photometry-source",
        choices=("none", "jpl", "obs"),
        default="none",
        help="Photometry source: none, obs (mag columns), or jpl (Horizons V mag).",
    )
    parser.add_argument(
        "--photometry-target",
        default=None,
        help="Horizons target for photometry (required for --photometry-source jpl).",
    )
    parser.add_argument(
        "--photometry-location",
        default=None,
        help="Horizons observer location override (e.g., 500@-163).",
    )
    parser.add_argument(
        "--photometry-sigma-mag",
        type=float,
        default=0.3,
        help="Default mag uncertainty if none provided in obs.",
    )
    parser.add_argument(
        "--photometry-h",
        type=float,
        default=None,
        help="Absolute magnitude H prior mean (defaults to Horizons H if available).",
    )
    parser.add_argument(
        "--photometry-h-sigma",
        type=float,
        default=0.3,
        help="Absolute magnitude H prior sigma.",
    )
    parser.add_argument(
        "--photometry-g",
        type=float,
        default=None,
        help="HG phase parameter G (defaults to Horizons G if available).",
    )
    parser.add_argument(
        "--admissible-bound",
        action="store_true",
        help="Reject ranged proposals with unbound heliocentric energy.",
    )
    parser.add_argument(
        "--admissible-q-min-au",
        type=float,
        default=None,
        help="Reject ranged proposals with perihelion below this AU.",
    )
    parser.add_argument(
        "--admissible-q-max-au",
        type=float,
        default=None,
        help="Reject ranged proposals with perihelion above this AU.",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=["kepler", "nbody"],
        default="kepler",
        help="Scoring mode for ranged sampling: kepler prefilter or nbody only.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for ranged proposal scoring (defaults to proposals//workers or 128).",
    )
    parser.add_argument(
        "--radec-chunk-size",
        type=int,
        default=2048,
        help="Chunk size for RA/Dec computation.",
    )
    parser.add_argument(
        "--top-k-nbody",
        type=int,
        default=2000,
        help="Top-K proposals to rescore with n-body after Kepler prefilter.",
    )
    parser.add_argument("--n-workers", type=int, default=8, help="Workers for ranged scoring.")
    parser.add_argument(
        "--emit-debug",
        action="store_true",
        help="If set, write ranging_debug.npz with proposals/weights/top-candidates for post-mortem.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Log progress every N ranged proposals (0 disables).",
    )
    parser.add_argument("--max-step", type=float, default=3600.0, help="Max step (seconds) for propagation.")
    parser.add_argument("--no-kepler", action="store_true", help="Disable Kepler propagation for ranged scoring.")
    parser.add_argument(
        "--local-spread-n",
        type=int,
        default=0,
        help="Per-state tangent jitter count (0 disables).",
    )
    parser.add_argument(
        "--local-spread-sigma-arcsec",
        type=float,
        default=0.5,
        help="Per-state tangent jitter sigma (arcsec).",
    )
    parser.add_argument(
        "--local-spread-mode",
        choices=("tangent", "multit", "attributable"),
        default="tangent",
        help=(
            "Which local-spread generator to use (tangent=existing, multit=multivariate-t, "
            "attributable=attributable-space)."
        ),
    )
    parser.add_argument(
        "--local-spread-vel-scale",
        type=float,
        default=1.0,
        help="velocity scale for local spread (multit/tangent modes).",
    )
    parser.add_argument(
        "--local-spread-df",
        type=float,
        default=4.0,
        help="degrees-of-freedom for multivariate-t.",
    )
    args = parser.parse_args()
    if args.ranging_multiobs_indices:
        try:
            args.ranging_multiobs_indices = [
                int(x.strip())
                for x in args.ranging_multiobs_indices.split(",")
                if x.strip() != ""
            ]
        except Exception as exc:
            raise SystemExit(f"Invalid --ranging-multiobs-indices: {exc}") from exc
    # Apply range-profile overrides (explicit profile wins)
    if args.range_profile is not None:
        prof = args.range_profile
        if prof in ("main", "mba"):
            prof = "main-belt"
        if prof == "neo":
            args.rho_min_au = 1e-4
            args.rho_max_au = 2.0
            args.rhodot_max_kms = 100.0
        elif prof == "main-belt":
            args.rho_min_au = 1.8
            args.rho_max_au = 4.5
            args.rhodot_max_kms = 20.0
        elif prof == "jupiter-trojan":
            args.rho_min_au = 4.5
            args.rho_max_au = 5.7
            args.rhodot_max_kms = 10.0
        elif prof == "tno":
            args.rho_min_au = 20.0
            args.rho_max_au = 100.0
            args.rhodot_max_kms = 5.0
        elif prof == "comet":
            args.rho_min_au = 0.01
            args.rho_max_au = 50.0
            args.rhodot_max_kms = 200.0
        elif prof == "wide":
            args.rho_min_au = 1e-4
            args.rho_max_au = 100.0
            args.rhodot_max_kms = 100.0
        print(
            f"[replica_cli] range_profile={args.range_profile} -> "
            f"rho=[{args.rho_min_au},{args.rho_max_au}] AU, rhodot_max={args.rhodot_max_kms} km/s"
        )

    posterior = load_posterior(args.posterior)
    nu_val = posterior.nu if posterior.nu is not None else args.nu
    if args.smc:
        if args.obs is None:
            raise SystemExit("--smc requires --obs (observation CSV).")
        observations = load_observations(args.obs, None)
        site_kappas = getattr(posterior, "site_kappas", {})
        sigma_vec = []
        for ob in observations:
            kappa = site_kappas.get(ob.site or "UNK", 1.0)
            sigma = max(1e-6, float(ob.sigma_arcsec) * float(kappa))
            sigma_vec.extend([sigma, sigma])
        sigma_vec = np.array(sigma_vec, dtype=float)

        def _loglikes_from_residuals(residuals: np.ndarray) -> np.ndarray:
            t = (residuals / sigma_vec) ** 2
            if args.mix_likelihood == "gaussian":
                return -0.5 * np.sum(t, axis=1)
            nu_like = max(1.0, float(nu_val))
            if _HAS_NUMBA:
                return _studentt_loglike_numba(residuals, sigma_vec, nu_like)
            return -0.5 * np.sum((nu_like + 1.0) * np.log1p(t / nu_like), axis=1)

        def score_states_serial(states: np.ndarray) -> np.ndarray:
            res = residuals_batch(
                states,
                posterior.epoch,
                observations,
                tuple(args.perturbers),
                args.max_step,
                use_kepler=not args.no_kepler,
                allow_unknown_site=True,
            )
            return _loglikes_from_residuals(res)

        def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            n = len(weights)
            positions = (rng.random() + np.arange(n)) / n
            cdf = np.cumsum(weights)
            return np.searchsorted(cdf, positions)

        def smc_cartesian() -> np.ndarray:
            rng = np.random.default_rng(int(args.seed))
            cov = np.array(posterior.cov, dtype=float) * (float(posterior.fit_scale) ** 2)
            cov *= float(args.smc_prior_scale) ** 2
            mean = np.array(posterior.state, dtype=float)
            parts = sample_replicas(
                posterior, args.n, seed=int(args.seed), method=args.method, nu=nu_val
            ).T
            parts = np.asarray(parts, dtype=float)
            loglikes = score_states_serial(parts)
            betas = np.linspace(0.0, 1.0, max(2, int(args.smc_steps)))
            stds = np.sqrt(np.maximum(np.diag(cov), 1e-12))
            prop_scale = float(args.smc_proposal_scale) * stds
            inv_cov = np.linalg.pinv(cov)
            mu_sun_km3_s2 = 1.32712440018e11

            def logprior(theta: np.ndarray) -> float:
                d = theta - mean
                lp = float(-0.5 * d.dot(inv_cov).dot(d))
                r = np.linalg.norm(theta[:3])
                v2 = float(np.dot(theta[3:], theta[3:]))
                if r <= 0.0:
                    return -np.inf
                energy = 0.5 * v2 - mu_sun_km3_s2 / r
                if args.smc_prior_bound_only and energy >= 0.0:
                    return -np.inf
                if args.smc_prior_a_min_au is not None or args.smc_prior_a_max_au is not None:
                    if energy >= 0.0:
                        return -np.inf
                    a_km = -mu_sun_km3_s2 / (2.0 * energy)
                    a_au = a_km / 149597870.7
                    if args.smc_prior_a_min_au is not None and a_au < float(args.smc_prior_a_min_au):
                        return -np.inf
                    if args.smc_prior_a_max_au is not None and a_au > float(args.smc_prior_a_max_au):
                        return -np.inf
                return lp

            weights = np.ones(len(parts), dtype=float) / len(parts)
            for b0, b1 in zip(betas[:-1], betas[1:]):
                delta = b1 - b0
                logw = delta * loglikes
                logw -= np.max(logw)
                weights = np.exp(logw)
                weights /= np.sum(weights)
                ess = 1.0 / np.sum(weights**2)
                if ess < float(args.smc_ess_frac) * len(parts):
                    idx = systematic_resample(weights, rng)
                    parts = parts[idx]
                    loglikes = loglikes[idx]
                    weights = np.ones(len(parts), dtype=float) / len(parts)
                for _ in range(int(args.smc_mcmc_steps)):
                    for i in range(len(parts)):
                        cur = parts[i]
                        prop = cur + rng.normal(scale=prop_scale, size=cur.shape)
                    props = parts + rng.normal(scale=prop_scale, size=parts.shape)
                    ll_props = score_states_serial(props)
                    logp_cur = np.array([logprior(p) for p in parts]) + b1 * loglikes
                    logp_prop = np.array([logprior(p) for p in props]) + b1 * ll_props
                    accept = np.log(rng.random(size=len(parts))) < (logp_prop - logp_cur)
                    parts[accept] = props[accept]
                    loglikes[accept] = ll_props[accept]
            return parts

        def smc_attributable() -> np.ndarray:
            rng = np.random.default_rng(int(args.seed))
            attrib = build_attributable(observations, posterior.epoch)
            obs_ref = _ranging_reference_observation(observations, posterior.epoch)
            obs_eval = observations
            if args.ranging_multiobs_indices:
                obs_eval = [observations[i] for i in args.ranging_multiobs_indices]
            n = int(args.n)
            log_rho_min = np.log(max(1e-12, float(args.rho_min_au)))
            log_rho_max = np.log(max(float(args.rho_min_au), float(args.rho_max_au)))
            rhos = np.exp(rng.uniform(log_rho_min, log_rho_max, size=n)) * 149597870.7
            rhodots = rng.uniform(-float(args.rhodot_max_kms), float(args.rhodot_max_kms), size=n)
            n_workers = max(1, int(args.n_workers))
            build_chunk = int(args.chunk_size) if args.chunk_size is not None else max(
                32, int(n // max(1, n_workers * 4))
            )
            resid_chunk = build_chunk
            if n_workers > 1:
                use_threads = str(args.smc_pool).lower() == "threads"
                if use_threads:
                    _init_smc_ctx(
                        obs_eval,
                        obs_ref,
                        posterior.epoch,
                        attrib,
                        tuple(args.perturbers),
                        args.max_step,
                        not args.no_kepler,
                        int(args.ranging_multiobs_max_iter),
                        bool(args.ranging_multiobs),
                        observations,
                        True,
                    )
                    executor_ctx = ThreadPoolExecutor(max_workers=n_workers)
                else:
                    ctx = _get_smc_context()
                    executor_ctx = ProcessPoolExecutor(
                        max_workers=n_workers,
                        mp_context=ctx,
                        initializer=_init_smc_ctx,
                        initargs=(
                            obs_eval,
                            obs_ref,
                            posterior.epoch,
                            attrib,
                            tuple(args.perturbers),
                            args.max_step,
                            not args.no_kepler,
                            int(args.ranging_multiobs_max_iter),
                            bool(args.ranging_multiobs),
                            observations,
                            True,
                        ),
                    )
                with executor_ctx as executor:
                    build_chunks = _chunk_pairs(rhos, rhodots, build_chunk)
                    parts = np.vstack(list(executor.map(_smc_build_chunk, build_chunks)))
                    res_chunks = _chunk_states(parts, resid_chunk)
                    res = np.vstack(list(executor.map(_smc_resid_chunk, res_chunks)))
                    loglikes = _loglikes_from_residuals(res)
                    betas = np.linspace(0.0, 1.0, max(2, int(args.smc_steps)))
                    weights = np.ones(n, dtype=float) / n
                    for b0, b1 in zip(betas[:-1], betas[1:]):
                        delta = b1 - b0
                        logw = delta * loglikes
                        logw -= np.max(logw)
                        weights = np.exp(logw)
                        weights /= np.sum(weights)
                        ess = 1.0 / np.sum(weights**2)
                        if ess < float(args.smc_ess_frac) * n:
                            idx = systematic_resample(weights, rng)
                            rhos = rhos[idx]
                            rhodots = rhodots[idx]
                            parts = parts[idx]
                            loglikes = loglikes[idx]
                            weights = np.ones(n, dtype=float) / n
                        for _ in range(int(args.smc_mcmc_steps)):
                            rho_prop = rhos + rng.normal(
                                scale=float(args.smc_proposal_scale) * rhos
                            )
                            rhodot_prop = rhodots + rng.normal(
                                scale=float(args.smc_proposal_scale)
                                * float(args.rhodot_max_kms),
                                size=n,
                            )
                            rho_prop = np.clip(
                                rho_prop,
                                float(args.rho_min_au) * 149597870.7,
                                float(args.rho_max_au) * 149597870.7,
                            )
                            rhodot_prop = np.clip(
                                rhodot_prop,
                                -float(args.rhodot_max_kms),
                                float(args.rhodot_max_kms),
                            )
                            prop_chunks = _chunk_pairs(rho_prop, rhodot_prop, build_chunk)
                            props = np.vstack(
                                list(executor.map(_smc_build_chunk, prop_chunks))
                            )
                            res_chunks = _chunk_states(props, resid_chunk)
                            res = np.vstack(list(executor.map(_smc_resid_chunk, res_chunks)))
                            ll_props = _loglikes_from_residuals(res)
                            logp_cur = b1 * loglikes
                            logp_prop = b1 * ll_props
                            accept = np.log(rng.random(size=n)) < (logp_prop - logp_cur)
                            rhos[accept] = rho_prop[accept]
                            rhodots[accept] = rhodot_prop[accept]
                            parts[accept] = props[accept]
                            loglikes[accept] = ll_props[accept]
                return parts
            parts = _build_states_serial(
                rhos,
                rhodots,
                obs_eval,
                obs_ref,
                posterior.epoch,
                attrib,
                tuple(args.perturbers),
                args.max_step,
                not args.no_kepler,
                int(args.ranging_multiobs_max_iter),
                bool(args.ranging_multiobs),
            )
            loglikes = score_states_serial(parts)
            betas = np.linspace(0.0, 1.0, max(2, int(args.smc_steps)))
            weights = np.ones(n, dtype=float) / n
            for b0, b1 in zip(betas[:-1], betas[1:]):
                delta = b1 - b0
                logw = delta * loglikes
                logw -= np.max(logw)
                weights = np.exp(logw)
                weights /= np.sum(weights)
                ess = 1.0 / np.sum(weights**2)
                if ess < float(args.smc_ess_frac) * n:
                    idx = systematic_resample(weights, rng)
                    rhos = rhos[idx]
                    rhodots = rhodots[idx]
                    parts = parts[idx]
                    loglikes = loglikes[idx]
                    weights = np.ones(n, dtype=float) / n
                for _ in range(int(args.smc_mcmc_steps)):
                    rho_prop = rhos + rng.normal(scale=float(args.smc_proposal_scale) * rhos)
                    rhodot_prop = rhodots + rng.normal(
                        scale=float(args.smc_proposal_scale) * float(args.rhodot_max_kms),
                        size=n,
                    )
                    rho_prop = np.clip(
                        rho_prop,
                        float(args.rho_min_au) * 149597870.7,
                        float(args.rho_max_au) * 149597870.7,
                    )
                    rhodot_prop = np.clip(
                        rhodot_prop, -float(args.rhodot_max_kms), float(args.rhodot_max_kms)
                    )
                    props = _build_states_serial(
                        rho_prop,
                        rhodot_prop,
                        obs_eval,
                        obs_ref,
                        posterior.epoch,
                        attrib,
                        tuple(args.perturbers),
                        args.max_step,
                        not args.no_kepler,
                        int(args.ranging_multiobs_max_iter),
                        bool(args.ranging_multiobs),
                    )
                    ll_props = score_states_serial(props)
                    logp_cur = b1 * loglikes
                    logp_prop = b1 * ll_props
                    accept = np.log(rng.random(size=n)) < (logp_prop - logp_cur)
                    rhos[accept] = rho_prop[accept]
                    rhodots[accept] = rhodot_prop[accept]
                    parts[accept] = props[accept]
                    loglikes[accept] = ll_props[accept]
            return parts

        if args.smc_param == "cartesian":
            replicas = smc_cartesian().T
        else:
            replicas = smc_attributable().T
    else:
        if args.ranged:
            if args.obs is None:
                raise SystemExit("--ranged requires --obs (observation CSV).")
            observations = load_observations(args.obs, None)
            photometry = None
            if args.photometry_source != "none":
                photometry = _attach_photometry(
                    observations,
                    source=args.photometry_source,
                    target=args.photometry_target,
                    location=args.photometry_location,
                    sigma_mag_default=float(args.photometry_sigma_mag),
                    h0_override=args.photometry_h,
                    h_sigma=float(args.photometry_h_sigma),
                    g_override=args.photometry_g,
                )
            scoring_mode = "nbody" if args.no_kepler else args.scoring_mode
            ranged = sample_ranged_replicas(
                observations=observations,
                epoch=posterior.epoch,
                n_replicas=args.n,
                n_proposals=args.n_proposals,
                rho_min_au=args.rho_min_au,
                rho_max_au=args.rho_max_au,
                rhodot_max_kms=args.rhodot_max_kms,
                perturbers=tuple(args.perturbers),
                max_step=args.max_step,
                nu=nu_val,
                site_kappas=getattr(posterior, "site_kappas", {}),
                seed=args.seed,
                log_every=args.log_every,
                scoring_mode=scoring_mode,
                n_workers=args.n_workers,
                chunk_size=args.chunk_size,
                top_k_nbody=args.top_k_nbody,
                rho_prior_power=args.rho_prior_power,
                rho_prior_mode=args.rho_prior,
                multiobs=args.ranging_multiobs,
                multiobs_indices=args.ranging_multiobs_indices,
                multiobs_max_iter=args.ranging_multiobs_max_iter,
                attrib_mode=args.ranging_attrib_mode,
                photometry=photometry,
                admissible_bound=args.admissible_bound,
                admissible_q_min_au=args.admissible_q_min_au,
                admissible_q_max_au=args.admissible_q_max_au,
            )
            weights = ranged["weights"]
            states = ranged["states"]
            if args.emit_debug:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                diag_path = args.output.parent / "ranging_debug.npz"
                top_k = min(len(weights), max(1, args.top_k_nbody))
                top_idx = np.argsort(-weights)[:top_k]
                try:
                    np.savez(
                        diag_path,
                        rhos=ranged.get("rhos"),
                        rhodots=ranged.get("rhodots"),
                        weights=weights,
                        top_idx=top_idx,
                        top_rhos=ranged.get("rhos")[top_idx],
                        top_rhodots=ranged.get("rhodots")[top_idx],
                        top_weights=weights[top_idx],
                        top_states=states[top_idx],
                    )
                    print("Wrote diagnostics to", diag_path)
                except Exception as exc:
                    print("Failed to write debug npz:", exc)
            if args.mix_posterior_frac and args.mix_posterior_frac > 0.0:
                mix_frac = float(args.mix_posterior_frac)
                mix_frac = min(1.0, max(0.0, mix_frac))
                pool_n = max(args.n, int(round(args.n * float(args.mix_oversample))))
                n_post = int(round(pool_n * mix_frac))
                n_range = max(0, pool_n - n_post)

                rng = np.random.default_rng(int(args.seed))
                post_states = sample_replicas(
                    posterior, n_post, seed=args.seed, method=args.method, nu=nu_val
                ).T

                if n_range > 0:
                    idx = rng.choice(len(states), size=n_range, replace=True, p=weights)
                    range_states = states[idx]
                    pool_states = np.vstack([post_states, range_states])
                else:
                    pool_states = post_states

                sigma_vec = []
                for ob in observations:
                    kappa = getattr(posterior, "site_kappas", {}).get(ob.site or "UNK", 1.0)
                    sigma = max(1e-6, float(ob.sigma_arcsec) * float(kappa))
                    sigma_vec.extend([sigma, sigma])
                sigma_vec = np.array(sigma_vec, dtype=float)

                res = residuals_parallel(
                    pool_states,
                    posterior.epoch,
                    observations,
                    tuple(args.perturbers),
                    args.max_step,
                    n_workers=int(args.n_workers),
                    chunk_size=int(args.chunk_size) if args.chunk_size is not None else None,
                    use_kepler=not args.no_kepler,
                    allow_unknown_site=True,
                )
                t = (res / sigma_vec) ** 2
                if args.mix_likelihood == "gaussian":
                    logw = -0.5 * np.sum(t, axis=1)
                else:
                    nu_like = max(1.0, float(nu_val))
                    logw = -0.5 * np.sum((nu_like + 1.0) * np.log1p(t / nu_like), axis=1)
                logw -= np.max(logw)
                w = np.exp(logw)
                if not np.all(np.isfinite(w)) or np.sum(w) <= 0:
                    raise RuntimeError("Mixture reweight produced invalid weights.")
                w /= np.sum(w)
                idx = rng.choice(pool_states.shape[0], size=args.n, replace=True, p=w)
                replicas = pool_states[idx].T
            else:
                ess = 1.0 / np.sum(weights**2)
                if ess < max(50, 0.05 * len(weights)):
                    from .ranging import stratified_resample

                    replicas = stratified_resample(
                        states,
                        weights,
                        nrep=args.n,
                        n_clusters=12,
                        jitter_scale=1e-6,
                        nu=nu_val,
                        seed=args.seed,
                    ).T
                else:
                    idx = np.random.default_rng(int(args.seed)).choice(
                        len(states), size=args.n, replace=True, p=weights
                    )
                    replicas = states[idx].T
        else:
            replicas = sample_replicas(posterior, args.n, seed=args.seed, method=args.method, nu=nu_val)

    # Optional: range jitter around posterior to explicitly span rho/rhodot.
    if args.range_jitter_n and args.range_jitter_n > 0:
        if args.obs is None:
            raise SystemExit("--range-jitter requires --obs (observation CSV).")
        obs_for_jitter = observations if args.ranged else args.obs
        base_states = replicas.T.copy()
        jittered = add_range_jitter(
            base_states,
            obs_for_jitter,
            posterior.epoch,
            n_per_state=int(args.range_jitter_n),
            rho_min_au=float(args.range_jitter_rho_min_au),
            rho_max_au=float(args.range_jitter_rho_max_au),
            rhodot_max_kms=float(args.range_jitter_rhodot_max_kms),
            seed=int(args.seed) if args.seed is not None else None,
        )
        pool_states = np.vstack([base_states, jittered])
        rng = np.random.default_rng(int(args.seed))
        if args.range_jitter_reweight:
            observations = observations if args.ranged else load_observations(args.obs, None)
            sigma_vec = []
            for ob in observations:
                kappa = getattr(posterior, "site_kappas", {}).get(ob.site or "UNK", 1.0)
                sigma = max(1e-6, float(ob.sigma_arcsec) * float(kappa))
                sigma_vec.extend([sigma, sigma])
            sigma_vec = np.array(sigma_vec, dtype=float)
            res = residuals_parallel(
                pool_states,
                posterior.epoch,
                observations,
                tuple(args.perturbers),
                args.max_step,
                n_workers=int(args.n_workers),
                chunk_size=int(args.chunk_size) if args.chunk_size is not None else None,
                use_kepler=not args.no_kepler,
                allow_unknown_site=True,
            )
            t = (res / sigma_vec) ** 2
            if args.range_jitter_likelihood == "gaussian":
                logw = -0.5 * np.sum(t, axis=1)
            else:
                nu_like = max(1.0, float(nu_val))
                logw = -0.5 * np.sum((nu_like + 1.0) * np.log1p(t / nu_like), axis=1)
            logw -= np.max(logw)
            w = np.exp(logw)
            if not np.all(np.isfinite(w)) or np.sum(w) <= 0:
                raise RuntimeError("Range jitter reweight produced invalid weights.")
            w /= np.sum(w)
            idx = rng.choice(pool_states.shape[0], size=args.n, replace=True, p=w)
            replicas = pool_states[idx].T
        else:
            idx = rng.choice(pool_states.shape[0], size=args.n, replace=True)
            replicas = pool_states[idx].T

    # replicas currently shaped (6, N)
    # Optionally expand each sampled state with local tangent-plane jitter to create thickness
    if args.local_spread_n and args.local_spread_n > 0:
        if args.ranged:
            obs_for_jitter = observations
        else:
            if args.obs is None:
                raise SystemExit("--local-spread requires --obs when not using --ranged")
            obs_for_jitter = args.obs

        states = replicas.T.copy()
        jittered = add_local_spread_parallel(
            states,
            obs_for_jitter,
            posterior,
            mode=args.local_spread_mode,
            n_per_state=int(args.local_spread_n),
            sigma_arcsec=float(args.local_spread_sigma_arcsec),
            fit_scale=float(getattr(posterior, "fit_scale", 1.0)),
            site_kappas=getattr(posterior, "site_kappas", {}),
            vel_scale_factor=float(args.local_spread_vel_scale),
            df=float(args.local_spread_df),
            n_workers=int(args.n_workers),
            chunk_size=int(args.chunk_size) if args.chunk_size is not None else None,
            seed=int(args.seed) if args.seed is not None else None,
        )
        combined = np.vstack([states, jittered])
        rng = np.random.default_rng(int(args.seed))
        if combined.shape[0] >= args.n:
            idxs = rng.choice(combined.shape[0], size=args.n, replace=False)
        else:
            idxs = rng.choice(combined.shape[0], size=args.n, replace=True)
        chosen = combined[idxs]
        replicas = chosen.T

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_name(args.output.stem + "_meta.json")
    with meta_path.open("w") as fh:
        json.dump(
            {
                "epoch_utc": posterior.epoch.isot,
                "n": args.n,
                "seed": args.seed,
                "posterior": str(args.posterior),
                "perturbers": args.perturbers,
                "method": args.method,
                "nu": nu_val,
                "fit_scale": float(getattr(posterior, "fit_scale", 1.0)),
                "ranged": args.ranged,
                "obs": str(args.obs) if args.obs else None,
                "n_proposals": args.n_proposals,
                "rho_min_au": args.rho_min_au,
                "rho_max_au": args.rho_max_au,
                "rhodot_max_kms": args.rhodot_max_kms,
                "rho_prior": args.rho_prior,
                "rho_prior_power": args.rho_prior_power,
                "scoring_mode": "nbody" if args.no_kepler else args.scoring_mode,
                "chunk_size": args.chunk_size,
                "top_k_nbody": args.top_k_nbody,
                "n_workers": args.n_workers,
                "log_every": args.log_every,
                "mix_posterior_frac": args.mix_posterior_frac,
                "mix_oversample": args.mix_oversample,
                "mix_likelihood": args.mix_likelihood,
                "range_jitter_n": args.range_jitter_n,
                "range_jitter_rho_min_au": args.range_jitter_rho_min_au,
                "range_jitter_rho_max_au": args.range_jitter_rho_max_au,
                "range_jitter_rhodot_max_kms": args.range_jitter_rhodot_max_kms,
                "range_jitter_reweight": args.range_jitter_reweight,
                "range_jitter_likelihood": args.range_jitter_likelihood,
            },
            fh,
            indent=2,
        )

    # Compute RA/Dec in parallel batches to avoid a slow serial loop.
    epoch_isot = posterior.epoch.isot
    states = replicas.T.copy()
    total = states.shape[0]
    radec_chunk = max(1, int(args.radec_chunk_size))

    schedule = []
    for start in range(0, total, radec_chunk):
        schedule.append((states[start : start + radec_chunk], epoch_isot))

    max_workers = max(1, int(args.n_workers or 1))
    actual_workers = min(max_workers, len(schedule))

    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "replica_id",
                "x_km",
                "y_km",
                "z_km",
                "vx_km_s",
                "vy_km_s",
                "vz_km_s",
                "ra_deg",
                "dec_deg",
            ]
        )
        idx_offset = 0
        if actual_workers <= 1 or len(schedule) == 1:
            for chunk_states, epoch_str in schedule:
                ra, dec = _radec_chunk_worker((chunk_states, epoch_str))
                for i in range(chunk_states.shape[0]):
                    state = chunk_states[i]
                    writer.writerow(
                        [
                            idx_offset + i,
                            *(f"{val:.6f}" for val in state),
                            f"{ra[i]:.6f}",
                            f"{dec[i]:.6f}",
                        ]
                    )
                idx_offset += chunk_states.shape[0]
        else:
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                for (chunk_states, _), (ra, dec) in zip(
                    schedule, executor.map(_radec_chunk_worker, schedule)
                ):
                    for i in range(chunk_states.shape[0]):
                        state = chunk_states[i]
                        writer.writerow(
                            [
                                idx_offset + i,
                                *(f"{val:.6f}" for val in state),
                                f"{ra[i]:.6f}",
                                f"{dec[i]:.6f}",
                            ]
                        )
                    idx_offset += chunk_states.shape[0]

    print(f"Wrote {args.n} replicas to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
