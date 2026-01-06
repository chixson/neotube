from __future__ import annotations

import argparse
from pathlib import Path

from .fit import load_posterior_json
from .infer import infer_cutouts, load_replicas_states_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential inference on cutouts using propagated replicas (v0).")
    parser.add_argument("--posterior-json", type=Path, required=True, help="Posterior JSON from neotube-fit (contains epoch).")
    parser.add_argument("--replicas", type=Path, required=True, help="Replica CSV from neotube-replicas.")
    parser.add_argument("--cutouts-index", type=Path, required=True, help="cutouts_index.csv from neotube-ztf fetch step.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for weighted replicas + evidence.")

    parser.add_argument("--perturbers", nargs="+", default=["earth", "mars", "jupiter"], help="Perturbers for propagation.")
    parser.add_argument("--workers", type=int, default=None, help="ProcessPool workers for propagation.")
    parser.add_argument("--batch-size", type=int, default=None, help="Replicas per batch for propagation.")
    parser.add_argument("--max-step", type=float, default=3600.0, help="Max step size (seconds) for propagation.")

    parser.add_argument("--fwhm-arcsec", type=float, default=2.0, help="Gaussian PSF FWHM (arcsec) for matched-filter SNR.")
    parser.add_argument("--snr-max", type=float, default=8.0, help="Clamp SNR in likelihood.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Likelihood temperature (higher = softer).")
    parser.add_argument("--search-radius-px", type=int, default=6, help="Search radius (pixels) for local SNR peak near prediction.")
    parser.add_argument("--pos-sigma-px", type=float, default=None, help="Position sigma (pixels) for gating penalty (default ~0.5*FWHM).")

    parser.add_argument("--pdet", type=float, default=0.95, help="Detection probability prior used in hit/miss mixture.")
    parser.add_argument("--miss-logl", type=float, default=0.0, help="Baseline log-likelihood for miss term.")

    parser.add_argument("--resample-ess-frac", type=float, default=0.0, help="Resample when ESS < frac*N (0 disables).")
    parser.add_argument("--jitter-pos-km", type=float, default=5.0, help="Position jitter (km) when resampling.")
    parser.add_argument("--jitter-vel-km-s", type=float, default=0.005, help="Velocity jitter (km/s) when resampling.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for resampling.")

    parser.add_argument("--max-exposures", type=int, default=None, help="Process at most N exposures.")
    parser.add_argument("--exposure-id", type=str, default=None, help="Process only this exposure_id.")

    args = parser.parse_args()

    posterior = load_posterior_json(args.posterior_json)
    replica_ids, states, logw = load_replicas_states_csv(args.replicas)

    infer_cutouts(
        posterior=posterior,
        replica_ids=replica_ids,
        replica_states=states,
        logw=logw,
        cutouts_index=args.cutouts_index,
        out_dir=args.out_dir,
        perturbers=tuple(args.perturbers),
        workers=args.workers,
        batch_size=args.batch_size,
        max_step=args.max_step,
        fwhm_arcsec=args.fwhm_arcsec,
        snr_max=args.snr_max,
        temperature=args.temperature,
        search_radius_px=args.search_radius_px,
        pos_sigma_px=args.pos_sigma_px,
        pdet=args.pdet,
        miss_logl=args.miss_logl,
        resample_ess_frac=args.resample_ess_frac,
        jitter_pos_km=args.jitter_pos_km,
        jitter_vel_km_s=args.jitter_vel_km_s,
        seed=args.seed,
        max_exposures=args.max_exposures,
        exposure_id=args.exposure_id,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
