from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neotube.resample import ResampleConfig, resample_replicas


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resample + rejuvenate replicas (Liu-West).")
    p.add_argument("--replicas", type=Path, required=True, help="Input replicas CSV (must include state + logw/w).")
    p.add_argument("--out", type=Path, required=True, help="Output replicas CSV (uniform weights).")
    p.add_argument("--seed", type=int, required=True, help="RNG seed.")
    p.add_argument("--liu-west-a", type=float, default=0.99, help="Liu-West shrinkage factor in (0,1).")
    p.add_argument("--ridge", type=float, default=1e-6, help="Diagonal ridge added to noise covariance.")
    p.add_argument("--plot", type=Path, default=None, help="Optional output PNG showing weight histogram (before).")
    p.add_argument("--meta-in", type=Path, default=None, help="Optional replicas_meta.json to copy/update.")
    p.add_argument("--meta-out", type=Path, default=None, help="Optional output meta json path.")
    return p


def _weights_from_df(df: pd.DataFrame) -> np.ndarray:
    if "w" in df.columns:
        w = df["w"].to_numpy(np.float64)
        w = np.clip(w, 0.0, np.inf)
        s = float(np.sum(w))
        if s > 0 and np.isfinite(s):
            return w / s
    if "logw" in df.columns:
        lw = df["logw"].to_numpy(np.float64)
        m = float(np.max(lw))
        w = np.exp(lw - m)
        s = float(np.sum(w))
        if s > 0 and np.isfinite(s):
            return w / s
    return np.full(len(df), 1.0 / max(len(df), 1), dtype=np.float64)


def main() -> None:
    args = build_argparser().parse_args()
    df = pd.read_csv(args.replicas)
    w = _weights_from_df(df)

    cfg = ResampleConfig(liu_west_a=args.liu_west_a, ridge=args.ridge)
    out = resample_replicas(df, seed=args.seed, cfg=cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(w, bins=60)
        ax.set_title("Replica weight histogram (before resample)")
        ax.set_xlabel("weight")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        plt.close(fig)

    if args.meta_in is not None and args.meta_out is not None:
        meta = json.loads(args.meta_in.read_text())
        meta["resample"] = {
            "method": "liu_west",
            "liu_west_a": args.liu_west_a,
            "ridge": args.ridge,
            "seed": args.seed,
            "input": str(args.replicas),
        }
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        args.meta_out.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

