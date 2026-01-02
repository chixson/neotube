import numpy as np
from pathlib import Path

CKPT = Path("runs/ceres-ground-test/fit_smc_ckpt.npz")
SMC_DEBUG_DIR = Path("runs/ceres-ground-test/smc_debug")


def load_checkpoint_states(ckpt_path):
    d = np.load(ckpt_path, allow_pickle=True)
    for k in d.files:
        if "state" in k or "states" in k:
            return d[k]
    raise RuntimeError("Could not find 'states' in checkpoint")


def find_latest_pre_resample(debug_dir):
    files = sorted(debug_dir.glob("obs2_pre_resample_*.npz"))
    if not files:
        files = sorted(debug_dir.glob("obs3_pre_resample_*.npz"))
    if not files:
        raise RuntimeError("No pre_resample files found in debug dir")
    return files[-1]


def extract_pos_vel_from_state(s):
    try:
        pos = np.asarray(s.pos, dtype=float)
        vel = np.asarray(s.vel, dtype=float)
        epoch = getattr(s, "epoch", None)
        return pos, vel, epoch
    except Exception:
        arr = np.asarray(s, dtype=float).ravel()
        if arr.size >= 6:
            return arr[:3].copy(), arr[3:6].copy(), None
        return None, None, None


def compare_arrays(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return None
    diffs = np.linalg.norm(a - b, axis=1)
    rel = diffs / (np.linalg.norm(a, axis=1) + 1e-30)
    return diffs, rel


def main():
    ckpt_states = load_checkpoint_states(CKPT)
    print("Loaded checkpoint states shape:", np.asarray(ckpt_states).shape)

    pre_file = find_latest_pre_resample(SMC_DEBUG_DIR)
    print("Using pre-resample dump:", pre_file)
    pre = np.load(pre_file, allow_pickle=True)
    smc_states = pre["states"]

    n_ck = len(ckpt_states)
    n_smc = len(smc_states)
    n = min(n_ck, n_smc)
    print(f"Comparing first {n} states (ckpt vs smc_pre_resample)")

    ck_pos = np.zeros((n, 3))
    ck_vel = np.zeros((n, 3))
    ck_epoch = [None] * n
    for i in range(n):
        p, v, e = extract_pos_vel_from_state(ckpt_states[i])
        ck_pos[i] = p
        ck_vel[i] = v
        ck_epoch[i] = e

    sm_pos = np.zeros((n, 3))
    sm_vel = np.zeros((n, 3))
    sm_epoch = [None] * n
    for i in range(n):
        p, v, e = extract_pos_vel_from_state(smc_states[i])
        sm_pos[i] = p
        sm_vel[i] = v
        sm_epoch[i] = e

    print(
        "Checkpoint pos norms: min/med/max:",
        np.nanmin(np.linalg.norm(ck_pos, axis=1)),
        np.nanmedian(np.linalg.norm(ck_pos, axis=1)),
        np.nanmax(np.linalg.norm(ck_pos, axis=1)),
    )
    print(
        "SMC pre-res pos norms: min/med/max:",
        np.nanmin(np.linalg.norm(sm_pos, axis=1)),
        np.nanmedian(np.linalg.norm(sm_pos, axis=1)),
        np.nanmax(np.linalg.norm(sm_pos, axis=1)),
    )

    diffs_pos, rel_pos = compare_arrays(ck_pos, sm_pos)
    diffs_vel, rel_vel = compare_arrays(ck_vel, sm_vel)
    if diffs_pos is None:
        print("Cannot compare: shapes differ", ck_pos.shape, sm_pos.shape)
        return

    print(
        "Position diffs (km) - min/median/mean/max:",
        np.nanmin(diffs_pos),
        np.nanmedian(diffs_pos),
        np.nanmean(diffs_pos),
        np.nanmax(diffs_pos),
    )
    print(
        "Position rel diffs - min/median/mean/max:",
        np.nanmin(rel_pos),
        np.nanmedian(rel_pos),
        np.nanmean(rel_pos),
        np.nanmax(rel_pos),
    )
    print(
        "Velocity diffs (km/s) - min/median/mean/max:",
        np.nanmin(diffs_vel),
        np.nanmedian(diffs_vel),
        np.nanmean(diffs_vel),
        np.nanmax(diffs_vel),
    )
    print(
        "Velocity rel diffs - min/median/mean/max:",
        np.nanmin(rel_vel),
        np.nanmedian(rel_vel),
        np.nanmean(rel_vel),
        np.nanmax(rel_vel),
    )

    idxs = np.argsort(diffs_pos)[::-1][:20]
    print("\nTop 20 position mismatches (index, abs_diff_km, rel_diff):")
    for i in idxs[:20]:
        print(
            i,
            diffs_pos[i],
            rel_pos[i],
            "ck_norm",
            np.linalg.norm(ck_pos[i]),
            "sm_norm",
            np.linalg.norm(sm_pos[i]),
            "ck_epoch",
            ck_epoch[i],
            "sm_epoch",
            sm_epoch[i],
        )

    ck_norms = np.linalg.norm(ck_pos, axis=1)
    sm_norms = np.linalg.norm(sm_pos, axis=1)
    nonzero = ck_norms > 0
    scale_ratios = sm_norms[nonzero] / ck_norms[nonzero]
    print(
        "Scale ratio stats (smc_norm / ckpt_norm) min/med/max:",
        np.min(scale_ratios),
        np.median(scale_ratios),
        np.max(scale_ratios),
    )

    sample_ratio = sm_pos[0] / (ck_pos[0] + 1e-30)
    print("Sample per-axis ratio for state 0:", sample_ratio)

    for key in ["resample_idx", "cov_diag", "jitter"]:
        if key in pre.files:
            print(f"pre-resample contains {key}")

    print("\nDone.")


if __name__ == "__main__":
    main()
