import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict

CSV = "runs/ceres/geom_validator_out.csv"  # change path if needed

def signed_deltas(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(np.asarray(ra1))
    ra2 = np.deg2rad(np.asarray(ra2))
    dec1 = np.deg2rad(np.asarray(dec1))
    dec2 = np.deg2rad(np.asarray(dec2))
    dra = ra1 - ra2
    dra = (dra + np.pi) % (2*np.pi) - np.pi
    dx = dra * np.cos(dec1) * 206265.0
    dy = (dec1 - dec2) * 206265.0
    return dx, dy

df = pd.read_csv(CSV)
# If your CSV already has dx/dy, skip recompute. Otherwise:
df['dx_arcsec'], df['dy_arcsec'] = signed_deltas(df['pred_ra'], df['pred_dec'], df['obs_ra'], df['obs_dec'])
df['sep_arcsec'] = np.hypot(df['dx_arcsec'], df['dy_arcsec'])

# 1) time series per site
sites = sorted(df['site'].unique())
plt.figure(figsize=(12, 3*len(sites)))
for i, s in enumerate(sites):
    ax = plt.subplot(len(sites), 1, i+1)
    sub = df[df['site'] == s].copy()
    times = pd.to_datetime(sub['time_utc'])
    ax.scatter(times, sub['dx_arcsec'], label='dx', marker='o')
    ax.scatter(times, sub['dy_arcsec'], label='dy', marker='x')
    if len(sub) > 1:
        # linear fit dx vs time
        x = (times - times.min()).dt.total_seconds().values
        A = np.vstack([np.ones_like(x), x]).T
        coef_dx, *_ = np.linalg.lstsq(A, sub['dx_arcsec'].values, rcond=None)
        coef_dy, *_ = np.linalg.lstsq(A, sub['dy_arcsec'].values, rcond=None)
        tline = np.linspace(0, x.max(), 50)
        ax.plot(times.min() + pd.to_timedelta(tline, unit='s'), coef_dx[0] + coef_dx[1]*tline, color='C0', alpha=0.6)
        ax.plot(times.min() + pd.to_timedelta(tline, unit='s'), coef_dy[0] + coef_dy[1]*tline, color='C1', alpha=0.6)
    ax.set_ylabel(f"{s} (\" )")
    ax.legend()
plt.tight_layout()
plt.savefig("runs/ceres/residuals_timeseries_by_site.png", dpi=180)
print("Wrote runs/ceres/residuals_timeseries_by_site.png")

# 2) quiver plot
plt.figure(figsize=(6,6))
for s in sites:
    sub = df[df['site']==s]
    plt.quiver(np.zeros(len(sub)), np.zeros(len(sub)), sub['dx_arcsec'], sub['dy_arcsec'], angles='xy', scale_units='xy', scale=1, alpha=0.6, label=f"{s} n={len(sub)}")
plt.scatter([],[], c='k', label='dx/dy vectors')
plt.xlabel('dx (arcsec)')
plt.ylabel('dy (arcsec)')
plt.legend()
plt.grid(True)
plt.title('Residual vectors by observation (pred - obs)')
plt.savefig("runs/ceres/residual_vectors.png", dpi=180)
print("Wrote runs/ceres/residual_vectors.png")

# 3) histograms
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(df['dx_arcsec'].dropna(), bins=40)
plt.title('dx distribution (arcsec)')
plt.subplot(1,2,2)
plt.hist(df['dy_arcsec'].dropna(), bins=40)
plt.title('dy distribution (arcsec)')
plt.savefig("runs/ceres/residual_histograms.png", dpi=180)
print("Wrote runs/ceres/residual_histograms.png")

# 4) per-site median table
summary = []
for s in sites:
    sub = df[df['site']==s]
    medx = np.median(sub['dx_arcsec'])
    medy = np.median(sub['dy_arcsec'])
    madx = np.median(np.abs(sub['dx_arcsec']-medx))*1.4826
    mady = np.median(np.abs(sub['dy_arcsec'] - medy))*1.4826
    rms = np.sqrt(np.mean(sub['sep_arcsec']**2))
    summary.append((s, len(sub), medx, medy, madx, mady, rms))
summary_df = pd.DataFrame(summary, columns=['site','n','median_dx','median_dy','mad_dx','mad_dy','rms'])
summary_df.to_csv("runs/ceres/per_site_summary.csv", index=False)
print("Wrote runs/ceres/per_site_summary.csv")
print(summary_df)
