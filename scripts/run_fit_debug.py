import csv
import json
from pathlib import Path
from astropy.time import Time
from neotube.fit import fit_orbit, Observation
path = Path('runs/ceres/obs_for_fit.csv')
obs = []
with path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        obs.append(
            Observation(
                time=Time(row['time_utc'], scale='utc'),
                ra_deg=float(row['ra_deg']),
                dec_deg=float(row['dec_deg']),
                sigma_arcsec=float(row.get('sigma_arcsec', 0.5)),
                obs_code=row.get('site'),
            )
        )
result = fit_orbit(
    '1',
    obs,
    sigma_arcsec=0.5,
    max_iter=10,
    prior_cov_scale=1e6,
    perturbers=['sun', 'earth', 'moon', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'],
)
outdir = Path('runs/ceres/00_fit_dense')
outdir.mkdir(parents=True, exist_ok=True)
with (outdir / 'posterior.json').open('w') as f:
    json.dump(
        {
            'epoch': result.epoch.utc.isot,
            'state': result.state.tolist(),
            'cov': result.covariance.tolist(),
            'rms': result.rms_arcsec,
        },
        f,
        indent=2,
    )
print('done fit RMS', result.rms_arcsec)
