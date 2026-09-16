"""
Design space for BlendedNet, from:

  "BlendedNet: A Blended Wing Body Aircraft Dataset and Surrogate Model for
  Aerodynamic Predictions," arXiv:2509.07209 (MIT DeCoDE Lab), dataset on
  Harvard Dataverse (doi:10.7910/DVN/VJT9EP).

Bounds below are the exact min/max observed across all 8830 training cases
(data/blendednet_train.csv, produced by data/fetch_blendednet_summary.py) -
not approximations from the paper text.

This is a DIFFERENT design space from the BWBUG problem in bwbug_bounds.py:
BlendedNet describes a subsonic BWB *aircraft* (compressible air, Mach
0.05-0.5), not the underwater glider from the Component-Sharing paper. The
two are kept separate rather than force-mapped onto one another.

Geometry parameters (mm; C1 is the fixed centerline/root chord length):
  B1, B2, B3       - span-station breakpoints
  C1               - fixed at 1000mm for every case (not sampled)
  C2, C3, C4       - chord lengths at span stations
  S1, S2, S3       - sweep angles (degrees) of three planform segments

Flight condition parameters:
  alt_kft    - altitude, kft
  Re         - Reynolds number
  M_inf      - freestream Mach number
  alpha_deg  - angle of attack, degrees
  beta_deg   - sideslip angle, degrees; always 0 in the dataset (not sampled)
"""

from collections import OrderedDict
from typing import Optional

import numpy as np

C1_FIXED_MM = 1000.0
BETA_FIXED_DEG = 0.0

GEOM_BOUNDS = OrderedDict([
    ("B1", (100.0, 200.0)),
    ("B2", (50.12, 199.9)),
    ("B3", (200.5, 700.0)),
    ("C2", (550.2, 849.8)),
    ("C3", (180.1, 280.0)),
    ("C4", (60.02, 90.0)),
    ("S1", (40.01, 59.99)),
    ("S2", (40.01, 60.0)),
    ("S3", (24.01, 39.99)),
])

FLIGHT_BOUNDS = OrderedDict([
    ("alt_kft", (0.0, 40.0)),
    ("Re", (67330.0, 1.055e8)),
    ("M_inf", (0.05, 0.5)),
    ("alpha_deg", (-10.0, 20.0)),
])

BOUNDS = OrderedDict(list(GEOM_BOUNDS.items()) + list(FLIGHT_BOUNDS.items()))
VAR_NAMES = list(BOUNDS.keys())
LOWER = np.array([b[0] for b in BOUNDS.values()])
UPPER = np.array([b[1] for b in BOUNDS.values()])


def sample_blendednet(n: int = 1, seed: Optional[int] = None, method: str = "lhs") -> np.ndarray:
    """Sample n points uniformly within BlendedNet's real observed design
    space (9 geometry vars + 4 flight-condition vars; C1 and beta_deg are
    fixed constants in the dataset, so they're not sampled here).

    Returns an (n, 13) array in the column order of VAR_NAMES.
    """
    rng = np.random.default_rng(seed)
    dim = len(VAR_NAMES)

    if method == "lhs":
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=dim, seed=rng)
        unit = sampler.random(n=n)
    else:
        unit = rng.random((n, dim))

    return LOWER + unit * (UPPER - LOWER)


def as_dict(row: np.ndarray) -> dict:
    d = {name: float(v) for name, v in zip(VAR_NAMES, row)}
    d["C1"] = C1_FIXED_MM
    d["beta_deg"] = BETA_FIXED_DEG
    return d


if __name__ == "__main__":
    sample = sample_blendednet(n=1, seed=0)
    print("BlendedNet sampled design + flight condition:")
    for name, value in as_dict(sample[0]).items():
        kind = "geometry" if name in GEOM_BOUNDS or name == "C1" else "flight-condition"
        print(f"  {name:>10} ({kind:>16}) = {value:.4f}")
