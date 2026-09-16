"""
Named fixed flight-condition operating points for the NSGA-II shape
optimization (optimization/bwb_problem.py). Only the 9 geometry variables
are optimized; the flight condition is held constant per run.

Values are representative points chosen within the real training envelope
of data/blendednet_train.csv (see bounds/blendednet_bounds.FLIGHT_BOUNDS
for the exact observed ranges: alt_kft in [0,40], M_inf in [0.05,0.5],
Re in [6.7e4, 1.055e8], alpha_deg in [-10,20]). They are illustrative
"cruise" / "takeoff" style points, not derived from an atmospheric model:
the dataset itself samples alt_kft, M_inf and alpha_deg near-independently
(checked - pairwise correlations ~0), with only a moderate Re-vs-alt/Mach
correlation, so no strict physical alt/Mach/Re relationship is enforced
here either.

beta_deg is not included: it is always 0 in the dataset
(bounds.blendednet_bounds.BETA_FIXED_DEG) and is not varied.
"""

FLIGHT_CONDITIONS = {
    "cruise": {
        "alt_kft": 35.0,
        "M_inf": 0.45,
        "Re": 4.0e7,
        "alpha_deg": 4.0,
    },
    "takeoff": {
        "alt_kft": 2.0,
        "M_inf": 0.15,
        "Re": 8.0e6,
        "alpha_deg": 14.0,
    },
}
