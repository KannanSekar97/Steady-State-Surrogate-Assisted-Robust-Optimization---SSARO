import math
import numpy as np

# The MATLAB engine is started lazily (on first evaluation) so that simply
# importing this problem definition -- e.g. via the PROBLEMS registry in
# run_experiment.py -- does not require MATLAB to be installed.
_eng = None


def _get_matlab_engine():
    global _eng
    if _eng is None:
        import matlab.engine
        _eng = matlab.engine.start_matlab()
        _eng.addpath(r'Problems\Constrained\Torque arm\Torque_arm', nargout=0)
        _eng.addpath(r'Problems\Constrained\Torque arm\DesignModule', nargout=0)
        _eng.addpath(r'Problems\Constrained\Torque arm\calfem', nargout=0)
        _eng.addpath(r'Problems\Constrained\Torque arm\calfem\fem', nargout=0)
        _eng.addpath(r'Problems\Constrained\Torque arm\calfem\geom', nargout=0)
    return _eng


ROUND_DECIMALS = 12


def round_array(arr, decimals=ROUND_DECIMALS):
    return np.round(np.asarray(arr, dtype=float), decimals)


def to_2d_rounded(x, decimals=ROUND_DECIMALS):
    xv = np.atleast_2d(x).astype(float)
    return np.round(xv, decimals)

def evaluate_torque_arm(x):
    import matlab

    eng = _get_matlab_engine()

    x = to_2d_rounded(x) # Ensure x is a 2D array with 9 columns


    stot_mat = matlab.double(x[:,:7].tolist())


    N = len(x)

    fx = x[:,7].reshape(-1, 1)
    fy = x[:,8].reshape(-1, 1)

    fx_mat = matlab.double(fx.tolist())
    fy_mat = matlab.double(fy.tolist())

    sreq, mass = eng.torque_arm(
        stot_mat,
        fx_mat,
        fy_mat,
        nargout=2
    )

    F = np.column_stack([
        np.asarray(sreq).flatten(),
        np.asarray(mass).flatten()
    ])

    G = np.asarray(sreq).reshape(-1,1) - 800
    return F, G

TORQUE_ARM = {
    "name": "TA",
    "func": evaluate_torque_arm,
    "f": [],
    "g": [],
    "nvar": 7,
    "nf": 2,
    "nc": 1,
    "bounds": [(-2, 3.5),(-0.2, 2.5),(-2, 6),(-0.2, 0.5),(-0.1, 2),(-1.5, 2),(-0.1, 2)],
    "random_variables": [
        {
            "name": "X_load",
            "distribution": "normal",
            "mean": -2789.0,
            "std": 278.9
        },
        {
            "name": "Y_load",
            "distribution": "normal",
            "mean": 5066.0,
            "std": 506.6
        },
    ]
}
