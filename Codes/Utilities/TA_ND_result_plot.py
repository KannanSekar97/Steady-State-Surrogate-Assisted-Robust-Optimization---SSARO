# Plot the stress plot of the knee point and the extreme solutions on the SSARO 95th percentile robust ND front

import math
import numpy as np
import matlab.engine
# Start MATLAB engine once here for the matlab-based torque arm problem
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\z5653370\OneDrive - UNSW\Documents\PhD\EMO\Codes\Torque arm\Torque_arm', nargout=0)
eng.addpath(r'C:\Users\z5653370\OneDrive - UNSW\Documents\PhD\EMO\Codes\Torque arm\DesignModule', nargout=0)
eng.addpath(r'C:\Users\z5653370\OneDrive - UNSW\Documents\PhD\EMO\Codes\Torque arm\calfem', nargout=0)
eng.addpath(r'C:\Users\z5653370\OneDrive - UNSW\Documents\PhD\EMO\Codes\Torque arm\calfem\fem', nargout=0)
eng.addpath(r'C:\Users\z5653370\OneDrive - UNSW\Documents\PhD\EMO\Codes\Torque arm\calfem\geom', nargout=0)
ROUND_DECIMALS = 12


def round_array(arr, decimals=ROUND_DECIMALS):
    return np.round(np.asarray(arr, dtype=float), decimals)


def to_2d_rounded(x, decimals=ROUND_DECIMALS):
    xv = np.atleast_2d(x).astype(float)
    return np.round(xv, decimals)

def evaluate_torque_arm(x):

    xv = to_2d_rounded(x)

    stot_mat = matlab.double(xv.tolist())

    N = len(xv)

    fx = np.full((N,1), -2789.0)
    fy = np.full((N,1), 5066.0)

    fx_mat = matlab.double(fx.tolist())
    fy_mat = matlab.double(fy.tolist())

    sreq, mass = eng.torque_arm(
        stot_mat,
        fx_mat,
        fy_mat,
        1,
        nargout=2
    )

    F = np.column_stack([
        np.asarray(sreq).flatten(),
        np.asarray(mass).flatten()
    ])

    # print(F)
    G = np.asarray(sreq).reshape(-1,1) - 800
    # print(G)
    return F, G



prob = "TA"
iter = 20
X_Dir = fr"Results\{prob}\Robust_ND_front_points\X_robust_nd_front_true_run_{iter}.npy"
F_Dir = fr"Results\{prob}\Robust_ND_front_points\robust_nd_front_true_run_{iter}.npy"
X_Rnd = np.load(X_Dir)
F_Rnd = np.load(F_Dir)
idx = np.argsort(F_Rnd[:,1]) #sort by mass
F_Rnd_sorted = F_Rnd[idx]
X_Rnd_sorted = X_Rnd[idx]

print(evaluate_torque_arm(X_Rnd_sorted[3,:]))
print(f"Robust_F: {F_Rnd_sorted[3,:]}")