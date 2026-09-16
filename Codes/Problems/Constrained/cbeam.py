import numpy as np

# ==========================================================
# Beam constants
# ==========================================================
L = 100.0
D0 = 2.25

# ==========================================================
# Physics
# ==========================================================
def stress(w, t, X, Y):

    return (
        (600.0 * Y) / (w * t**2)
        +
        (600.0 * X) / (w**2 * t)
    )


def displacement(w, t, X, Y, E):

    return (
        (4.0 * L**3)
        /
        (E * w * t)
        *
        np.sqrt(
            (Y / t**2)**2
            +
            (X / w**2)**2
        )
    )


# ==========================================================
# Main evaluation
# ==========================================================
def _cantilever_beam_problem(x):

    x = np.atleast_2d(
        np.asarray(x, dtype=float)
    )

    w = x[:, 0]
    t = x[:, 1]

    X = x[:, 2]
    Y = x[:, 3]

    R = x[:, 4]
    E = x[:, 5]

    sigma = stress(
        w,
        t,
        X,
        Y
    )

    disp = displacement(
        w,
        t,
        X,
        Y,
        E
    )

    # ----------------------------
    # objectives
    # ----------------------------

    f1 = w * t

    f2 = sigma

    F = np.column_stack([
        f1,
        f2
    ])

    # ----------------------------
    # constraints
    # ----------------------------

    g1 = sigma - R

    g2 = disp - D0

    G = np.column_stack([
        g1,
        g2
    ])

    return F, G


# ==========================================================
# Objective helpers
# ==========================================================
def f1_area(x):

    x = np.atleast_2d(x)

    return x[:, 0] * x[:, 1]


def f2_stress(x):

    x = np.atleast_2d(x)

    w = x[:, 0]
    t = x[:, 1]

    X = x[:, 2]
    Y = x[:, 3]

    return stress(
        w,
        t,
        X,
        Y
    )


# ==========================================================
# Constraint helpers
# ==========================================================
def g1_strength(x):

    _, G = _cantilever_beam_problem(x)

    return G[:, 0]


def g2_displacement(x):

    _, G = _cantilever_beam_problem(x)

    return G[:, 1]


# ==========================================================
# RTP-style dictionary
# ==========================================================
# ==========================================================
# RTP-style dictionary
# ==========================================================
CANTILEVER_BEAM = {

    "name": "CBEAM",

    "func": _cantilever_beam_problem,

    "f": [
        f1_area,
        f2_stress
    ],

    "g": [
        g1_strength,
        g2_displacement
    ],

    "nvar": 2,

    "nf": 2,

    "nc": 2,

    "bounds": [
        (1.0, 5.0),     # w
        (1.0, 5.0),     # t
    ],

    # uncertainty variables
    "random_variables": [
        {
            "name": "X",
            "distribution": "normal",
            "mean": 500.0,
            "std": 100.0
        },
        {
            "name": "Y",
            "distribution": "normal",
            "mean": 1000.0,
            "std": 100.0
        },
        {
            "name": "R",
            "distribution": "normal",
            "mean": 40000.0,
            "std": 2000.0
        },
        {
            "name": "E",
            "distribution": "normal",
            "mean": 2.9e7,
            "std": 1.45e6
        }
    ],

}