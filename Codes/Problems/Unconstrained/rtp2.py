import numpy as np


# ============================================================
# RTP2
# ============================================================

def _rtp2(x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    nvar = x.shape[1]

    x1 = x[:, 0]
    x2 = x[:, 1]
    xi = x[:, 2:]

    h = (
        2.0
        - 1.2 / (1.0 + ((x2 - 0.3) / 0.01) ** 2)
        - 1.0 / (1.0 + ((x2 - 0.8) / 0.1) ** 2)
    )

    g = 1.0 + (9.0 / (nvar - 2)) * np.sum(xi, axis=1)

    f1 = x1
    f2 = h * g * (2.0 - x1 ** 2)

    F = np.column_stack([f1, f2])
    G = np.zeros((x.shape[0], 0))

    return F, G


def f1_rtp2(x):
    x = np.atleast_2d(np.asarray(x))
    return x[:, 0]


def f2_rtp2(x):
    x = np.atleast_2d(np.asarray(x))

    nvar = x.shape[1]

    x1 = x[:, 0]
    x2 = x[:, 1]
    xi = x[:, 2:]

    h = (
        2.0
        - 1.2 / (1.0 + ((x2 - 0.3) / 0.01) ** 2)
        - 1.0 / (1.0 + ((x2 - 0.8) / 0.1) ** 2)
    )

    g = 1.0 + (9.0 / (nvar - 2)) * np.sum(xi, axis=1)

    return h * g * (2.0 - x1 ** 2)


RTP2 = {
    "name": "RTP2",
    "func": _rtp2,
    "f": [f1_rtp2, f2_rtp2],
    "g": [],
    "nvar": 5,
    "nf": 2,
    "nc": 0,
    "bounds": [(0, 1)] * 5,
    "random_variables": []
}
