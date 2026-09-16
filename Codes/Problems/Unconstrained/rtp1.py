import numpy as np
def _rtp1(x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    nvar = x.shape[1]

    x1 = x[:, 0]
    x2 = x[:, 1]
    xi = x[:, 2:]

    # numerical safety
    x1_safe = np.clip(x1, 0.0, 1.0)

    # h(x)
    h = (
        2.0
        - 1.2 / (1.0 + ((x2 - 0.3) / 0.01) ** 2)
        - 1.0 / (1.0 + ((x2 - 0.8) / 0.1) ** 2)
    )

    # g(x)   <-- row-wise
    g = 1.0 + (9.0 / (nvar - 2)) * np.sum(xi, axis=1)

    # objectives
    f1 = x1_safe
    f2 = h * g * (2.0 - np.sqrt(x1_safe))

    # pymoo expects (N,2)
    F = np.column_stack([f1, f2])
    G = np.zeros((x.shape[0], 0))

    return F, G

def f1_rtp1(x):
    x = np.atleast_2d(np.asarray(x))
    return x[:, 0]


def f2_rtp1(x):
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

    return h * g * (2.0 - np.sqrt(x1))


RTP1 = {
    "name": "RTP1",
    "func": _rtp1,
    "f": [f1_rtp1, f2_rtp1],
    "g": [],
    "nvar": 5,
    "nf": 2,
    "nc": 0,
    "bounds": [(0, 1)] * 5,
    "random_variables": []
}
