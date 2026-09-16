import numpy as np

def safe_true_evaluate(eval_func, x, n_obj, n_con):
    """
    Robust wrapper around the 'true' evaluation function.

    Returns:
        F (1D, length n_obj),
        G (1D, length n_con, or length 0 if n_con==0),
        success (bool).
    On failure it returns NaNs in F/G (except unconstrained problems where G
    is zero-length) and success=False.
    """
    try:
        F_raw, G_raw = eval_func(x)
    except Exception as e:
        F = np.full((n_obj,), np.nan, dtype=float)
        G = np.full((n_con,), np.nan, dtype=float) if n_con > 0 else np.zeros((0,), dtype=float)
        print(f"[safe_true_evaluate] Exception during eval: {e}. Marking as failure.")
        return F, G, False

    F = np.atleast_1d(F_raw).astype(float).flatten()
    if F.size != n_obj:
        print(f"[safe_true_evaluate] F size mismatch ({F.size} != {n_obj}). Marking as failure.")
        F = np.full((n_obj,), np.nan, dtype=float)
        G = np.full((n_con,), np.nan, dtype=float) if n_con > 0 else np.zeros((0,), dtype=float)
        return F, G, False

    if n_con > 0:
        G = np.atleast_1d(G_raw).astype(float).flatten()
        if G.size != n_con:
            print(f"[safe_true_evaluate] G size mismatch ({G.size} != {n_con}). Marking as failure.")
            G = np.full((n_con,), np.nan, dtype=float)
            return F, G, False
    else:
        G = np.zeros((0,), dtype=float)

    finite_F = np.all(np.isfinite(F))
    finite_G = (G.size == 0) or np.all(np.isfinite(G))
    success = finite_F and finite_G

    if not success:
        print("[safe_true_evaluate] Non-finite F/G detected. Marking as failure.")
        if not finite_F:
            F[~np.isfinite(F)] = np.nan
        if G.size > 0 and not finite_G:
            G[~np.isfinite(G)] = np.nan

    return F, G, success

def safe_true_evaluate_batch(eval_func, X, n_obj, n_con):
    n = X.shape[0]

    F_all = np.full((n, n_obj), np.nan)
    G_all = np.full((n, n_con), np.nan) if n_con > 0 else np.zeros((n, 0))
    success = np.zeros((n, 1), dtype=int)

    for i, x in enumerate(X):
        F, G, s = safe_true_evaluate(eval_func, x, n_obj, n_con)
        F_all[i] = F
        if n_con > 0:
            G_all[i] = G
        success[i, 0] = int(s)

    return F_all, G_all, success
    
    