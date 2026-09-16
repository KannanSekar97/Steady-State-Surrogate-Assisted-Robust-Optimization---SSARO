import os
import json
import math
import warnings
import random
from pathlib import Path
from functools import reduce
import operator

import numpy as np
import pandas as pd
import sympy as sp
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from scipy.stats import qmc
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
# Import problems and evaluation functions

# ----------------------------------------------------
# Environment & warnings
# ----------------------------------------------------
#os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/Cellar/gcc/14.2.0_1/lib/gcc/14"
warnings.filterwarnings("ignore", category=UserWarning, module="pysr.sr")

ROUND_DECIMALS = 12
IDEAL_NADIR_PREC = 1e-6   # precision for ideal/nadir & F-quantization


def round_array(arr, decimals=ROUND_DECIMALS):
    """Round numpy array or scalar to the given decimals."""
    return np.round(np.asarray(arr, dtype=float), decimals)


def round_to_precision(arr, prec=IDEAL_NADIR_PREC):
    """
    Quantize array to a given precision, e.g. 1e-6.
    Used for F-values when computing/global ideal & nadir, and for scaling.
    """
    arr = np.asarray(arr, dtype=float)
    return np.round(arr / prec) * prec


def to_2d_rounded(x, decimals=ROUND_DECIMALS):
    """Ensure x is 2D float array and rounded."""
    xv = np.atleast_2d(x).astype(float)
    return np.round(xv, decimals)


# ----------------------------------------------------
# LHS (QMC) sampler
# ----------------------------------------------------
def sample_lhs_qmc(n_samples, dim, lb, ub):
    sampler = qmc.LatinHypercube(d=dim)
    U = sampler.random(n=n_samples)
    return qmc.scale(U, lb, ub)

def is_nondominated(F):
    """
    Standard (unconstrained) Pareto nondominance mask in objective space.
    F: (n_points, n_obj)
    """
    F = np.asarray(F, float)
    n = F.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[i]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dominated[i] = True
                break
    return ~dominated
# ====================================================
# SymPy / PySR helpers
# ====================================================

def sympy_to_pyomo(sym_expr, pyomo_model):
    """Convert a SymPy expression to a Pyomo expression."""
    if isinstance(sym_expr, (sp.Float, sp.Integer)) or sym_expr.is_Number:
        return float(sym_expr)

    if isinstance(sym_expr, sp.Symbol):
        name = str(sym_expr)
        if name.startswith('x'):
            idx = int(name[1:])
            return pyomo_model.x[idx]
        try:
            return float(name)
        except Exception:
            raise NotImplementedError(f"Unhandled sympy symbol '{name}'")

    if isinstance(sym_expr, sp.Add):
        return reduce(operator.add, [sympy_to_pyomo(a, pyomo_model) for a in sym_expr.args])
    if isinstance(sym_expr, sp.Mul):
        return reduce(operator.mul, [sympy_to_pyomo(a, pyomo_model) for a in sym_expr.args])
    if isinstance(sym_expr, sp.Pow):
        b, e = sym_expr.args
        return sympy_to_pyomo(b, pyomo_model) ** sympy_to_pyomo(e, pyomo_model)

    func, args = sym_expr.func, sym_expr.args
    if func == sp.sin:
        return pyo.sin(sympy_to_pyomo(args[0], pyomo_model))
    if func == sp.cos:
        return pyo.cos(sympy_to_pyomo(args[0], pyomo_model))
    if func == sp.exp:
        return pyo.exp(sympy_to_pyomo(args[0], pyomo_model))
    if func == sp.log:
        return pyo.log(sympy_to_pyomo(args[0], pyomo_model))
    if func == sp.sqrt:
        return pyo.sqrt(sympy_to_pyomo(args[0], pyomo_model))
    if func == sp.atan:
        return pyo.atan(sympy_to_pyomo(args[0], pyomo_model))
    if func == sp.atan2:
        y = sympy_to_pyomo(args[0], pyomo_model)
        x = sympy_to_pyomo(args[1], pyomo_model)
        return pyo.atan2(y, x) if hasattr(pyo, "atan2") else pyo.atan(y / (x + 1e-6))

    # last resort: numeric approximation
    try:
        return float(sp.N(sym_expr))
    except Exception:
        raise NotImplementedError(f"Cannot convert sympy expr: {sym_expr}")


def extract_sympy(model):
    """
    Lightweight SymPy extractor from PySR model.

    Now supports a custom chosen equation index (model._chosen_equation_index)
    selected via validation MSE across the nondominated (loss, complexity) front.
    """
    # If we manually selected an equation index, use that first
    if hasattr(model, "_chosen_equation_index"):
        idx = model._chosen_equation_index
        try:
            eqs = model.equations_
            row = eqs.iloc[int(idx)]
            for key in ("sympy_format", "equation"):
                try:
                    val = row.get(key, None)
                except AttributeError:
                    val = None
                if val is not None:
                    return sp.sympify(val)
        except Exception as e:
            print(f"[extract_sympy] Failed to use chosen equation index {idx}: {e}; falling back to get_best().")

    # Fallback: original behavior using get_best()
    best = model.get_best()

    for key in ("sympy_format", "equation"):
        try:
            val = best.get(key, None)
        except AttributeError:
            val = None
        if val is not None:
            return sp.sympify(val)

    if isinstance(best, str):
        return sp.sympify(best)

    raise RuntimeError(f"Could not extract sympy from best model: {best}")


def get_sympy_exprs_from_pysr_models(symbolic_models):
    """Convert a dict of trained PySR models to SymPy expressions."""
    return {k: extract_sympy(m) for k, m in symbolic_models.items()}


def add_classifier_constraint(sympy_exprs):
    """
    If a classifier 'clf_success' is present, add it as an extra constraint:
        g_new(x) = 0.5 - clf_success(x) <= 0
    """
    if "clf_success" not in sympy_exprs:
        return sympy_exprs

    clf_expr = sympy_exprs["clf_success"]
    existing_g = [k for k in sympy_exprs if k.startswith("g")]
    next_idx = len(existing_g) + 1
    new_g_key = f"g{next_idx}"
    sympy_exprs[new_g_key] = 0.5 - clf_expr
    print(f"[add_classifier_constraint] Added classifier-based constraint as '{new_g_key}'.")
    return sympy_exprs

# ====================================================
# PySR model training (regressors + classifier)
# ====================================================

def _train_single_pysr(X, y, iters=60, rng_seed=1, val_fraction=0.2):
    """
    Train a single PySRRegressor with an 80/20 train/validation split.

    Steps:
    - Filter to finite y.
    - Split into train/val (80/20 by default).
    - Fit PySR on the training subset.
    - From model.equations_, take ALL nondominated equations in
      (train loss, complexity) space.
    - Evaluate each of these on the validation set and compute validation MSE.
    - Select the equation with lowest validation MSE and store its index as
      `model._chosen_equation_index` so that extract_sympy() uses it.

    If a validation set is not available or something fails, we fall back to
    PySR's internal best-equation logic (model.get_best()).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # X must be 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # y must be 1D
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    elif y.ndim != 1:
        raise ValueError(
            f"[y must be 1D (single objective). "
            f"Got shape {y.shape}."
        )

    mask = np.isfinite(y)
    if mask.sum() < 3:
        print(f"[train_symbolic_models] Skipping training: only {mask.sum()} finite points.")
        return None

    X_clean = X[mask]
    y_clean = y[mask]
    n = X_clean.shape[0]

    rng_args = {} if rng_seed is None else {"random_state": rng_seed}

    model = PySRRegressor(
        niterations=iters,
        population_size=60,
        parsimony=0.001,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp"], #"log", "sqrt"
        optimizer_iterations=25,
        model_selection="best",
        verbosity=0,
        temp_equation_file=True,
        delete_tempfiles=True,
        output_directory=None,
        **rng_args,
    )
    
    model.fit(X_clean, y_clean)

    # If we have a validation set, perform nondominated selection + val MSE
    if X_clean is not None and X_clean.shape[0] > 0:
        try:
            eqs = model.equations_
        except Exception as e:
            eqs = None
            print(f"[train_symbolic_models]: could not access equations_ ({e}); skipping val-based selection.")

        if eqs is not None and len(eqs) > 0 and "loss" in eqs.columns and "complexity" in eqs.columns:
            try:
                # Non-dominated set in (loss, complexity) space (both to be minimized)
                metrics = eqs[["loss", "complexity"]].to_numpy(dtype=float)
                nd_mask = is_nondominated(metrics)
                nd_indices = np.where(nd_mask)[0]
                if nd_indices.size == 0:
                    nd_indices = np.arange(len(eqs))

                D = X_clean.shape[1]
                x_syms = sp.symbols(f"x0:{D}")

                best_idx = None
                best_mse = np.inf
                n_evaluated = 0

                for idx in nd_indices:
                    row = eqs.iloc[int(idx)]
                    expr_str = None
                    for key in ("sympy_format", "equation"):
                        try:
                            expr_str = row.get(key, None)
                        except AttributeError:
                            expr_str = None
                        if expr_str is not None:
                            break
                    if expr_str is None:
                        continue

                    try:
                        expr = sp.sympify(expr_str)
                        f_lam = sp.lambdify(x_syms, expr, modules=["numpy"])
                        # Evaluate on validation set
                        Xv = np.asarray(X_clean, dtype=float)
                        args = [Xv[:, j] for j in range(D)]
                        y_pred = f_lam(*args)
                        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
                        if y_pred.shape[0] != y_clean.shape[0]:
                            continue
                        mse = float(np.mean((y_pred - y_clean) ** 2))
                    except Exception:
                        continue

                    if not np.isfinite(mse):
                        continue

                    n_evaluated += 1
                    if mse < best_mse:
                        best_mse = mse
                        best_idx = int(idx)

                if best_idx is not None:
                    model._chosen_equation_index = best_idx
                    print(
                        f"[train_symbolic_models]: "
                        f"selected equation idx {best_idx} from {len(nd_indices)} ND models "
                        f"based on validation MSE = {best_mse:.4g} "
                    )
                else:
                    print(
                        f"[train_symbolic_models]: "
                        f"could not select a ND equation by validation MSE; "
                        f"falling back to PySR's internal best."
                    )
            except Exception as e:
                print(
                    f"[train_symbolic_models]: error during ND+val selection "
                    f"({e}); falling back to PySR's internal best."
                )
        else:
            print(
                f"[train_symbolic_models]: equations_ missing or without "
                f"'loss'/'complexity'; skipping ND+val selection."
            )

    return model