"""
Robust evaluation of the final nondominated front of one optimization run.

For a given run, this module:

1. Loads the final optimization archive (``final_archives.json``).
2. Trains PySR surrogate models on the complete archive, or loads a
   previously saved evaluator bundle if one already exists.
3. Uses the fitted surrogate evaluator to predict the robust objective and
   constraint values of all archived designs.
4. Extracts the predicted robust feasible nondominated front.
5. Evaluates those designs using the true robust evaluator.
6. Extracts the true robust feasible nondominated front (the "SSARO 95th
   percentile" front, since alpha=0.95 defines the robustness level).
7. Saves fronts/design variables (.npy) and an evaluator bundle (.joblib).

``compute_robust_nd_front`` and ``plot_ssaro_95th_front`` are the public
entry points used by ``run_experiment.py`` right after an optimization run
finishes. Running this file directly reprocesses every existing run folder
for a chosen problem (useful for regenerating fronts without repeating the
optimization).
"""

import sys
import json
import platform
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error

from Utilities.ND import nondominated
from Utilities.Problem import MyProblem
from Utilities.Problem_Robust_True import MyProblem_Robust
from Utilities.Load_Json import load_json
from Utilities.pySR_Model import _train_single_pysr
from Utilities.Concept_Evaluator import ConceptEvaluator

# Robust-evaluation defaults (match Optimization.py's defaults).
DEFAULT_DEL_X = 0.01
DEFAULT_ALPHA = 0.95
DEFAULT_SAMPLES_SCALAR = 100
DEFAULT_N_MODELS = 5


# =========================================================================
# Helpers
# =========================================================================

def ensure_2d(array, name):
    """Convert an array to a two-dimensional NumPy array."""
    array = np.asarray(array, dtype=float)

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Received shape {array.shape}.")

    return array


def load_archive(problem_dir: Path, run_id: int):
    """Load and validate the final archive for one run."""
    archive_path = problem_dir / f"Run_{run_id:02d}" / "final_archives.json"

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive file does not exist: {archive_path}")

    raw_archive = load_json(str(archive_path))

    required_keys = {"X", "F", "G"}
    missing_keys = required_keys.difference(raw_archive.keys())
    if missing_keys:
        raise KeyError(f"Archive {archive_path} is missing keys: {sorted(missing_keys)}")

    X = ensure_2d(np.vstack(raw_archive["X"]), "X")
    F = ensure_2d(np.vstack(raw_archive["F"]), "F")
    G = ensure_2d(np.vstack(raw_archive["G"]), "G")

    if not (X.shape[0] == F.shape[0] == G.shape[0]):
        raise ValueError(
            "Archive arrays have inconsistent numbers of rows: "
            f"X={X.shape[0]}, F={F.shape[0]}, G={G.shape[0]}."
        )

    return {"X": X, "F": F, "G": G}


def train_best_pysr_model_full_data(X, y, n_models=DEFAULT_N_MODELS):
    """Train several PySR models and retain the one with the smallest RMSE."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional. Received shape {X.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y contain different numbers of samples: {X.shape[0]} and {y.shape[0]}."
        )
    if X.shape[0] == 0:
        raise ValueError("Cannot train a model using an empty dataset.")
    if n_models < 1:
        raise ValueError("n_models must be at least 1.")

    best_model = None
    best_error = np.inf

    for model_index in range(n_models):
        print(f"  Training candidate model {model_index + 1}/{n_models}")

        model = _train_single_pysr(X, y)
        y_pred = model.predict(X)

        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        if y_pred.shape[0] != y.shape[0]:
            raise ValueError(
                "The PySR model returned an unexpected number of predictions: "
                f"expected {y.shape[0]}, received {y_pred.shape[0]}."
            )

        if not np.all(np.isfinite(y_pred)):
            print(f"  Candidate {model_index + 1} produced non-finite predictions and will be ignored.")
            continue

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print(f"  Candidate {model_index + 1}/{n_models}: training RMSE = {rmse:.6e}")

        if rmse < best_error:
            best_error = rmse
            best_model = model

    if best_model is None:
        raise RuntimeError("All PySR model-training attempts failed or produced non-finite predictions.")

    return best_model, best_error


def build_evaluator_model(archive, concept, n_models=DEFAULT_N_MODELS):
    """Train objective and constraint PySR models using all valid archive rows."""
    nf = int(concept["nf"])
    nc = int(concept["nc"])

    X_all = ensure_2d(archive["X"], "archive X")
    F_all = ensure_2d(archive["F"], "archive F")

    if F_all.shape[1] != nf:
        raise ValueError(f"Expected {nf} objective columns, but archive F has {F_all.shape[1]} columns.")

    if nc > 0:
        G_all = ensure_2d(archive["G"], "archive G")
        if G_all.shape[1] != nc:
            raise ValueError(f"Expected {nc} constraint columns, but archive G has {G_all.shape[1]} columns.")
        targets = np.column_stack([F_all, G_all])
    else:
        G_all = None
        targets = F_all

    finite_mask = np.isfinite(X_all).all(axis=1) & np.isfinite(targets).all(axis=1)

    X_valid = X_all[finite_mask]
    F_valid = F_all[finite_mask]
    G_valid = G_all[finite_mask] if nc > 0 else None

    print(f"Valid training samples: {X_valid.shape[0]} of {X_all.shape[0]}")

    if X_valid.shape[0] == 0:
        raise ValueError("No finite archive samples are available for model training.")

    pysr_objectives, pysr_constraints = [], []
    objective_errors, constraint_errors = [], []

    for objective_index in range(nf):
        print(f"\nTraining objective model f{objective_index + 1}")
        best_model, best_error = train_best_pysr_model_full_data(
            X=X_valid, y=F_valid[:, objective_index], n_models=n_models
        )
        pysr_objectives.append(best_model)
        objective_errors.append(best_error)
        print(f"Best f{objective_index + 1} training RMSE = {best_error:.6e}")

    for constraint_index in range(nc):
        print(f"\nTraining constraint model g{constraint_index + 1}")
        best_model, best_error = train_best_pysr_model_full_data(
            X=X_valid, y=G_valid[:, constraint_index], n_models=n_models
        )
        pysr_constraints.append(best_model)
        constraint_errors.append(best_error)
        print(f"Best g{constraint_index + 1} training RMSE = {best_error:.6e}")

    evaluator = ConceptEvaluator(
        concept=concept, nf=nf, nc=nc, PySR_obj=pysr_objectives, PySR_con=pysr_constraints
    )
    evaluator.obj_errors = objective_errors
    evaluator.con_errors = constraint_errors

    return evaluator


def get_model_path(model_dir: Path, run_id: int):
    return model_dir / f"evaluator_run_{run_id:02d}.joblib"


def save_evaluator_bundle(evaluator, concept, run_id, model_path, del_x, alpha, nsamples_scalar, n_models):
    bundle = {
        "run": int(run_id),
        "problem_name": concept["name"],
        "evaluator": evaluator,
        "concept": concept,
        "del_x": float(del_x),
        "alpha": float(alpha),
        "nsamples_scalar": int(nsamples_scalar),
        "n_models": int(n_models),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    joblib.dump(bundle, model_path)
    print(f"Saved evaluator bundle to: {model_path}")


def load_evaluator_bundle(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Saved evaluator does not exist: {model_path}")

    bundle = joblib.load(model_path)

    required_keys = {"evaluator", "concept", "del_x", "alpha", "nsamples_scalar"}
    missing_keys = required_keys.difference(bundle.keys())
    if missing_keys:
        raise KeyError(f"Saved evaluator bundle is missing keys: {sorted(missing_keys)}")

    print(f"Loaded evaluator bundle from: {model_path}")
    return bundle


def get_or_train_evaluator(
    archive, concept, run_id, model_dir: Path,
    del_x=DEFAULT_DEL_X, alpha=DEFAULT_ALPHA, nsamples_scalar=DEFAULT_SAMPLES_SCALAR,
    n_models=DEFAULT_N_MODELS, load_saved_models=True,
):
    """Load a saved evaluator if available and requested; otherwise train and save it."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = get_model_path(model_dir, run_id)

    if load_saved_models and model_path.exists():
        bundle = load_evaluator_bundle(model_path)
        evaluator = bundle["evaluator"]
        parameters = {
            "del_x": bundle["del_x"],
            "alpha": bundle["alpha"],
            "nsamples_scalar": bundle["nsamples_scalar"],
        }
        return evaluator, parameters

    print(f"Training new evaluator for run {run_id:02d}")

    evaluator = build_evaluator_model(archive=archive, concept=concept, n_models=n_models)

    save_evaluator_bundle(
        evaluator=evaluator, concept=concept, run_id=run_id, model_path=model_path,
        del_x=del_x, alpha=alpha, nsamples_scalar=nsamples_scalar, n_models=n_models,
    )

    parameters = {"del_x": del_x, "alpha": alpha, "nsamples_scalar": nsamples_scalar}
    return evaluator, parameters


def feasible_nondominated(F, G, X=None):
    """Extract feasible nondominated points, mapping indices back to the original arrays."""
    F = ensure_2d(F, "F")
    G = ensure_2d(G, "G")

    if F.shape[0] != G.shape[0]:
        raise ValueError(f"F and G row counts differ: {F.shape[0]} and {G.shape[0]}.")

    if X is not None:
        X = ensure_2d(X, "X")
        if X.shape[0] != F.shape[0]:
            raise ValueError(f"X and F row counts differ: {X.shape[0]} and {F.shape[0]}.")

    finite_mask = np.isfinite(F).all(axis=1) & np.isfinite(G).all(axis=1)
    feasible_mask = finite_mask & np.all(G <= 0.0, axis=1)
    feasible_indices = np.flatnonzero(feasible_mask)

    if feasible_indices.size == 0:
        empty_F = np.empty((0, F.shape[1]), dtype=float)
        empty_X = None if X is None else np.empty((0, X.shape[1]), dtype=float)
        return empty_F, empty_X, np.array([], dtype=int)

    F_feasible = F[feasible_indices]
    F_nd, local_nd_indices = nondominated(F_feasible)
    local_nd_indices = np.asarray(local_nd_indices, dtype=int).ravel()
    global_nd_indices = feasible_indices[local_nd_indices]

    X_nd = None if X is None else X[global_nd_indices]

    return F_nd, X_nd, global_nd_indices


def evaluate_predicted_robust_front(evaluator, archive, robust_parameters, concept):
    """Use the archive-trained evaluator to obtain the predicted robust front."""
    archive_problem = MyProblem(
        evaluator, concept,
        robust_parameters["del_x"], robust_parameters["alpha"], robust_parameters["nsamples_scalar"],
    )

    output = {}
    archive_problem._evaluate(archive["X"], output)

    if "F" not in output or "G" not in output:
        raise KeyError("MyProblem evaluation did not return both F and G.")

    F_predicted = ensure_2d(output["F"], "predicted robust F")
    G_predicted = ensure_2d(output["G"], "predicted robust G")

    F_nd_predicted, X_nd_predicted, predicted_indices = feasible_nondominated(
        F=F_predicted, G=G_predicted, X=archive["X"],
    )

    return {
        "F_all": F_predicted, "G_all": G_predicted,
        "F_nd": F_nd_predicted, "X_nd": X_nd_predicted, "indices": predicted_indices,
    }


def evaluate_true_robust_front(X_candidates, robust_parameters, concept):
    """Evaluate predicted robust ND designs with the true robust evaluator."""
    X_candidates = ensure_2d(X_candidates, "true robust candidate X")

    if X_candidates.shape[0] == 0:
        return {
            "F_all": np.empty((0, int(concept["nf"]))),
            "G_all": np.empty((0, int(concept["nc"]))),
            "F_nd": np.empty((0, int(concept["nf"]))),
            "X_nd": np.empty((0, int(concept["nvar"]))),
            "indices": np.array([], dtype=int),
        }

    true_problem = MyProblem_Robust(
        concept, robust_parameters["del_x"], robust_parameters["alpha"], robust_parameters["nsamples_scalar"],
    )

    output = {}
    true_problem._evaluate(X_candidates, output)

    if "F" not in output or "G" not in output:
        raise KeyError("MyProblem_Robust evaluation did not return both F and G.")

    F_true = ensure_2d(output["F"], "true robust F")
    G_true = ensure_2d(output["G"], "true robust G")

    F_nd_true, X_nd_true, true_indices = feasible_nondominated(F=F_true, G=G_true, X=X_candidates)

    return {
        "F_all": F_true, "G_all": G_true,
        "F_nd": F_nd_true, "X_nd": X_nd_true, "indices": true_indices,
    }


def get_true_base_front(archive):
    """Extract the feasible nondominated front from the original true archive."""
    F_nd_base, X_nd_base, base_indices = feasible_nondominated(
        F=archive["F"], G=archive["G"], X=archive["X"],
    )
    return {"F_nd": F_nd_base, "X_nd": X_nd_base, "indices": base_indices}


def save_run_results(run_id, base_result, predicted_result, true_result, front_dir: Path):
    """Save objective fronts and decision vectors for one run."""
    front_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"run_{run_id:02d}"

    np.save(front_dir / f"true_nd_base_{prefix}.npy", base_result["F_nd"])

    np.save(front_dir / f"robust_nd_front_predicted_{prefix}.npy", predicted_result["F_nd"])
    np.save(front_dir / f"X_robust_nd_front_predicted_{prefix}.npy", predicted_result["X_nd"])

    np.save(front_dir / f"robust_nd_front_true_{prefix}.npy", true_result["F_nd"])
    np.save(front_dir / f"X_robust_nd_front_true_{prefix}.npy", true_result["X_nd"])

    print(f"Saved numerical results for run {run_id:02d}")


def plot_ssaro_95th_front(run_id, base_result, predicted_result, true_result, plot_dir: Path):
    """
    Plot the base archive front, the predicted robust (alpha=0.95) front and
    the true robust (alpha=0.95, i.e. "SSARO 95th percentile") front.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    if base_result["F_nd"].shape[0] > 0:
        plt.scatter(
            base_result["F_nd"][:, 0], base_result["F_nd"][:, 1],
            c="green", marker="s", label="True ND Front",
        )

    if predicted_result["F_nd"].shape[0] > 0:
        plt.scatter(
            predicted_result["F_nd"][:, 0], predicted_result["F_nd"][:, 1],
            c="blue", marker="o", label="Predicted 95th Percentile Robust ND Front",
        )

    if true_result["F_nd"].shape[0] > 0:
        plt.scatter(
            true_result["F_nd"][:, 0], true_result["F_nd"][:, 1],
            c="red", marker="x", label="True 95th Percentile Robust ND Front",
        )

    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.title(f"Run {run_id:02d}: SSARO 95th Percentile Robust Nondominated Front")
    plt.grid(alpha=0.25)
    plt.legend()

    save_path = plot_dir / f"ssaro_95th_front_run_{run_id:02d}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {save_path}")
    return save_path


# =========================================================================
# Public entry point
# =========================================================================

def compute_robust_nd_front(
    concept, run_id, problem_dir,
    del_x=DEFAULT_DEL_X, alpha=DEFAULT_ALPHA, nsamples_scalar=DEFAULT_SAMPLES_SCALAR,
    n_models=DEFAULT_N_MODELS, load_saved_models=True,
):
    """
    Compute (and save) the robust nondominated front for one finished
    optimization run.

    Parameters
    ----------
    concept : dict
        Problem definition (as used by Optimization.main).
    run_id : int
        Run index; ``problem_dir / f"Run_{run_id:02d}" / "final_archives.json"``
        must already exist (produced by Optimization.main).
    problem_dir : str or Path
        Root results directory for this problem, e.g. ``Results/RTP1``.

    Returns
    -------
    dict with keys "base", "predicted", "true", "front_dir", "plot_dir",
    "model_dir".
    """
    problem_dir = Path(problem_dir)
    model_dir = problem_dir / "Saved_Evaluator_Models"
    front_dir = problem_dir / "Robust_ND_front_points"
    plot_dir = problem_dir / "Robust_ND_front"

    archive = load_archive(problem_dir, run_id)
    print(f"Archive shapes: X={archive['X'].shape}, F={archive['F'].shape}, G={archive['G'].shape}")

    base_result = get_true_base_front(archive)
    print(f"True archive feasible ND points: {base_result['F_nd'].shape[0]}")

    evaluator, robust_parameters = get_or_train_evaluator(
        archive=archive, concept=concept, run_id=run_id, model_dir=model_dir,
        del_x=del_x, alpha=alpha, nsamples_scalar=nsamples_scalar,
        n_models=n_models, load_saved_models=load_saved_models,
    )

    predicted_result = evaluate_predicted_robust_front(
        evaluator=evaluator, archive=archive, robust_parameters=robust_parameters, concept=concept,
    )
    print(f"Predicted robust feasible ND points: {predicted_result['F_nd'].shape[0]}")

    true_result = evaluate_true_robust_front(
        X_candidates=predicted_result["X_nd"], robust_parameters=robust_parameters, concept=concept,
    )
    print(f"True robust feasible ND points: {true_result['F_nd'].shape[0]}")

    save_run_results(run_id, base_result, predicted_result, true_result, front_dir)

    return {
        "base": base_result,
        "predicted": predicted_result,
        "true": true_result,
        "front_dir": front_dir,
        "plot_dir": plot_dir,
        "model_dir": model_dir,
    }


# =========================================================================
# Standalone batch reprocessing (does not repeat the optimization)
# =========================================================================

def _discover_run_ids(problem_dir: Path):
    run_ids = []
    for entry in sorted(problem_dir.glob("Run_*")):
        if entry.is_dir() and (entry / "final_archives.json").exists():
            try:
                run_ids.append(int(entry.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(run_ids)


def main():
    """Reprocess every existing run folder for one problem."""
    import argparse
    from run_experiment import PROBLEMS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="RTP1", help="Problem key from run_experiment.PROBLEMS")
    parser.add_argument("--result-root", default="Results")
    parser.add_argument("--load-saved-models", action="store_true", default=True)
    args = parser.parse_args()

    problem_registry = dict(PROBLEMS)
    if args.problem not in problem_registry:
        raise SystemExit(f"Unknown problem '{args.problem}'. Options: {sorted(problem_registry)}")

    concept = problem_registry[args.problem]
    problem_dir = Path(args.result_root) / args.problem

    run_ids = _discover_run_ids(problem_dir)
    if not run_ids:
        raise SystemExit(f"No completed runs found beneath {problem_dir}")

    print(f"Reprocessing {len(run_ids)} run(s) for {args.problem}: {run_ids}")

    for run_id in run_ids:
        print("\n" + "=" * 72)
        print(f"Processing run {run_id:02d}")
        print("=" * 72)
        try:
            result = compute_robust_nd_front(
                concept=concept, run_id=run_id, problem_dir=problem_dir,
                load_saved_models=args.load_saved_models,
            )
            plot_ssaro_95th_front(
                run_id=run_id, base_result=result["base"], predicted_result=result["predicted"],
                true_result=result["true"], plot_dir=result["plot_dir"],
            )
        except Exception as error:
            print(f"Run {run_id:02d} failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
