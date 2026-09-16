import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import erf

# import optuna
import optuna
import warnings
from sklearn.exceptions import ConvergenceWarning
import optuna
import logging
warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

#Pymoo packages
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
import matplotlib.pyplot as plt

#Import all the required uitility functions
from Utilities.ND import nondominated
from Utilities.Generate_Samples import generate_samples
from Utilities.Concept_Evaluator import ConceptEvaluator
from Utilities.pySR_Model import _train_single_pysr
from Utilities.True_Evaluation import safe_true_evaluate, safe_true_evaluate_batch
from Utilities.Problem import MyProblem
from Utilities.Check_Candidate import check_candidate
from Utilities.ND import nondominated
from Utilities.Json_saver import ensure_dir, save_json
from Utilities.Load_Json import load_json

ROUND_DECIMALS = 12
tol = 1e-6
IDEAL_NADIR_PREC = 1e-6 

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
# ============================================================
# Build evaluator model
# ============================================================
def build_evaluator_model(archive, concept):
    nf, nc = concept["nf"], concept["nc"]
    # Extract archive data
    X_all = archive["X"]
    F_all = archive["F"]
    G_all = archive["G"] if nc > 0 else None
    # Filter valid rows
    if nc == 0:
        targets = F_all
    else:
        targets = np.column_stack([F_all, G_all])
    mask = ~np.isnan(targets).any(axis=1)
    X_valid = X_all[mask]
    F_valid = F_all[mask]
    G_valid = G_all[mask] if nc > 0 else None
    mode  = "SA"
    if mode == "SA":
        #Objectives
        PySR_obj = []
        for i in range(nf):
            PySR_obj.append(_train_single_pysr(X_valid, F_valid[:, i]))
        #Constraints
        PySR_con=[]
        for j in range(nc):
            PySR_con.append(_train_single_pysr(X_valid, G_valid[:, j]))
    else:
        # TRUE mode → no surrogates
        PySR_obj = [None] * nf
        PySR_con = [None] * nc
    # ================================================================
    # 2. Build evaluator
    # ================================================================
    evaluator = ConceptEvaluator(
                    concept=concept,
                    nf=nf,
                    nc=nc,
                    PySR_obj=PySR_obj,
                    PySR_con=PySR_con,
                    )

    return evaluator

def Run_optimization(evaluator, pop_size, generations, concept, del_x, alpha, nsamples_scalar, crossover_prob, crossover_eta,mutation_prob, mutation_eta):
    # Setup crossover and mutation
    sbx = SBX(prob=crossover_prob, eta=crossover_eta)
    pm  = PolynomialMutation(prob=mutation_prob, eta=mutation_eta)

    # Setup NSGA2 and Problem
    algorithm = NSGA2(pop_size=pop_size, crossover=sbx, mutation=pm,eliminate_duplicates=True)
    problem = MyProblem(evaluator, concept, del_x=del_x, alpha=alpha, nsamples_scalar=nsamples_scalar)

    res_true = minimize(
        problem,
        algorithm,
        termination=("n_gen", generations),
        seed=1,
        verbose=False
    )

    X = res_true.X
    F = res_true.F
    G = res_true.G

    return X, F, G

def sample_random_variables(concept, n_sample):
    """
    Samples random variables defined in concept["random_variables"].
    """
    random_variables = concept.get("random_variables", [])
    if len(random_variables) == 0:
        return None
    samples = []
    for rv in random_variables:
        dist = rv["distribution"].lower()
        if dist == "normal":
            values = np.random.normal(
                loc=rv["mean"],
                scale=rv["std"],
                size=n_sample
            )
        elif dist == "uniform":
            values = np.random.uniform(
                low=rv["lower"],
                high=rv["upper"],
                size=n_sample
            )
        else:
            raise ValueError(
                f"Unsupported distribution '{rv['distribution']}' "
                f"for random variable '{rv['name']}'"
            )
        samples.append(values)
    return np.column_stack(samples)

def initial_sampling(concept, archive, run_id, base_dir):
    initial_results = {}
    json_path = os.path.join(
        base_dir,
        "initial_solutions.json"
    )
    random_variables = concept.get("random_variables", [])
    n_rand = len(random_variables)
    # Number of initial samples
    if n_rand == 0:
        n_sample = concept["nvar"] * 11 - 1
    else:
        n_sample = (concept["nvar"] + n_rand) * 11 - 1

    # Design-variable sampling
    X_design = generate_samples(run_id, concept["bounds"], n_sample)

    # Random-variable sampling, if available
    X_random = sample_random_variables(concept, n_sample)
    if X_random is None:
        X_initial = X_design
    else:
        X_initial = np.column_stack([ X_design, X_random])

    # Evaluation
    F, G, success = safe_true_evaluate_batch(concept["func"], X_initial, concept["nf"], concept["nc"])
  
    # Save archive
    archive["X"] = np.vstack([archive["X"], round_array(X_initial)])
    archive["F"] = np.vstack([archive["F"], round_array(F)])
    archive["G"] = np.vstack([ archive["G"],round_array(G)])

    # Save initial results
    initial_results = {
        "concept_name": concept["name"],
        "initial_X": X_initial.tolist(),
        "initial_F": F.tolist(),
        "initial_G": G.tolist(),
    }
    save_json(initial_results,json_path)
    return n_sample

#Robust nondominated front of archives (95th delta perturbed point)
def archives_nd_front(evaluator, archive, concept, del_x, alpha, nsamples_scalar, k_dom):
    """
    Robust nondominated front of archive solutions.

    Handles:
        - design-variable-only problems
        - design + random-variable problems
        - constrained and unconstrained problems
    """
    nf = concept["nf"]
    nc = concept["nc"]
    nvar = concept["nvar"]
    X_archive = archive["X"]
    # Use only design variables for pymoo decision variables
    X_design = X_archive[:, :nvar]
    Archive_evaluator = MyProblem(evaluator=evaluator, concept=concept, del_x=del_x, alpha = alpha, nsamples_scalar=nsamples_scalar)
    out = {}
    Archive_evaluator._evaluate(X_design, out)
    archive_F = out["F"]
    if nc > 0:
        archive_G = out["G"]
    else:
        archive_G = np.empty((archive_F.shape[0], 0))
    # Remove NaN rows
    if archive_F.size > 0:
        valid_mask = ~np.isnan(archive_F).any(axis=1)
        if archive_G.size > 0:
            valid_mask &= ~np.isnan(archive_G).any(axis=1)
        archive_F = round_to_precision(archive_F[valid_mask])
        archive_G = archive_G[valid_mask]
    else:
        return np.empty((0, nf)), np.array([], dtype=bool)
    # ==================================================
    # Constrained case
    # ==================================================
    if nc > 0:
        if archive_G.size == 0:
            print("⚠️ No constraint data available yet. Skipping ND sorting.")
            return np.empty((0, nf)), np.array([], dtype=bool)
        feasible_mask = np.all(
            archive_G <= tol,
            axis=1
        )
        feasible_F = archive_F[feasible_mask]
        feasible_F = feasible_F[~np.isnan(feasible_F).any(axis=1)]
        if feasible_F.shape[0] < 2:
            print("⚠️ Not enough feasible points for ND sorting.")
            True_nd_front = feasible_F
        else:
            True_nd_front, _ = nondominated(feasible_F)
    # ==================================================
    # Unconstrained case
    # ==================================================
    else:
        feasible_mask = np.ones(archive_F.shape[0], dtype=bool)
        if archive_F.shape[0] < 2:
            print("⚠️ Not enough points for ND sorting.")
            True_nd_front = archive_F
        else:
            True_nd_front, _ = nondominated(archive_F)

    return round_to_precision(True_nd_front), feasible_mask

def p_obj_a_b_vec(muA, sigmaA, muB, sigmaB):
    """Probability that A dominates B per objective (independent)."""
    p_each = 0.5 + 0.5 * erf((muB - muA) / np.sqrt(2 * (sigmaA**2 + sigmaB**2)))
    return np.prod(p_each)


def select_candidate(
    True_nd_front: np.ndarray,
    cand_F_mean: np.ndarray,
    cand_G_mean: np.ndarray,
    method: str = "ED",
    nc: int = 0,
    nf: int = 2,
    n_cand_per_run: int = 1,
    atol: float = 1e-8,
    eps: float = 1e-12,
):
    candidate = np.asarray(cand_F_mean)

    # Feasibility Handling
    if nc > 0:
        idx_cand_feas = np.where((cand_G_mean <= 0).all(axis=1))[0]
        candidate_nd_front, candidate_nds = nondominated(candidate[idx_cand_feas])
        candidate_nd_idx = idx_cand_feas[candidate_nds]
    else:
        candidate_nd_front, candidate_nds = nondominated(candidate)
        candidate_nd_idx = candidate_nds

    combined_ndf = np.vstack([True_nd_front, candidate_nd_front])
    combined_nd_front, _ = nondominated(combined_ndf)

    # Setup
    fmin = np.min(True_nd_front, axis=0)
    fmax = np.max(True_nd_front, axis=0)
    rng = np.where(np.abs(fmax - fmin) < eps, 1.0, fmax - fmin)

    candidate_nd_front = round_to_precision(np.atleast_2d(candidate_nd_front))
    combined_nd_front =round_to_precision(np.atleast_2d(combined_nd_front))

    # ---------------- Membership check ----------------
    mask = np.isclose(
        candidate_nd_front[:, None, :],
        combined_nd_front[None, :, :],
        atol=atol,
    ).all(-1).any(axis=1)

    mu_sel_idx = candidate_nd_idx[mask]
    cand_dom_idx = candidate_nd_idx[~mask]

    mu_sel = round_to_precision(candidate_nd_front[mask])
    cand_dom = round_to_precision(candidate_nd_front[~mask])
    true_points = round_to_precision(True_nd_front)
    
    # ---------------- Selection Method ----------------
    m = method.lower()
    if m == "ed":
        true_norm = (true_points - fmin) / rng

        if len(mu_sel) > 0:
            cand_norm = (mu_sel - fmin) / rng
            dists = cdist(cand_norm, true_norm)
            min_dists = np.min(dists, axis=1)
            best_local = np.argsort(-min_dists)
            best_idx = mu_sel_idx[best_local]

        else:
            cand_dom_norm = (cand_dom - fmin) / rng
            dists_dom = cdist(cand_dom_norm, true_norm)
            min_dists_dom = np.min(dists_dom, axis=1)
            best_local = np.argsort(min_dists_dom)
            best_idx = cand_dom_idx[best_local]
    # Return indices relative to cand_mean
    return best_idx[:n_cand_per_run]

# ============================================================
# Plotting
# ============================================================
def plot_nd_front(F_nd, base_dir):
    
    if F_nd is None or len(F_nd) == 0:
        print("⚠️ No ND solutions to plot.")
        return

    plt.figure()

    if F_nd.shape[1] == 1:
        plt.scatter(range(len(F_nd)), F_nd[:, 0], c="r")
        plt.ylabel("Objective")
        plt.xlabel("Solution index")

    elif F_nd.shape[1] == 2:
        plt.scatter(F_nd[:, 0], F_nd[:, 1], c="r")
        plt.xlabel("f1")
        plt.ylabel("f2")

    else:
        print("⚠️ Plotting only supported for nf ≤ 2")

    plt.title("Final Non-Dominated Front")
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(base_dir, "final_nd_front.png")
    plt.savefig(path, dpi=300)
    plt.show()

    print(f"📈 ND front plot saved to {path}")

def evaluate_and_update(concept, archive, x, iteration):
    nf, nc = concept["nf"], concept["nc"]
    F, G, success = safe_true_evaluate(
        concept["func"], x, nf, nc
    )
    archive["X"] = np.vstack([archive["X"], round_array(x)])
    archive["F"] = np.vstack([archive["F"], round_array(F)])
    archive["G"] = np.vstack([archive["G"], round_array(G)])
    print(f"Iteration {iteration}: X: {x}, F: {F}, G: {G}")
    return F, G


import numpy as np

def x_updated_with_random_variables(x, concept):
    x = np.asarray(x, dtype=float).reshape(1, -1)
    random_variables = concept.get("random_variables", [])
    # No random variables
    if len(random_variables) == 0:
        return x
    uncertainty = []
    for rv in random_variables:
        dist = rv["distribution"].lower()
        if dist == "normal":
            value = np.random.normal(rv["mean"],rv["std"])
        elif dist == "uniform":
            value = np.random.uniform(rv["lower"],rv["upper"])
        else:
            raise ValueError(
                f"Unsupported distribution '{rv['distribution']}' "
                f"for random variable '{rv['name']}'"
            )
        uncertainty.append(value)
    uncertainty = np.asarray(uncertainty).reshape(1, -1)
    X_updated = np.hstack([x, uncertainty])
    return X_updated


def main(concept, base_dir, **params):
    nvar = concept['nvar']
    random_variables = concept.get("random_variables", [])
    nrandvar = len(random_variables)
    total_var = nvar + nrandvar
    nf = concept['nf']
    nc = concept['nc']
    crossover_prob = params.get('crossover_prob', 0.95)
    crossover_eta = params.get('crossover_eta', 20)
    mutation_prob = 1 / nvar
    mutation_eta = params.get('mutation_eta', 20)
    generations = params.get('generations', 100)
    pop_size = params.get('pop_size', 100)
    budget = min(params.get('budget_scale_params', 100)*nvar, 500)
    n_cand_per_run = params.get('n_cand_per_run', 1)
    run_id = params.get('run_id', 1)
    del_x = params.get('del_x', 0.05)
    alpha = params.get('alpha', 0.95)
    nsamples_scalar = params.get('nsamples_scalar', 100)
    k_dom = params.get('k_dom', 50)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    archive = {
        "X": np.empty((0, total_var)),
        "F": np.empty((0, concept["nf"])),
        "G": np.empty((0, concept["nc"])),
        "LB": np.array([b[0] for b in concept["bounds"]]),
        "UB": np.array([b[1] for b in concept["bounds"]]),
    }

    # ==========================================================
    # Resume Logic using initial_solutions.json + iteration_*.json
    # ==========================================================
    initial_file = os.path.join(base_dir, "initial_solutions.json")

    iter_files = sorted(
        [
            f for f in os.listdir(base_dir)
            if f.startswith("iteration_") and f.endswith(".json")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    if os.path.exists(initial_file):
        print("Loading previous optimization state...")

        # Load initial DOE
        data = load_json(initial_file)

        archive["X"] = np.array(data["initial_X"])
        archive["F"] = np.array(data["initial_F"])
        archive["G"] = np.array(data["initial_G"])

        # Load all evaluated iterations
        for file in iter_files:
            loaded = load_json(os.path.join(base_dir, file))
            Inter_solutions = loaded.get("newly_eval", [])
            for d in Inter_solutions:
                x = np.array(d["X"]).reshape(1, -1)
                f = np.array(d["F"]).reshape(1, -1)
                g = np.array(d["G"]).reshape(1, -1)

                archive["X"] = np.vstack([archive["X"], round_array(x)])
                archive["F"] = np.vstack([archive["F"], round_array(f)])
                archive["G"] = np.vstack([archive["G"], round_array(g)])

        nf_eval = len(archive["X"])
        iter = len(iter_files) + 1
        print(f"Resuming from iteration {iter}")
        print(f"Evaluations: {nf_eval}/{budget}")

    else:
        nf_eval = initial_sampling(concept, archive, run_id, base_dir)
        print(f"Initial sampling done. Evaluations: {nf_eval}/{budget}")
        iter = 1

    # ==========================================================
    # Main optimization loop
    # ==========================================================
    try:
        while nf_eval < budget:
            #Build surrogate models using evaluator function
            evaluator = build_evaluator_model(
                archive=archive,
                concept=concept
            )
            True_nd_front, feasible = archives_nd_front(evaluator, archive, concept, del_x, alpha, nsamples_scalar, k_dom)
            #Generate candidates using surrrogate-assisted optimization for 95th delta perturbation (robust optimization)
            x_pred, f_pred, g_pred = Run_optimization(evaluator, pop_size, generations, concept,
                                                       del_x, alpha, nsamples_scalar, crossover_prob,
                                                       crossover_eta, mutation_prob, mutation_eta)
            
            #Remove duplicate candidates that is already in the archive and return the unique index of candidates    
            # idx_unique = check_candidate(x_pred, archive["X"], archive["LB"], archive["UB"])
            idx_unique = check_candidate(
                x_pred,
                archive["X"][:, :nvar],
                archive["LB"][:nvar],
                archive["UB"][:nvar]
            )
            # Unique candidate solutions
            X_cand_unique = x_pred[idx_unique]
            F_cand_unique = f_pred[idx_unique]
            G_cand_unique = g_pred[idx_unique]
            #Perform DSS to select a single candidate to truly evaluate returns the selected index of the unique candidates
            sel_idx = select_candidate(
                True_nd_front,
                F_cand_unique,
                G_cand_unique,
                method="ED",
                nc=nc,
                nf=nf,
                n_cand_per_run=n_cand_per_run
            )

            x_sel = X_cand_unique[sel_idx]
            f_sel = F_cand_unique[sel_idx]
            g_sel = G_cand_unique[sel_idx]
            #Evaluate the selected candidate solution
            newly_eval = []
            for x,f,g in zip(x_sel,f_sel, g_sel):
                print(f"Selected candidate for evaluation: {x}")
                #Evaluate and return the function and constraint values
                x_updated = x_updated_with_random_variables(x, concept)
                F,G = evaluate_and_update(concept, archive, x_updated, iter)
                Iter_results = {
                    "X": x.tolist(),
                    "F": F.tolist(),
                    "G": G.tolist(),
                    "F_pred":f,
                    "G_pred":g}
                newly_eval.append(Iter_results)
                nf_eval += 1

                if nf_eval >= budget:
                    print(
                        "Budget reached after evaluating candidate. "
                        "Stopping optimization."
                    )
                    break

            save_json({"newly_eval": newly_eval,
                       "Candidates":f_pred,
                       "True_Robust_ND": True_nd_front}, os.path.join(base_dir, f"iteration_{iter}.json"))
            iter += 1
            print(f"Evaluations: {nf_eval}/{budget}")

    # ==========================================================
    # Exceptions
    # ==========================================================
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Saving final results...")

    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================================
    # Final Save
    # ==========================================================
    finally:
        try:
            archive_F = archive["F"]
            if nc > 0:
                G_all = archive["G"]
                feasible_mask = (G_all <= 0).all(axis=1)
                if np.any(feasible_mask):
                    F_true_nds, _ = nondominated(archive_F[feasible_mask])
                else:
                    F_true_nds = np.empty((0, nf))

            else:
                F_true_nds, _ = nondominated(archive_F)
            if len(F_true_nds) > 0:
                plot_nd_front(F_true_nds, base_dir)
            final_archive = {
                "X": archive["X"],
                "F": archive["F"],
                "G": archive["G"],
            }
            save_json(final_archive,os.path.join(base_dir,"final_archives.json"))
            print(f"✅ Final results saved to "f"{os.path.join(base_dir, 'final_archives.json')}")
        except Exception as e2:
            print(f"⚠️ Could not save final results: {e2}")
    return archive