import os
from pathlib import Path

from Utilities.Optimization import main as run_optimization
from Utilities.SSARO_robust_ND_front import compute_robust_nd_front, plot_ssaro_95th_front

# ==========================================================
# Problem imports
# ==========================================================
from Problems.Unconstrained.test_problem_4 import TEST_PROBLEM_4
from Problems.Unconstrained.rtp1 import RTP1
from Problems.Unconstrained.rtp2 import RTP2

from Problems.Constrained.cbeam import CANTILEVER_BEAM
from Problems.Constrained.torque_arm import TORQUE_ARM
from Problems.Constrained.bwb_vsp_aero_problem import BWB_VSP_AERO

# ==========================================================
# Problem registry
# ==========================================================
PROBLEMS = [
    ("TP4", TEST_PROBLEM_4),
    ("RTP1", RTP1),
    ("RTP2", RTP2),
    ("CBEAM", CANTILEVER_BEAM),
    ("TA", TORQUE_ARM),
    ("BWB_VSP_AERO", BWB_VSP_AERO),
]

RESULT_ROOT = Path("Results")

# Default optimization parameters (production settings).
DEFAULT_PARAMS = {
    "budget_scale_params": 100,
    "del_x": 0.01,  # 1 percent of range of the design bounds for perturbation
    "alpha": 0.95,  # "mean", 0.95
    "nsamples_scalar": 100,  # scalar * n_var = number of perturbed samples per candidate
    "pop_size": 100,
    "generations": 100,
    "crossover_prob": 0.95,
    "crossover_eta": 20,
    "mutation_eta": 20,
    "n_cand_per_run": 1,  # Steady state evaluation (1 true evaluation per run)
}


def run_problem(problem_idx, run_id, params=None, load_saved_models=True):
    """
    Run the optimization for one problem/run, then compute and plot its
    SSARO 95th-percentile robust nondominated front.

    Returns (concept_name, archives, robust_front_result).
    """
    concept_name, concept = PROBLEMS[problem_idx]
    run_params = {**DEFAULT_PARAMS, **(params or {}), "run_id": run_id}

    problem_dir = RESULT_ROOT / concept_name
    run_dir = problem_dir / f"Run_{run_id:02d}"

    print("=" * 60)
    print(f"Problem : {concept_name}")
    print(f"Run ID  : {run_id}")
    print("=" * 60)

    # ======================================================
    # Run optimization -> final_archives.json in run_dir
    # ======================================================
    archives = run_optimization(concept=concept, base_dir=str(run_dir), **run_params)
    print(f"Finished {concept_name} Run {run_id}")

    # ======================================================
    # Robust nondominated front (SSARO_robust_ND_front)
    # ======================================================
    robust_front_result = compute_robust_nd_front(
        concept=concept,
        run_id=run_id,
        problem_dir=problem_dir,
        del_x=run_params["del_x"],
        alpha=run_params["alpha"],
        nsamples_scalar=run_params["nsamples_scalar"],
        load_saved_models=load_saved_models,
    )

    # ======================================================
    # SSARO 95th-percentile robust ND front plot
    # ======================================================
    plot_ssaro_95th_front(
        run_id=run_id,
        base_result=robust_front_result["base"],
        predicted_result=robust_front_result["predicted"],
        true_result=robust_front_result["true"],
        plot_dir=robust_front_result["plot_dir"],
    )

    return


if __name__ == "__main__":

    # ======================================================
    # Array index from PBS
    # ======================================================
    run_id = 1  # int(os.environ["PBS_ARRAY_INDEX"])
    problem_idx = 4

    run_problem(problem_idx, run_id)
