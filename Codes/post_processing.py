"""
Post-processing, run once every optimization run (across all problems) is
finished.

For each problem folder beneath Results/:

1. Hypervolume and statistics
   Computes globally normalized hypervolume per run and summary statistics
   (mean/std/median/best/worst), identifying the median-HV run.
   -> Results/Statistics/{problem}_HV_statistics.{json,txt}
   -> Results/Median_HV_figure/{problem}_median_HV.png

2. Median run SSARO 95th-percentile plot
   Plots the median-HV run's true robust ND front (alpha=0.95) against an
   external robust reference front, if one is available
   (see Utilities/Robust_reference_set.py).
   -> Results/95th_percentile_median_HV_figure/{problem}_median_HV.png

3. Knee-point marking
   Marks the knee point and extreme solutions on the median run's front.
   -> Results/95th_percentile_median_HV_figure/{problem}_knee_marked.png
"""

from pathlib import Path

from run_experiment import PROBLEMS
from Utilities.Hypervolume_and_statistics import run_hypervolume_and_statistics
from Utilities.SSARO_95th_percentile_domination_plot import plot_median_run_vs_reference
from Utilities.Knee_point_marking import plot_knee_point

RESULT_ROOT = Path("Results")


def run_post_processing(problem_names=None, result_root=RESULT_ROOT):
    problem_names = problem_names or [name for name, _ in PROBLEMS]

    print("=" * 72)
    print("Step 1/3: Hypervolume and statistics")
    print("=" * 72)
    hv_results = run_hypervolume_and_statistics(
        problem_names=problem_names, result_root=result_root
    )

    print("\n" + "=" * 72)
    print("Step 2/3: Median run SSARO 95th-percentile plot")
    print("=" * 72)
    for problem_name in problem_names:
        result = hv_results.get(problem_name)
        median_run = result["statistics"]["median_run"] if result else None

        if median_run is None:
            print(f"  Skipped {problem_name}: no median-HV run available.")
            continue

        print(f"Processing {problem_name} (median run {median_run})...")
        plot_median_run_vs_reference(
            problem_name=problem_name,
            result_folder=problem_name,
            median_run=int(median_run),
            result_root=result_root,
        )

    print("\n" + "=" * 72)
    print("Step 3/3: Knee-point marking")
    print("=" * 72)
    for problem_name in problem_names:
        result = hv_results.get(problem_name)
        median_run = result["statistics"]["median_run"] if result else None

        if median_run is None:
            print(f"  Skipped {problem_name}: no median-HV run available.")
            continue

        print(f"Processing {problem_name} (median run {median_run})...")
        plot_knee_point(
            problem_name=problem_name,
            result_folder=problem_name,
            median_run=int(median_run),
            result_root=result_root,
        )

    print("\nPost-processing finished.")
    return hv_results


if __name__ == "__main__":
    run_post_processing()
