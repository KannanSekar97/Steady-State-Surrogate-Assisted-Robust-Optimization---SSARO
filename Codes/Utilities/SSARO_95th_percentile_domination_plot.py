"""
SSARO 95th-percentile robust ND front vs. an external "ground truth"
robust reference front (obtained by running NSGA-II directly on the true
robust problem, see Utilities/Robust_reference_set.py).

Used by post_processing.py for the median-HV run of each problem, once all
runs are finished. If no reference front is available for a problem, the
SSARO front is still plotted on its own.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_median_run_vs_reference(
    problem_name,
    result_folder,
    median_run,
    result_root="Results",
    reference_path=None,
    save_folder=None,
):
    """
    Plot the SSARO true robust ND front for one (median) run against an
    external robust reference front, if available.

    Parameters
    ----------
    problem_name : str
        Label used in the plot legend / filenames.
    result_folder : str
        Sub-folder of ``result_root`` holding this problem's results
        (usually the same as ``problem_name``).
    median_run : int
        Run id whose front is plotted (typically the median-HV run).
    reference_path : str or Path, optional
        Path to a ``Robust_front_{problem}_F.npy`` reference file. Defaults
        to ``Robust_front_NSGA-II/Robust_front_{problem_name}_F.npy``.
        Skipped silently if missing.
    save_folder : str or Path, optional
        Defaults to ``{result_root}/95th_percentile_median_HV_figure``.

    Returns
    -------
    Path to the saved figure, or None if the SSARO front was unavailable.
    """
    result_root = Path(result_root)
    front_dir = result_root / result_folder / "Robust_ND_front_points"

    ssaro_path = front_dir / f"robust_nd_front_true_run_{median_run:02d}.npy"
    if not ssaro_path.exists():
        print(f"  Skipped {problem_name}: SSARO front not found at {ssaro_path}")
        return None

    Rnd_SSARO = np.load(ssaro_path)

    if reference_path is None:
        reference_path = Path("Robust_front_NSGA-II") / f"Robust_front_{problem_name}_F.npy"
    reference_path = Path(reference_path)

    save_folder = Path(save_folder) if save_folder else result_root / "95th_percentile_median_HV_figure"
    save_folder.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    if reference_path.exists():
        Rnd = np.load(reference_path)
        plt.scatter(
            Rnd[:, 0], Rnd[:, 1],
            label="95th percentile robust reference front",
            c="red", marker="o", s=45,
        )
    else:
        print(f"  Note: no reference front found at {reference_path}; plotting SSARO front only.")

    plt.scatter(
        Rnd_SSARO[:, 0], Rnd_SSARO[:, 1],
        label="95th percentile robust ND front",
        c="black", marker="*", s=90,
    )

    plt.xlabel(r"$f_1$", fontsize=16)
    plt.ylabel(r"$f_2$", fontsize=16)
    plt.legend(fontsize=12)
    plt.tick_params(axis="both", labelsize=14, length=6, width=1.2)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = save_folder / f"{result_folder}_median_HV.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    # Legacy standalone usage: hardcoded median runs from a completed 30-run
    # study. For a new study, drive this from post_processing.py instead,
    # using the median run reported by Hypervolume_and_statistics.py.
    problem_result_folder = ["RTP1", "RTP2", "TP4", "CBEAM", "TA", "BWB_VSP_AERO"]
    median_runs = [28, 27, 19, 25, 20, 20]

    for result_folder, median_run in zip(problem_result_folder, median_runs):
        print(f"Processing {result_folder}...")
        plot_median_run_vs_reference(
            problem_name=result_folder,
            result_folder=result_folder,
            median_run=median_run,
        )

    print("\nFinished.")
