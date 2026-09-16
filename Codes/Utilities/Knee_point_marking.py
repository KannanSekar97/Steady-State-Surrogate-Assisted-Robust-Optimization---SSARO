"""
Knee point and extreme solutions on the SSARO 95th-percentile robust ND
front (paper's definition, Zou et al. 2019, Section 3): hyperplane built
from the ideal and nadir points of the front; hyperdistance = distance
along the ideal->nadir direction (d1) + perpendicular distance to that
line (d2); the knee point has the shortest hyperdistance.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def normalize(F):
    z_min = F.min(axis=0)
    z_max = F.max(axis=0)
    return (F - z_min) / (z_max - z_min + 1e-12)


def find_knee_and_extremes(F):
    """
    Parameters
    ----------
    F : ndarray, shape (n, 2)
        Objective values of a nondominated front.

    Returns
    -------
    dict with keys "idx_extreme1", "idx_extreme2", "idx_knee",
    "hyperdistance".
    """
    idx_extreme1 = int(np.argmin(F[:, 0]))
    idx_extreme2 = int(np.argmin(F[:, 1]))

    F_norm = normalize(F)
    ideal = np.zeros(F_norm.shape[1])
    nadir = np.ones(F_norm.shape[1])

    normal = nadir - ideal
    normal_unit = normal / np.linalg.norm(normal)

    vecs = F_norm - ideal
    d1 = vecs @ normal_unit
    proj_points = ideal + np.outer(d1, normal_unit)
    d2 = np.linalg.norm(F_norm - proj_points, axis=1)

    hyperdistance = d1 + d2
    idx_knee = int(np.argmin(hyperdistance))

    return {
        "idx_extreme1": idx_extreme1,
        "idx_extreme2": idx_extreme2,
        "idx_knee": idx_knee,
        "hyperdistance": hyperdistance,
    }


def plot_knee_point(
    problem_name,
    result_folder,
    median_run,
    result_root="Results",
    reference_path=None,
    save_folder=None,
):
    """
    Mark the knee point and extreme solutions on one (median) run's SSARO
    95th-percentile robust ND front, optionally against an external
    reference front (see Utilities/Robust_reference_set.py).

    Returns the save path, or None if the SSARO front was unavailable.
    """
    result_root = Path(result_root)
    front_dir = result_root / result_folder / "Robust_ND_front_points"

    ssaro_path = front_dir / f"robust_nd_front_true_run_{median_run:02d}.npy"
    ssaro_x_path = front_dir / f"X_robust_nd_front_true_run_{median_run:02d}.npy"

    if not ssaro_path.exists():
        print(f"  Skipped {problem_name}: SSARO front not found at {ssaro_path}")
        return None

    F = np.load(ssaro_path)
    X = np.load(ssaro_x_path) if ssaro_x_path.exists() else None

    if F.shape[0] < 2:
        print(f"  Skipped {problem_name}: front has fewer than 2 points.")
        return None

    knee = find_knee_and_extremes(F)
    idx_extreme1, idx_extreme2, idx_knee = knee["idx_extreme1"], knee["idx_extreme2"], knee["idx_knee"]

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
        F[:, 0], F[:, 1],
        label="95th percentile robust ND front",
        c="black", marker="*", s=90,
    )
    plt.scatter(
        F[[idx_extreme1, idx_extreme2], 0], F[[idx_extreme1, idx_extreme2], 1],
        facecolors="none", edgecolors="blue", marker="o",
        s=180, linewidths=2, label="Extreme solutions", zorder=5,
    )
    plt.scatter(
        F[idx_knee, 0], F[idx_knee, 1],
        facecolors="none", edgecolors="green", marker="s",
        s=180, linewidths=2, label="Knee-point", zorder=5,
    )

    plt.xlabel(r"$f_1$", fontsize=16)
    plt.ylabel(r"$f_2$", fontsize=16)
    plt.legend(fontsize=11)
    plt.tick_params(axis="both", labelsize=14, length=6, width=1.2)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = save_folder / f"{result_folder}_knee_marked.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {save_path}")
    print(f"  Extreme 1 (min f1): {F[idx_extreme1]}" + ("" if X is None else f" X: {X[idx_extreme1]}"))
    print(f"  Extreme 2 (min f2): {F[idx_extreme2]}" + ("" if X is None else f" X: {X[idx_extreme2]}"))
    print(f"  Knee point: {F[idx_knee]}" + ("" if X is None else f" X: {X[idx_knee]}"))
    print(f"  Knee point hyperdistance: {knee['hyperdistance'][idx_knee]}")

    return save_path


if __name__ == "__main__":
    # Legacy standalone usage for a single problem/run.
    plot_knee_point(problem_name="CBEAM", result_folder="CBEAM", median_run=25)
