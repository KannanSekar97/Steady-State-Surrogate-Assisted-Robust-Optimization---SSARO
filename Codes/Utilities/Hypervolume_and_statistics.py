"""Compute hypervolume values and statistics for multiple problem folders.

Expected layout (default):

    Results/
        RTP1/
            .../*.npy
        RTP2/
            .../*.npy
        ...
        Statistics/

By default, the script recursively searches each problem folder for files named
``robust_nd_front_true_run_*.npy``. Change FRONT_GLOB if your files use a
different name.

For each problem, all run fronts are normalized using one global ideal/nadir
computed from the nondominated union of that problem's run fronts. Hypervolume
is then calculated with a normalized reference point of 1.1 in every objective.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import Hypervolume


# =============================================================================
# Configuration
# =============================================================================

RESULT_ROOT = Path("Results")

PROBLEM_NAMES = [
    "RTP1",
    "RTP2",
    "TP4",
    "TA",
    "CBEAM",
    "BWB_VSP_AERO",
]


STATISTICS_DIR = RESULT_ROOT / "Statistics"
MEDIAN_FIGURE_DIR = RESULT_ROOT / "Median_HV_figure"

FRONT_GLOB = "**/robust_nd_front_true_run_*.npy"

NORMALIZED_REFERENCE_VALUE = 1.1

# Sample standard deviation when at least two valid runs are available.
STD_DDOF = 1


# =============================================================================
# Helpers
# =============================================================================


def ensure_2d_front(array: np.ndarray, source: Path) -> np.ndarray:
    """Validate and return one objective front as a finite 2D float array."""
    front = np.asarray(array, dtype=float)

    if front.ndim == 1:
        # Interpret a 1D array as one objective vector / one point.
        front = front.reshape(1, -1)

    if front.ndim != 2:
        raise ValueError(
            f"{source} must contain a 2D objective array; got shape {front.shape}."
        )

    # Remove rows containing NaN or infinity.
    front = front[np.isfinite(front).all(axis=1)]
    return front


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    """Return a mask selecting nondominated points for minimization."""
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    keep = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not keep[i]:
            continue

        # Point j dominates i when j <= i in all objectives and j < i in at
        # least one objective.
        dominates_i = np.all(points <= points[i], axis=1) & np.any(
            points < points[i], axis=1
        )
        dominates_i[i] = False

        if np.any(dominates_i):
            keep[i] = False

    return keep


def extract_run_label(path: Path) -> str:
    """Extract a stable run label from a filename."""
    match = re.search(r"run[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return f"{int(match.group(1)):02d}"
    return path.stem


def discover_front_files(problem_dir: Path) -> list[Path]:
    """Find run-front .npy files beneath one problem directory."""
    files = sorted(problem_dir.glob(FRONT_GLOB))

    # Defensive exclusions in case FRONT_GLOB is changed to something broad.
    excluded_tokens = (
        "x_robust",
        "design",
        "ideal",
        "nadir",
        "combined",
        "hypervolume",
        "run_numbers",
    )

    return [
        path
        for path in files
        if not any(token in path.name.lower() for token in excluded_tokens)
    ]


def load_problem_fronts(problem_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load all valid run fronts and record files that could not be used."""
    fronts: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    objective_count: int | None = None

    for path in discover_front_files(problem_dir):
        run_label = extract_run_label(path)

        try:
            front = ensure_2d_front(np.load(path, allow_pickle=False), path)

            if front.shape[0] == 0:
                skipped[str(path)] = "empty front after removing non-finite rows"
                continue

            if objective_count is None:
                objective_count = front.shape[1]
            elif front.shape[1] != objective_count:
                skipped[str(path)] = (
                    f"objective count {front.shape[1]} does not match "
                    f"expected count {objective_count}"
                )
                continue

            # Avoid silently overwriting duplicate run labels.
            if run_label in fronts:
                run_label = path.stem

            fronts[run_label] = front

        except Exception as exc:  # Continue processing the remaining runs.
            skipped[str(path)] = f"{type(exc).__name__}: {exc}"

    return fronts, skipped


def calculate_problem_hv(fronts: dict[str, np.ndarray], problem_name: str) -> dict[str, Any]:
    """Compute globally normalized HV values and summary statistics."""
    if not fronts:
        raise ValueError("No valid fronts were supplied.")
    
    if problem_name == "RTP1_mean":
        combined_nd = np.load("Reference_set/RTP1_reference_100.npy")
    elif problem_name == "RTP2_mean":
        combined_nd = np.load("Reference_set/RTP2_reference_100.npy")
    elif problem_name == "TP4_mean":
        combined_nd = np.load("Reference_set/TP4_reference_100.npy")
    else:
        combined = np.vstack(list(fronts.values()))
        combined_nd = combined[nondominated_mask(combined)]

    ideal = np.min(combined_nd, axis=0)
    nadir = np.max(combined_nd, axis=0)
    ranges = nadir - ideal

    # A constant objective contributes no discrimination. Mapping it to zero
    # avoids division by zero while keeping all points valid.
    constant_objectives = np.isclose(ranges, 0.0)
    safe_ranges = ranges.copy()
    safe_ranges[constant_objectives] = 1.0

    reference_point = np.full(
        combined_nd.shape[1], NORMALIZED_REFERENCE_VALUE, dtype=float
    )
    hv_indicator = Hypervolume(ref_point=reference_point)

    hv_by_run: dict[str, float] = {}
    points_by_run: dict[str, int] = {}

    for run_label, front in sorted(fronts.items()):
        normalized = (front - ideal) / safe_ranges
        normalized[:, constant_objectives] = 0.0

        # Points at or beyond the reference point do not contribute useful
        # dominated volume and may violate the indicator's assumptions.
        inside_reference = np.all(normalized < reference_point, axis=1)
        normalized = normalized[inside_reference]

        if normalized.shape[0] == 0:
            hv_by_run[run_label] = float("nan")
            points_by_run[run_label] = 0
            continue

        normalized_nd = normalized[nondominated_mask(normalized)]
        hv_by_run[run_label] = float(hv_indicator(normalized_nd))
        points_by_run[run_label] = int(normalized_nd.shape[0])

    valid_values = np.array(
        [value for value in hv_by_run.values() if np.isfinite(value)],
        dtype=float,
    )

    median_run = None

    if valid_values.size:
        median_value = float(np.median(valid_values))
        valid_run_values = {
            run: value
            for run, value in hv_by_run.items()
            if np.isfinite(value)
        }
        median_run = min(
            valid_run_values,
            key=lambda run: (
                abs(valid_run_values[run] - median_value),
                str(run),
            ),
        )

        statistics = {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values, ddof=STD_DDOF))
            if valid_values.size > STD_DDOF
            else 0.0,
            "median": median_value,
            "median_run": median_run,
            "median_run_hv": float(valid_run_values[median_run]),
            "minimum": float(np.min(valid_values)),
            "maximum": float(np.max(valid_values)),
            "q1": float(np.percentile(valid_values, 25)),
            "q3": float(np.percentile(valid_values, 75)),
            "count": int(valid_values.size),
        }
    else:
        statistics = {
            "mean": None,
            "std": None,
            "median": None,
            "median_run": None,
            "median_run_hv": None,
            "minimum": None,
            "maximum": None,
            "q1": None,
            "q3": None,
            "count": 0,
        }

    return {
        "objective_count": int(combined_nd.shape[1]),
        "reference_point": reference_point.tolist(),
        "ideal": ideal.tolist(),
        "nadir": nadir.tolist(),
        "constant_objective_indices": np.flatnonzero(constant_objectives).tolist(),
        "number_of_combined_nd_points": int(combined_nd.shape[0]),
        "hypervolume_by_run": {
            run: (float(value) if np.isfinite(value) else None)
            for run, value in hv_by_run.items()
        },
        "nondominated_points_used_by_run": points_by_run,
        "statistics": statistics,
        "_combined_nd": combined_nd,
    }



def plot_median_hv_front(
    problem_name: str,
    fronts: dict[str, np.ndarray],
    result: dict[str, Any],
    median_figure_dir: Path = MEDIAN_FIGURE_DIR,
) -> Path | None:
    """Plot the median-HV run together with the combined ND solution."""
    combined_nd = np.asarray(result["_combined_nd"], dtype=float)
    stats = result["statistics"]
    median_run = stats.get("median_run")

    if median_run is None:
        print(f"  Median plot skipped for {problem_name}: no valid HV run.")
        return None

    if combined_nd.ndim != 2 or combined_nd.shape[1] != 2:
        print(
            f"  Median plot skipped for {problem_name}: "
            f"expected two objectives, got shape {combined_nd.shape}."
        )
        return None

    median_front = ensure_2d_front(
        fronts[median_run],
        Path(f"{problem_name}:{median_run}"),
    )
    median_front_nd = median_front[nondominated_mask(median_front)]

    median_figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        combined_nd[:, 0],
        combined_nd[:, 1],
        marker="o",
        s=60,
        facecolors="none",
        edgecolors="black",
        label="Combined ND solution",
    )

    plt.scatter(
        median_front_nd[:, 0],
        median_front_nd[:, 1],
        marker="d",
        facecolors="red",
        s=75,
        label=f"Median-HV run {median_run} (HV={stats['median_run_hv']:.6f})",
    )

    # Axis labels
    plt.xlabel(r"$f_1$", fontsize=22)
    plt.ylabel(r"$f_2$", fontsize=22)

    # Tick label size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Alternatively:
    # plt.tick_params(axis='both', which='major', labelsize=16)

    # Legend
    plt.legend(fontsize=16)

    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = median_figure_dir / f"{problem_name}_median_HV.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved median-HV figure: {output_path}")
    return output_path


def save_problem_statistics(
    problem_name: str,
    result: dict[str, Any],
    source_files: list[Path],
    skipped: dict[str, str],
    statistics_dir: Path = STATISTICS_DIR,
) -> None:
    """Write one JSON file and one human-readable TXT file."""
    statistics_dir.mkdir(parents=True, exist_ok=True)

    public_result = {
        key: value
        for key, value in result.items()
        if not key.startswith("_")
    }

    result = {
        "problem": problem_name,
        "front_glob": FRONT_GLOB,
        "source_files": [str(path) for path in source_files],
        "skipped_files": skipped,
        **public_result,
    }

    json_path = statistics_dir / f"{problem_name}_HV_statistics.json"
    txt_path = statistics_dir / f"{problem_name}_HV_statistics.txt"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, allow_nan=False)

    stats = result["statistics"]
    with txt_path.open("w", encoding="utf-8") as file:
        file.write(f"Problem: {problem_name}\n")
        file.write(f"Objectives: {result['objective_count']}\n")
        file.write(f"Reference point: {result['reference_point']}\n")
        file.write(f"Ideal: {result['ideal']}\n")
        file.write(f"Nadir: {result['nadir']}\n")
        file.write(
            "Constant objective indices: "
            f"{result['constant_objective_indices']}\n\n"
        )

        file.write("Hypervolume by run\n")
        file.write("------------------\n")
        for run, value in result["hypervolume_by_run"].items():
            text = "NA" if value is None else f"{value:.12g}"
            file.write(f"Run {run}: {text}\n")

        file.write("\nStatistics\n")
        file.write("----------\n")
        for key in (
            "count",
            "mean",
            "std",
            "median",
            "median_run",
            "median_run_hv",
            "minimum",
            "maximum",
            "q1",
            "q3",
        ):
            value = stats[key]
            if isinstance(value, float):
                file.write(f"{key}: {value:.12g}\n")
            else:
                file.write(f"{key}: {value}\n")

        if skipped:
            file.write("\nSkipped files\n")
            file.write("-------------\n")
            for path, reason in skipped.items():
                file.write(f"{path}: {reason}\n")

    print(f"  Saved: {json_path}")
    print(f"  Saved: {txt_path}")


# =============================================================================
# Main
# =============================================================================


def run_hypervolume_and_statistics(
    problem_names: list[str] = PROBLEM_NAMES,
    result_root: Path = RESULT_ROOT,
) -> dict[str, dict[str, Any]]:
    """
    Compute globally normalized hypervolume + statistics for every problem
    folder beneath ``result_root``, save JSON/TXT summaries and a
    median-HV-run figure per problem.

    Returns a dict keyed by problem name, mapping to the ``result`` dict
    from ``calculate_problem_hv`` (includes "statistics", with
    "median_run" identifying the run closest to the median hypervolume).
    Problems that could not be processed are omitted; see the returned
    failures via the printed log / ``HV_statistics_failures.json``.
    """
    result_root = Path(result_root)
    statistics_dir = result_root / "Statistics"
    median_figure_dir = result_root / "Median_HV_figure"

    statistics_dir.mkdir(parents=True, exist_ok=True)
    median_figure_dir.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}

    for problem_name in problem_names:
        problem_dir = result_root / problem_name
        print(f"\nProcessing {problem_name}: {problem_dir}")

        if not problem_dir.is_dir():
            failures[problem_name] = f"problem folder not found: {problem_dir}"
            print(f"  Skipped: {failures[problem_name]}")
            continue

        source_files = discover_front_files(problem_dir)

        fronts, skipped = load_problem_fronts(problem_dir)

        if not fronts:
            failures[problem_name] = (
                f"no valid files matched {FRONT_GLOB!r} beneath {problem_dir}"
            )
            print(f"  Skipped: {failures[problem_name]}")
            continue

        try:
            result = calculate_problem_hv(fronts, problem_name)

            plot_median_hv_front(
                problem_name=problem_name,
                fronts=fronts,
                result=result,
                median_figure_dir=median_figure_dir,
            )

            save_problem_statistics(
                problem_name=problem_name,
                result=result,
                source_files=source_files,
                skipped=skipped,
                statistics_dir=statistics_dir,
            )
            print(
                f"  Valid runs: {result['statistics']['count']}; "
                f"mean HV: {result['statistics']['mean']}"
            )
            results[problem_name] = result
        except Exception as exc:
            failures[problem_name] = f"{type(exc).__name__}: {exc}"
            print(f"  Failed: {failures[problem_name]}")

    if failures:
        failure_path = statistics_dir / "HV_statistics_failures.json"
        with failure_path.open("w", encoding="utf-8") as file:
            json.dump(failures, file, indent=4)
        print(f"\nFailure report saved to: {failure_path}")

    print(f"\nFinished. Statistics folder: {statistics_dir}")

    return results


def main() -> None:
    run_hypervolume_and_statistics()


if __name__ == "__main__":
    main()