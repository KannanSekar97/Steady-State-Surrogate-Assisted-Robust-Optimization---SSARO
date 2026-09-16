"""Plots the BWB VSPAERO model of the median-HV run's non-dominated front.

Picks three representative designs off the true robust ND front of the
median-hypervolume run - Results/Statistics/BWB_VSP_AERO_HV_statistics.json
states which run that is directly ("median_run": run 20 across the full
31-run cohort; don't use Results/BWB_VSP_AERO/Hypervolume/hypervolume_
results.json, a stale file with only one run recorded under a different
ideal/nadir): the min-CD extreme, the max-CL extreme, and the knee point
(minimum normalized distance to the ideal point). Each is re-run through the
real VSPAERO VLM solver (not the surrogate) at the design cruise condition,
with CpSlicer swept spanwise to recover a genuine surface pressure-
coefficient field. Local surface velocity ratio V/Vinf is then derived from
that Cp via the standard compressible (isentropic) Bernoulli relation -
VSPAERO's VLM solver has no off-body flow field of its own, only
circulation/Cp on the thin lifting surface, so this is the honest way to get
a "velocity contour" out of it.

Produces, under optimization/bwb_cfd_median_hv/:
  bwb_run<NN>_tess<TW>_<label>.npz       - cached per-(run, tessellation,
                                           design) Cp slice data - keyed by
                                           all three, not just label, since
                                           the three labels (min_CD/knee/
                                           max_CL) mean different designs in
                                           different runs, and a different
                                           Tess_W changes the solved CD
  BWB_run<NN>_front_highlighted.png      - ND front with the 3 points marked
                                            (CL axis kept as the raw -CL
                                            objective, matching F[:,1];
                                            markers plotted at the archived
                                            front point itself, so they land
                                            exactly on it)
  BWB_run<NN>_velocity_ratio.png         - 3-panel surface V/Vinf field, each
                                            panel framed by the actual solved
                                            (mirrored) geometry bounding box
                                            with a freestream direction arrow
  BWB_run<NN>_median_HV_with_CFD.png     - front + velocity field combined

plot_field(..., field="Cp", ...) still works if a pressure-coefficient
figure is wanted, it's just not called by run_and_plot() by default.

Requires the `openvsp` package only for the (cached) solver runs - if the
.npz cache already exists for a (run, tess_w, design) combination, plotting
reuses it without invoking VSPAERO again.

Usage:
  python plot_median_hv_bwb_cfd.py                    # auto: current median-HV run, Tess_W=20
  python plot_median_hv_bwb_cfd.py --run 20            # a specific run
  python plot_median_hv_bwb_cfd.py --run 20 --tess-w 25

  # or from Python / a notebook:
  from plot_median_hv_bwb_cfd import run_and_plot
  run_and_plot(20)          # a specific run number
  run_and_plot()            # auto-detect the current median-HV run
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
BWB_REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "Results" / "BWB_VSP_AERO"
OUT_DIR = Path(__file__).resolve().parent / "bwb_cfd_median_hv"
OUT_DIR.mkdir(exist_ok=True)

DESIGN_KEYS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]
FLIGHT_CONDITION = {"alt_kft": 35.0, "M_inf": 0.45, "Re": 4.0e7, "alpha_deg": 4.0}
GAMMA = 1.4
# Matches bwb_vsp_aero_problem.py's TESS_W default (the pipeline that actually
# produced the archived fronts) - keep these in sync, or CD comes out a
# consistent few % high/low relative to the archive (CL/CMy barely move,
# CD is what's sensitive to spanwise panel count).
TESS_W = 20.0

_PANELS = [("XSec_1", "B1", "S1", "C2"), ("XSec_2", "B2", "S2", "C3"), ("XSec_3", "B3", "S3", "C4")]


def median_hv_run_number() -> int:
    """Run number identified as the median-HV run for BWB_VSP_AERO.

    Results/Statistics/BWB_VSP_AERO_HV_statistics.json is the authoritative
    source - it aggregates all 31 runs and states "median_run" directly.
    (Results/BWB_VSP_AERO/Hypervolume/hypervolume_results.json looks similar
    but only has one run ("17") recorded under a different ideal/nadir - a
    stale/partial file, not the full-cohort statistics - don't use it here.)
    """
    with open(REPO_ROOT / "Results" / "Statistics" / "BWB_VSP_AERO_HV_statistics.json") as f:
        stats = json.load(f)
    return int(stats["statistics"]["median_run"])


# def representative_designs(run_number: int):
#     """min-CD extreme, knee point, and max-CL extreme off the run's true ND front.

#     Returns (designs, idx): idx maps each label to its row in the front's F/X
#     arrays, so callers can plot the marker at the *exact* archived front
#     point rather than at a freshly re-solved (and therefore slightly
#     different, since the VLM re-solve here uses a different tessellation
#     than whatever produced the archive) CL/CD.
#     """
#     front_dir = RESULTS_DIR / "Robust_ND_front_points"
#     tag = f"run_{run_number:02d}"
#     X = np.load(front_dir / f"X_robust_nd_front_true_{tag}.npy")
#     F = np.load(front_dir / f"robust_nd_front_true_{tag}.npy")  # F[:,0]=CD, F[:,1]=-CL

#     ideal, nadir = F.min(axis=0), F.max(axis=0)
#     dist = np.linalg.norm((F - ideal) / (nadir - ideal), axis=1)

#     idx = {"min_CD": int(np.argmin(F[:, 0])), "max_CL": int(np.argmin(F[:, 1])), "knee": int(np.argmin(dist))}
#     designs = {label: dict(zip(DESIGN_KEYS, X[i])) for label, i in idx.items()}
#     return designs, idx

def representative_designs(run_number: int):
    """min-CD extreme, knee point, and max-CL extreme off the run's true ND front.

    Knee point follows the paper's hyperdistance definition (Zou et al. 2019,
    Sec. 3.3): build a hyperplane from the ideal and nadir points in
    normalized objective space, then pick the solution with the SHORTEST
    combined distance to that hyperplane (d1 = projection length along the
    ideal->nadir direction, d2 = perpendicular distance to the hyperplane).
    This is different from (and geometrically more correct than) simply
    picking the point closest to the ideal point.

    Returns (designs, idx): idx maps each label to its row in the front's F/X
    arrays, so callers can plot the marker at the *exact* archived front
    point rather than at a freshly re-solved (and therefore slightly
    different, since the VLM re-solve here uses a different tessellation
    than whatever produced the archive) CL/CD.
    """
    front_dir = RESULTS_DIR / "Robust_ND_front_points"
    tag = f"run_{run_number:02d}"
    X = np.load(front_dir / f"X_robust_nd_front_true_{tag}.npy")
    F = np.load(front_dir / f"robust_nd_front_true_{tag}.npy")  # F[:,0]=CD, F[:,1]=-CL

    ideal, nadir = F.min(axis=0), F.max(axis=0)
    F_norm = (F - ideal) / (nadir - ideal)  # ideal -> (0,...,0), nadir -> (1,...,1)

    normal = np.ones(F_norm.shape[1])       # nadir - ideal in normalized space
    normal_unit = normal / np.linalg.norm(normal)

    d1 = F_norm @ normal_unit                                # projection length onto ideal->nadir direction
    proj_points = np.outer(d1, normal_unit)
    d2 = np.linalg.norm(F_norm - proj_points, axis=1)        # perpendicular distance to the hyperplane

    hyperdistance = d1 + d2
    idx_knee = int(np.argmin(hyperdistance))                 # shortest hyperdistance = knee point

    idx = {"min_CD": int(np.argmin(F[:, 0])), "max_CL": int(np.argmin(F[:, 1])), "knee": idx_knee}
    designs = {label: dict(zip(DESIGN_KEYS, X[i])) for label, i in idx.items()}
    return designs, idx


def cp_to_velocity_ratio(cp: np.ndarray, mach: float, gamma: float = GAMMA) -> np.ndarray:
    """V/Vinf from surface Cp via the compressible (isentropic) Bernoulli relation.

    Cp = (2/(gamma*M^2)) * [(1 + (gamma-1)/2 * M^2 * (1-(V/Vinf)^2))^(gamma/(gamma-1)) - 1]
    solved for V/Vinf. VLM has no off-body velocity field to sample directly,
    so this is the standard way to turn a solved surface Cp into a velocity
    contour.
    """
    a = 1.0 + cp * gamma * mach ** 2 / 2.0
    a = np.clip(a, 1e-9, None)  # guard against Cp below the physical floor
    bracket = a ** ((gamma - 1.0) / gamma)
    v_ratio_sq = 1.0 - (bracket - 1.0) * 2.0 / ((gamma - 1.0) * mach ** 2)
    return np.sqrt(np.clip(v_ratio_sq, 0.0, None))


def _run_vspaero_design(label: str, design: dict, fc: dict, run_number: int, tess_w: float = TESS_W,
                         n_y_slices: int = 41) -> Path:
    """Full VSPAERO VLM solve + spanwise CpSlicer sweep for one design; caches to .npz."""
    import openvsp as vsp
    from vspaero_lowfidelity import VSP3_PATH

    devnull = open(os.devnull, "w")
    tmp = tempfile.mkdtemp()
    vsp3_copy = Path(tmp) / "model.vsp3"
    shutil.copyfile(VSP3_PATH, vsp3_copy)

    vsp.VSPCheckSetup()
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(str(vsp3_copy))
    geom_id = vsp.FindGeomsWithName("WingGeom1")[0]
    vsp.SetParmVal(geom_id, "Tess_W", "Shape", float(tess_w))
    for xsec, b_key, s_key, tip_key in _PANELS:
        vsp.SetParmVal(geom_id, "Span", xsec, design[b_key])
        vsp.SetParmVal(geom_id, "Sweep", xsec, design[s_key])
        vsp.SetParmVal(geom_id, "Tip_Chord", xsec, design[tip_key])
    vsp.Update()

    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [-1])
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "ThinGeomSet", [1])
    vsp.Update()
    with contextlib.redirect_stdout(devnull):
        vsp.ExecAnalysis("VSPAEROComputeGeometry")

    analysis = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(analysis)
    vsp.SetIntAnalysisInput(analysis, "GeomSet", [-1])
    vsp.SetIntAnalysisInput(analysis, "ThinGeomSet", [1])
    vsp.SetIntAnalysisInput(analysis, "RefFlag", [1])
    vsp.SetStringAnalysisInput(analysis, "WingID", [geom_id])
    vsp.SetDoubleAnalysisInput(analysis, "AlphaStart", [fc["alpha_deg"]])
    vsp.SetIntAnalysisInput(analysis, "AlphaNpts", [1])
    vsp.SetDoubleAnalysisInput(analysis, "MachStart", [fc["M_inf"]])
    vsp.SetIntAnalysisInput(analysis, "MachNpts", [1])
    vsp.SetDoubleAnalysisInput(analysis, "ReCref", [fc["Re"]])
    vsp.SetIntAnalysisInput(analysis, "NCPU", [4])
    vsp.Update()
    with contextlib.redirect_stdout(devnull):
        vsp.ExecAnalysis(analysis)

    hist_id = vsp.FindLatestResultsID("VSPAERO_History")
    cl = vsp.GetDoubleResults(hist_id, "CLtot")[-1]
    cd = vsp.GetDoubleResults(hist_id, "CDtot")[-1]
    cmy = vsp.GetDoubleResults(hist_id, "CMytot")[-1]

    ymax = 1086.93  # half-span extent of this model's Bref; slice just inboard of the tips
    # (CpSlicer degenerates exactly at the tip edge, so 0.995 rather than 1.0 - this
    # is close enough that the "missing" sliver at each tip is sub-pixel at plot scale)
    y_positions = np.linspace(-ymax * 0.995, ymax * 0.995, n_y_slices)
    vsp.SetAnalysisInputDefaults("CpSlicer")
    vsp.SetDoubleAnalysisInput("CpSlicer", "YSlicePosVec", list(y_positions))
    with contextlib.redirect_stdout(devnull):
        cp_res = vsp.ExecAnalysis("CpSlicer")
    case_ids = vsp.GetStringResults(cp_res, "CpSlice_Case_ID_Vec")

    X, Y, Z, CP = [], [], [], []
    for cid in case_ids:
        X.extend(vsp.GetDoubleResults(cid, "X_Loc"))
        Y.extend(vsp.GetDoubleResults(cid, "Y_Loc"))
        Z.extend(vsp.GetDoubleResults(cid, "Z_Loc"))
        CP.extend(vsp.GetDoubleResults(cid, "Cp"))

    out_path = OUT_DIR / f"bwb_run{run_number:02d}_tess{int(tess_w)}_{label}.npz"
    np.savez(out_path, X=np.array(X), Y=np.array(Y), Z=np.array(Z), CP=np.array(CP),
              CL=cl, CD=cd, CMy=cmy, design=json.dumps(design))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{label}: CL={cl:.5f} CD={cd:.5f} CMy={cmy:.5f} n_pts={len(X)}")
    return out_path


def solve_or_load(designs: dict, fc: dict, run_number: int, tess_w: float = TESS_W) -> dict:
    """Cache filenames are keyed by run_number and tess_w as well as label -
    otherwise switching which run is "median" (e.g. after fixing
    median_hv_run_number()) or changing the mesh resolution would silently
    reuse stale cached aero data under labels that now mean something else."""
    data = {}
    for label, design in designs.items():
        cache = OUT_DIR / f"bwb_run{run_number:02d}_tess{int(tess_w)}_{label}.npz"
        if not cache.exists():
            sys.path.insert(0, str(BWB_REPO / "optimization"))
            _run_vspaero_design(label, design, fc, run_number, tess_w)
        data[label] = np.load(cache)
    return data


def structured_chord_grid(d: dict, n_chord: int = 60):
    """Resample each spanwise Cp slice onto a common normalized-chord grid so the
    field can be drawn as a proper (span x chord) mesh instead of a naive
    Delaunay triangulation (which bridges across the taper/sweep and produces
    spurious spikes near the root)."""
    X, Y, CP = d["X"], d["Y"], d["CP"]
    rows = []
    for yv in np.unique(np.round(Y, 3)):
        mask = np.isclose(Y, yv, atol=1e-2)
        if mask.sum() < 3:
            continue
        xs, cps = X[mask], CP[mask]
        order = np.argsort(xs)
        xs, cps = xs[order], cps[order]
        xs, uidx = np.unique(xs, return_index=True)
        cps = cps[uidx]
        if len(xs) < 3:
            continue
        s = (xs - xs.min()) / (xs.max() - xs.min())
        s_grid = np.linspace(0, 1, n_chord)
        rows.append((yv, xs.min() + s_grid * (xs.max() - xs.min()), np.interp(s_grid, s, cps)))
    rows.sort(key=lambda r: r[0])
    Ymesh = np.repeat(np.array([r[0] for r in rows])[:, None], n_chord, axis=1)
    Xgrid = np.array([r[1] for r in rows])
    CPgrid = np.array([r[2] for r in rows])
    return Ymesh, Xgrid, CPgrid


LABELS = [("min_CD", "Min-Drag Extreme", "#2ca02c", "o"),
          ("knee", "Knee-Point", "#d62728", "D"),
          ("max_CL", "Max-Lift Extreme", "#9467bd", "s")]


def add_geometry_frame(ax, d: dict):
    """Draws the actual solved (mirrored) geometry bounding box around the
    field plot, plus a freestream direction arrow.

    VSPAERO itself computes an exact Xmin/Xmax, Ymin/Ymax, Zmin/Zmax bounding
    box of the geometry internally (visible in its solver log) to size the
    wake/far-field domain - it isn't rendered anywhere in the model or GUI,
    so nothing shows it by default. Here it's taken directly from the extents
    of the solved CpSlice points (X, Y already cover the true, symmetry-
    mirrored surface), not an idealized/parametric approximation.
    """
    x_min, x_max = float(d["X"].min()), float(d["X"].max())
    y_min, y_max = float(d["Y"].min()), float(d["Y"].max())
    x_span, y_span = x_max - x_min, y_max - y_min

    rect = plt.Rectangle((y_min, x_min), y_span, x_span,
                          fill=False, edgecolor="0.25", linewidth=1.1,
                          linestyle="--", alpha=0.8, zorder=6)
    ax.add_patch(rect)

    # Freestream arrow: flow runs root-LE (small X) -> TE (large X); the axis
    # is inverted so that direction reads as pointing down the page.
    arrow_x = 0.0
    x_tail, x_head = x_min - 0.22 * x_span, x_min - 0.04 * x_span
    ax.annotate("", xy=(arrow_x, x_head), xytext=(arrow_x, x_tail),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.6), zorder=7)
    ax.text(arrow_x, x_tail - 0.05 * x_span, r"$V_\infty$", ha="center", va="top",
            fontsize=9.5, color="0.15")

    # Margin on all 4 sides so the box doesn't hide behind the axes spines.
    ax.set_xlim(y_min - 0.06 * y_span, y_max + 0.06 * y_span)
    ax.set_ylim(x_max + 0.05 * x_span, x_min - 0.32 * x_span)


def plot_front(run_number: int, idx: dict, out_path: Path):
    front_dir = RESULTS_DIR / "Robust_ND_front_points"
    F = np.load(front_dir / f"robust_nd_front_true_run_{run_number:02d}.npy")
    CD, CL_obj = F[:, 0], F[:, 1]  # CL kept negative, matching the optimizer's own objective (F[:,1] = -CL)

    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.scatter(CD, CL_obj, s=22, c="#4C72B0", alpha=0.55, edgecolor="none",
               label=f"Run {run_number} (median-HV) ND front")
    for label, title, color, marker in LABELS:
        # Plot the archived front point itself (not the freshly re-solved CL/CD from
        # `data`, which uses a different VLM tessellation and so lands slightly off
        # the front curve) - this guarantees the marker sits exactly on the front.
        cd, cl_obj = float(CD[idx[label]]), float(CL_obj[idx[label]])
        ax.scatter(cd, cl_obj, s=140, c=color, marker=marker, edgecolor="black", linewidth=1.1, zorder=5, label=title)
        ax.annotate(title, (cd, cl_obj), textcoords="offset points", xytext=(8, 6),
                    fontsize=10, fontweight="bold", color=color)
    ax.set_xlabel("$C_D$"); ax.set_ylabel("$-C_L$")
    ax.set_title(f"BWB VSP-Aero - Median-HV Run (Run {run_number}) Robust ND Front")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_field(data: dict, field: str, cbar_label: str, cmap: str, out_path: Path, fc: dict, title_suffix: str = ""):
    grids, values = {}, {}
    for label, _, _, _ in LABELS:
        Ymesh, Xg, CPg = structured_chord_grid(data[label])
        grids[label] = (Ymesh, Xg)
        if field == "Cp":
            values[label] = CPg
        else:
            values[label] = cp_to_velocity_ratio(CPg, fc["M_inf"])

    all_vals = np.concatenate([v.ravel() for v in values.values()])
    if field == "Cp":
        vabs = np.percentile(np.abs(all_vals), 98)
        vmin, vmax = -vabs, vabs
    else:
        vmin, vmax = np.percentile(all_vals, [1, 99])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)
    for ax, (label, title, color, _) in zip(axes, LABELS):
        Ymesh, Xg = grids[label]
        d = data[label]
        pc = ax.pcolormesh(Ymesh, Xg, values[label], shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.contour(Ymesh, Xg, values[label], levels=14, colors="k", linewidths=0.25, alpha=0.35)
        add_geometry_frame(ax, d)
        ax.set_title(f"{title}\nCL={float(d['CL']):.3f}  CD={float(d['CD']):.4f}  CMy={float(d['CMy']):.3f}",
                     fontsize=10, color=color)
        ax.set_xlabel("Y (mm, span)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("X (mm, chord, LE at top)")
    cbar = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label)
    fig.suptitle(f"BWB VSP-Aero - Median-HV Run: {title_suffix}"
                 f" alpha={fc['alpha_deg']:.0f}deg, M={fc['M_inf']}", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined(run_number: int, data: dict, idx: dict, fc: dict, out_path: Path):
    front_dir = RESULTS_DIR / "Robust_ND_front_points"
    F = np.load(front_dir / f"robust_nd_front_true_run_{run_number:02d}.npy")
    CD, CL_obj = F[:, 0], F[:, 1]  # CL kept negative, matching the optimizer's own objective (F[:,1] = -CL)

    grids, v_values = {}, {}
    for label, _, _, _ in LABELS:
        Ymesh, Xg, CPg = structured_chord_grid(data[label])
        grids[label] = (Ymesh, Xg)
        v_values[label] = cp_to_velocity_ratio(CPg, fc["M_inf"])
    vmin, vmax = np.percentile(np.concatenate([v.ravel() for v in v_values.values()]), [1, 99])

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.05, 1], hspace=0.42, wspace=0.25)

    ax_front = fig.add_subplot(gs[0, :])
    ax_front.scatter(CD, CL_obj, s=22, c="#4C72B0", alpha=0.55, edgecolor="none",
                      label=f"Run {run_number} (median-HV) ND front")
    for label, title, color, marker in LABELS:
        # Archived front point (see plot_front) so the marker sits exactly on the front.
        cd, cl_obj = float(CD[idx[label]]), float(CL_obj[idx[label]])
        ax_front.scatter(cd, cl_obj, s=140, c=color, marker=marker, edgecolor="black", linewidth=1.1, zorder=5, label=title)
        ax_front.annotate(title, (cd, cl_obj), textcoords="offset points", xytext=(8, 6),
                           fontsize=9.5, fontweight="bold", color=color)
    ax_front.set_xlabel("$C_D$"); ax_front.set_ylabel("$-C_L$")
    ax_front.set_title(f"BWB VSP-Aero -- Median-HV Run (Run {run_number}) Robust Non-Dominated Front")
    ax_front.grid(alpha=0.3)
    ax_front.legend(loc="upper right", fontsize=8.5)

    axes_v = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax, (label, title, color, _) in zip(axes_v, LABELS):
        Ymesh, Xg = grids[label]
        d = data[label]
        pc = ax.pcolormesh(Ymesh, Xg, v_values[label], shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.contour(Ymesh, Xg, v_values[label], levels=14, colors="k", linewidths=0.25, alpha=0.35)
        add_geometry_frame(ax, d)
        ax.set_title(f"{title}\nCL={float(d['CL']):.3f}  CD={float(d['CD']):.4f}  CMy={float(d['CMy']):.3f}",
                     fontsize=10, color=color)
        ax.set_xlabel("Y (mm, span)")
        ax.set_aspect("equal")
    axes_v[0].set_ylabel("X (mm, chord, LE at top)")
    cbar = fig.colorbar(pc, ax=axes_v, shrink=0.85, pad=0.02)
    cbar.set_label("Local surface velocity ratio, $V/V_\\infty$")

    fig.suptitle(f"BWB VSP-Aero Median-HV Run: Robust ND Front & Surface Velocity (VLM), "
                 f"alpha={fc['alpha_deg']:.0f}deg, M={fc['M_inf']}", fontsize=13, y=0.98)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_and_plot(run_number: int | None = None, tess_w: float = TESS_W, fc: dict = FLIGHT_CONDITION) -> dict:
    """Entry point: feed a run number (or None for the current median-HV run)
    and this solves + plots the min-CD/knee/max-CL designs off that run's
    true ND front.

    Returns {"run_number", "data", "idx"} in case you want to inspect the
    solved CL/CD/CMy or the front-index mapping afterward.

    Examples:
        run_and_plot()      # auto: whichever run is median-HV right now
        run_and_plot(20)    # a specific run, e.g. to compare against the median
    """
    if run_number is None:
        run_number = median_hv_run_number()
    print(f"Run: {run_number}  (Tess_W={tess_w})")

    designs, idx = representative_designs(run_number)
    data = solve_or_load(designs, fc, run_number, tess_w)

    plot_front(run_number, idx, OUT_DIR / f"BWB_run{run_number:02d}_front_highlighted.png")
    plot_field(data, "V", "Local surface velocity ratio, $V/V_\\infty$", "viridis",
               OUT_DIR / f"BWB_run{run_number:02d}_velocity_ratio.png", fc,
               title_suffix="Surface Velocity Ratio (from Cp via isentropic relation),")
    plot_combined(run_number, data, idx, fc, OUT_DIR / f"BWB_run{run_number:02d}_median_HV_with_CFD.png")

    print(f"Figures written to {OUT_DIR}")
    return {"run_number": run_number, "data": data, "idx": idx}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, default=None,
                         help="Run number to plot (default: auto-detect the current median-HV run)")
    parser.add_argument("--tess-w", type=float, default=TESS_W,
                         help=f"VSPAERO chordwise tessellation (default: {TESS_W}, matching "
                              "bwb_vsp_aero_problem.py's TESS_W)")
    args = parser.parse_args()
    run_and_plot(args.run, args.tess_w)


if __name__ == "__main__":
    main()
