"""
Low-fidelity aerodynamic evaluator for BlendedNet BWB designs, using OpenVSP's
VSPAERO vortex-lattice solver directly on a (coarsened-tessellation, see
tess_w) copy of data/model.vsp3.

Role has shifted since this module was first written: evaluate_vspaero()
here is no longer called live inside the optimization loop. It was used to
batch-generate data/LF.csv (optimization/run_vspaero_lf_batch.py, all 9700
dataset cases) and data/vspaero_ref_values.csv (optimization/
run_vspaero_cref_batch.py's Sref/Cref/Bref extraction), which
surrogate/train_multifidelity_surrogates.py then trained
surrogate/lf_surrogate.pkl on - that fast surrogate is what
optimization/bwb_mfbo_problem.py actually queries per-iteration now. Call
evaluate_vspaero() directly only when you need a fresh live VLM point
(e.g. regenerating LF.csv, or spot-checking lf_surrogate.pkl against the
real solver) - HF is surrogate/hf_surrogate.pkl (optimization/bwb_problem.py
and bwb_mfbo_problem.py), not this module.

Geometry mapping (verified directly against model.vsp3's XSec values, see
geometry/blendednet_geometry.py's docstring for the same mapping):
  XSec_1: Span=B1, Sweep=S1, Root_Chord=C1(fixed 1000mm), Tip_Chord=C2
  XSec_2: Span=B2, Sweep=S2, Root_Chord=C2,               Tip_Chord=C3
  XSec_3: Span=B3, Sweep=S3, Root_Chord=C3,               Tip_Chord=C4

Requires the `openvsp` Python package (not on PyPI/conda - see
https://www.openvsp.org/download.php for a platform build bundling prebuilt
wheels/.so files for a specific Python version). Not added to
requirements.txt since it's a heavyweight, platform/Python-version-pinned
dependency meant to live in its own environment (e.g. `conda create -n
bwb-vsp python=3.13` on macOS ARM64 + OpenVSP 3.51.0's Python3.13 build).

NOTE on moment reference (resolved, was open when this module was first
written): CMy here uses VSPAERO's default moment reference point
(Xcg=Ycg=Zcg=0, the model origin). BlendedNet's own CMy turned out to need
a bigger fix first - their CFD used a placeholder Aref=cref=1 instead of
the aircraft's real reference area/chord (see data/note.txt) - and once
that's corrected (data/HF.csv), CMy correlates with this module's LF
output at r=0.974 across all 9700 cases (data/vspaero_lf_results_corrected.csv),
which wouldn't hold up if the two moment reference points were seriously
misaligned. Treat as empirically validated, not just informative.
"""

from pathlib import Path

import openvsp as vsp

VSP3_PATH = Path(__file__).resolve().parent.parent / "data" / "model.vsp3"

_PANELS = [
    ("XSec_1", "B1", "S1", "C2"),
    ("XSec_2", "B2", "S2", "C3"),
    ("XSec_3", "B3", "S3", "C4"),
]


def evaluate_vspaero(design: dict, flight_condition: dict, vsp3_path: Path = VSP3_PATH,
                      ncpu: int = 4, tess_w: int | None = None) -> dict:
    """Run a single-point VSPAERO VLM sweep for one BWB design.

    design: dict with keys B1,B2,B3,C2,C3,C4,S1,S2,S3 (C1 fixed at 1000mm,
        set directly on the template - not overridden here).
    flight_condition: dict with keys alt_kft (unused by VSPAERO directly),
        M_inf, Re, alpha_deg.
    vsp3_path: which .vsp3 file to drive. VSPAERO writes sibling output files
        (model.vspgeom, model.history, ...) next to this path, so parallel
        callers must each pass a distinct copy to avoid clobbering.
    ncpu: threads for the VSPAERO solver itself; keep low when running many
        of these concurrently to avoid oversubscription.
    tess_w: if set, overrides the template's chordwise tessellation
        (Shape/Tess_W, authored at 201 - CFD-surface-mesh resolution, not
        typical coarse-VLM resolution). At 201 a single case needs ~45.6k
        panels, ~20GB peak RSS and ~150s; that makes any real batch either
        memory-thrashing under concurrency or infeasibly slow serially. A
        much smaller value (e.g. 20-30) is standard VLM chordwise
        resolution and cuts panel count/memory/runtime roughly
        proportionally. Leave None to preserve the template's fidelity
        as-authored.

    Returns {"CL": ..., "CD": ..., "CMy": ...}.
    """
    vsp.VSPCheckSetup()
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(str(vsp3_path))
    geom_id = vsp.FindGeomsWithName("WingGeom1")[0]

    if tess_w is not None:
        vsp.SetParmVal(geom_id, "Tess_W", "Shape", float(tess_w))

    for xsec, b_key, s_key, tip_key in _PANELS:
        vsp.SetParmVal(geom_id, "Span", xsec, design[b_key])
        vsp.SetParmVal(geom_id, "Sweep", xsec, design[s_key])
        vsp.SetParmVal(geom_id, "Tip_Chord", xsec, design[tip_key])
    vsp.Update()

    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [-1])      # skip thick/panel mesh
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "ThinGeomSet", [1])   # thin/degen -> VLM
    vsp.Update()
    vsp.ExecAnalysis("VSPAEROComputeGeometry")

    analysis = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(analysis)
    vsp.SetIntAnalysisInput(analysis, "GeomSet", [-1])
    vsp.SetIntAnalysisInput(analysis, "ThinGeomSet", [1])
    vsp.SetIntAnalysisInput(analysis, "RefFlag", [1])          # auto Sref/Cref/Bref from the wing
    vsp.SetStringAnalysisInput(analysis, "WingID", [geom_id])
    vsp.SetDoubleAnalysisInput(analysis, "AlphaStart", [flight_condition["alpha_deg"]])
    vsp.SetIntAnalysisInput(analysis, "AlphaNpts", [1])
    vsp.SetDoubleAnalysisInput(analysis, "MachStart", [flight_condition["M_inf"]])
    vsp.SetIntAnalysisInput(analysis, "MachNpts", [1])
    vsp.SetDoubleAnalysisInput(analysis, "ReCref", [flight_condition["Re"]])
    vsp.SetIntAnalysisInput(analysis, "NCPU", [ncpu])
    vsp.Update()

    res_id = vsp.ExecAnalysis(analysis)
    hist_id = vsp.FindLatestResultsID("VSPAERO_History")

    # last row of the history (final wake iteration) is the converged result
    cl = vsp.GetDoubleResults(hist_id, "CLtot")[-1]
    cd = vsp.GetDoubleResults(hist_id, "CDtot")[-1]
    cmy = vsp.GetDoubleResults(hist_id, "CMytot")[-1]
    return {"CL": cl, "CD": cd, "CMy": cmy}


if __name__ == "__main__":
    fc = {"alt_kft": 35.0, "M_inf": 0.45, "Re": 4.0e7, "alpha_deg": 4.0}
    design = {"B1": 150.0, "B2": 100.0, "B3": 500.0, "C2": 720.0, "C3": 280.0, "C4": 90.0,
              "S1": 60.0, "S2": 50.0, "S3": 40.0}
    print(evaluate_vspaero(design, fc))
