"""
Draws one random BWB design + flight condition from bounds.blendednet_bounds,
runs it through the real VSPAERO VLM solver (vspaero_lowfidelity.evaluate_vspaero),
and reports CL/CD/CMy. Also re-runs VSPAERO on the first 3 cases already
recorded in data/LF.csv, as a sanity check that a fresh solve reproduces the
stored values for known designs.

Requires the `openvsp` package (see geometry/../optimization/vspaero_lowfidelity.py's
docstring - install OpenVSP's Windows Python 3.11 build and run this under a
matching Python 3.11 environment).

Run with: python evaluate_bwb_sample.py
"""

import csv
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bounds.blendednet_bounds import sample_blendednet, as_dict
from optimization.vspaero_lowfidelity import VSP3_PATH, evaluate_vspaero

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LF_CSV = DATA_DIR / "LF.csv"
DESIGN_KEYS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]


def _run_vspaero(design: dict, flight_condition: dict, tess_w: int = 20, ncpu: int = 4) -> dict:
    """Runs evaluate_vspaero on a throwaway copy of model.vsp3 so concurrent/
    repeated calls never clobber each other's VSPAERO output files."""
    with tempfile.TemporaryDirectory() as tmp:
        vsp3_copy = Path(tmp) / "model.vsp3"
        shutil.copyfile(VSP3_PATH, vsp3_copy)
        return evaluate_vspaero(design, flight_condition, vsp3_path=vsp3_copy, ncpu=ncpu, tess_w=tess_w)


def evaluate_random_bwb_design(seed: int | None = None, tess_w: int = 20, ncpu: int = 4) -> dict:
    """Samples one random BWB design + flight condition within the BlendedNet
    bounds (bounds/blendednet_bounds.py) and evaluates it with VSPAERO.

    Returns a dict with the sampled `design`, `flight_condition`, and the
    resulting `CL`, `CD`, `CMy`.
    """
    sample = sample_blendednet(n=1, seed=seed)[0]
    full = as_dict(sample)
    design = {k: full[k] for k in DESIGN_KEYS}
    flight_condition = {k: full[k] for k in ["alt_kft", "M_inf", "Re", "alpha_deg"]}

    aero = _run_vspaero(design, flight_condition, tess_w=tess_w, ncpu=ncpu)
    return {"design": design, "flight_condition": flight_condition, **aero}


def evaluate_reference_designs(n: int = 3, tess_w: int = 20, ncpu: int = 4) -> list[dict]:
    """Re-runs VSPAERO on the first `n` cases in data/LF.csv (i.e. the top n
    rows already present in the repo's sample data) and compares the fresh
    result against the CL/CD/CMy already recorded there for the same design +
    flight condition.
    """
    with open(LF_CSV, newline="") as f:
        rows = [row for _, row in zip(range(n), csv.DictReader(f))]

    results = []
    for row in rows:
        design = {k: float(row[k]) for k in DESIGN_KEYS}
        flight_condition = {k: float(row[k]) for k in ["alt_kft", "M_inf", "Re", "alpha_deg"]}
        recorded = {"CL": float(row["CL"]), "CD": float(row["CD"]), "CMy": float(row["CMy"])}

        fresh = _run_vspaero(design, flight_condition, tess_w=tess_w, ncpu=ncpu)

        results.append({
            "case_name": row["case_name"],
            "geom_name": row["geom_name"],
            "design": design,
            "flight_condition": flight_condition,
            "recorded": recorded,
            "fresh": fresh,
        })
    return results


def _print_aero(label: str, aero: dict):
    print(f"  {label}: CL={aero['CL']:.5f}  CD={aero['CD']:.5f}  CMy={aero['CMy']:.5f}")


if __name__ == "__main__":
    print("=== Random BWB design, sampled within BlendedNet bounds ===")
    result = evaluate_random_bwb_design(seed=0)
    print("Design:", {k: round(v, 2) for k, v in result["design"].items()})
    print("Flight condition:", {k: round(v, 4) for k, v in result["flight_condition"].items()})
    _print_aero("VSPAERO", result)

    print("\n=== Top 3 sample solutions from data/LF.csv, re-evaluated ===")
    for r in evaluate_reference_designs(n=3):
        print(f"\n{r['case_name']} ({r['geom_name']}):")
        _print_aero("recorded (data/LF.csv)", r["recorded"])
        _print_aero("fresh VSPAERO run     ", r["fresh"])
