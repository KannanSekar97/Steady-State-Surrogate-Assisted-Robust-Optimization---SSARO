"""Minimal parallel BWB VSPAERO evaluator.

Each design runs in its own spawned process (so a native VSPAERO crash only
kills that design) and VSPAERO's console output is silenced. Concurrency is
bounded by a ThreadPoolExecutor whose threads each block on one subprocess.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

BWB_REPO = Path(
    os.environ.get(
        "BWB_VSP_REPO",
        r"Problems/Constrained/BWB_VSP_Aero",
    )
).resolve()
sys.path.insert(0, str(BWB_REPO))

from Problems.Constrained.BWB_VSP_Aero.bounds.blendednet_bounds import BETA_FIXED_DEG, C1_FIXED_MM, GEOM_BOUNDS  # noqa: E402
from Problems.Constrained.BWB_VSP_Aero.optimization.vspaero_lowfidelity import VSP3_PATH  # noqa: E402
from Problems.Constrained.BWB_VSP_Aero.optimization.flight_conditions import FLIGHT_CONDITIONS  
# Import OpenVSP-related modules inside the isolated child.
from Problems.Constrained.BWB_VSP_Aero.optimization.vspaero_lowfidelity import evaluate_vspaero

N_WORKERS = int(os.environ.get("BWB_VSP_N_WORKERS", "4")) # keep the number of worker core based on available RAM. 4 worker cores for 16GB RAM
NCPU_PER_WORKER = int(os.environ.get("BWB_VSP_NCPU_PER_WORKER", "4")) # Keep the NCPU_per worker to be 4.
TESS_W = int(os.environ.get("BWB_VSP_TESS_W", "20"))
TIMEOUT_S = float(os.environ.get("BWB_VSP_TIMEOUT_S", "600"))

_geom_names = list(GEOM_BOUNDS.keys())
_bounds = [GEOM_BOUNDS[name] for name in _geom_names]
_cm_max = 0.3

_FLIGHT_CONDITION_CRUISE = {"alt_kft": 35.0, "M_inf": 0.45, "Re": 4.0e7, "alpha_deg": 4.0}
_FAILED_AERO = {"CL": float("nan"), "CD": float("nan"), "CMy": float("nan")}

CASE_ROOT = Path(tempfile.gettempdir()) / "bwb_vspaero_cases"
CASE_ROOT.mkdir(parents=True, exist_ok=True)

_CTX = mp.get_context("spawn")


def _worker(conn, case_dir: str, source_vsp3: str, design: dict, flight_condition: dict, ncpu: int, tess_w: int) -> None:
    """Isolated subprocess: runs one VSPAERO evaluation with output silenced."""
    try:
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        vsp3_copy = Path(case_dir) / "model.vsp3"
        shutil.copy2(source_vsp3, vsp3_copy)
        os.chdir(case_dir)

        # vspaero is a native subprocess that inherits fd 1/2 directly, so
        # silencing it requires redirecting the real file descriptors.
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved_out, saved_err = os.dup(1), os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            result = evaluate_vspaero(
                design, flight_condition, vsp3_path=vsp3_copy,
                ncpu=int(ncpu), tess_w=int(tess_w),
            )
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(devnull)
            os.close(saved_out)
            os.close(saved_err)

        cl, cd, cmy = float(result["CL"]), float(result["CD"]), float(result["CMy"])
        if cl == 0.0 and cd == 0.0 and cmy == 0.0:
            # OpenVSP can fail internally without raising a Python exception,
            # returning fabricated zeros instead. A real solution never lands
            # on this exact triple.
            raise RuntimeError("Suspicious exact-zero result; likely a silent OpenVSP failure.")

        conn.send(("ok", {"CL": cl, "CD": cd, "CMy": cmy}))
    except BaseException as exc:
        conn.send(("error", str(exc)))
    finally:
        conn.close()


def _evaluate_one(design: dict, flight_condition: dict, ncpu: int, tess_w: int, timeout_s: float, retries: int = 5) -> dict:
    source_vsp3 = str(Path(VSP3_PATH).resolve())

    for attempt in range(retries + 1):
        case_dir = CASE_ROOT / f"design_{uuid.uuid4().hex[:8]}"
        case_dir.mkdir(parents=True)

        recv, send = _CTX.Pipe(duplex=False)
        proc = _CTX.Process(target=_worker, args=(send, str(case_dir), source_vsp3, design, flight_condition, ncpu, tess_w))
        proc.start()
        send.close()

        proc.join(timeout_s)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            status, payload = "error", f"timed out after {timeout_s}s"
        elif recv.poll(5):
            status, payload = recv.recv()
        else:
            status, payload = "error", f"worker died silently (exitcode={proc.exitcode})"

        recv.close()
        shutil.rmtree(case_dir, ignore_errors=True)

        if status == "ok":
            return payload

    return dict(_FAILED_AERO)


def _bwb_aero(
    x: np.ndarray,
    flight_condition: Optional[dict] = None,
    n_workers: Optional[int] = None,
    ncpu_per_worker: Optional[int] = None,
    tess_w: Optional[int] = None,
    timeout_s: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """F[:, 0] = CD, F[:, 1] = -CL, G[:, 0] = abs(CMy) - 0.3"""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[0]

    fc = dict(_FLIGHT_CONDITION_CRUISE if flight_condition is None else flight_condition)
    fc.setdefault("beta_deg", BETA_FIXED_DEG)

    designs = []
    for row in x:
        design = {name: float(row[j]) for j, name in enumerate(_geom_names)}
        design["C1"] = float(C1_FIXED_MM)
        designs.append(design)

    n_workers = min(n_workers or N_WORKERS, n)
    ncpu_per_worker = ncpu_per_worker or NCPU_PER_WORKER
    tess_w = tess_w or TESS_W
    timeout_s = timeout_s or TIMEOUT_S

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        aero = list(pool.map(
            lambda d: _evaluate_one(d, fc, ncpu_per_worker, tess_w, timeout_s),
            designs,
        ))

    cl = np.asarray([r["CL"] for r in aero], dtype=float)
    cd = np.asarray([r["CD"] for r in aero], dtype=float)
    cmy = np.asarray([r["CMy"] for r in aero], dtype=float)
    failed = ~np.isfinite(cl) | ~np.isfinite(cd) | ~np.isfinite(cmy)

    f1, f2, g1 = cd.copy(), -cl, np.abs(cmy) - _cm_max
    f1[failed] = f2[failed] = g1[failed] = 1.0e6

    return np.column_stack((f1, f2)), g1.reshape(n, 1)


BWB_VSP_AERO = {
    "name": "BWB_VSP_AERO",
    "func": _bwb_aero,
    "f": [], 
    "g": [],
    "nvar": len(_geom_names),
    "nf": 2, "nc": 1,
    "bounds": _bounds,
    "var_names": _geom_names,
    "random_variables": [],
}


def main() -> None:
    mp.set_executable(sys.executable)
    rng = np.random.default_rng(0)
    lower = np.asarray([b[0] for b in _bounds])
    upper = np.asarray([b[1] for b in _bounds])
    X = lower + rng.random((2000, len(_bounds))) * (upper - lower)
    F, G = _bwb_aero(X)
    print("F =\n", F)
    print("G =\n", G)


if __name__ == "__main__":
    main()
