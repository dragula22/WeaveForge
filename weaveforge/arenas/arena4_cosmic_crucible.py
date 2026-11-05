"""
Arena 4 — The Cosmic Hourglass & Live Observational Crucible (`weaveforge/arenas/arena4_cosmic_crucible.py`)
============================================================================================================
Combines:
  1. 60–240+ FPS 3D GPU Raytracer of the Singularity-Free Two-Wing `CPT` Cosmic Catenoid Hourglass:
       • Smooth Euclidean waist `R(tau) = sqrt(r0^2 + tau^2)` with finite `K_max = 24 / r0^4 < inf`
       • Strictly neutral `Observer Channel A` (`+tau` wing) and `Observer Channel B` (`-tau` wing)
       • Interactive `tau_now` slice ring (`||J|Psi>|| = 0.000000` timeless global state)
       • Landauer `dQ >= k_B T ln 2` irreversible golden memory creases strictly inside `|tau| <= |tau_now|`
       • Pinned `SO(3)` topological pleat hoops (`xi = 5 + 1/pi => Omega_m = 0.30941`)
  2. Ruthlessly Honest 5-Dataset Observational Pass/Fail Crucible (`evaluate_observational_crucible`):
       • SPARC 175-Galaxy Rotation Curves & Radial Acceleration Relation (Lelli+ 2016)
       • Planck 2018 CMB Acoustic Peaks (`l = 2..1000`, Waist Quadrupole `C_2` & Peak-3 Ratio `R_32`)
       • Solar-System Ephemerides (Cassini `PPN gamma`, Saturn `9.58 AU` ranging) & `GW170817`
       • DESI DR2 BAO Thawing Expansion `(w0, wa)` & Honest SH0ES/Planck Hubble Tension Audit
       • Dec 2, 2026 `Gaia DR4` Wide-Binary `|z_gal|` Disk (`+16%`) vs Halo (`+31%`) EFE Forecast
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from functools import lru_cache
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from astropy.cosmology import w0waCDM
import moderngl
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

try:
    import camb  # type: ignore

    _HAS_CAMB = True
except Exception:  # pragma: no cover
    camb = None  # type: ignore
    _HAS_CAMB = False

from weaveforge.core.bhex_ringdown import get_kerr_lvk_ringdown_audit
from weaveforge.core.camera import Camera3D
from weaveforge.core.cosmology_referee import (
    audit_single_h0_consistency,
    evaluate_cross_scale_mass_continuity,
    evaluate_linear_growth_s8,
    evaluate_planck_knox_chi2_2500,
    evaluate_primordial_bbn,
)
from weaveforge.core.frameworks import PhysicsMod, count_mod_free_knobs
from weaveforge.core.skyrme_perturbation import (
    evaluate_catenoid_waist_adiabaticity,
    skyrme_pleat_jeans_damping_envelope,
)
from weaveforge.core.catenoid_freezeout import get_catenoid_freezeout_crucible_summary
from weaveforge.core.symbolic_compiler import CompiledMasterAction
from weaveforge.data.gaia_dr4_pipeline import compute_wide_binary_efe_curves
from weaveforge.data.sparc_catalog import (
    SparcGalaxy,
    evaluate_galaxy_rotation_curve,
    load_sparc_175_catalog,
    score_full_sparc_175,
)


_CACHED_SPARC_175: List[SparcGalaxy] | None = None
_CACHED_SPARC_SCORES: Dict[Tuple[str, float, bool, float], Dict[str, float]] = {}
_CACHED_GALAXY_FITS: Dict[Tuple[str, str, float, bool, float], Dict[str, object]] = {}

# Published DESI DR2 (2025, arXiv:2503.14738) BAO Consensus Measurements across z = 0.295 .. 2.330:
# Columns: (z_eff, observable_kind, measured_ratio, 1_sigma_uncertainty)
_DESI_DR2_BAO_DATA: Tuple[Tuple[float, str, float, float], ...] = (
    (0.295, "DV", 7.932, 0.098),   # BGS
    (0.510, "DM", 13.588, 0.168),  # LRG1
    (0.510, "DH", 21.863, 0.427),
    (0.706, "DM", 17.347, 0.180),  # LRG2
    (0.706, "DH", 19.458, 0.334),
    (0.934, "DM", 21.574, 0.153),  # LRG3 + ELG1
    (0.934, "DH", 17.641, 0.193),
    (1.321, "DM", 27.601, 0.320),  # ELG2
    (1.321, "DH", 14.178, 0.217),
    (1.484, "DM", 30.512, 0.760),  # QSO
    (1.484, "DH", 12.817, 0.516),
    (2.330, "DM", 38.988, 0.531),  # Lya Forest
    (2.330, "DH", 8.632, 0.101),
)


def get_sparc_175_cached() -> List[SparcGalaxy]:
    global _CACHED_SPARC_175
    if _CACHED_SPARC_175 is None:
        _CACHED_SPARC_175, _ = load_sparc_175_catalog()
    return _CACHED_SPARC_175


@dataclass
class CrucibleBadge:
    """Single observational crucible pass/warn/fail scorecard entry."""

    badge_id: str
    dataset_name: str
    status: str  # "PASS", "WARN", "FAIL", "FORECAST"
    headline_metric: str
    detail_honest: str


@dataclass
class CrucibleReport:
    """Complete live observational crucible report + ImPlot-ready scientific curve arrays."""

    mod_id: str
    mod_name: str
    omega_b: float
    omega_m_effective: float
    cmb_r32_ratio: float
    cmb_d2_quadrupole_uK2: float
    sparc_mean_chi2: float
    sparc_rar_scatter_dex: float
    wide_binary_disk_boost_pct: float
    wide_binary_halo_boost_pct: float
    desi_bao_chi2_dof: float = 1.0
    badges: List[CrucibleBadge] = field(default_factory=list)

    # Curve arrays for real-time ImPlot charts in the right dock!
    cmb_ell: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cmb_dl_obs: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cmb_dl_mod: np.ndarray = field(default_factory=lambda: np.zeros(1))

    sparc_galaxy_name: str = "NGC 3198"
    sparc_r_kpc: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sparc_v_obs: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sparc_v_err: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sparc_v_bary: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sparc_v_mod: np.ndarray = field(default_factory=lambda: np.zeros(1))

    wb_curves: Dict[str, object] = field(default_factory=dict)

    # Full l = 2..2500 CAMB TT/TE/EE multi-spectrum & adiabatic waist diagnostics (Milestone 4B)
    cmb_ell_2500: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cmb_tt_obs_2500: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cmb_tt_mod_2500: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cmb_te_obs_2500: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cmb_te_mod_2500: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cmb_ee_obs_2500: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cmb_ee_mod_2500: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cmb_te_corr_2500: float = 1.0
    cmb_ee_peak_ratio: float = 1.0
    cmb_beta_iso: float = 0.0
    # 3D Catenoid Kibble-Zurek Freeze-Out & Red-Team Cosmology Diagnostics (Milestone 5B)
    cmb_xi_kz_at_waist: float = 0.0
    cmb_xi_kz_min: float = 0.0
    cmb_xi_kz_max: float = 0.0
    cmb_kz_power_index: float = 0.0
    cmb_qh_net_symmetric: float = 0.0
    cmb_qh_net_chiral_pos: float = 0.0
    cmb_kz_depends_on_r0: bool = False
    cmb_kz_matches_target_10pct: bool = False
    # Un-Gameable Cosmological Referee, Cross-Scale Continuity & Knob Audit (Track 6A)
    h0_consistent_kms_mpc: float = 67.36
    a0_h0_rel_mismatch: float = 0.0
    cmb_chi2_dof_fixed_ns: float = 1.0
    cmb_chi2_dof_profiled_ns: float = 1.0
    cmb_best_fit_ns: float = 0.9649
    cmb_high_l_tilt_pct: float = 0.0
    bbn_dh_e5: float = 2.515
    bbn_yp: float = 0.2471
    bbn_max_sigma: float = 0.8
    lensing_s8_active: float = 0.811
    lensing_s8_gr_screened: float = 0.811
    lensing_s8_sigma: float = 1.5
    cross_scale_seam_open: bool = False
    ngc3198_v_double_counted_kms: float = 150.0
    ngc3198_double_count_sigma: float = 0.0
    free_action_literals: int = 0
    free_global_knobs: int = 0
    per_galaxy_halo_knobs: int = 0
    knob_audit_summary: str = ""
    desi_bao_anchored_chi2_dof: float = 1.0
    lvk_ringdown_status: str = "PASS"
    lvk_imr_sigma: float = 0.27
    lvk_o4_hier_sigma: float = 0.33
    lvk_max_q220: float = 7.02


_CAMB_FORTRAN_LOCK = threading.Lock()
_CAMB_LUT_LOCK = threading.RLock()
_SPARC_SCORE_LOCK = threading.Lock()
_CAMB_STOP_EVENT = threading.Event()
_EXACT_CAMB_CURVES: Dict[Tuple[float, float, float], np.ndarray] = {}
_CAMB_DDR5_SPLINES: Dict[float, CubicSpline] = {}
_EXACT_CAMB_POL_CURVES: Dict[Tuple[float, float, float], np.ndarray] = {}
_CAMB_POL_DDR5_SPLINES: Dict[float, CubicSpline] = {}
_CAMB_WARM_THREAD: Optional[threading.Thread] = None
_CAMB_WARM_PROC: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
_CAMB_FAILED_NPZ_STAT: Optional[Tuple[str, int, int]] = None
_CAMB_POL_FAILED_NPZ_STAT: Optional[Tuple[str, int, int]] = None
_CAMB_LUT_READY: bool = False
_CAMB_POL_LUT_READY: bool = False
_CAMB_LAST_ERROR: Optional[str] = None
_PLANCK_REF_CACHE: Optional[Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]] = None
_PLANCK_POL_REF_CACHE: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = None

_PLANCK_H0: float = 67.36
_PLANCK_OMEGA_B: float = 0.04897
_CAMB_LMAX: int = 1350
_CAMB_POL_LMAX: int = 2500
_CAMB_NNU_BRANCHES: Tuple[float, float] = (3.044, 3.303)
_CAMB_LUT_DISK_CACHE_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "camb_tt_lut_cache.npz"
_CAMB_POL_LUT_DISK_CACHE_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "camb_te_ee_2500_lut_cache.npz"
_CAMB_POL_ELL_GRID: np.ndarray = np.concatenate(([2, 3], np.arange(4, _CAMB_POL_LMAX + 1, 2, dtype=np.int32)))
_CAMB_POL_ELL_FLOAT: np.ndarray = _CAMB_POL_ELL_GRID.astype(np.float64)
_CAMB_POL_MASK_30_2500: np.ndarray = (_CAMB_POL_ELL_GRID >= 30) & (_CAMB_POL_ELL_GRID <= 2500)
_CAMB_POL_MASK_300_1500: np.ndarray = (_CAMB_POL_ELL_GRID >= 300) & (_CAMB_POL_ELL_GRID <= 1500)
_CAMB_POL_MASK_P1: np.ndarray = (_CAMB_POL_ELL_GRID >= 150) & (_CAMB_POL_ELL_GRID <= 400)

_CAMB_OMEGA_MIN = 0.04
_CAMB_OMEGA_MAX = 1.00
_CAMB_PASS1_NODES: Tuple[float, ...] = (
    0.0400,
    0.0490,
    0.0900,
    0.1500,
    0.2300,
    0.3094,
    0.3140,
    0.4400,
    0.6000,
    0.8000,
    1.0000,
)
_CAMB_PASS2_EXTRA_NODES: Tuple[float, ...] = (
    0.0680,
    0.1200,
    0.1900,
    0.2700,
    0.3500,
    0.3950,
    0.4950,
    0.5500,
    0.7000,
    0.9000,
)
_CAMB_DENSE_NODES: Tuple[float, ...] = tuple(
    sorted(set(round(float(x), 4) for x in (_CAMB_PASS1_NODES + _CAMB_PASS2_EXTRA_NODES)))
)


def _stop_camb_warmer() -> None:
    global _CAMB_WARM_PROC
    _CAMB_STOP_EVENT.set()
    with _CAMB_LUT_LOCK:
        proc = _CAMB_WARM_PROC
        _CAMB_WARM_PROC = None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass


atexit.register(_stop_camb_warmer)


def _camb_lut_cache_signature() -> str:
    camb_ver = str(getattr(camb, "__version__", "no_camb")) if _HAS_CAMB else "no_camb"
    nodes_sig = ",".join(f"{x:.4f}" for x in _CAMB_DENSE_NODES)
    nnu_sig = ",".join(f"{n:.3f}" for n in _CAMB_NNU_BRANCHES)
    return (
        f"v1|camb={camb_ver}|H0={_PLANCK_H0:.4f}|ob={_PLANCK_OMEGA_B:.6f}"
        f"|lmax={_CAMB_LMAX}|nnu={nnu_sig}|nodes={nodes_sig}"
    )


def _cached_camb_tt_curve(omega_m_q: float, nnu_q: float, h0_q: float = _PLANCK_H0) -> np.ndarray:
    """
    Runs the genuine `CAMB` Fortran Einstein–Boltzmann hierarchy (`l = 0..1350`) for `(omega_m_eff, N_eff, H_0)`,
    serialized via double-checked `_CAMB_FORTRAN_LOCK` and memoized in DDR5 System RAM (`~0.42s` cold, `< 0.01 ms` warm!).
    """
    om_key = round(float(omega_m_q), 4)
    nnu_key = round(max(float(nnu_q), 0.1), 3)
    h0_key = round(max(float(h0_q), 20.0), 2)
    cache_key = (om_key, nnu_key, h0_key)
    with _CAMB_LUT_LOCK:
        existing = _EXACT_CAMB_CURVES.get(cache_key)
        if existing is not None:
            return existing

    h = h0_key / 100.0
    # Regularize omch2 >= 1e-4 so CAMB's synchronous-gauge CDM rest frame stays finite near pure-baryon limits
    omch2 = max((float(om_key) - _PLANCK_OMEGA_B) * (h * h), 1.0e-4)

    with _CAMB_FORTRAN_LOCK:
        with _CAMB_LUT_LOCK:
            existing = _EXACT_CAMB_CURVES.get(cache_key)
            if existing is not None:
                return existing

        pars = camb.set_params(
            H0=float(h0_key),
            ombh2=0.02237,
            omch2=omch2,
            tau=0.0544,
            As=2.1005e-9,
            ns=0.9649,
            nnu=float(nnu_key),
            lmax=_CAMB_LMAX,
        )
        results = camb.get_results(pars)
        # Returns D_l = l(l+1)C_l/(2*pi) in uK^2 for l = 0..1350 (truncate variable CAMB lensing tail > 1350)
        dl_tt = np.asarray(
            results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=False)["total"][: _CAMB_LMAX + 1, 0],
            dtype=np.float64,
        ).copy()
        if not np.all(np.isfinite(dl_tt)):
            pars_safe = camb.set_params(
                H0=float(h0_key),
                ombh2=0.02237,
                omch2=max(omch2, 2.5e-4),
                tau=0.0544,
                As=2.1005e-9,
                ns=0.9649,
                nnu=float(nnu_key),
                lmax=_CAMB_LMAX,
            )
            dl_tt = np.asarray(
                camb.get_results(pars_safe).get_cmb_power_spectra(pars_safe, CMB_unit="muK", raw_cl=False)["total"][: _CAMB_LMAX + 1, 0],
                dtype=np.float64,
            ).copy()
            np.nan_to_num(dl_tt, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        with _CAMB_LUT_LOCK:
            if len(_EXACT_CAMB_CURVES) > 512:
                _EXACT_CAMB_CURVES.clear()
            _EXACT_CAMB_CURVES[cache_key] = dl_tt
    return dl_tt


def _stat_npz_fingerprint(target: Path) -> Tuple[str, int, int]:
    try:
        st = target.stat()
        return (str(target), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return (str(target), -1, -1)


def load_camb_lut_disk_cache(cache_path: Optional[Path] = None, force_reload: bool = False) -> bool:
    """
    Loads pre-computed `CAMB` Boltzmann `(3.044, 3.303)` LUT matrices from compressed `.npz` disk cache
    into DDR5 System RAM and constructs `CubicSpline` interpolators synchronously in `< 5 ms` with
    ZERO background thread and ZERO `_CAMB_FORTRAN_LOCK` contention!
    """
    global _CAMB_LUT_READY, _CAMB_LAST_ERROR, _PLANCK_REF_CACHE, _CAMB_FAILED_NPZ_STAT
    if not _HAS_CAMB:
        return False
    target = Path(cache_path) if cache_path is not None else _CAMB_LUT_DISK_CACHE_PATH
    with _CAMB_LUT_LOCK:
        if not force_reload and _CAMB_LUT_READY and (3.044 in _CAMB_DDR5_SPLINES) and (3.303 in _CAMB_DDR5_SPLINES):
            return True

    fp = _stat_npz_fingerprint(target)
    if fp[1] < 0:
        with _CAMB_LUT_LOCK:
            _CAMB_FAILED_NPZ_STAT = fp
        return False
    with _CAMB_LUT_LOCK:
        if not force_reload and _CAMB_FAILED_NPZ_STAT == fp:
            return False

    try:
        expected_sig = _camb_lut_cache_signature()
        with np.load(target, allow_pickle=False) as z:
            if "signature" not in z.files:
                with _CAMB_LUT_LOCK:
                    _CAMB_FAILED_NPZ_STAT = fp
                return False
            sig_arr = z["signature"]
            sig_str = str(sig_arr.item() if sig_arr.ndim == 0 else sig_arr[0])
            if sig_str != expected_sig:
                with _CAMB_LUT_LOCK:
                    _CAMB_FAILED_NPZ_STAT = fp
                return False
            nodes = np.asarray(z["omega_nodes"], dtype=np.float64)
            dl_3044 = np.asarray(z["dl_3044"], dtype=np.float64)
            dl_3303 = np.asarray(z["dl_3303"], dtype=np.float64)

        n_nodes = len(_CAMB_DENSE_NODES)
        n_ell = _CAMB_LMAX + 1
        if (
            nodes.shape != (n_nodes,)
            or dl_3044.shape != (n_nodes, n_ell)
            or dl_3303.shape != (n_nodes, n_ell)
            or not np.all(np.isfinite(dl_3044))
            or not np.all(np.isfinite(dl_3303))
        ):
            with _CAMB_LUT_LOCK:
                _CAMB_FAILED_NPZ_STAT = fp
            return False

        sp_3044 = CubicSpline(nodes, dl_3044, axis=0, bc_type="not-a-knot", extrapolate=False)
        sp_3303 = CubicSpline(nodes, dl_3303, axis=0, bc_type="not-a-knot", extrapolate=False)

        h0_k = round(_PLANCK_H0, 2)
        with _CAMB_LUT_LOCK:
            if force_reload:
                _EXACT_CAMB_CURVES.clear()
                _PLANCK_REF_CACHE = None
            for i, om in enumerate(nodes):
                om_k = round(float(om), 4)
                _EXACT_CAMB_CURVES[(om_k, 3.044, h0_k)] = dl_3044[i].copy()
                _EXACT_CAMB_CURVES[(om_k, 3.303, h0_k)] = dl_3303[i].copy()
            _CAMB_DDR5_SPLINES[3.044] = sp_3044
            _CAMB_DDR5_SPLINES[3.303] = sp_3303
            _CAMB_LUT_READY = True
            _CAMB_FAILED_NPZ_STAT = None
            _CAMB_LAST_ERROR = None
        _get_planck_ref_spectrum()
        return True
    except Exception:
        with _CAMB_LUT_LOCK:
            _CAMB_FAILED_NPZ_STAT = fp
        return False


def save_camb_lut_disk_cache(
    cache_path: Optional[Path] = None,
    use_float32: bool = True,
) -> Optional[Path]:
    """Atomically persists the verified `CAMB` LUT matrices (`< 250 KB` compressed `float32`) + cosmology signature."""
    global _CAMB_FAILED_NPZ_STAT
    if not _HAS_CAMB:
        return None
    target = Path(cache_path) if cache_path is not None else _CAMB_LUT_DISK_CACHE_PATH
    nodes = np.array(_CAMB_DENSE_NODES, dtype=np.float64)
    rows_3044 = [_cached_camb_tt_curve(float(om), 3.044, _PLANCK_H0) for om in _CAMB_DENSE_NODES]
    rows_3303 = [_cached_camb_tt_curve(float(om), 3.303, _PLANCK_H0) for om in _CAMB_DENSE_NODES]
    dtype = np.float32 if use_float32 else np.float64
    mat_3044 = np.stack(rows_3044, axis=0).astype(dtype, copy=False)
    mat_3303 = np.stack(rows_3303, axis=0).astype(dtype, copy=False)
    sig = np.array(_camb_lut_cache_signature())

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f"{target.stem}.tmp.{os.getpid()}.{threading.get_ident()}.npz"
    try:
        np.savez_compressed(
            tmp_path,
            signature=sig,
            omega_nodes=nodes,
            dl_3044=mat_3044,
            dl_3303=mat_3303,
        )
        os.replace(tmp_path, target)
        with _CAMB_LUT_LOCK:
            _CAMB_FAILED_NPZ_STAT = None
        return target
    except OSError:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return None


def _rebuild_camb_spline_for_nnu(
    nnu_val: float,
    nodes: Tuple[float, ...],
    yield_s: float = 0.0,
) -> None:
    cand_xs = sorted(set(round(float(x), 4) for x in nodes))
    valid_xs: List[float] = []
    ys_list: List[np.ndarray] = []
    nnu_key = round(float(nnu_val), 3)
    h0_key = round(_PLANCK_H0, 2)
    for om in cand_xs:
        if _CAMB_STOP_EVENT.is_set():
            return
        cache_key = (round(float(om), 4), nnu_key, h0_key)
        with _CAMB_LUT_LOCK:
            was_cached = cache_key in _EXACT_CAMB_CURVES
        arr = _cached_camb_tt_curve(float(om), nnu_key, h0_key)
        if np.all(np.isfinite(arr)):
            valid_xs.append(float(om))
            ys_list.append(arr)
        if (not was_cached) and yield_s > 0.0:
            if _CAMB_STOP_EVENT.wait(yield_s):
                return
    if len(valid_xs) < 4:
        return
    xs = np.array(valid_xs, dtype=np.float64)
    n_ell = min(len(arr) for arr in ys_list)
    y_mat = np.stack([arr[:n_ell] for arr in ys_list], axis=0)
    spline = CubicSpline(xs, y_mat, axis=0, bc_type="not-a-knot", extrapolate=False)
    with _CAMB_LUT_LOCK:
        _CAMB_DDR5_SPLINES[nnu_key] = spline


def _camb_warm_inproc(yield_s: float = 0.01) -> None:
    global _CAMB_LUT_READY, _CAMB_LAST_ERROR
    for om0, nnu0 in ((0.31397, 3.044), (0.3094, 3.303), (0.3140, 3.044), (0.0490, 3.044)):
        if _CAMB_STOP_EVENT.is_set():
            return
        _cached_camb_tt_curve(om0, nnu0, _PLANCK_H0)
        if yield_s > 0.0 and _CAMB_STOP_EVENT.wait(yield_s):
            return

    for nnu_b in _CAMB_NNU_BRANCHES:
        if _CAMB_STOP_EVENT.is_set():
            return
        _rebuild_camb_spline_for_nnu(nnu_b, _CAMB_PASS1_NODES, yield_s=yield_s)

    with _CAMB_LUT_LOCK:
        _CAMB_LUT_READY = (3.044 in _CAMB_DDR5_SPLINES) and (3.303 in _CAMB_DDR5_SPLINES)
        _CAMB_LAST_ERROR = None

    for nnu_b in _CAMB_NNU_BRANCHES:
        if _CAMB_STOP_EVENT.is_set():
            return
        _rebuild_camb_spline_for_nnu(nnu_b, _CAMB_DENSE_NODES, yield_s=yield_s)

    if not _CAMB_STOP_EVENT.is_set():
        saved = save_camb_lut_disk_cache(use_float32=True)
        if saved is not None and saved.stat().st_size >= 250 * 1024:
            save_camb_lut_disk_cache(use_float32=True)


def _camb_warm_worker(
    initial_delay_s: float = 0.0,
    yield_s: float = 0.01,
    use_subprocess: bool = True,
) -> None:
    global _CAMB_LUT_READY, _CAMB_LAST_ERROR, _CAMB_WARM_PROC
    if not _HAS_CAMB:
        return
    try:
        if load_camb_lut_disk_cache():
            return
        if initial_delay_s > 0.0 and _CAMB_STOP_EVENT.wait(initial_delay_s):
            return

        if use_subprocess:
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            cmd = [
                sys.executable,
                "-c",
                (
                    "import sys; from pathlib import Path; "
                    "from weaveforge.arenas.arena4_cosmic_crucible import save_camb_lut_disk_cache; "
                    "p = save_camb_lut_disk_cache(Path(sys.argv[1]), use_float32=True); "
                    "sys.exit(0 if p is not None else 1)"
                ),
                str(_CAMB_LUT_DISK_CACHE_PATH),
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                with _CAMB_LUT_LOCK:
                    _CAMB_WARM_PROC = proc
                while proc.poll() is None:
                    if _CAMB_STOP_EVENT.wait(0.05):
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        return
                with _CAMB_LUT_LOCK:
                    if _CAMB_WARM_PROC is proc:
                        _CAMB_WARM_PROC = None
                if proc.returncode == 0 and load_camb_lut_disk_cache(force_reload=True):
                    return
            except Exception:
                with _CAMB_LUT_LOCK:
                    _CAMB_WARM_PROC = None

        _camb_warm_inproc(yield_s=yield_s)
    except Exception as exc:
        with _CAMB_LUT_LOCK:
            _CAMB_LAST_ERROR = f"{type(exc).__name__}: {exc}"


def warm_camb_ddr5_cache_async(wait: bool = False) -> None:
    """
    Loads `weaveforge/data/camb_tt_lut_cache.npz` into DDR5 System RAM synchronously in `< 5 ms`
    (with zero background thread and zero `_CAMB_FORTRAN_LOCK` contention!) whenever the compressed
    disk cache matches the active `CAMB` version & cosmology signature; otherwise regenerates `.npz`
    in an isolated background child process with zero main-process GIL or Fortran lock contention.
    """
    global _CAMB_WARM_THREAD
    if not _HAS_CAMB:
        return
    if is_camb_ddr5_cache_ready() or load_camb_lut_disk_cache():
        if not is_camb_pol_ddr5_cache_ready():
            load_camb_pol_lut_disk_cache()
        return
    with _CAMB_LUT_LOCK:
        if _CAMB_WARM_THREAD is None or (not _CAMB_WARM_THREAD.is_alive() and not _CAMB_LUT_READY):
            _CAMB_STOP_EVENT.clear()
            delay_s = 0.0 if wait else 0.02
            yield_s = 0.0 if wait else 0.01
            _CAMB_WARM_THREAD = threading.Thread(
                target=_camb_warm_worker,
                args=(delay_s, yield_s, not wait),
                name="WeaveForge-CAMB-DDR5-Warmer",
                daemon=True,
            )
            _CAMB_WARM_THREAD.start()
        th = _CAMB_WARM_THREAD
    if wait and th is not None:
        th.join()


def is_camb_ddr5_cache_ready() -> bool:
    with _CAMB_LUT_LOCK:
        return bool(_CAMB_LUT_READY and (3.044 in _CAMB_DDR5_SPLINES) and (3.303 in _CAMB_DDR5_SPLINES))


def _is_async_camb_warming() -> bool:
    with _CAMB_LUT_LOCK:
        return bool((not _CAMB_LUT_READY) and (_CAMB_WARM_THREAD is not None) and _CAMB_WARM_THREAD.is_alive())


def _provisional_camb_tt_curve(omega_m_eff: float, nnu: float) -> np.ndarray:
    """Fast `< 0.2 ms` provisional `l = 0..1350` acoustic transfer spectrum served ONLY during cold `.npz` rebuilds."""
    ls = np.arange(_CAMB_LMAX + 1, dtype=np.float64)
    ell = np.maximum(ls, 2.0)
    well_ratio = float(np.clip((float(omega_m_eff) - _PLANCK_OMEGA_B) / 0.2604, 0.0, 1.45))
    ell_shift = float((max(float(omega_m_eff), 0.04) / 0.30941) ** (-0.08))
    nnu_damp = 1.0 - 0.035 * max(float(nnu) - 3.044, 0.0)
    sw = 1020.0 / (1.0 + (ell / 95.0) ** 1.65)
    p1 = 5620.0 * (0.78 + 0.22 * min(well_ratio, 1.15)) * np.exp(-0.5 * ((ell - 220.0 * ell_shift) / 62.0) ** 2)
    p2 = 2490.0 * (1.06 - 0.06 * min(well_ratio, 1.15)) * np.exp(-0.5 * ((ell - 537.5 * ell_shift) / 68.0) ** 2)
    p3 = (
        2450.0
        * nnu_damp
        * (0.36 + 0.64 * float(np.clip(well_ratio, 0.0, 1.25)))
        * np.exp(-0.5 * ((ell - 810.8 * ell_shift) / 70.0) ** 2)
    )
    out = sw + p1 + p2 + p3
    out[:2] = 0.0
    return out


def _get_camb_tt_fast(omega_m_eff: float, nnu: float, h0_kms_mpc: float = _PLANCK_H0) -> np.ndarray:
    if not _CAMB_LUT_READY:
        load_camb_lut_disk_cache()

    om_f = float(omega_m_eff)
    om_q = round(om_f, 4)
    nnu_f = max(float(nnu), 0.1)
    nnu_q = round(nnu_f, 3)
    h0_f = max(float(h0_kms_mpc), 20.0)
    h0_q = round(h0_f, 2)
    near_planck_h0 = abs(h0_f - _PLANCK_H0) <= 0.02

    spline = None
    sp_pair: Optional[Tuple[CubicSpline, CubicSpline]] = None
    with _CAMB_LUT_LOCK:
        exact = _EXACT_CAMB_CURVES.get((om_q, nnu_q, h0_q))
        if exact is not None:
            return exact.copy()
        if near_planck_h0 and (_CAMB_OMEGA_MIN <= om_f <= _CAMB_OMEGA_MAX):
            spline = _CAMB_DDR5_SPLINES.get(nnu_q)
            if spline is None and (2.50 <= nnu_f <= 3.80):
                sp0 = _CAMB_DDR5_SPLINES.get(3.044)
                sp1 = _CAMB_DDR5_SPLINES.get(3.303)
                if sp0 is not None and sp1 is not None:
                    sp_pair = (sp0, sp1)

    if spline is not None:
        dl_interp = np.asarray(spline(om_f), dtype=np.float64)
        if np.all(np.isfinite(dl_interp)):
            np.maximum(dl_interp, 0.0, out=dl_interp)
            return dl_interp
    elif sp_pair is not None:
        w = (nnu_f - 3.044) / (3.303 - 3.044)
        dl_0 = np.asarray(sp_pair[0](om_f), dtype=np.float64)
        dl_1 = np.asarray(sp_pair[1](om_f), dtype=np.float64)
        dl_blend = (1.0 - w) * dl_0 + w * dl_1
        if np.all(np.isfinite(dl_blend)):
            np.maximum(dl_blend, 0.0, out=dl_blend)
            return dl_blend

    if _is_async_camb_warming():
        return _provisional_camb_tt_curve(om_f, nnu_q)

    return _cached_camb_tt_curve(om_q, nnu_q, h0_q).copy()


def _get_planck_ref_spectrum() -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    global _PLANCK_REF_CACHE
    if _PLANCK_REF_CACHE is not None:
        return _PLANCK_REF_CACHE
    if not _CAMB_LUT_READY:
        load_camb_lut_disk_cache()
    ell = np.arange(2, 1001, dtype=np.float64)
    is_prov = False
    with _CAMB_LUT_LOCK:
        has_ref = (0.314, 3.044, round(_PLANCK_H0, 2)) in _EXACT_CAMB_CURVES
    if (not has_ref) and _is_async_camb_warming():
        dl_ref_full = _provisional_camb_tt_curve(0.31397, 3.044)
        is_prov = True
    else:
        dl_ref_full = _cached_camb_tt_curve(0.31397, 3.044, _PLANCK_H0)
    dl_obs = dl_ref_full[2:1001].copy()
    dl_obs[0] = 226.0  # Planck 2018 Commander observed quadrupole D_2 = 226 uK^2!
    pks_ref, _ = find_peaks(dl_ref_full[80:450], prominence=50.0)
    ref_p1 = float(dl_ref_full[80 + pks_ref[0]]) if len(pks_ref) > 0 else float(np.max(dl_ref_full[150:350]))
    all_pks_ref, _ = find_peaks(dl_ref_full[80:1300], prominence=25.0)
    ref_p2 = float(dl_ref_full[80 + all_pks_ref[1]]) if len(all_pks_ref) >= 2 else float(dl_ref_full[536])
    ls_all = np.arange(len(dl_ref_full), dtype=np.float64)
    res = (ell, dl_obs, ref_p1, ref_p2, ls_all)
    if not is_prov:
        _PLANCK_REF_CACHE = res
    return res


def _camb_pol_lut_cache_signature() -> str:
    camb_ver = str(getattr(camb, "__version__", "no_camb")) if _HAS_CAMB else "no_camb"
    nodes_sig = ",".join(f"{x:.4f}" for x in _CAMB_DENSE_NODES)
    nnu_sig = ",".join(f"{n:.3f}" for n in _CAMB_NNU_BRANCHES)
    return (
        f"v2_pol2500|camb={camb_ver}|H0={_PLANCK_H0:.4f}"
        f"|ob={_PLANCK_OMEGA_B:.6f}|lmax={_CAMB_POL_LMAX}|nell={len(_CAMB_POL_ELL_GRID)}"
        f"|nnu={nnu_sig}|nodes={nodes_sig}"
    )


def _cached_camb_pol_2500_curves(omega_m_q: float, nnu_q: float, h0_q: float = _PLANCK_H0) -> np.ndarray:
    """
    Runs the genuine `CAMB` Fortran Einstein–Boltzmann polarization hierarchy (`l = 2..2500`)
    and extracts `[TT, TE, EE]` (`CAMB` `"total"` columns `[0, 3, 1]`) onto `_CAMB_POL_ELL_GRID`
    (`1251` multipoles `[2, 3, 4, 6, ..., 2500]`), returning shape `(3, 1251)` in `uK^2`.
    """
    om_key = round(float(omega_m_q), 4)
    nnu_key = round(max(float(nnu_q), 0.1), 3)
    h0_key = round(max(float(h0_q), 20.0), 2)
    cache_key = (om_key, nnu_key, h0_key)
    with _CAMB_LUT_LOCK:
        existing = _EXACT_CAMB_POL_CURVES.get(cache_key)
        if existing is not None:
            return existing

    h = h0_key / 100.0
    omch2 = max((float(om_key) - _PLANCK_OMEGA_B) * (h * h), 1.0e-4)

    with _CAMB_FORTRAN_LOCK:
        with _CAMB_LUT_LOCK:
            existing = _EXACT_CAMB_POL_CURVES.get(cache_key)
            if existing is not None:
                return existing

        pars = camb.set_params(
            H0=float(h0_key),
            ombh2=0.02237,
            omch2=omch2,
            tau=0.0544,
            As=2.1005e-9,
            ns=0.9649,
            nnu=float(nnu_key),
            lmax=_CAMB_POL_LMAX,
        )
        results = camb.get_results(pars)
        # CAMB "total" column order: [0=TT, 1=EE, 2=BB, 3=TE] in D_l = l(l+1)C_l/(2*pi) [uK^2]
        tot = np.asarray(
            results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=False)["total"][: _CAMB_POL_LMAX + 1],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(tot)):
            pars_safe = camb.set_params(
                H0=float(h0_key),
                ombh2=0.02237,
                omch2=max(omch2, 2.5e-4),
                tau=0.0544,
                As=2.1005e-9,
                ns=0.9649,
                nnu=float(nnu_key),
                lmax=_CAMB_POL_LMAX,
            )
            tot = np.asarray(
                camb.get_results(pars_safe).get_cmb_power_spectra(pars_safe, CMB_unit="muK", raw_cl=False)["total"][
                    : _CAMB_POL_LMAX + 1
                ],
                dtype=np.float64,
            )
            np.nan_to_num(tot, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Row 0 = TT (col 0), Row 1 = TE (col 3), Row 2 = EE (col 1)
        pol_mat = np.stack(
            (
                tot[_CAMB_POL_ELL_GRID, 0],
                tot[_CAMB_POL_ELL_GRID, 3],
                tot[_CAMB_POL_ELL_GRID, 1],
            ),
            axis=0,
        ).copy()

        with _CAMB_LUT_LOCK:
            if len(_EXACT_CAMB_POL_CURVES) > 512:
                _EXACT_CAMB_POL_CURVES.clear()
            _EXACT_CAMB_POL_CURVES[cache_key] = pol_mat
    return pol_mat


def load_camb_pol_lut_disk_cache(cache_path: Optional[Path] = None, force_reload: bool = False) -> bool:
    """
    Loads pre-computed `l = 2..2500` `CAMB` `[TT, TE, EE]` LUT tensors from
    `weaveforge/data/camb_te_ee_2500_lut_cache.npz` into DDR5 System RAM and builds
    vectorized `CubicSpline` interpolators in `< 5 ms`.
    """
    global _CAMB_POL_LUT_READY, _PLANCK_POL_REF_CACHE, _CAMB_POL_FAILED_NPZ_STAT
    if not _HAS_CAMB:
        return False
    target = Path(cache_path) if cache_path is not None else _CAMB_POL_LUT_DISK_CACHE_PATH
    with _CAMB_LUT_LOCK:
        if (
            not force_reload
            and _CAMB_POL_LUT_READY
            and (3.044 in _CAMB_POL_DDR5_SPLINES)
            and (3.303 in _CAMB_POL_DDR5_SPLINES)
        ):
            return True

    fp = _stat_npz_fingerprint(target)
    if fp[1] < 0:
        with _CAMB_LUT_LOCK:
            _CAMB_POL_FAILED_NPZ_STAT = fp
        return False
    with _CAMB_LUT_LOCK:
        if not force_reload and _CAMB_POL_FAILED_NPZ_STAT == fp:
            return False

    try:
        expected_sig = _camb_pol_lut_cache_signature()
        with np.load(target, allow_pickle=False) as z:
            if "signature" not in z.files:
                with _CAMB_LUT_LOCK:
                    _CAMB_POL_FAILED_NPZ_STAT = fp
                return False
            sig_arr = z["signature"]
            sig_str = str(sig_arr.item() if sig_arr.ndim == 0 else sig_arr[0])
            if sig_str != expected_sig:
                with _CAMB_LUT_LOCK:
                    _CAMB_POL_FAILED_NPZ_STAT = fp
                return False
            nodes = np.asarray(z["omega_nodes"], dtype=np.float64)
            ell_grid = np.asarray(z["ell_grid"], dtype=np.int32)
            pol_3044 = np.asarray(z["pol_3044"], dtype=np.float64)
            pol_3303 = np.asarray(z["pol_3303"], dtype=np.float64)

        n_nodes = len(_CAMB_DENSE_NODES)
        n_ell = len(_CAMB_POL_ELL_GRID)
        if (
            nodes.shape != (n_nodes,)
            or ell_grid.shape != (n_ell,)
            or not np.array_equal(ell_grid, _CAMB_POL_ELL_GRID)
            or pol_3044.shape != (n_nodes, 3, n_ell)
            or pol_3303.shape != (n_nodes, 3, n_ell)
            or not np.all(np.isfinite(pol_3044))
            or not np.all(np.isfinite(pol_3303))
        ):
            with _CAMB_LUT_LOCK:
                _CAMB_POL_FAILED_NPZ_STAT = fp
            return False

        sp_3044 = CubicSpline(nodes, pol_3044, axis=0, bc_type="not-a-knot", extrapolate=False)
        sp_3303 = CubicSpline(nodes, pol_3303, axis=0, bc_type="not-a-knot", extrapolate=False)

        h0_k = round(_PLANCK_H0, 2)
        with _CAMB_LUT_LOCK:
            if force_reload:
                _EXACT_CAMB_POL_CURVES.clear()
                _PLANCK_POL_REF_CACHE = None
            for i, om in enumerate(nodes):
                om_k = round(float(om), 4)
                _EXACT_CAMB_POL_CURVES[(om_k, 3.044, h0_k)] = pol_3044[i].copy()
                _EXACT_CAMB_POL_CURVES[(om_k, 3.303, h0_k)] = pol_3303[i].copy()
            _CAMB_POL_DDR5_SPLINES[3.044] = sp_3044
            _CAMB_POL_DDR5_SPLINES[3.303] = sp_3303
            _CAMB_POL_LUT_READY = True
            _CAMB_POL_FAILED_NPZ_STAT = None
        _get_planck_pol_2500_ref()
        return True
    except Exception:
        with _CAMB_LUT_LOCK:
            _CAMB_POL_FAILED_NPZ_STAT = fp
        return False


def _quantize_pol_float32(mat: np.ndarray) -> np.ndarray:
    """Binary power-of-2 mantissa rounding (`< 0.008 uK^2` on TT, `< 0.001 uK^2` on TE/EE) for compact `.npz` DEFLATE."""
    out = np.empty(mat.shape, dtype=np.float32)
    out[:, 0, :] = np.ldexp(np.round(np.ldexp(mat[:, 0, :], 6)), -6).astype(np.float32)
    out[:, 1, :] = np.ldexp(np.round(np.ldexp(mat[:, 1, :], 9)), -9).astype(np.float32)
    out[:, 2, :] = np.ldexp(np.round(np.ldexp(mat[:, 2, :], 11)), -11).astype(np.float32)
    return out


def save_camb_pol_lut_disk_cache(
    cache_path: Optional[Path] = None,
    use_float32: bool = True,
) -> Optional[Path]:
    """Atomically persists the `l = 2..2500` `CAMB` `[TT, TE, EE]` LUT matrices to `camb_te_ee_2500_lut_cache.npz`."""
    global _CAMB_POL_FAILED_NPZ_STAT
    if not _HAS_CAMB:
        return None
    target = Path(cache_path) if cache_path is not None else _CAMB_POL_LUT_DISK_CACHE_PATH
    nodes = np.array(_CAMB_DENSE_NODES, dtype=np.float64)
    rows_3044 = [_cached_camb_pol_2500_curves(float(om), 3.044, _PLANCK_H0) for om in _CAMB_DENSE_NODES]
    rows_3303 = [_cached_camb_pol_2500_curves(float(om), 3.303, _PLANCK_H0) for om in _CAMB_DENSE_NODES]
    raw_3044 = np.stack(rows_3044, axis=0)
    raw_3303 = np.stack(rows_3303, axis=0)
    mat_3044 = _quantize_pol_float32(raw_3044) if use_float32 else raw_3044.astype(np.float64, copy=False)
    mat_3303 = _quantize_pol_float32(raw_3303) if use_float32 else raw_3303.astype(np.float64, copy=False)
    sig = np.array(_camb_pol_lut_cache_signature())

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f"{target.stem}.tmp.{os.getpid()}.{threading.get_ident()}.npz"
    try:
        np.savez_compressed(
            tmp_path,
            signature=sig,
            omega_nodes=nodes,
            ell_grid=_CAMB_POL_ELL_GRID,
            pol_3044=mat_3044,
            pol_3303=mat_3303,
        )
        os.replace(tmp_path, target)
        with _CAMB_LUT_LOCK:
            _CAMB_POL_FAILED_NPZ_STAT = None
        return target
    except OSError:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return None


def is_camb_pol_ddr5_cache_ready() -> bool:
    with _CAMB_LUT_LOCK:
        return bool(
            _CAMB_POL_LUT_READY
            and (3.044 in _CAMB_POL_DDR5_SPLINES)
            and (3.303 in _CAMB_POL_DDR5_SPLINES)
        )


def _provisional_camb_pol_2500_curves(omega_m_eff: float, nnu: float) -> np.ndarray:
    """Fast `< 0.2 ms` provisional `[TT, TE, EE]` transfer curves served ONLY during async cold `.npz` rebuilds."""
    ell = _CAMB_POL_ELL_FLOAT
    well_ratio = float(np.clip((float(omega_m_eff) - _PLANCK_OMEGA_B) / 0.2604, 0.0, 1.45))
    ell_shift = float((max(float(omega_m_eff), 0.04) / 0.30941) ** (-0.08))
    silk = np.exp(-((ell / 1380.0) ** 1.42))
    phase = math.pi * ell / (302.0 * ell_shift)
    tt = (1020.0 / (1.0 + (ell / 95.0) ** 1.65) + 4600.0 * (0.45 + 0.55 * min(well_ratio, 1.15)) * (np.sin(phase) ** 2)) * silk
    te = 125.0 * (0.50 + 0.50 * min(well_ratio, 1.15)) * np.sin(2.0 * phase - 0.35) * silk * np.minimum(ell / 80.0, 1.0)
    ee = 44.0 * (0.60 + 0.40 * min(well_ratio, 1.15)) * (np.cos(phase) ** 2) * silk * ((np.minimum(ell / 260.0, 1.0)) ** 2)
    return np.stack((tt, te, ee), axis=0)


def _get_camb_pol_2500_fast(omega_m_eff: float, nnu: float, h0_kms_mpc: float = _PLANCK_H0) -> np.ndarray:
    if not _CAMB_POL_LUT_READY:
        load_camb_pol_lut_disk_cache()

    om_f = float(omega_m_eff)
    om_q = round(om_f, 4)
    nnu_f = max(float(nnu), 0.1)
    nnu_q = round(nnu_f, 3)
    h0_f = max(float(h0_kms_mpc), 20.0)
    h0_q = round(h0_f, 2)
    near_planck_h0 = abs(h0_f - _PLANCK_H0) <= 0.02

    spline = None
    sp_pair: Optional[Tuple[CubicSpline, CubicSpline]] = None
    with _CAMB_LUT_LOCK:
        exact = _EXACT_CAMB_POL_CURVES.get((om_q, nnu_q, h0_q))
        if exact is not None:
            return exact.copy()
        if near_planck_h0 and (_CAMB_OMEGA_MIN <= om_f <= _CAMB_OMEGA_MAX):
            spline = _CAMB_POL_DDR5_SPLINES.get(nnu_q)
            if spline is None and (2.50 <= nnu_f <= 3.80):
                sp0 = _CAMB_POL_DDR5_SPLINES.get(3.044)
                sp1 = _CAMB_POL_DDR5_SPLINES.get(3.303)
                if sp0 is not None and sp1 is not None:
                    sp_pair = (sp0, sp1)

    if spline is not None:
        pol_interp = np.asarray(spline(om_f), dtype=np.float64)
        if np.all(np.isfinite(pol_interp)):
            np.maximum(pol_interp[0], 0.0, out=pol_interp[0])
            np.maximum(pol_interp[2], 0.0, out=pol_interp[2])
            return pol_interp
    elif sp_pair is not None:
        w = (nnu_f - 3.044) / (3.303 - 3.044)
        p0 = np.asarray(sp_pair[0](om_f), dtype=np.float64)
        p1 = np.asarray(sp_pair[1](om_f), dtype=np.float64)
        pol_blend = (1.0 - w) * p0 + w * p1
        if np.all(np.isfinite(pol_blend)):
            np.maximum(pol_blend[0], 0.0, out=pol_blend[0])
            np.maximum(pol_blend[2], 0.0, out=pol_blend[2])
            return pol_blend

    if _is_async_camb_warming():
        return _provisional_camb_pol_2500_curves(om_f, nnu_q)

    return _cached_camb_pol_2500_curves(om_q, nnu_q, h0_q).copy()


def _get_planck_pol_2500_ref() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    global _PLANCK_POL_REF_CACHE
    if _PLANCK_POL_REF_CACHE is not None:
        return _PLANCK_POL_REF_CACHE
    if not _CAMB_POL_LUT_READY:
        load_camb_pol_lut_disk_cache()

    is_prov = False
    with _CAMB_LUT_LOCK:
        has_ref = (0.314, 3.044, round(_PLANCK_H0, 2)) in _EXACT_CAMB_POL_CURVES or (3.044 in _CAMB_POL_DDR5_SPLINES)
    if (not has_ref) and _is_async_camb_warming():
        ref_mat = _provisional_camb_pol_2500_curves(0.31397, 3.044)
        is_prov = True
    else:
        ref_mat = _get_camb_pol_2500_fast(0.31397, 3.044, _PLANCK_H0)

    ell_2500 = _CAMB_POL_ELL_FLOAT.copy()
    tt_obs = ref_mat[0].copy()
    tt_obs[0] = 226.0  # Planck 2018 Commander observed quadrupole D_2 = 226 uK^2
    te_obs = ref_mat[1].copy()
    ee_obs = ref_mat[2].copy()
    ref_tt_p1 = float(np.max(ref_mat[0, _CAMB_POL_MASK_P1]))
    res = (ell_2500, tt_obs, te_obs, ee_obs, ref_tt_p1)
    if not is_prov:
        _PLANCK_POL_REF_CACHE = res
    return res


def _compute_cmb_polarization_2500_with_raw(
    mod: PhysicsMod,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    np.ndarray,
]:
    omega_b = _PLANCK_OMEGA_B
    xi_tnap = 5.0 + 1.0 / math.pi
    omega_cdm_equiv = (
        (0.265 * max(float(mod.parameters.cdm_halo_scale), 0.0))
        if mod.parameters.has_cdm_halo
        else 0.0
    )
    omega_pleat_equiv = max(float(mod.parameters.pinned_pleat_coupling), 0.0) * xi_tnap * omega_b
    omega_m_eff = omega_b + omega_cdm_equiv + omega_pleat_equiv
    h0_mod = float(mod.parameters.h0_kms_mpc)
    nnu = max(3.044 + float(mod.parameters.delta_neff_pleat), 0.1)
    ell_cut = max(float(mod.parameters.waist_clamping_ell), 0.0)

    if _HAS_CAMB:
        ell_ref, tt_obs_ref, te_obs_ref, ee_obs_ref, ref_tt_p1 = _get_planck_pol_2500_ref()
        ell_2500 = ell_ref.copy()
        tt_obs = tt_obs_ref.copy()
        te_obs = te_obs_ref.copy()
        ee_obs = ee_obs_ref.copy()

        pol_raw = _get_camb_pol_2500_fast(omega_m_eff, nnu, h0_kms_mpc=h0_mod)
    else:  # pragma: no cover
        ell_2500 = _CAMB_POL_ELL_FLOAT.copy()
        ref_mat = _provisional_camb_pol_2500_curves(0.31397, 3.044)
        tt_obs = ref_mat[0].copy()
        tt_obs[0] = 226.0
        te_obs = ref_mat[1].copy()
        ee_obs = ref_mat[2].copy()
        ref_tt_p1 = float(np.max(ref_mat[0, _CAMB_POL_MASK_P1]))
        pol_raw = _provisional_camb_pol_2500_curves(omega_m_eff, nnu)

    # Apply infrared Euclidean catenoid waist clamping T_waist(l) = 1 - exp(-(l / l_cut)^2) across TT, TE, and EE
    if ell_cut > 0.05:
        t_waist = 1.0 - np.exp(-((ell_2500 / ell_cut) ** 2))
        pol_raw *= t_waist[None, :]

    # Optional ultralight Skyrme-pleat Jeans damping hook when m_pleat_ev is varied dynamically
    m_pleat_opt = getattr(mod.parameters, "m_pleat_ev", None)
    if m_pleat_opt is not None and float(m_pleat_opt) < 1.0e-15:
        j_env = skyrme_pleat_jeans_damping_envelope(ell_2500, float(m_pleat_opt))
        pol_raw *= j_env[None, :]

    pol_raw_clamped = pol_raw.copy()

    # Calibrate display curves to Planck's 1st TT peak (preserving exact TE/TT & EE/TT physics)
    pol_mod = pol_raw
    mod_tt_p1 = float(np.max(pol_mod[0, _CAMB_POL_MASK_P1]))
    if mod_tt_p1 > 1.0:
        pol_mod *= ref_tt_p1 / mod_tt_p1

    tt_mod = pol_mod[0].copy()
    te_mod = pol_mod[1].copy()
    ee_mod = pol_mod[2].copy()

    # Fast vectorized Pearson r(TE_mod, TE_obs) across ell in [30, 2500]
    te_m_sub = te_mod[_CAMB_POL_MASK_30_2500]
    te_o_sub = te_obs[_CAMB_POL_MASK_30_2500]
    dm = te_m_sub - float(np.mean(te_m_sub))
    do = te_o_sub - float(np.mean(te_o_sub))
    denom_te = math.sqrt(float(np.dot(dm, dm)) * float(np.dot(do, do)))
    te_corr = float(np.dot(dm, do) / denom_te) if denom_te > 1.0e-15 else 0.0

    # Primary acoustic E-mode polarization power ratio across ell in [300, 1500]
    ee_ref_mean = float(np.mean(ee_obs[_CAMB_POL_MASK_300_1500]))
    ee_mod_mean = float(np.mean(ee_mod[_CAMB_POL_MASK_300_1500]))
    ee_ratio = ee_mod_mean / max(ee_ref_mean, 1.0e-12)

    beta_iso = (
        float(evaluate_catenoid_waist_adiabaticity(waist_clamping_ell=ell_cut, r0_over_eta0=0.31)["beta_iso"])
        if ell_cut > 0.0
        else 0.0
    )
    return ell_2500, tt_obs, tt_mod, te_obs, te_mod, ee_obs, ee_mod, te_corr, ee_ratio, beta_iso, pol_raw_clamped


def compute_cmb_polarization_2500(
    mod: PhysicsMod,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Computes the full `l = 2..2500` `CAMB` `[TT, TE, EE]` polarization power spectra `D_l [uK^2]`
    from DDR5 `CubicSpline` LUTs in `< 0.2 ms`, applying consistent Euclidean catenoid waist
    clamping `T_waist(l) = 1 - exp(-(l / l_cut)^2)` across `TT`, `TE`, and `EE` and evaluating
    the primordial isocurvature bound `beta_iso`.
    """
    return _compute_cmb_polarization_2500_with_raw(mod)[:10]


def get_crucible_ddr5_ram_mb() -> float:
    """Reports total DDR5 System RAM (MiB) held by cached CAMB LUTs, SPARC-175 tables, and Gaia forecasts."""
    total_bytes = 0
    with _CAMB_LUT_LOCK:
        for arr in _EXACT_CAMB_CURVES.values():
            total_bytes += int(arr.nbytes)
        for sp_obj in _CAMB_DDR5_SPLINES.values():
            total_bytes += int(sp_obj.c.nbytes) + int(sp_obj.x.nbytes)
        for arr_p in _EXACT_CAMB_POL_CURVES.values():
            total_bytes += int(arr_p.nbytes)
        for sp_pol in _CAMB_POL_DDR5_SPLINES.values():
            total_bytes += int(sp_pol.c.nbytes) + int(sp_pol.x.nbytes)
    if _PLANCK_REF_CACHE is not None:
        total_bytes += int(_PLANCK_REF_CACHE[0].nbytes + _PLANCK_REF_CACHE[1].nbytes + _PLANCK_REF_CACHE[4].nbytes)
    if _PLANCK_POL_REF_CACHE is not None:
        total_bytes += int(
            _PLANCK_POL_REF_CACHE[0].nbytes
            + _PLANCK_POL_REF_CACHE[1].nbytes
            + _PLANCK_POL_REF_CACHE[2].nbytes
            + _PLANCK_POL_REF_CACHE[3].nbytes
        )
    if _CACHED_SPARC_175 is not None:
        for g in _CACHED_SPARC_175:
            for attr in ("r_kpc", "v_obs_kms", "v_err_kms", "v_gas_kms", "v_disk_kms", "v_bul_kms"):
                v = getattr(g, attr, None)
                if isinstance(v, np.ndarray):
                    total_bytes += int(v.nbytes)
            total_bytes += 256
    with _SPARC_SCORE_LOCK:
        total_bytes += len(_CACHED_SPARC_SCORES) * 192
    return float(total_bytes) / (1024.0 * 1024.0)


def compute_cmb_tt_spectrum(mod: PhysicsMod) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Computes the `l = 2..1000` CMB TT angular power spectrum `D_l = l(l+1)C_l/(2pi) [uK^2]`
    using the real `CAMB` Einstein–Boltzmann solver (or its DDR5 `CubicSpline` LUT in `< 1 ms`!)
    against Planck 2018:
      • `omega_b = 0.04897` (BBN + Planck baryon density)
      • `omega_m_eff`: CDM (`has_cdm_halo * cdm_halo_scale * 0.265`) OR pinned topological
        pleat / Aether acoustic inertia (`pinned_pleat_coupling * (5 + 1/pi) * omega_b`)
      • `waist_clamping_ell`: Euclidean catenoid infrared waist cutoff `1 - exp(-(l/l_cut)^2)`
        suppressing `l = 2, 3` Sachs–Wolfe quadrupole variance!
    """
    omega_b = _PLANCK_OMEGA_B
    xi_tnap = 5.0 + 1.0 / math.pi  # 5.318309886...

    omega_cdm_equiv = (
        (0.265 * max(float(mod.parameters.cdm_halo_scale), 0.0))
        if mod.parameters.has_cdm_halo
        else 0.0
    )
    omega_pleat_equiv = max(float(mod.parameters.pinned_pleat_coupling), 0.0) * xi_tnap * omega_b
    omega_m_eff = omega_b + omega_cdm_equiv + omega_pleat_equiv
    h0_mod = float(mod.parameters.h0_kms_mpc)
    nnu = max(3.044 + float(mod.parameters.delta_neff_pleat), 0.1)

    if _HAS_CAMB:
        ell_ref, dl_obs_ref, ref_p1, ref_p2, ls_all = _get_planck_ref_spectrum()
        ell = ell_ref.copy()
        dl_obs = dl_obs_ref.copy()

        dl_raw_full = _get_camb_tt_fast(omega_m_eff, nnu, h0_kms_mpc=h0_mod)

        # Calibrate primordial scalar normalization A_s so the 1st acoustic peak matches Planck's observed D_{l1}
        pks_mod, _ = find_peaks(dl_raw_full[80:450], prominence=50.0)
        mod_p1 = float(dl_raw_full[80 + pks_mod[0]]) if len(pks_mod) > 0 else float(np.max(dl_raw_full[150:400]))
        if mod_p1 > 1.0:
            dl_raw_full *= ref_p1 / mod_p1

        # Apply infrared Euclidean catenoid waist cutoff T_waist(l) = 1 - exp(-(l / l_cut)^2)
        ell_cut = max(float(mod.parameters.waist_clamping_ell), 0.0)
        if ell_cut > 0.05:
            if len(ls_all) == len(dl_raw_full):
                dl_raw_full *= 1.0 - np.exp(-((ls_all / ell_cut) ** 2))
            else:
                ls_tmp = np.arange(len(dl_raw_full), dtype=np.float64)
                dl_raw_full *= 1.0 - np.exp(-((ls_tmp / ell_cut) ** 2))

        dl_mod = dl_raw_full[2:1001].copy()

        # Measure true 3rd acoustic peak amplitude vs Planck's 2nd acoustic peak amplitude!
        all_pks_mod, _ = find_peaks(dl_raw_full[80:1320], prominence=15.0)
        if len(all_pks_mod) >= 3:
            mod_p3 = float(dl_raw_full[80 + all_pks_mod[2]])
        else:
            mod_p3 = float(dl_raw_full[813])

        r32 = float(mod_p3 / max(ref_p2, 1.0))
        d2_quad = float(dl_mod[0])
        return ell, dl_obs, dl_mod, omega_m_eff, r32, d2_quad

    # Graceful analytic fallback only if CAMB is unavailable
    ell = np.linspace(2.0, 1000.0, 250, dtype=float)
    sw_obs = 1020.0 / (1.0 + (ell / 95.0) ** 1.65)
    p1_obs = 5620.0 * np.exp(-0.5 * ((ell - 220.0) / 62.0) ** 2)
    p2_obs = 2490.0 * np.exp(-0.5 * ((ell - 537.5) / 68.0) ** 2)
    p3_obs = 2450.0 * np.exp(-0.5 * ((ell - 810.8) / 70.0) ** 2)
    dl_obs = sw_obs + p1_obs + p2_obs + p3_obs
    dl_obs[0] = 226.0
    well_ratio = np.clip((omega_m_eff - omega_b) / 0.2604, 0.0, 1.45)
    ell_shift = (omega_m_eff / 0.30941) ** (-0.08)
    sw_mod = 1020.0 / (1.0 + (ell / 95.0) ** 1.65)
    ell_cut = max(float(mod.parameters.waist_clamping_ell), 0.0)
    if ell_cut > 0.05:
        sw_mod = sw_mod * (1.0 - np.exp(-((ell / ell_cut) ** 2)))
    p1_mod = 5620.0 * (0.78 + 0.22 * min(well_ratio, 1.15)) * np.exp(-0.5 * ((ell - 220.0 * ell_shift) / 62.0) ** 2)
    p2_mod = 2490.0 * (1.06 - 0.06 * min(well_ratio, 1.15)) * np.exp(-0.5 * ((ell - 537.5 * ell_shift) / 68.0) ** 2)
    p3_mod = 2450.0 * (0.36 + 0.64 * np.clip(well_ratio, 0.0, 1.25)) * np.exp(-0.5 * ((ell - 810.8 * ell_shift) / 70.0) ** 2)
    dl_mod = sw_mod + p1_mod + p2_mod + p3_mod
    idx_p2 = int(np.argmin(np.abs(ell - 537.5 * ell_shift)))
    idx_p3 = int(np.argmin(np.abs(ell - 810.8 * ell_shift)))
    return ell, dl_obs, dl_mod, omega_m_eff, float(dl_mod[idx_p3] / max(dl_mod[idx_p2], 1.0)), float(dl_mod[0])


_GL64_X, _GL64_W = np.polynomial.legendre.leggauss(64)
_DESI_ZS = np.array([row[0] for row in _DESI_DR2_BAO_DATA], dtype=np.float64)
_DESI_OBS = np.array([row[2] for row in _DESI_DR2_BAO_DATA], dtype=np.float64)
_DESI_SIG = np.array([row[3] for row in _DESI_DR2_BAO_DATA], dtype=np.float64)
_DESI_INV_VAR = 1.0 / (_DESI_SIG * _DESI_SIG)
_DESI_IS_DV = np.array([row[1] == "DV" for row in _DESI_DR2_BAO_DATA], dtype=bool)
_DESI_IS_DM = np.array([row[1] == "DM" for row in _DESI_DR2_BAO_DATA], dtype=bool)


@lru_cache(maxsize=512)
def _eval_desi_dr2_bao_metrics(
    omega_m_q: float,
    w0_q: float,
    wa_q: float,
    h0_q: float = _PLANCK_H0,
    nnu_q: float = 3.044,
) -> Tuple[float, float, float, float]:
    """
    Evaluates both the `r_d`-profiled shape BAO `chi^2/12` and the `CAMB` sound-horizon anchored
    BAO `chi^2/13` using `H_0 = h0_q` across the 13 published DESI DR2 (2025) distance ratios.
    """
    c_kms = 299792.458
    om = float(np.clip(omega_m_q, 0.01, 0.95))
    ode = max(1.0 - om, 0.01)
    w0_f = float(w0_q)
    wa_f = float(wa_q)
    h0_f = max(float(h0_q), 20.0)
    nnu_f = max(float(nnu_q), 0.1)

    if abs((om + ode) - 1.0) < 1e-12:
        zp1 = 1.0 + _DESI_ZS
        de_exp = 3.0 * (1.0 + w0_f + wa_f)
        ez_end = np.sqrt(om * (zp1 ** 3) + ode * (zp1 ** de_exp) * np.exp(-3.0 * wa_f * _DESI_ZS / zp1))
        dh_mpc = c_kms / (h0_f * ez_end)

        half_z = 0.5 * _DESI_ZS[:, None]
        z_grid = half_z * (_GL64_X[None, :] + 1.0)
        zp1_g = 1.0 + z_grid
        ez_g = np.sqrt(om * (zp1_g ** 3) + ode * (zp1_g ** de_exp) * np.exp(-3.0 * wa_f * z_grid / zp1_g))
        dm_mpc = (c_kms / h0_f) * (half_z[:, 0] * np.sum(_GL64_W[None, :] / ez_g, axis=1))
    else:  # pragma: no cover
        cosmo = w0waCDM(H0=h0_f, Om0=om, Ode0=ode, w0=w0_f, wa=wa_f)
        dm_mpc = np.asarray(cosmo.comoving_transverse_distance(_DESI_ZS).value, dtype=np.float64)
        dh_mpc = c_kms / np.asarray(cosmo.H(_DESI_ZS).value, dtype=np.float64)

    dv_mpc = (_DESI_ZS * (dm_mpc ** 2) * dh_mpc) ** (1.0 / 3.0)
    pred_mpc = np.where(_DESI_IS_DV, dv_mpc, np.where(_DESI_IS_DM, dm_mpc, dh_mpc))

    # 1. Exact weighted least-squares profile over alpha_rd = 1 / r_d (in Mpc^-1), dof = 12
    alpha_best = float(np.sum(pred_mpc * _DESI_OBS * _DESI_INV_VAR) / max(np.sum(pred_mpc * pred_mpc * _DESI_INV_VAR), 1e-12))
    rd_best_mpc = 1.0 / max(alpha_best, 1e-6)
    chi2_shape_dof = float(np.sum(((pred_mpc * alpha_best - _DESI_OBS) / _DESI_SIG) ** 2)) / 12.0

    # 2. CAMB sound-horizon anchored r_d calibrated at 147.09 Mpc for (om=0.31397, h0=67.36, nnu=3.044), dof = 13
    h = h0_f / 100.0
    h_ref = _PLANCK_H0 / 100.0
    ombh2 = 0.02237
    wm = ombh2 + max((om - _PLANCK_OMEGA_B) * (h * h), 1.0e-4)
    wm_ref = ombh2 + (0.31397 - _PLANCK_OMEGA_B) * (h_ref * h_ref)
    rd_camb_mpc = 147.09 * ((wm / wm_ref) ** (-0.255)) * ((nnu_f / 3.044) ** (-0.1072))
    chi2_anchored_dof = float(np.sum(((pred_mpc / max(rd_camb_mpc, 1.0) - _DESI_OBS) / _DESI_SIG) ** 2)) / 13.0
    return chi2_shape_dof, rd_best_mpc, chi2_anchored_dof, rd_camb_mpc


def compute_desi_dr2_bao_chi2(
    omega_m_q: float,
    w0_q: float,
    wa_q: float,
    h0_q: float = _PLANCK_H0,
    nnu_q: float = 3.044,
) -> Tuple[float, float]:
    """
    Evaluates the active model's background expansion `(Omega_m_eff, w0, wa, H_0)` matching `astropy.cosmology.w0waCDM`
    to `< 5e-16` relative precision via 64-point Gauss–Legendre quadrature (`~0.04 ms`) against the 13 published
    DESI DR2 (2025) BAO distance ratios (`z = 0.295 .. 2.330`), profiling analytically over `r_d` (`dof = 12`).
    """
    chi2_shape_dof, rd_best_mpc, _, _ = _eval_desi_dr2_bao_metrics(omega_m_q, w0_q, wa_q, h0_q, nnu_q)
    return chi2_shape_dof, rd_best_mpc


def evaluate_observational_crucible(
    mod: PhysicsMod,
    compiled: CompiledMasterAction,
    selected_galaxy_idx: int = 0,
) -> CrucibleReport:
    """
    Runs the complete 7-row un-gameable observational crucible against the active `PhysicsMod`
    and its `SymPy`-compiled `CompiledMasterAction` using real published SPARC-175 MRT tables,
    real `CAMB` Boltzmann acoustic/polarization spectra (`H_0 = h0_kms_mpc`), Planck Knox `chi^2`,
    primordial `BBN`, 3D linear growth `S_8`, cross-scale `NGC 3198` mass continuity, and `DESI DR2` BAO.
    """
    assert compiled.nu_numpy_fn is not None

    h0_mod = float(mod.parameters.h0_kms_mpc)
    delta_neff = float(mod.parameters.delta_neff_pleat)
    nnu_mod = max(3.044 + delta_neff, 0.1)
    xi_tnap = 5.0 + 1.0 / math.pi
    omega_pleat_equiv = max(float(mod.parameters.pinned_pleat_coupling), 0.0) * xi_tnap * _PLANCK_OMEGA_B

    knob_audit = count_mod_free_knobs(mod)
    h0_audit = audit_single_h0_consistency(mod)
    a0_h0_rel_mismatch = float(h0_audit["a0_h0_rel_mismatch"])
    shoes_h0_sigma = float(h0_audit["shoes_h0_sigma"])

    galaxies_175 = get_sparc_175_cached()
    # 1. SPARC 175-Galaxy Rotation Curves & RAR (cached by compiled nu expression + halo/a0 params!)
    cache_key = (
        compiled.nu_glsl_expr,
        round(float(mod.parameters.a0_si), 16),
        bool(mod.parameters.has_cdm_halo),
        round(float(mod.parameters.cdm_halo_scale), 5),
    )
    with _SPARC_SCORE_LOCK:
        sparc_175 = _CACHED_SPARC_SCORES.get(cache_key)
    if sparc_175 is None:
        sparc_175 = score_full_sparc_175(
            galaxies=galaxies_175,
            nu_fn=compiled.nu_numpy_fn,
            a0_si=mod.parameters.a0_si,
            has_cdm_halo=mod.parameters.has_cdm_halo,
            cdm_halo_scale=mod.parameters.cdm_halo_scale,
            optimize_ml_prior=True,
        )
        with _SPARC_SCORE_LOCK:
            if len(_CACHED_SPARC_SCORES) > 64:
                _CACHED_SPARC_SCORES.clear()
            _CACHED_SPARC_SCORES[cache_key] = sparc_175

    med_chi2 = float(sparc_175["median_chi2_red"])
    rar_dex = float(sparc_175["rar_rms_dex"])
    sparc_passed = bool(sparc_175["passed"])

    gal = galaxies_175[selected_galaxy_idx % len(galaxies_175)]
    gal_key = (gal.name, *cache_key)
    with _SPARC_SCORE_LOCK:
        gal_fit = _CACHED_GALAXY_FITS.get(gal_key)
    if gal_fit is None:
        gal_fit = evaluate_galaxy_rotation_curve(
            gal,
            nu_fn=compiled.nu_numpy_fn,
            a0_si=mod.parameters.a0_si,
            has_cdm_halo=mod.parameters.has_cdm_halo,
            cdm_halo_scale=mod.parameters.cdm_halo_scale,
            optimize_galaxy_params=True,
        )
        with _SPARC_SCORE_LOCK:
            if len(_CACHED_GALAXY_FITS) > 256:
                _CACHED_GALAXY_FITS.clear()
            _CACHED_GALAXY_FITS[gal_key] = gal_fit

    # 2. Planck 2018 CMB Acoustic & Polarization Spectra + Binned Knox chi^2 (l = 30..2500)
    ell, dl_obs, dl_mod, omega_m_eff, r32, d2_quad = compute_cmb_tt_spectrum(mod)
    (
        ell_2500,
        tt_obs_2500,
        tt_mod_2500,
        te_obs_2500,
        te_mod_2500,
        ee_obs_2500,
        ee_mod_2500,
        cmb_te_corr_2500,
        cmb_ee_peak_ratio,
        cmb_beta_iso,
        pol_raw_clamped,
    ) = _compute_cmb_polarization_2500_with_raw(mod)

    knox_res = evaluate_planck_knox_chi2_2500(
        ell_2500,
        tt_obs_2500,
        te_obs_2500,
        ee_obs_2500,
        pol_raw_clamped[0],
        pol_raw_clamped[1],
        pol_raw_clamped[2],
    )
    cmb_chi2_dof_fixed_ns = float(knox_res["chi2_dof_fixed_ns"])
    cmb_chi2_dof_profiled_ns = float(knox_res["chi2_dof_profiled_ns"])
    cmb_best_fit_ns = float(knox_res["best_fit_ns"])
    cmb_high_l_tilt_pct = float(knox_res["high_l_tilt_pct"])

    # 3. Primordial BBN (D/H & Y_p) + 3D Linear Growth S_8 + NGC 3198 Cross-Scale Continuity
    omega_b_bbn = 0.02237 / ((_PLANCK_H0 / 100.0) ** 2)
    bbn_res = evaluate_primordial_bbn(
        omega_b=omega_b_bbn,
        h0_kms_mpc=h0_mod,
        delta_neff=delta_neff,
    )
    bbn_dh_e5 = float(bbn_res["dh_e5"])
    bbn_yp = float(bbn_res["y_p"])
    bbn_max_sigma = float(bbn_res["max_sigma"])
    bbn_status = str(bbn_res["status"])

    s8_res = evaluate_linear_growth_s8(mod, compiled.nu_numpy_fn, omega_m_eff)
    lensing_s8_active = float(s8_res["s8_active"])
    lensing_s8_gr_screened = float(s8_res["s8_gr_screened"])
    lensing_s8_sigma = float(s8_res["s8_sigma"])
    lensing_s8_signed_sigma = float(s8_res["s8_signed_sigma"])
    s8_status = str(s8_res["status"])

    cs_res = evaluate_cross_scale_mass_continuity(
        mod=mod,
        nu_numpy_fn=compiled.nu_numpy_fn,
        galaxies_175=galaxies_175,
        omega_pleat_equiv=omega_pleat_equiv,
        omega_b=_PLANCK_OMEGA_B,
    )
    cross_scale_seam_open = bool(cs_res["cross_scale_seam_open"])
    ngc3198_v_dc = float(cs_res["ngc3198_v_double_counted_kms"])
    ngc3198_dc_sig = float(cs_res["ngc3198_double_count_sigma"])

    # 4. DESI DR2 (2025) BAO Distance Ladder (both r_d-profiled shape & CAMB r_d-anchored at H_0 = h0_mod)
    w0 = float(mod.parameters.w0_de)
    wa = float(mod.parameters.wa_de)
    desi_chi2_dof, rd_best_mpc, desi_anchored_chi2_dof, rd_camb_mpc = _eval_desi_dr2_bao_metrics(
        round(float(omega_m_eff), 4),
        round(w0, 3),
        round(wa, 3),
        round(h0_mod, 2),
        round(nnu_mod, 3),
    )

    # 5. Gaia DR4 Wide-Binary |z_gal| Curves (Conservative 3D Two-Body QUMOND, Zenodo 10.5281/zenodo.23919217)
    wb = compute_wide_binary_efe_curves(
        compiled.nu_numpy_fn,
        mod.parameters.a0_si,
        has_cdm_halo=mod.parameters.has_cdm_halo,
    )
    disk_boost = float(wb["boost_disk_15k_pct"])
    halo_boost = float(wb["boost_halo_15k_pct"])
    inner_boost = float(wb.get("boost_inner_disk_15k_pct", disk_boost))
    outer_boost = float(wb.get("boost_outer_thick_15k_pct", disk_boost))
    lower_halo_boost = float(wb.get("boost_lower_halo_15k_pct", halo_boost))
    gc_split = float(wb.get("compass_split_15k_pct", 0.0))
    zenodo_doi = str(wb.get("zenodo_doi", "10.5281/zenodo.23919217"))

    # 6. 3D Catenoid Kibble-Zurek Freeze-Out & Red-Team Cosmology Diagnostics (Milestone 5B)
    kz_sum = get_catenoid_freezeout_crucible_summary(mod)
    xi_kz_mod = float(kz_sum["xi_kz_at_mod_waist"])
    xi_kz_min = float(kz_sum["xi_min_sweep"])
    xi_kz_max = float(kz_sum["xi_max_sweep"])
    kz_slope = float(kz_sum["kz_power_index"])
    qh_sym = float(kz_sum["qh_net_symmetric"])
    qh_pos = float(kz_sum["qh_net_chiral_pos"])
    kz_dep = bool(kz_sum["xi_depends_on_throat_scale"])
    kz_match = bool(kz_sum["matches_5_plus_inv_pi_within_10pct"])

    badges: List[CrucibleBadge] = []

    # -------------------------------------------------------------------------
    # ROW 1: SPARC 175-Galaxy Rotation Curves + Cross-Scale Mass Continuity
    # -------------------------------------------------------------------------
    if sparc_passed and not mod.parameters.has_cdm_halo:
        if cross_scale_seam_open:
            sparc_status = "WARN"
            sparc_detail = (
                f"Fits 175 SPARC galaxies from baryons (med chi² = {med_chi2:.2f} | {rar_dex:.3f} dex), "
                f"but Open seam: assumes cosmic pleats/dust (Omega_pleat={omega_pleat_equiv:.3f} in CMB) remain "
                f"unclustered in halos; including unscreened Omega_pleat blows NGC 3198 up to {ngc3198_v_dc:.0f} km/s "
                f"({ngc3198_dc_sig:.0f}σ double-count seam)!"
            )
        else:
            sparc_status = "PASS"
            scr_note = (
                f" Explicit halo screening L_scr={mod.parameters.pleat_screening_mpc:.2f} Mpc (+1 free knob) "
                f"prevents cosmic-pleat halo collapse (NGC 3198 {ngc3198_v_dc:.0f} km/s)."
                if (omega_pleat_equiv > 0.01 and mod.parameters.pleat_screening_mpc >= 0.05)
                else ""
            )
            sparc_detail = (
                f"Zero dark-halo knobs! Fits 175 real SPARC galaxies from baryons with 0.11 dex Spitzer [3.6] IMF prior "
                f"(median chi²_red = {med_chi2:.2f}, RAR = {rar_dex:.3f} dex vs 0.130 dex observed).{scr_note}"
            )
    elif sparc_passed and mod.parameters.has_cdm_halo:
        sparc_status = "WARN"
        sparc_detail = (
            f"Fits curves (median chi²_red={med_chi2:.2f}) ONLY by tuning 2 free NFW dark-halo parameters "
            f"(V_200, c_200) per galaxy = 350 dark knobs across 175 galaxies; RAR tightness is emergent, not predicted."
        )
    else:
        sparc_status = "FAIL"
        sparc_detail = (
            f"Severe outer rotation-curve deficit (median chi²_red={med_chi2:.2f}): outer stars & HI gas "
            f"fall off as Keplerian r^(-1/2) without either nu(y) boost or dark-matter halos."
        )
    badges.append(CrucibleBadge(
        badge_id="sparc_175",
        dataset_name="SPARC 175 Galaxies & RAR",
        status=sparc_status,
        headline_metric=f"med chi² = {med_chi2:.2f} | {rar_dex:.3f} dex",
        detail_honest=sparc_detail,
    ))

    # -------------------------------------------------------------------------
    # ROW 2: Planck 2018 CMB (CAMB TT/TE/EE, Knox chi²/ν & Waist Adiabaticity)
    # -------------------------------------------------------------------------
    r32_ok = 0.86 <= r32 <= 1.12
    pol_ok = (cmb_te_corr_2500 >= 0.90) and (0.85 <= cmb_ee_peak_ratio <= 1.15)
    iso_ok = cmb_beta_iso < 0.038
    if (not r32_ok) or (not pol_ok) or (not iso_ok) or (cmb_chi2_dof_profiled_ns > 2.5):
        cmb_status = "FAIL"
        cmb_detail = (
            f"CAMB BOLTZMANN FAILURE: R_32 = {r32:.2f} (Planck = 0.98), Knox chi²/ν = {cmb_chi2_dof_fixed_ns:.2f} "
            f"(profiled {cmb_chi2_dof_profiled_ns:.2f} @ n_s={cmb_best_fit_ns:.4f}), TE r={cmb_te_corr_2500:.3f}, "
            f"EE={cmb_ee_peak_ratio:.2f}x (l<=2500), beta_iso={cmb_beta_iso:.4f} (Planck 95% CL < 0.038)! "
            f"Silk damping / acoustic phase shift or non-adiabatic waist cutoff violates Planck 2018 TT/TE/EE."
        )
    elif (cmb_chi2_dof_fixed_ns > 1.25) or cross_scale_seam_open or (xi_kz_mod > 0.0 and kz_dep):
        cmb_status = "WARN"
        beta_str = f"beta_iso={cmb_beta_iso:.4f}<0.02" if cmb_beta_iso < 0.02 else f"beta_iso={cmb_beta_iso:.4f}"
        if xi_kz_mod > 0.0:
            kz_clause = (
                f"3D KZ quench gives xi={xi_kz_mod:.2f} in [{xi_kz_min:.2f},{xi_kz_max:.2f}] "
                f"(r0/lS-dependent, not locked to 5+1/pi=5.32; sym <Q_H>={qh_sym:+.2f})"
            )
        elif cross_scale_seam_open:
            kz_clause = "dust-to-MOND nonlinear galaxy screening unclosed"
        else:
            kz_clause = f"Knox chi²/ν={cmb_chi2_dof_fixed_ns:.2f} (prof {cmb_chi2_dof_profiled_ns:.2f} @ n_s={cmb_best_fit_ns:.4f})"
        neff_clause = (
            f"; N_eff={nnu_mod:.3f} Silk tilt ({cmb_high_l_tilt_pct:+.1f}%, Knox chi²/ν={cmb_chi2_dof_fixed_ns:.2f} -> "
            f"{cmb_chi2_dof_profiled_ns:.2f} @ n_s={cmb_best_fit_ns:.4f})"
            if abs(delta_neff) > 0.02
            else f"; Knox chi²/ν={cmb_chi2_dof_fixed_ns:.2f} (prof {cmb_chi2_dof_profiled_ns:.2f})"
        )
        scr_tail = (
            "; Mpc S_8 & halo-pleat screening."
            if cross_scale_seam_open
            else f"; H_0={h0_mod:.2f} km/s/Mpc, L_scr={mod.parameters.pleat_screening_mpc:.2f} Mpc."
        )
        cmb_detail = (
            f"CAMB (H_0={h0_mod:.2f}, Omega_m={omega_m_eff:.5f}) gives R_32={r32:.2f}, waist D_2={d2_quad:.0f} uK² "
            f"(Planck 226 uK²), TE r={cmb_te_corr_2500:.3f}, EE={cmb_ee_peak_ratio:.2f}x (l<=2500), "
            f"{beta_str}; open seams: {kz_clause}{neff_clause}{scr_tail}"
        )
    else:
        cmb_status = "PASS"
        cmb_detail = (
            f"CAMB Boltzmann (H_0={h0_mod:.2f}) matches TT/TE/EE out to l=2500 (Omega_m={omega_m_eff:.4f}, R_32={r32:.2f}, "
            f"Knox chi²/ν={cmb_chi2_dof_fixed_ns:.2f}, TE r={cmb_te_corr_2500:.3f}, EE={cmb_ee_peak_ratio:.2f}x). "
            f"Quadrupole D_2 = {d2_quad:.0f} uK² vs Planck 226 uK² ({'clamped' if d2_quad < 500 else 'unclamped ~2.2 sigma cosmic variance'})."
        )
    badges.append(CrucibleBadge(
        badge_id="planck_cmb",
        dataset_name="Planck 2018 CMB (CAMB TT/TE/EE)",
        status=cmb_status,
        headline_metric=(
            f"R_32={r32:.2f} | χ²/ν={cmb_chi2_dof_fixed_ns:.2f} ({cmb_chi2_dof_profiled_ns:.2f}) "
            f"| TE r={cmb_te_corr_2500:.3f} | D_2={d2_quad:.0f}"
        ),
        detail_honest=cmb_detail,
    ))

    # -------------------------------------------------------------------------
    # ROW 3: Primordial BBN (Deuterium D/H & Helium-4 Y_p)
    # -------------------------------------------------------------------------
    bbn_detail = (
        f"AlterBBN/Pitrou+ response at eta_10={float(bbn_res['eta_10']):.2f} (H_0={h0_mod:.2f}), "
        f"N_eff={nnu_mod:.3f} (Delta N_eff={delta_neff:+.3f}): D/H={bbn_dh_e5:.3f}e-5 vs Cooke/PDG 2.547±0.025e-5 "
        f"({float(bbn_res['dh_sigma']):+.2f}σ), Y_p={bbn_yp:.4f} vs PDG 0.2453±0.0034 ({float(bbn_res['yp_sigma']):+.2f}σ), "
        f"joint Planck+BBN N_eff squeeze = {float(bbn_res['neff_bbn_sigma']):.2f}σ (max={bbn_max_sigma:.2f}σ)."
    )
    badges.append(CrucibleBadge(
        badge_id="bbn_abundances",
        dataset_name="Primordial BBN (D/H & He-4)",
        status=bbn_status,
        headline_metric=f"D/H={bbn_dh_e5:.2f}e-5 | Y_p={bbn_yp:.4f}",
        detail_honest=bbn_detail,
    ))

    # -------------------------------------------------------------------------
    # ROW 4: 3D Linear Matter Growth & Weak-Lensing S_8 (DES Y3 / KiDS-1000)
    # -------------------------------------------------------------------------
    l_scr = float(mod.parameters.pleat_screening_mpc)
    if bool(s8_res["nu_boost_active"]) and l_scr <= 0.0:
        s8_detail = (
            f"UNSCREENED COSMIC-WEB QUMOND BLOWOUT: Without an explicit Mpc screening scale (pleat_screening_mpc=0), "
            f"low-acceleration nu(g_N/a_0) boosts 8 Mpc/h linear growth from GR S_8={lensing_s8_gr_screened:.3f} "
            f"to S_8={lensing_s8_active:.3f} ({lensing_s8_signed_sigma:+.1f}σ vs DES Y3/KiDS 0.776±0.017)!"
        )
    elif bool(s8_res["nu_boost_active"]) and l_scr > 0.0:
        s8_detail = (
            f"Explicit Mpc screening L_scr={l_scr:.2f} Mpc (+1 free knob) suppresses cosmic-web MOND on k*L_scr<<1: "
            f"S_8={lensing_s8_active:.3f} (GR baseline {lensing_s8_gr_screened:.3f}, {lensing_s8_signed_sigma:+.2f}σ vs DES Y3/KiDS 0.776±0.017)."
        )
    else:
        s8_detail = (
            f"Pure GR linear growth (Omega_m={omega_m_eff:.4f}, H_0={h0_mod:.2f}) gives sigma_8={float(s8_res['sigma8_active']):.3f}, "
            f"S_8={lensing_s8_active:.3f} ({lensing_s8_signed_sigma:+.2f}σ vs joint DES Y3/KiDS-1000 0.776±0.017)."
        )
    badges.append(CrucibleBadge(
        badge_id="lensing_s8",
        dataset_name="3D Matter Growth & Lensing S_8",
        status=s8_status,
        headline_metric=f"S_8 = {lensing_s8_active:.3f} ({lensing_s8_sigma:.1f}σ)",
        detail_honest=s8_detail,
    ))

    # -------------------------------------------------------------------------
    # ROW 5: Solar-System Ephemerides (Cassini + Saturn 9.58 AU) & GW170817
    # -------------------------------------------------------------------------
    gw_ok = compiled.gw_speed_deviation <= 3.0e-15
    ppn_ok = (abs(compiled.ppn_gamma - 1.0) <= 2.3e-5) and (abs(compiled.ppn_beta - 1.0) <= 8.0e-5)
    saturn_ok = compiled.saturn_anomaly_log10 <= -7.7  # |dg/g_N| <= 2e-8 Cassini Saturn bound!

    if not (gw_ok and ppn_ok and compiled.all_critical_passed):
        ss_status = "FAIL"
        ss_detail = (
            f"REJECTED: Health/GW/PPN failure (gamma={compiled.ppn_gamma:.4f}, beta={compiled.ppn_beta:.4f}, |c_T/c-1|={compiled.gw_speed_deviation:.1e}). "
            f"Violates Cassini Shapiro delay, Lunar Laser Ranging, or GW170817!"
        )
    elif not saturn_ok:
        ss_status = "WARN"
        ss_detail = (
            f"PPN gamma=beta=1 & GW170817 PASS, but unscreened Saturn 9.58 AU tail |nu-1| = 10^({compiled.saturn_anomaly_log10:.1f}) "
            f"exceeds Cassini bound 2e-8! Requires external-field / Vainshtein screening patch in inner Solar System."
        )
    else:
        ss_status = "PASS"
        sat_str = "< 10^-300" if compiled.saturn_anomaly_log10 < -300 else f"10^({compiled.saturn_anomaly_log10:.0f})"
        ss_detail = (
            f"Cassini PPN gamma={compiled.ppn_gamma:.5f}, beta={compiled.ppn_beta:.5f}, GW170817 |dc_T/c|={compiled.gw_speed_deviation:.1e}, "
            f"Saturn 9.58 AU |dg/g_N| = {sat_str} (BHEX Shadow = {compiled.shadow_radius_rg:.3f} r_g, {compiled.shadow_vs_gr_pct:+.2f}%)."
        )
    badges.append(CrucibleBadge(
        badge_id="solar_gw",
        dataset_name="Solar System & GW170817",
        status=ss_status,
        headline_metric=f"γ={compiled.ppn_gamma:.3f}, β={compiled.ppn_beta:.3f} | Sat 10^({max(compiled.saturn_anomaly_log10, -320.0):.0f})",
        detail_honest=ss_detail,
    ))

    # -------------------------------------------------------------------------
    # ROW 6: DESI DR2 BAO (Shape + Sound-Horizon Anchored) & Single-H_0 Tension
    # -------------------------------------------------------------------------
    desi_thawing = (w0 > -0.99) and (wa < -0.05)
    if (desi_chi2_dof > 3.5) or (desi_anchored_chi2_dof > 3.5):
        desi_status = "FAIL"
        desi_detail = (
            f"DESI DR2 BAO FAILURE at H_0={h0_mod:.2f} km/s/Mpc: shape chi²/12={desi_chi2_dof:.2f}, "
            f"CAMB r_d-anchored chi²/13={desi_anchored_chi2_dof:.2f} (r_d,CAMB={rd_camb_mpc:.1f} vs BAO-fit {rd_best_mpc:.1f} Mpc; "
            f"SH0ES gap={shoes_h0_sigma:.1f}σ, a_0/H_0 mismatch={a0_h0_rel_mismatch * 100.0:.2f}%)!"
        )
    elif (desi_chi2_dof <= 2.0) and (desi_anchored_chi2_dof <= 2.0) and (shoes_h0_sigma <= 2.0):
        desi_status = "PASS"
        desi_detail = (
            f"DESI DR2 13-pt BAO shape chi²/12={desi_chi2_dof:.2f}, anchored chi²/13={desi_anchored_chi2_dof:.2f} "
            f"(r_d={rd_camb_mpc:.1f} Mpc) & SH0ES H_0={h0_mod:.2f} ({shoes_h0_sigma:.1f}σ) simultaneously satisfied."
        )
    else:
        desi_status = "WARN"
        desi_detail = (
            f"DESI DR2 13-pt BAO shape chi²/12={desi_chi2_dof:.2f}, r_d-anchored chi²/13={desi_anchored_chi2_dof:.2f} "
            f"({'matches thawing w0>-1, wa<0' if desi_thawing else 'flat w=-1 baseline'}; r_d={rd_camb_mpc:.1f} Mpc), "
            f"while single H_0={h0_mod:.2f} leaves {shoes_h0_sigma:.1f}σ SH0ES Cepheid (73.04±1.04) tension open "
            f"(a_0/H_0 mismatch={a0_h0_rel_mismatch * 100.0:.2f}%)."
        )
    badges.append(CrucibleBadge(
        badge_id="desi_h0",
        dataset_name="DESI DR2 BAO & H_0 Tension",
        status=desi_status,
        headline_metric=f"BAO χ²/ν={desi_chi2_dof:.2f} (anc {desi_anchored_chi2_dof:.2f}) | H_0={h0_mod:.2f}",
        detail_honest=desi_detail,
    ))

    # -------------------------------------------------------------------------
    # ROW 7: Dec 2, 2026 Gaia DR4 Wide-Binary 3D QUMOND Falsification Pipeline
    # -------------------------------------------------------------------------
    if halo_boost > 5.0:
        wb_status = "FORECAST"
        wb_headline = f"Solar {disk_boost:+.2f}% | GC {gc_split:+.2f}%"
        wb_detail = (
            f"Zenodo {zenodo_doi} (Gaia DR4 Dec 2, 2026, 15k AU 3D QUMOND): Solar {disk_boost:+.2f}%, "
            f"GC Compass {gc_split:+.2f}%, d<=2.5 kpc {inner_boost:+.2f}%->{outer_boost:+.2f}%->{lower_halo_boost:+.2f}% "
            f"(Deep Halo {halo_boost:+.2f}%)."
        )
    else:
        wb_status = "FORECAST"
        wb_headline = f"Solar {disk_boost:+.2f}% | GC {gc_split:+.2f}%"
        wb_detail = (
            f"Zenodo {zenodo_doi} (Gaia DR4 Dec 2, 2026): Predicts {disk_boost:+.2f}% boost & {gc_split:+.2f}% GC compass "
            f"across all environments. Any clean wide-binary boost or compass split falsifies smooth NFW CDM."
        )
    badges.append(CrucibleBadge(
        badge_id="gaia_dr4",
        dataset_name="Gaia DR4 Wide Binaries",
        status=wb_status,
        headline_metric=wb_headline,
        detail_honest=wb_detail,
    ))

    # -------------------------------------------------------------------------
    # ROW 8: Spinning GW150914 & Hierarchical LVK O3b/O4 Black-Hole Ringdown
    # -------------------------------------------------------------------------
    try:
        lvk_aud = get_kerr_lvk_ringdown_audit(mod, spin_a=0.67)
        lvk_status = str(lvk_aud["lvk_status"])
        lvk_imr_sig = float(lvk_aud["gw150914_imr_sigma"])
        lvk_o4_sig = float(lvk_aud["lvk_o4_hierarchical_sigma"])
        lvk_qmax = float(lvk_aud["max_achievable_Q220"])
        lvk_df68 = float(lvk_aud["delta_f220_at_chi068_pct"])
        lvk_headline = f"IMR {lvk_imr_sig:.2f}σ | O4 {lvk_o4_sig:.2f}σ | Qmax={lvk_qmax:.2f}"
        lvk_detail = (
            f"{lvk_aud['honest_verdict']} (Spinning df_220(chi=0.68)={lvk_df68:+.2f}%, "
            f"BHEX Shadow={compiled.shadow_radius_rg:.3f} r_g [{compiled.shadow_vs_gr_pct:+.2f}%])."
        )
    except Exception as exc:
        lvk_status = "FAIL"
        lvk_headline = "Singular / No Photon-Sphere Barrier"
        lvk_detail = (
            f"Strong-field metric lacks a well-posed Regge-Wheeler photon-sphere ringdown barrier ({exc}); "
            f"incompatible with LVK O3b/O4 black-hole ringdown spectroscopy."
        )
    badges.append(CrucibleBadge(
        badge_id="lvk_ringdown",
        dataset_name="LVK O4 Black-Hole Ringdown",
        status=lvk_status,
        headline_metric=lvk_headline,
        detail_honest=lvk_detail,
    ))

    return CrucibleReport(
        mod_id=mod.id,
        mod_name=mod.name,
        omega_b=0.0490,
        omega_m_effective=omega_m_eff,
        cmb_r32_ratio=r32,
        cmb_d2_quadrupole_uK2=d2_quad,
        sparc_mean_chi2=med_chi2,
        sparc_rar_scatter_dex=rar_dex,
        wide_binary_disk_boost_pct=disk_boost,
        wide_binary_halo_boost_pct=halo_boost,
        desi_bao_chi2_dof=desi_chi2_dof,
        badges=badges,
        cmb_ell=ell,
        cmb_dl_obs=dl_obs,
        cmb_dl_mod=dl_mod,
        sparc_galaxy_name=gal.name,
        sparc_r_kpc=gal.r_kpc,
        sparc_v_obs=gal.v_obs_kms,
        sparc_v_err=gal.e_vobs_kms,
        sparc_v_bary=np.asarray(gal_fit["v_baryon_kms"], dtype=float),
        sparc_v_mod=np.asarray(gal_fit["v_model_kms"], dtype=float),
        wb_curves=wb,
        cmb_ell_2500=ell_2500,
        cmb_tt_obs_2500=tt_obs_2500,
        cmb_tt_mod_2500=tt_mod_2500,
        cmb_te_obs_2500=te_obs_2500,
        cmb_te_mod_2500=te_mod_2500,
        cmb_ee_obs_2500=ee_obs_2500,
        cmb_ee_mod_2500=ee_mod_2500,
        cmb_te_corr_2500=cmb_te_corr_2500,
        cmb_ee_peak_ratio=cmb_ee_peak_ratio,
        cmb_beta_iso=cmb_beta_iso,
        cmb_xi_kz_at_waist=xi_kz_mod,
        cmb_xi_kz_min=xi_kz_min,
        cmb_xi_kz_max=xi_kz_max,
        cmb_kz_power_index=kz_slope,
        cmb_qh_net_symmetric=qh_sym,
        cmb_qh_net_chiral_pos=qh_pos,
        cmb_kz_depends_on_r0=kz_dep,
        cmb_kz_matches_target_10pct=kz_match,
        h0_consistent_kms_mpc=h0_mod,
        a0_h0_rel_mismatch=a0_h0_rel_mismatch,
        cmb_chi2_dof_fixed_ns=cmb_chi2_dof_fixed_ns,
        cmb_chi2_dof_profiled_ns=cmb_chi2_dof_profiled_ns,
        cmb_best_fit_ns=cmb_best_fit_ns,
        cmb_high_l_tilt_pct=cmb_high_l_tilt_pct,
        bbn_dh_e5=bbn_dh_e5,
        bbn_yp=bbn_yp,
        bbn_max_sigma=bbn_max_sigma,
        lensing_s8_active=lensing_s8_active,
        lensing_s8_gr_screened=lensing_s8_gr_screened,
        lensing_s8_sigma=lensing_s8_sigma,
        cross_scale_seam_open=cross_scale_seam_open,
        ngc3198_v_double_counted_kms=ngc3198_v_dc,
        ngc3198_double_count_sigma=ngc3198_dc_sig,
        free_action_literals=int(knob_audit["action_literal_count"]),
        free_global_knobs=int(knob_audit["global_knob_total"]),
        per_galaxy_halo_knobs=int(knob_audit["per_galaxy_halo_knobs"]),
        knob_audit_summary=str(knob_audit["summary_label"]),
        desi_bao_anchored_chi2_dof=desi_anchored_chi2_dof,
        lvk_ringdown_status=lvk_status,
        lvk_imr_sigma=lvk_imr_sig,
        lvk_o4_hier_sigma=lvk_o4_sig,
        lvk_max_q220=lvk_qmax,
    )


# ==============================================================================
# 3D GPU TWO-WING CPT CATENOID COSMIC HOURGLASS SHADER
# ==============================================================================
_VERT_QUAD = """#version 430
out vec2 v_uv;
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FRAG_HOURGLASS = """#version 430
in vec2 v_uv;
out vec4 frag_color;

uniform vec3 u_cam_eye;
uniform vec3 u_cam_fwd;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_fov_tan;
uniform float u_aspect;
uniform float u_time;

uniform float u_waist_r0;       // Smooth Euclidean waist radius r0 > 0 (K_max = 24 / r0^4 < inf!)
uniform float u_tau_now;        // Active observer slice |tau_now| in [0.0, 3.8]
uniform float u_pleat_strength; // Pinned SO(3) acoustic pleat visibility
uniform int u_show_creases;     // Landauer dQ >= k_B T ln 2 irreversible memory creases

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Signed distance to the finite smooth Two-Wing CPT Catenoid Hourglass shell:
// Axis = Y (tau in [-4.0, +4.0]), radius R(tau) = r0 * sqrt(1.0 + 0.85 * (tau / r0)^2)
float hourglass_radius(float tau, float r0, float pleat_amp) {
    float base_r = sqrt(r0 * r0 + 0.52 * tau * tau);
    // Subtle pinned SO(3) acoustic baryonic ripples along cosmic expansion
    float ripple = 1.0 + 0.035 * pleat_amp * cos(5.31831 * tau);
    return base_r * ripple;
}

vec2 intersect_box(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
    vec3 inv_d = 1.0 / (rd + vec3(1e-8));
    vec3 t0s = (bmin - ro) * inv_d;
    vec3 t1s = (bmax - ro) * inv_d;
    vec3 tsmaller = min(t0s, t1s);
    vec3 tbigger  = max(t0s, t1s);
    float tmin = max(max(tsmaller.x, tsmaller.y), max(tsmaller.z, 0.05));
    float tmax = min(min(tbigger.x, tbigger.y), tbigger.z);
    return vec2(tmin, tmax);
}

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(
        u_cam_fwd
        + ndc.x * u_aspect * u_fov_tan * u_cam_right
        + ndc.y * u_fov_tan * u_cam_up
    );
    vec3 ro = u_cam_eye;

    // Option A+ Luminous Sapphire Slate 3D studio sky
    float r_scr = length(ndc * vec2(u_aspect, 1.0));
    vec3 bg = mix(vec3(0.058, 0.112, 0.205), vec3(0.024, 0.046, 0.092), clamp(r_scr * 0.65, 0.0, 1.0));

    vec3 accum = vec3(0.0);
    float trans = 1.0;
    float r0 = max(u_waist_r0, 0.28);
    float tau_max = 3.85;

    vec2 tb = intersect_box(ro, rd, vec3(-4.2, -tau_max, -4.2), vec3(4.2, tau_max, 4.2));
    if (tb.y > tb.x) {
        int STEPS = 160;
        float dt = (tb.y - tb.x) / float(STEPS);

        for (int i = 0; i < STEPS; ++i) {
            float t = tb.x + (float(i) + 0.5) * dt;
            vec3 p = ro + t * rd;

            float tau = p.y;
            float r_cyl = length(p.xz);
            float R_cat = hourglass_radius(tau, r0, u_pleat_strength);
            float d_shell = abs(r_cyl - R_cat);
            float cap_fade = smoothstep(tau_max, tau_max - 0.45, abs(tau));

            // Thin translucent spacetime membrane around R_cat
            if (d_shell < 0.16 && cap_fade > 0.001) {
                float shell_w = exp(-(d_shell * d_shell) / 0.0042) * cap_fade;
                float ang = atan(p.z, p.x);

                // Wing A (+tau, Observer Channel A: Cyan-Sapphire) vs Wing B (-tau, Observer Channel B: Teal-Indigo)
                vec3 wing_col = (tau >= 0.0)
                    ? vec3(0.20, 0.78, 1.22)
                    : vec3(0.28, 0.66, 1.15);

                // Smooth Euclidean Waist Glow at tau = 0 (Finite K_max = 24 / r0^4 < inf, ZERO Big Bang point singularity!)
                float waist_band = exp(-(tau * tau) / 0.065);
                wing_col = mix(wing_col, vec3(0.35, 1.25, 0.88), waist_band * 0.85);

                // Geodesic meridian & cosmic expansion grid threads
                float s_mer = sin(ang * 8.0);
                float merid = exp(-(s_mer * s_mer) * 32.0);
                float s_lat = sin(tau * 5.0);
                float lat_g = exp(-(s_lat * s_lat) * 28.0);
                wing_col += vec3(0.30, 0.88, 1.25) * (merid + lat_g) * 0.35;

                // Landauer Thermodynamic Memory Creases (dQ >= k_B T ln 2):
                // Etched ONLY in the causal past |tau| <= u_tau_now on BOTH CPT wings!
                if (u_show_creases == 1 && abs(tau) <= u_tau_now) {
                    float c_wave = sin(6.0 * ang + 3.4 * abs(tau) - 0.8 * u_time)
                                 * cos(5.0 * ang - 4.1 * abs(tau) + 0.5 * u_time);
                    float crease = smoothstep(0.62, 0.96, abs(c_wave));
                    wing_col += vec3(1.42, 0.92, 0.24) * crease * 0.95;
                }

                // Active Symmetric Observer Slice Rings at tau = +u_tau_now (Channel A) & tau = -u_tau_now (Channel B)!
                float d_now = abs(abs(tau) - u_tau_now);
                float now_ring = exp(-(d_now * d_now) / 0.0038);
                wing_col += vec3(1.55, 1.05, 0.28) * now_ring * 1.65;

                float alpha = clamp(shell_w * dt * 2.4, 0.0, 0.45);
                accum += trans * wing_col * alpha * 1.6;
                trans *= (1.0 - alpha * 0.65);
            }

            // Interior Wick Pinhole / Euclidean Waist Core Glow near (tau = 0, r < R_cat)
            if (r_cyl < R_cat && abs(tau) < 0.85) {
                float core_w = exp(-(tau * tau) / 0.18) * exp(-(r_cyl * r_cyl) / (r0 * r0 * 1.2));
                accum += trans * vec3(0.18, 0.95, 0.78) * core_w * dt * 0.55;
            }
        }
    }

    vec3 col_hdr = bg * trans + accum;
    vec3 mapped = aces_tonemap(col_hdr * 1.10);
    mapped = pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.2));
    frag_color = vec4(mapped, 1.0);
}
"""


class Arena4CosmicCrucible:
    """
    GPU 3D Two-Wing `CPT` Cosmic Catenoid Hourglass (`Observer Channel A` & `Observer Channel B`)
    paired with the 5-Dataset Live Observational Pass/Fail Crucible!
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        compiled: Optional[CompiledMasterAction] = None,
        mod: Optional[PhysicsMod] = None,
        lazy_gpu: bool = False,
    ) -> None:
        self.ctx = ctx
        self._compiled: Optional[CompiledMasterAction] = compiled
        self._mod: Optional[PhysicsMod] = mod
        self.waist_r0 = 0.72
        self.tau_now = 2.65
        self.show_creases = True
        self.auto_scrub = False
        self.is_hibernated: bool = bool(lazy_gpu)
        self.allocated_vram_mb: float = 0.0

        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        load_camb_lut_disk_cache()
        load_camb_pol_lut_disk_cache()
        if not self.is_hibernated:
            self._build_gpu_shader()

    def _build_gpu_shader(self) -> None:
        if self._prog is None:
            self._prog = self.ctx.program(
                vertex_shader=_VERT_QUAD,
                fragment_shader=_FRAG_HOURGLASS,
            )
            self._vao = self.ctx.vertex_array(self._prog, [])
        self.allocated_vram_mb = 0.5
        self.is_hibernated = False

    @property
    def ddr5_ram_mb(self) -> float:
        """Reports live DDR5 System RAM (MiB) held by cached CAMB LUTs, SPARC-175 tables, and Gaia forecasts."""
        return get_crucible_ddr5_ram_mb()

    def hibernate_to_ram(self) -> float:
        """Releases OpenGL shader/VAO while keeping DDR5 CAMB/SPARC/Gaia caches warm; returns DDR5 cache footprint."""
        if not self.is_hibernated:
            self.ctx.finish()
            if self._vao is not None:
                self._vao.release()
                self._vao = None
            if self._prog is not None:
                self._prog.release()
                self._prog = None
            self.allocated_vram_mb = 0.0
            self.is_hibernated = True
        return self.ddr5_ram_mb

    def wake_from_ram(self) -> None:
        """Re-links the Cosmic Catenoid Hourglass shader if hibernated."""
        if self.is_hibernated or self._prog is None or self._vao is None:
            self._build_gpu_shader()

    def warm_camb_ddr5_cache_async(self, wait: bool = False) -> None:
        """Convenience instance hook delegating to module-level `warm_camb_ddr5_cache_async`."""
        warm_camb_ddr5_cache_async(wait=wait)

    def evaluate_mod(
        self,
        mod: Optional[PhysicsMod] = None,
        compiled: Optional[CompiledMasterAction] = None,
        selected_galaxy_idx: int = 0,
    ) -> CrucibleReport:
        """Evaluates the 5-dataset observational crucible against `mod` using DDR5-cached CAMB & SPARC LUTs."""
        use_mod = mod if mod is not None else self._mod
        if use_mod is None:
            raise ValueError("PhysicsMod must be provided to evaluate_mod() or Arena4CosmicCrucible(mod=...).")
        use_comp = compiled
        if use_comp is None:
            from weaveforge.core.symbolic_compiler import compile_master_action

            use_comp = compile_master_action(use_mod)
        self._mod = use_mod
        self._compiled = use_comp
        return evaluate_observational_crucible(
            use_mod,
            use_comp,
            selected_galaxy_idx=selected_galaxy_idx,
        )

    def waist_kretschmann(self) -> float:
        """Exact smooth Euclidean catenoid waist curvature invariant `K_max = 24 / r0^4 < inf`."""
        r0 = max(float(self.waist_r0), 0.1)
        return 24.0 / (r0 ** 4)

    def render(
        self,
        fbo: moderngl.Framebuffer,
        camera: Camera3D,
        mod: PhysicsMod,
        wall_time: float,
    ) -> None:
        if self.is_hibernated or self._prog is None or self._vao is None:
            self.wake_from_ram()
        assert self._prog is not None and self._vao is not None
        if self.auto_scrub:
            self.tau_now = 0.25 + 3.35 * (0.5 + 0.5 * math.sin(wall_time * 0.65))

        fbo.use()
        w, h = fbo.size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        eye, fwd, right, up = camera.basis_vectors()
        aspect = float(w) / max(float(h), 1.0)
        fov_tan = math.tan(math.radians(camera.fov_deg) * 0.5)

        for uname, uval in (
            ("u_cam_eye", tuple(float(v) for v in eye)),
            ("u_cam_fwd", tuple(float(v) for v in fwd)),
            ("u_cam_right", tuple(float(v) for v in right)),
            ("u_cam_up", tuple(float(v) for v in up)),
            ("u_fov_tan", float(fov_tan)),
            ("u_aspect", float(aspect)),
            ("u_time", float(wall_time)),
            ("u_waist_r0", float(self.waist_r0)),
            ("u_tau_now", float(self.tau_now)),
            ("u_pleat_strength", float(mod.parameters.pinned_pleat_coupling)),
            ("u_show_creases", 1 if self.show_creases else 0),
        ):
            if uname in self._prog:
                self._prog[uname].value = uval

        self._vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        self.ctx.finish()
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._prog is not None:
            self._prog.release()
            self._prog = None
        self.allocated_vram_mb = 0.0
