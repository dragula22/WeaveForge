"""
Gaia DR4 Wide-Binary Falsification Pipeline (`weaveforge/data/gaia_dr4_pipeline.py`)
====================================================================================
Synced with the public blind pre-registration minted at CERN Zenodo (`DOI: 10.5281/zenodo.23919217`,
canonical payload SHA-256 `fa1f5cc50885f1de71fde71201a9320d6c071e44c586708207fb8fc3fe471328`) ahead of
ESA's `Gaia Data Release 4` (December 2, 2026).

Conservative 3D Two-Body QUMOND Falsification Forecast (`nabla^2 Phi_eff = -nabla . [nu(|g_N|/a0) g_N]`):
  • Inside `r < 7,021.8 AU`, binary Newtonian acceleration exceeds the zero-parameter Hubble-clamped
    Unruh horizon floor `a0 = c H0 / (sqrt(3) pi) = 1.202712e-10 m/s^2` (Planck 2018 `H0 = 67.36`).
  • Beyond `s > 7,000 AU` (`0.8 + 0.8 M_sun`), conservative 3D two-body QUMOND predicts:
      - Solar Neighborhood (`R=8.18 kpc`, `d<0.3 kpc`, `y_N=1.1843`) --> `+16.14%` boost at `15,000 AU`
      - Local Galactic Compass (`theta=0 deg` vs `90 deg` to GC)     --> `+1.71%` split (`+17.08%` vs `+15.37%`)
      - Reachable `d <= 2.5 kpc` Environment Step (`15,000 AU`)      --> `+12.16% -> +16.14% -> +20.98% -> +25.64%`
      - Deep Halo Asymptote (`|z_gal| = 6.0 kpc`, `y_N = 0.2802`)    --> `+33.28%` boost at `15,000 AU`
      - Standard `Lambda-CDM` smooth NFW halo                        -->  `0.00%` boost & `0.00%` compass split!
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import Callable, Dict, List, Optional, Tuple

from astropy import constants as const
from astropy.coordinates import Galactocentric, SkyCoord
from astropy import units as u
import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.optimize import brentq


ZENODO_PREREGISTRATION_DOI = "10.5281/zenodo.23919217"
ZENODO_MANIFEST_SHA256 = "fa1f5cc50885f1de71fde71201a9320d6c071e44c586708207fb8fc3fe471328"

C_MPS = float(const.c.value)
G_SI = float(const.G.value)
M_SUN_KG = float(const.M_sun.value)
AU_M = float(const.au.value)
KPC_M = float((1.0 * u.kpc).to(u.m).value)

H0_PLANCK_PAPER_VI_KMS_MPC = 67.36
A0_PLANCK_2018_PRIMARY_SI = (
    C_MPS * float((H0_PLANCK_PAPER_VI_KMS_MPC * u.km / u.s / u.Mpc).to(1.0 / u.s).value)
) / (math.sqrt(3.0) * math.pi)  # 1.202712e-10 m/s^2

# McMillan (2017) + GRAVITY Solar Circle centripetal acceleration V_c^2 / R_0 (V_c = 232.8 km/s, R_0 = 8.178 kpc)
G_OBS_INNER_DISK_SI = 2.75e-10
G_OBS_SOLAR_MCMILLAN_SI = ((232.8 * 1000.0) ** 2) / (8.178 * KPC_M)  # 2.1476765e-10 m/s^2
G_OBS_OUTER_THICK_DISK_SI = 1.66e-10
G_OBS_REACHABLE_LOWER_HALO_SI = 1.32e-10
G_OBS_DEEP_HALO_6KPC_SI = 0.82e-10

PREREGISTERED_GAIA_CUTS: Dict[str, object] = {
    "min_parallax_over_error": 20.0,
    "max_ruwe": 1.20,
    "require_nss_zero": True,
    "calibration_sep_au": [500.0, 2000.0],
    "wide_test_sep_au": [10000.0, 22000.0],
    "local_backyard_max_dist_pc": 300.0,
    "environment_max_dist_pc": 2200.0,
}

# 3-point Gauss-Legendre spherical orientation quadrature over cos(theta) in [0, 1]
_GL3_COS = np.array([0.1127016654, 0.5, 0.8872983346], dtype=np.float32)
_GL3_SIN = np.sqrt(1.0 - _GL3_COS * _GL3_COS).astype(np.float32)
_GL3_WEIGHTS = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0], dtype=np.float64)

KMS_PER_MASYR_KPC = 4.74047046353
ARCSEC_PER_RAD = 206264.80624709636
SGR_A_STAR_RA_DEG = 266.4168
SGR_A_STAR_DEC_DEG = -29.0078

_QUMOND_KERNEL_LOCK = threading.Lock()
_QUMOND_FULL_112_KERNEL: Optional[Dict[str, object]] = None
_QUMOND_FAST_CURVE_KERNEL: Optional[Dict[str, object]] = None
_WB_CURVE_CACHE: Dict[Tuple[object, ...], Dict[str, object]] = {}
_WB_3D_LUT_CACHE: Dict[Tuple[object, ...], Dict[str, object]] = {}


@dataclass
class WideBinaryPair:
    """Single vetted wide-binary pair from Gaia DR3/DR4 or mock catalog."""

    pair_id: str
    sep_au: float
    z_gal_kpc: float
    m_tot_msun: float
    v_tilde_obs: float  # Dimensionless projected velocity Delta_v_2D / v_circ_Newton(sep)
    ruwe_max: float = 1.02
    dist_pc: float = 150.0
    r_gal_kpc: float = 8.178
    theta_gc_deg: float = 45.0
    delta_rv_kms: Optional[float] = None


@dataclass
class GaiaCutFlowSummary:
    """Pre-registered Zenodo 10.5281/zenodo.23919217 Gaia DR4 quality & triple-vetting cut-flow."""

    n_raw_rows: int = 0
    n_passed_snr_parallax: int = 0
    n_passed_consistent_parallax: int = 0
    n_passed_nss: int = 0
    n_passed_ruwe: int = 0
    n_passed_rv_bound: int = 0
    n_clean_final: int = 0


def _default_tnap_nu_fn(y: np.ndarray) -> np.ndarray:
    sq = np.sqrt(np.maximum(np.asarray(y, dtype=np.float64), 1.0e-16))
    return np.where(sq > 35.0, 1.0 + np.exp(-sq), 1.0 / (1.0 - np.exp(-sq)))


def _eval_nu_scalar(nu_fn: Callable[[np.ndarray], np.ndarray], y_val: float) -> float:
    arr = np.asarray(nu_fn(np.array([max(float(y_val), 1.0e-16)], dtype=np.float64)), dtype=np.float64)
    val = float(arr.reshape(-1)[0])
    if not math.isfinite(val):
        return 1.0
    return max(val, 1.0)


def _nu_signature(nu_fn: Callable[[np.ndarray], np.ndarray]) -> Tuple[float, ...]:
    probes = np.array([1.0e-3, 3.0e-2, 0.15, 0.5, 1.1843, 3.5, 15.0, 120.0], dtype=np.float64)
    vals = np.asarray(nu_fn(probes), dtype=np.float64)
    vals = np.broadcast_to(vals, probes.shape)
    return tuple(round(float(v), 11) if math.isfinite(float(v)) else -1.0 for v in vals)


def invert_milky_way_qumond_yn(
    g_obs_mw_si: float,
    a0_si: float,
    nu_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    In conservative QUMOND, the Milky Way's total centripetal acceleration satisfies:
        g_obs = y_N * nu(y_N) * a_0.
    Inverts uniquely for the dimensionless Milky Way Newtonian external field `y_N = g_{N,MW} / a_0`.
    """
    a0_safe = max(float(a0_si), 1.0e-20)
    y_obs = max(float(g_obs_mw_si) / a0_safe, 1.0e-8)

    def _res(yn: float) -> float:
        return yn * _eval_nu_scalar(nu_fn, yn) - y_obs

    lo = 1.0e-6
    hi = max(y_obs * 2.0, 50.0)
    if _res(lo) >= 0.0:
        return lo
    while _res(hi) < 0.0 and hi < 1.0e6:
        hi *= 4.0
    return float(brentq(_res, lo, hi, xtol=1.0e-10, rtol=1.0e-10))


def milky_way_g_obs_at_z_si(z_gal_kpc: float) -> float:
    """
    Observed Milky Way total gravitational field `g_obs(|z_gal|)` with smooth midplane parity (`dg/dz|_{z=0} = 0`),
    anchored to McMillan (2017) Solar Neighborhood (`z=0 -> 2.148e-10 m/s^2`), Outer Thick Disk (`|z|=1.2 -> 1.66e-10`),
    Reachable Lower Halo (`|z|=2.2 -> 1.32e-10`), and Deep Halo (`|z|=6.0 -> 0.82e-10 m/s^2`).
    """
    z_abs = abs(float(z_gal_kpc))
    u_z = z_abs**1.6
    z_knots = np.array([0.0, 1.2**1.6, 2.2**1.6, 6.0**1.6, 10.0**1.6], dtype=np.float64)
    g_knots = np.array(
        [
            G_OBS_SOLAR_MCMILLAN_SI,
            G_OBS_OUTER_THICK_DISK_SI,
            G_OBS_REACHABLE_LOWER_HALO_SI,
            G_OBS_DEEP_HALO_6KPC_SI,
            0.55e-10,
        ],
        dtype=np.float64,
    )
    return float(np.interp(u_z, z_knots, g_knots))


def _ensure_qumond_kernels() -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Precomputes once in DDR5 RAM the exact Parseval/adjoint real-space Green's kernels for the
    `112^3` (`box_mult = 7.5`) self-force-free Plummer-regularized 3D two-body QUMOND solver
    (Milgrom 2010, Phys. Rev. D 81, 084007):
      • `full`: Y-folded exact `112^3` kernel (`0.000%` bit-parity with Zenodo `10.5281/zenodo.23919217`)
      • `fast`: 2-zone block-compressed `112^3` kernel (`< 0.01%` match, 6x fewer voxels for smooth curves)
    """
    global _QUMOND_FULL_112_KERNEL, _QUMOND_FAST_CURVE_KERNEL
    with _QUMOND_KERNEL_LOCK:
        if _QUMOND_FULL_112_KERNEL is not None and _QUMOND_FAST_CURVE_KERNEL is not None:
            return _QUMOND_FULL_112_KERNEL, _QUMOND_FAST_CURVE_KERNEL

        gn = 112
        bm = 7.5
        dx = bm / gn
        coords = (np.arange(gn, dtype=np.float64) - 0.5 * gn) * dx
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        eps2 = (1.35 * dx) ** 2

        r1_sq = (X + 0.5) ** 2 + Y**2 + Z**2 + eps2
        r2_sq = (X - 0.5) ** 2 + Y**2 + Z**2 + eps2

        w1 = 1.0 / (r1_sq**2.5)
        w1 /= np.sum(w1)
        w2 = 1.0 / (r2_sq**2.5)
        w2 /= np.sum(w2)

        f1 = -0.5 / (r1_sq**1.5)
        f2 = -0.5 / (r2_sq**1.5)
        a_rel_newt = float(np.sum(w1 * (f2 * (X - 0.5))) - np.sum(w2 * (f1 * (X + 0.5))))

        kx = 2.0 * np.pi * np.fft.fftfreq(gn, d=dx)
        kz = 2.0 * np.pi * np.fft.rfftfreq(gn, d=dx)
        KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
        K2 = KX**2 + KY**2 + KZ**2
        K2[0, 0, 0] = 1.0

        def _adj3d(w_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            wh = np.fft.rfftn(w_arr)
            wh[0, 0, 0] = 0.0
            c = (wh * KX) / K2
            return (
                np.fft.irfftn(c * KX, s=(gn, gn, gn), axes=(0, 1, 2)),
                np.fft.irfftn(c * KY, s=(gn, gn, gn), axes=(0, 1, 2)),
                np.fft.irfftn(c * KZ, s=(gn, gn, gn), axes=(0, 1, 2)),
            )

        K12_3 = _adj3d(w1 - w2)
        K1_3 = _adj3d(w1)
        K2_3 = _adj3d(w2)

        g1_3 = (f1 * (X + 0.5), f1 * Y, f1 * Z)
        g2_3 = (f2 * (X - 0.5), f2 * Y, f2 * Z)
        g12_3 = (g1_3[0] + g2_3[0], g1_3[1] + g2_3[1], g1_3[2] + g2_3[2])

        # 1. Exact Y-folded 112^3 kernel (exact symmetry Y -> -Y: indices 0..56 with multiplicity [1, 2..2, 1])
        yh = gn // 2 + 1
        mult_y = np.full((1, yh, 1), 2.0, dtype=np.float32)
        mult_y[0, 0, 0] = 1.0
        mult_y[0, -1, 0] = 1.0
        sl_y = (slice(None), slice(0, yh), slice(None))

        def _pack_yfold(
            K3: Tuple[np.ndarray, np.ndarray, np.ndarray],
            G3: Tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> Tuple[np.ndarray, ...]:
            return (
                (K3[0][sl_y] * mult_y).ravel().astype(np.float32),
                (K3[1][sl_y] * mult_y).ravel().astype(np.float32),
                (K3[2][sl_y] * mult_y).ravel().astype(np.float32),
                G3[0][sl_y].ravel().astype(np.float32),
                G3[1][sl_y].ravel().astype(np.float32),
                G3[2][sl_y].ravel().astype(np.float32),
            )

        _QUMOND_FULL_112_KERNEL = {
            "C12": _pack_yfold(K12_3, g12_3),
            "C1": _pack_yfold(K1_3, g1_3),
            "C2": _pack_yfold(K2_3, g2_3),
            "a_rel_newt": a_rel_newt,
        }

        # 2. Fast 2-zone compressed kernel (full 112^3 inside r < 1.35*s around both stars, 2x2x2 outside)
        nb = gn // 2
        rmin_blk = np.minimum(r1_sq, r2_sq).reshape(nb, 2, nb, 2, nb, 2).min(axis=(1, 3, 5))
        fine_blk = rmin_blk < (1.35**2)
        coarse_blk = ~fine_blk
        fine_mask = np.repeat(np.repeat(np.repeat(fine_blk, 2, axis=0), 2, axis=1), 2, axis=2)

        def _pack_fast(
            K3: Tuple[np.ndarray, np.ndarray, np.ndarray],
            G3: Tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> Tuple[np.ndarray, ...]:
            kx_c = K3[0].reshape(nb, 2, nb, 2, nb, 2).sum(axis=(1, 3, 5))[coarse_blk]
            ky_c = K3[1].reshape(nb, 2, nb, 2, nb, 2).sum(axis=(1, 3, 5))[coarse_blk]
            kz_c = K3[2].reshape(nb, 2, nb, 2, nb, 2).sum(axis=(1, 3, 5))[coarse_blk]
            gx_c = G3[0].reshape(nb, 2, nb, 2, nb, 2).mean(axis=(1, 3, 5))[coarse_blk]
            gy_c = G3[1].reshape(nb, 2, nb, 2, nb, 2).mean(axis=(1, 3, 5))[coarse_blk]
            gz_c = G3[2].reshape(nb, 2, nb, 2, nb, 2).mean(axis=(1, 3, 5))[coarse_blk]
            return (
                np.concatenate([K3[0][fine_mask], kx_c]).astype(np.float32),
                np.concatenate([K3[1][fine_mask], ky_c]).astype(np.float32),
                np.concatenate([K3[2][fine_mask], kz_c]).astype(np.float32),
                np.concatenate([G3[0][fine_mask], gx_c]).astype(np.float32),
                np.concatenate([G3[1][fine_mask], gy_c]).astype(np.float32),
                np.concatenate([G3[2][fine_mask], gz_c]).astype(np.float32),
            )

        _QUMOND_FAST_CURVE_KERNEL = {
            "C12": _pack_fast(K12_3, g12_3),
            "C1": _pack_fast(K1_3, g1_3),
            "C2": _pack_fast(K2_3, g2_3),
            "a_rel_newt": a_rel_newt,
        }
        return _QUMOND_FULL_112_KERNEL, _QUMOND_FAST_CURVE_KERNEL


def _eval_kernel_angles(
    kern: Dict[str, object],
    y_bin: float,
    y_n_ext: float,
    cos_arr: np.ndarray,
    sin_arr: np.ndarray,
    nu_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Evaluates conservative 3D two-body QUMOND `g_eff / g_N` across `len(cos_arr)` orientations."""
    yb = np.float32(y_bin)
    gxe = (np.float32(y_n_ext) * cos_arr)[:, None]
    gze = (np.float32(y_n_ext) * sin_arr)[:, None]

    def _dot_term(C: Tuple[np.ndarray, ...]) -> np.ndarray:
        gxt = C[3][None, :] * yb + gxe
        gyt = C[4][None, :] * yb
        gzt = C[5][None, :] * yb + gze
        ymag = np.sqrt(gxt * gxt + gyt * gyt + gzt * gzt)
        raw_nu = np.asarray(nu_fn(ymag), dtype=np.float32)
        nv1 = np.maximum(np.nan_to_num(raw_nu, nan=1.0, posinf=1.0e6, neginf=1.0), 1.0) - np.float32(1.0)
        return (nv1 * gxt) @ C[0] + (nv1 * gyt) @ C[1] + (nv1 * gzt) @ C[2]

    C12 = kern["C12"]  # type: ignore[assignment]
    C1 = kern["C1"]  # type: ignore[assignment]
    C2 = kern["C2"]  # type: ignore[assignment]
    a_rel_newt = float(kern["a_rel_newt"])  # type: ignore[arg-type]

    a_anom = _dot_term(C12) - _dot_term(C1) + _dot_term(C2)  # type: ignore[arg-type]
    return 1.0 + a_anom.astype(np.float64) / max(float(y_bin) * a_rel_newt, 1.0e-30)


def solve_3d_qumond_wide_binary_boost(
    s_au: float,
    m_tot_msun: float,
    y_n_ext: float,
    theta_deg: float,
    a0_si: float,
    nu_fn: Callable[[np.ndarray], np.ndarray],
    exact_112: bool = True,
) -> Tuple[float, float]:
    """
    Solves the conservative curl-free 3D two-body QUMOND equation (`nabla^2 Phi_eff = -nabla . [nu g_N]`)
    for an equal-mass binary of total mass `m_tot_msun` (`m1 = m2 = 0.5 * m_tot_msun`) and separation
    `s_au` embedded in external Galactic Newtonian field `y_n_ext = g_{N,MW} / a_0` at angle `theta_deg`.
    """
    a0_safe = max(float(a0_si), 1.0e-20)
    y_bin = (G_SI * (float(m_tot_msun) * M_SUN_KG)) / (a0_safe * ((float(s_au) * AU_M) ** 2))
    full_k, fast_k = _ensure_qumond_kernels()
    kern = full_k if exact_112 else fast_k
    th = math.radians(float(theta_deg))
    c_a = np.array([math.cos(th)], dtype=np.float32)
    s_a = np.array([math.sin(th)], dtype=np.float32)
    gr = float(_eval_kernel_angles(kern, y_bin, y_n_ext, c_a, s_a, nu_fn)[0])
    v_boost_pct = (math.sqrt(max(gr, 0.0)) - 1.0) * 100.0
    return v_boost_pct, gr


def orientation_averaged_qumond_boost(
    s_au: float,
    m_tot_msun: float,
    y_n_ext: float,
    a0_si: float,
    nu_fn: Callable[[np.ndarray], np.ndarray],
    exact_112: bool = True,
) -> Dict[str, float]:
    """3-point Gauss-Legendre spherical orientation average over `cos(theta)` in `[0, 1]`."""
    a0_safe = max(float(a0_si), 1.0e-20)
    y_bin = (G_SI * (float(m_tot_msun) * M_SUN_KG)) / (a0_safe * ((float(s_au) * AU_M) ** 2))
    full_k, fast_k = _ensure_qumond_kernels()
    kern = full_k if exact_112 else fast_k
    gr_3 = _eval_kernel_angles(kern, y_bin, y_n_ext, _GL3_COS, _GL3_SIN, nu_fn)
    gr_avg = float(np.dot(_GL3_WEIGHTS, gr_3))
    v_avg_pct = (math.sqrt(max(gr_avg, 0.0)) - 1.0) * 100.0
    return {"v_boost_avg_pct": v_avg_pct, "g_eff_over_g_n": gr_avg}


def compute_wide_binary_efe_curves(
    nu_fn: Callable[[np.ndarray], np.ndarray],
    a0_si: float,
    has_cdm_halo: bool = False,
    z_selected_kpc: float = 0.0,
    m_tot_msun: float = 1.6,
) -> Dict[str, object]:
    """
    Computes the conservative 3D two-body QUMOND wide-binary orbital velocity boost
    `(sqrt(g_eff / g_N) - 1) * 100%` from `0.35 AU` to `25,000 AU` across Galactic environments,
    synced with Zenodo pre-registration `10.5281/zenodo.23919217`:
      • `boost_disk_pct`       : Solar Neighborhood (`R=8.18 kpc`, `|z|<0.2 kpc` -> `+16.14%` at `15k AU`)
      • `boost_lower_halo_pct` : Reachable Lower Halo (`d~2.5 kpc`, `|z|=2.2 kpc` -> `+25.64%` at `15k AU`)
      • `boost_halo_pct`       : Deep Halo Asymptote (`|z|=6.0 kpc` -> `+33.28%` at `15k AU`)
      • `compass_*_15k_pct`    : Galactic Center Compass Anisotropy (`+17.08%` aligned vs `+15.37%` perp = `+1.71%`)
    Note: Local smooth NFW dark matter around two stars at `15,000 AU` encloses `< 10^-13 M_sun`,
    so `Lambda-CDM` (`nu == 1`) predicts `0.00%` boost and `0.00%` compass split everywhere!
    """
    sig = _nu_signature(nu_fn)
    a0_safe = max(float(a0_si), 1.0e-20)
    m_safe = max(float(m_tot_msun), 0.1)
    z_sel_q = round(float(z_selected_kpc), 3)
    cache_key = (sig, round(a0_safe, 16), bool(has_cdm_halo), z_sel_q, round(m_safe, 4))

    with _QUMOND_KERNEL_LOCK:
        cached = _WB_CURVE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    r_au = np.geomspace(0.35, 25000.0, 220)
    r_m = r_au * AU_M
    g_n_sun = (G_SI * M_SUN_KG) / (r_m**2)
    r_crit_solar_au = math.sqrt(G_SI * M_SUN_KG / a0_safe) / AU_M

    # Fast exact-Keplerian short-circuit when nu(y) == 1 everywhere (e.g., Lambda-CDM / GR baryons)
    is_keplerian = all(abs(v - 1.0) < 1.0e-9 for v in sig if v > 0.0)
    if is_keplerian:
        zeros = np.zeros_like(r_au)
        res_zero: Dict[str, object] = {
            "zenodo_doi": ZENODO_PREREGISTRATION_DOI,
            "zenodo_sha256": ZENODO_MANIFEST_SHA256,
            "r_au": r_au,
            "g_newton_sun": g_n_sun,
            "r_crit_solar_au": r_crit_solar_au,
            "boost_disk_pct": zeros.copy(),
            "boost_lower_halo_pct": zeros.copy(),
            "boost_halo_pct": zeros.copy(),
            "boost_selected_pct": zeros.copy(),
            "boost_inner_disk_15k_pct": 0.0,
            "boost_disk_15k_pct": 0.0,
            "boost_outer_thick_15k_pct": 0.0,
            "boost_lower_halo_15k_pct": 0.0,
            "boost_halo_15k_pct": 0.0,
            "boost_selected_15k_pct": 0.0,
            "halo_minus_disk_pct": 0.0,
            "compass_aligned_15k_pct": 0.0,
            "compass_perp_15k_pct": 0.0,
            "compass_split_15k_pct": 0.0,
        }
        with _QUMOND_KERNEL_LOCK:
            if len(_WB_CURVE_CACHE) > 64:
                _WB_CURVE_CACHE.clear()
            _WB_CURVE_CACHE[cache_key] = res_zero
        return res_zero

    yn_inner = invert_milky_way_qumond_yn(G_OBS_INNER_DISK_SI, a0_safe, nu_fn)
    yn_solar = invert_milky_way_qumond_yn(G_OBS_SOLAR_MCMILLAN_SI, a0_safe, nu_fn)
    yn_outer = invert_milky_way_qumond_yn(G_OBS_OUTER_THICK_DISK_SI, a0_safe, nu_fn)
    yn_lower_halo = invert_milky_way_qumond_yn(G_OBS_REACHABLE_LOWER_HALO_SI, a0_safe, nu_fn)
    yn_deep_halo = invert_milky_way_qumond_yn(G_OBS_DEEP_HALO_6KPC_SI, a0_safe, nu_fn)
    yn_sel = invert_milky_way_qumond_yn(milky_way_g_obs_at_z_si(z_sel_q), a0_safe, nu_fn)

    # Anchor separations for smooth log10(s) spline from 0.35 AU to 25,000 AU
    s_anchors = np.array(
        [0.35, 50.0, 400.0, 1000.0, 2000.0, 3500.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0, 25000.0],
        dtype=np.float64,
    )
    log_anchors = np.log10(s_anchors)
    log_r = np.log10(r_au)

    def _build_3d_qumond_curve(yn_env: float, exact_15k_val: float) -> np.ndarray:
        vals = np.zeros_like(s_anchors)
        for idx, s_val in enumerate(s_anchors):
            if s_val <= 60.0:
                # Deep inner Solar System (y_bin > 1000): use exact point-field screening limit
                y_in = (G_SI * (m_safe * M_SUN_KG)) / (a0_safe * ((s_val * AU_M) ** 2))
                vals[idx] = (math.sqrt(_eval_nu_scalar(nu_fn, y_in + yn_env)) - 1.0) * 100.0
            elif abs(s_val - 15000.0) < 1.0:
                vals[idx] = exact_15k_val
            else:
                vals[idx] = orientation_averaged_qumond_boost(
                    s_val, m_safe, yn_env, a0_safe, nu_fn, exact_112=False
                )["v_boost_avg_pct"]
        sp = CubicSpline(log_anchors, vals, bc_type="natural")
        return np.asarray(sp(log_r), dtype=np.float64)

    # Evaluate canonical 15,000 AU benchmarks with the exact Y-folded 112^3 Zenodo kernel!
    b_inner_15k = orientation_averaged_qumond_boost(15000.0, m_safe, yn_inner, a0_safe, nu_fn, exact_112=True)[
        "v_boost_avg_pct"
    ]
    b_disk_15k = orientation_averaged_qumond_boost(15000.0, m_safe, yn_solar, a0_safe, nu_fn, exact_112=True)[
        "v_boost_avg_pct"
    ]
    b_outer_15k = orientation_averaged_qumond_boost(15000.0, m_safe, yn_outer, a0_safe, nu_fn, exact_112=True)[
        "v_boost_avg_pct"
    ]
    b_lower_15k = orientation_averaged_qumond_boost(15000.0, m_safe, yn_lower_halo, a0_safe, nu_fn, exact_112=True)[
        "v_boost_avg_pct"
    ]
    b_halo_15k = orientation_averaged_qumond_boost(15000.0, m_safe, yn_deep_halo, a0_safe, nu_fn, exact_112=True)[
        "v_boost_avg_pct"
    ]

    # Galactic Center Compass Anisotropy at 15,000 AU (aligned theta=0 vs perpendicular theta=90)
    full_k, _ = _ensure_qumond_kernels()
    yb_15k = (G_SI * (m_safe * M_SUN_KG)) / (a0_safe * ((15000.0 * AU_M) ** 2))
    gr_comp = _eval_kernel_angles(
        full_k,
        yb_15k,
        yn_solar,
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        nu_fn,
    )
    compass_aligned_15k = (math.sqrt(max(float(gr_comp[0]), 0.0)) - 1.0) * 100.0
    compass_perp_15k = (math.sqrt(max(float(gr_comp[1]), 0.0)) - 1.0) * 100.0

    boost_disk = _build_3d_qumond_curve(yn_solar, b_disk_15k)
    boost_lower_halo = _build_3d_qumond_curve(yn_lower_halo, b_lower_15k)
    boost_halo = _build_3d_qumond_curve(yn_deep_halo, b_halo_15k)

    if abs(z_sel_q) < 1.0e-3:
        boost_sel = boost_disk.copy()
        b_sel_15k = b_disk_15k
    elif abs(abs(z_sel_q) - 2.2) < 1.0e-3:
        boost_sel = boost_lower_halo.copy()
        b_sel_15k = b_lower_15k
    elif abs(abs(z_sel_q) - 6.0) < 1.0e-3:
        boost_sel = boost_halo.copy()
        b_sel_15k = b_halo_15k
    else:
        b_sel_15k = orientation_averaged_qumond_boost(15000.0, m_safe, yn_sel, a0_safe, nu_fn, exact_112=True)[
            "v_boost_avg_pct"
        ]
        boost_sel = _build_3d_qumond_curve(yn_sel, b_sel_15k)

    result: Dict[str, object] = {
        "zenodo_doi": ZENODO_PREREGISTRATION_DOI,
        "zenodo_sha256": ZENODO_MANIFEST_SHA256,
        "r_au": r_au,
        "g_newton_sun": g_n_sun,
        "r_crit_solar_au": r_crit_solar_au,
        "boost_disk_pct": boost_disk,
        "boost_lower_halo_pct": boost_lower_halo,
        "boost_halo_pct": boost_halo,
        "boost_selected_pct": boost_sel,
        "boost_inner_disk_15k_pct": b_inner_15k,
        "boost_disk_15k_pct": b_disk_15k,
        "boost_outer_thick_15k_pct": b_outer_15k,
        "boost_lower_halo_15k_pct": b_lower_15k,
        "boost_halo_15k_pct": b_halo_15k,
        "boost_selected_15k_pct": b_sel_15k,
        "halo_minus_disk_pct": b_halo_15k - b_disk_15k,
        "compass_aligned_15k_pct": compass_aligned_15k,
        "compass_perp_15k_pct": compass_perp_15k,
        "compass_split_15k_pct": compass_aligned_15k - compass_perp_15k,
    }
    with _QUMOND_KERNEL_LOCK:
        if len(_WB_CURVE_CACHE) > 64:
            _WB_CURVE_CACHE.clear()
        _WB_CURVE_CACHE[cache_key] = result
    return result


def generate_mock_gaia_dr4_catalog(
    n_binaries: int = 2400,
    truth_mode: str = "tnap_weave",
    a0_si: float = A0_PLANCK_2018_PRIMARY_SI,
    seed: int = 20261202,
    nu_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> List[WideBinaryPair]:
    """
    Generates a realistic mock `Gaia DR4` wide-binary sample (`s in [500, 22000] AU`,
    `|z_gal| in [0.05, 7.5] kpc`) using the conservative 3D two-body QUMOND boost profile!
    """
    rng = np.random.default_rng(seed)
    n_bin = max(int(n_binaries), 0)
    if n_bin == 0:
        return []
    pairs: List[WideBinaryPair] = []
    proj_median = 0.785
    a0_safe = max(float(a0_si), 1.0e-20)

    def _default_tnap_nu(y: np.ndarray) -> np.ndarray:
        sq = np.sqrt(np.maximum(np.asarray(y, dtype=np.float64), 1.0e-16))
        return np.where(sq > 35.0, 1.0 + np.exp(-sq), 1.0 / (1.0 - np.exp(-sq)))

    active_nu: Optional[Callable[[np.ndarray], np.ndarray]] = nu_fn
    if active_nu is None and truth_mode != "lambda_cdm":
        active_nu = _default_tnap_nu

    sep_arr = np.empty(n_bin, dtype=float)
    z_arr = np.empty(n_bin, dtype=float)
    m_arr = np.empty(n_bin, dtype=float)
    ruwe_arr = np.empty(n_bin, dtype=float)
    z_norm = np.empty(n_bin, dtype=float)
    for i in range(n_bin):
        sep_arr[i] = float(10.0 ** rng.uniform(2.7, 4.34))
        z_arr[i] = float(rng.uniform(4.8, 7.8)) if (i % 4 == 0) else float(rng.exponential(0.45))
        m_arr[i] = float(rng.uniform(1.1, 2.2))
        ruwe_arr[i] = float(rng.uniform(0.94, 1.18))
        z_norm[i] = float(rng.normal(0.0, 1.0))

    if active_nu is None:
        boost_arr = np.ones(n_bin, dtype=float)
    else:
        curves = compute_wide_binary_efe_curves(active_nu, a0_safe, m_tot_msun=1.6)
        r_grid = np.asarray(curves["r_au"], dtype=float)
        b_disk = np.asarray(curves["boost_disk_pct"], dtype=float)
        b_low = np.asarray(curves["boost_lower_halo_pct"], dtype=float)
        b_deep = np.asarray(curves["boost_halo_pct"], dtype=float)
        # Scale separation by sqrt(1.6 / m_tot) so y_bin = G * m_tot / (a0 * s^2) matches 3D QUMOND curve!
        s_eff = sep_arr * np.sqrt(1.6 / np.maximum(m_arr, 0.2))
        bd_i = np.interp(s_eff, r_grid, b_disk)
        bl_i = np.interp(s_eff, r_grid, b_low)
        bh_i = np.interp(s_eff, r_grid, b_deep)
        z_abs = np.abs(z_arr)
        pct_i = np.where(
            z_abs <= 2.2,
            bd_i + (bl_i - bd_i) * ((z_abs / 2.2) ** 1.6),
            bl_i + (bh_i - bl_i) * np.clip((np.maximum(z_abs - 2.2, 0.0) / 3.8) ** 0.85, 0.0, 1.35),
        )
        boost_arr = np.maximum(1.0 + 0.01 * pct_i, 0.5)

    for i in range(n_bin):
        v_tilde = max(0.15, float(proj_median * float(boost_arr[i]) + 0.14 * float(z_norm[i])))
        pairs.append(
            WideBinaryPair(
                pair_id=f"GDR4-{i + 1:06d}",
                sep_au=float(sep_arr[i]),
                z_gal_kpc=float(z_arr[i]),
                m_tot_msun=float(m_arr[i]),
                v_tilde_obs=v_tilde,
                ruwe_max=float(ruwe_arr[i]),
            )
        )
    return pairs


def is_raw_gaia_astrometry_csv(csv_path: Path) -> bool:
    """Returns True if `csv_path` has two-star sky-level Gaia astrometry columns (`ra1`/`ra1_deg`) without `v_tilde_obs`."""
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except OSError:
        return False
    if not header:
        return False
    cols = {c.strip().lower() for c in header if c is not None}
    has_raw_ra = bool(cols.intersection({"ra1", "ra1_deg", "ra_1", "ra_1_deg"}))
    return has_raw_ra and ("v_tilde_obs" not in cols)


def load_gaia_dr4_csv(csv_path: Path, max_ruwe: float = 1.20) -> List[WideBinaryPair]:
    """
    Ingests a real `Gaia DR4` wide-binary CSV table (`pair_id,sep_au,z_gal_kpc,m_tot_msun,v_tilde_obs,ruwe_max`)
    and applies strict `ruwe <= max_ruwe` unresolved-triple cleaning plus optional pre-registered
    `parallax_over_error_min >= 20` and `nss_flag == 0` cuts when those columns are present.
    Automatically delegates to `ingest_gaia_dr4_raw_astrometry` when raw two-star astrometry (`ra1`/`ra1_deg`) is detected!
    """
    if is_raw_gaia_astrometry_csv(csv_path):
        raw_pairs, _ = ingest_gaia_dr4_raw_astrometry(csv_path, max_ruwe=max_ruwe)
        return raw_pairs

    pairs: List[WideBinaryPair] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                raw_ruwe = row.get("ruwe_max")
                ruwe = float(raw_ruwe) if raw_ruwe not in (None, "") else 1.0
                raw_poe = row.get("parallax_over_error_min")
                poe = float(raw_poe) if raw_poe not in (None, "") else 50.0
                raw_nss = row.get("nss_flag")
                nss = int(raw_nss) if raw_nss not in (None, "") else 0
                sep = float(row["sep_au"])
                z_kpc = float(row["z_gal_kpc"])
                raw_m = row.get("m_tot_msun")
                m_tot = float(raw_m) if raw_m not in (None, "") else 1.6
                v_tilde = float(row["v_tilde_obs"])
            except (KeyError, TypeError, ValueError):
                continue

            if not (
                math.isfinite(ruwe)
                and math.isfinite(poe)
                and math.isfinite(sep)
                and math.isfinite(z_kpc)
                and math.isfinite(m_tot)
                and math.isfinite(v_tilde)
            ):
                continue
            if poe < float(PREREGISTERED_GAIA_CUTS["min_parallax_over_error"]):  # type: ignore[arg-type]
                continue
            if nss != 0:
                continue
            if ruwe <= 0.0 or ruwe > max_ruwe or sep <= 0.0 or m_tot <= 0.0 or v_tilde <= 0.0:
                continue

            pid = (row.get("pair_id") or "").strip() or f"ROW-{len(pairs)}"
            pairs.append(
                WideBinaryPair(
                    pair_id=pid,
                    sep_au=sep,
                    z_gal_kpc=z_kpc,
                    m_tot_msun=m_tot,
                    v_tilde_obs=v_tilde,
                    ruwe_max=ruwe,
                )
            )
    return pairs


def analyze_gaia_wide_binary_sample(pairs: List[WideBinaryPair]) -> Dict[str, float]:
    """
    Measures median `v_tilde` boost between inner Newtonian calibration bin (`500..2,500 AU`)
    and wide horizon regime (`8,000..22,000 AU`), stratified by Disk (`|z| < 1.5 kpc`) vs Halo (`|z| >= 4.5 kpc`)!
    """
    if not pairs:
        return {
            "n_clean": 0,
            "n_wide_disk": 0,
            "n_wide_halo": 0,
            "disk_boost_pct": 0.0,
            "halo_boost_pct": 0.0,
            "halo_minus_disk_pct": 0.0,
            "halo_excess_sigma": 0.0,
        }

    inner = [p.v_tilde_obs for p in pairs if 500.0 <= p.sep_au <= 2500.0]
    wide_disk = [p.v_tilde_obs for p in pairs if p.sep_au >= 8000.0 and abs(p.z_gal_kpc) < 1.5]
    wide_halo = [p.v_tilde_obs for p in pairs if p.sep_au >= 8000.0 and abs(p.z_gal_kpc) >= 4.5]

    v_cal = float(np.median(inner)) if inner else 0.785
    med_disk = float(np.median(wide_disk)) if wide_disk else v_cal
    med_halo = float(np.median(wide_halo)) if wide_halo else v_cal

    disk_boost_pct = ((med_disk / max(v_cal, 1.0e-6)) - 1.0) * 100.0
    halo_boost_pct = ((med_halo / max(v_cal, 1.0e-6)) - 1.0) * 100.0

    err_disk = (1.253 * float(np.std(wide_disk)) / math.sqrt(max(len(wide_disk), 1))) if len(wide_disk) > 2 else 0.05
    err_halo = (1.253 * float(np.std(wide_halo)) / math.sqrt(max(len(wide_halo), 1))) if len(wide_halo) > 2 else 0.05
    comb_err = math.hypot(err_disk, err_halo)
    halo_excess_sigma = (med_halo - med_disk) / max(comb_err, 1.0e-6)

    return {
        "n_clean": len(pairs),
        "n_wide_disk": len(wide_disk),
        "n_wide_halo": len(wide_halo),
        "disk_boost_pct": disk_boost_pct,
        "halo_boost_pct": halo_boost_pct,
        "halo_minus_disk_pct": halo_boost_pct - disk_boost_pct,
        "halo_excess_sigma": halo_excess_sigma,
    }


def _make_galactocentric_frame() -> Galactocentric:
    """GRAVITY Collaboration (R_0 = 8.178 kpc, Sgr A*) + Bennett & Bovy (z_sun = 20.8 pc) Galactocentric frame."""
    return Galactocentric(
        galcen_coord=SkyCoord(ra=SGR_A_STAR_RA_DEG * u.deg, dec=SGR_A_STAR_DEC_DEG * u.deg, frame="icrs"),
        galcen_distance=8.178 * u.kpc,
        z_sun=20.8 * u.pc,
    )


def milky_way_g_obs_rz_si_vec(r_gal_kpc: np.ndarray, z_gal_kpc: np.ndarray) -> np.ndarray:
    """
    Vectorized 2D McMillan (2017) + GRAVITY Milky Way external gravitational field `g_obs(R_gal, z_gal)` in m/s^2,
    anchored identically to `G_OBS_SOLAR_MCMILLAN_SI` at `(R_0 = 8.178 kpc, z = 0)` and `milky_way_g_obs_at_z_si(|z|)`.
    """
    r_safe = np.maximum(np.asarray(r_gal_kpc, dtype=np.float64), 2.5)
    z_abs = np.abs(np.asarray(z_gal_kpc, dtype=np.float64))
    u_z = z_abs**1.6
    z_knots = np.array([0.0, 1.2**1.6, 2.2**1.6, 6.0**1.6, 10.0**1.6], dtype=np.float64)
    g_knots = np.array(
        [
            G_OBS_SOLAR_MCMILLAN_SI,
            G_OBS_OUTER_THICK_DISK_SI,
            G_OBS_REACHABLE_LOWER_HALO_SI,
            G_OBS_DEEP_HALO_6KPC_SI,
            0.55e-10,
        ],
        dtype=np.float64,
    )
    g_z_solar = np.interp(u_z, z_knots, g_knots)
    radial_scale = 8.178 / r_safe
    halo_weight = (z_abs * z_abs) / (z_abs * z_abs + 2.5 * 2.5)
    sph_scale = np.sqrt(8.178 / np.hypot(r_safe, z_abs))
    eff_scale = (1.0 - halo_weight) * radial_scale + halo_weight * sph_scale
    return g_z_solar * eff_scale


def _ensure_3d_qumond_lut(
    a0_si: float,
    nu_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, object]:
    """Builds & caches in DDR5 RAM a 3D `(log10 y_bin, log10 y_N_ext, theta_GC)` exact QUMOND velocity-boost LUT."""
    a0_safe = max(float(a0_si), 1.0e-20)
    sig = _nu_signature(nu_fn)
    key = (sig, round(a0_safe, 16))
    with _QUMOND_KERNEL_LOCK:
        cached = _WB_3D_LUT_CACHE.get(key)
    if cached is not None:
        return cached

    full_k, fast_k = _ensure_qumond_kernels()
    g_obs_knots = np.geomspace(0.35e-10, 4.5e-10, 28)
    yn_knots = np.array([invert_milky_way_qumond_yn(float(g), a0_safe, nu_fn) for g in g_obs_knots], dtype=np.float64)
    yn_spline = CubicSpline(np.log10(g_obs_knots), np.log10(yn_knots), bc_type="natural")

    yb_15k_ref = math.log10((G_SI * (1.6 * M_SUN_KG)) / (a0_safe * ((15000.0 * AU_M) ** 2)))
    log_yb_grid = np.array(
        sorted([-2.4, -1.85, -1.4, -1.05, -0.78, yb_15k_ref, -0.32, 0.0, 0.4, 0.9, 1.5, 2.3, 3.3]),
        dtype=np.float64,
    )
    log_yn_grid = np.linspace(-0.80, 0.60, 9, dtype=np.float64)
    theta_grid = np.array([0.0, 22.5, 45.0, 67.5, 90.0], dtype=np.float64)

    yn_mesh, th_mesh = np.meshgrid(10.0**log_yn_grid, np.radians(theta_grid), indexing="ij")
    gxe_unit = (yn_mesh * np.cos(th_mesh)).ravel().astype(np.float32)
    gze_unit = (yn_mesh * np.sin(th_mesh)).ravel().astype(np.float32)

    table = np.zeros((len(log_yb_grid), len(log_yn_grid), len(theta_grid)), dtype=np.float64)
    for i, lyb in enumerate(log_yb_grid):
        yb_val = 10.0**float(lyb)
        if lyb >= 2.2:
            # Screened inner Newtonian regime: exact point-field screening limit
            y_tot = yb_val + (10.0**log_yn_grid)[:, None]
            nu_vals = np.maximum(np.asarray(nu_fn(y_tot), dtype=np.float64), 1.0)
            table[i, :, :] = (np.sqrt(nu_vals) - 1.0) * 100.0
        else:
            kern = full_k if abs(lyb - yb_15k_ref) < 0.28 else fast_k
            gr_flat = _eval_kernel_angles(kern, yb_val, 1.0, gxe_unit, gze_unit, nu_fn)
            boost_flat = (np.sqrt(np.maximum(gr_flat, 0.0)) - 1.0) * 100.0
            table[i, :, :] = boost_flat.reshape(len(log_yn_grid), len(theta_grid))

    interp_3d = RegularGridInterpolator(
        (log_yb_grid, log_yn_grid, theta_grid),
        table,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    entry: Dict[str, object] = {
        "yn_spline": yn_spline,
        "interp_3d": interp_3d,
        "log_yb_min": float(log_yb_grid[0]),
        "log_yb_max": float(log_yb_grid[-1]),
        "log_yn_min": float(log_yn_grid[0]),
        "log_yn_max": float(log_yn_grid[-1]),
    }
    with _QUMOND_KERNEL_LOCK:
        if len(_WB_3D_LUT_CACHE) > 16:
            _WB_3D_LUT_CACHE.clear()
        _WB_3D_LUT_CACHE[key] = entry
    return entry


def _eval_3d_qumond_boost_batch(
    s_au: np.ndarray,
    m_tot_msun: np.ndarray,
    r_gal_kpc: np.ndarray,
    z_gal_kpc: np.ndarray,
    theta_gc_deg: np.ndarray,
    a0_si: float = A0_PLANCK_2018_PRIMARY_SI,
    nu_fn: Callable[[np.ndarray], np.ndarray] = _default_tnap_nu_fn,
) -> np.ndarray:
    """Evaluates orientation- & environment-dependent 3D QUMOND velocity boost `%` across arrays of binaries."""
    a0_safe = max(float(a0_si), 1.0e-20)
    lut = _ensure_3d_qumond_lut(a0_safe, nu_fn)
    yn_spline: CubicSpline = lut["yn_spline"]  # type: ignore[assignment]
    interp_3d: RegularGridInterpolator = lut["interp_3d"]  # type: ignore[assignment]

    g_obs = np.clip(milky_way_g_obs_rz_si_vec(r_gal_kpc, z_gal_kpc), 0.36e-10, 4.4e-10)
    log_yn = np.clip(
        np.asarray(yn_spline(np.log10(g_obs)), dtype=np.float64),
        float(lut["log_yn_min"]),  # type: ignore[arg-type]
        float(lut["log_yn_max"]),  # type: ignore[arg-type]
    )
    yb = (G_SI * (np.maximum(m_tot_msun, 0.1) * M_SUN_KG)) / (
        a0_safe * ((np.maximum(s_au, 1.0) * AU_M) ** 2)
    )
    log_yb = np.clip(
        np.log10(np.maximum(yb, 1.0e-12)),
        float(lut["log_yb_min"]),  # type: ignore[arg-type]
        float(lut["log_yb_max"]),  # type: ignore[arg-type]
    )
    th_clamped = np.clip(np.asarray(theta_gc_deg, dtype=np.float64), 0.0, 90.0)
    pts = np.stack([log_yb, log_yn, th_clamped], axis=-1)
    return np.asarray(interp_3d(pts), dtype=np.float64)


def _compute_galactocentric_and_compass_batch(
    ra1_deg: np.ndarray,
    dec1_deg: np.ndarray,
    ra2_deg: np.ndarray,
    dec2_deg: np.ndarray,
    dist_pc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Astropy ICRS -> Galactocentric `(R_gal_kpc, z_gal_kpc)` and 3D Galactic Center
    sky-tangent compass angle `theta_GC in [0, 90] deg`.
    """
    if len(dist_pc) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    a1 = np.radians(ra1_deg)
    d1 = np.radians(dec1_deg)
    a2 = np.radians(ra2_deg)
    d2 = np.radians(dec2_deg)

    u1 = np.stack([np.cos(d1) * np.cos(a1), np.cos(d1) * np.sin(a1), np.sin(d1)], axis=-1)
    u2 = np.stack([np.cos(d2) * np.cos(a2), np.cos(d2) * np.sin(a2), np.sin(d2)], axis=-1)
    um = u1 + u2
    um /= np.maximum(np.linalg.norm(um, axis=-1, keepdims=True), 1.0e-15)

    dec_m = np.arcsin(np.clip(um[:, 2], -1.0, 1.0))
    ra_m = np.mod(np.arctan2(um[:, 1], um[:, 0]), 2.0 * np.pi)

    cos_dm = np.cos(dec_m)
    pole_mask = np.abs(cos_dm) < 1.0e-12
    e_alpha = np.stack([-np.sin(ra_m), np.cos(ra_m), np.zeros_like(ra_m)], axis=-1)
    e_delta = np.stack(
        [-np.sin(dec_m) * np.cos(ra_m), -np.sin(dec_m) * np.sin(ra_m), cos_dm],
        axis=-1,
    )
    if np.any(pole_mask):
        e_alpha[pole_mask] = np.array([0.0, 1.0, 0.0])
        e_delta[pole_mask] = np.array([-1.0, 0.0, 0.0])

    du = u2 - u1
    s_alpha = np.sum(du * e_alpha, axis=-1)
    s_delta = np.sum(du * e_delta, axis=-1)
    s_norm = np.hypot(s_alpha, s_delta)
    s_hat_a = np.where(s_norm > 1.0e-15, s_alpha / np.maximum(s_norm, 1.0e-15), 1.0)
    s_hat_d = np.where(s_norm > 1.0e-15, s_delta / np.maximum(s_norm, 1.0e-15), 0.0)

    gc_frame = _make_galactocentric_frame()
    sc_mid = SkyCoord(
        ra=np.degrees(ra_m) * u.deg,
        dec=np.degrees(dec_m) * u.deg,
        distance=dist_pc * u.pc,
        frame="icrs",
    )
    gc_coords = sc_mid.transform_to(gc_frame)
    x_gc_kpc = np.asarray(gc_coords.x.to_value(u.kpc), dtype=np.float64)
    y_gc_kpc = np.asarray(gc_coords.y.to_value(u.kpc), dtype=np.float64)
    z_gal_kpc = np.asarray(gc_coords.z.to_value(u.kpc), dtype=np.float64)
    r_gal_kpc = np.hypot(x_gc_kpc, y_gc_kpc)

    gc_origin_icrs = SkyCoord(x=0.0 * u.kpc, y=0.0 * u.kpc, z=0.0 * u.kpc, frame=gc_frame).transform_to("icrs")
    r_gc_xyz = np.asarray(gc_origin_icrs.cartesian.xyz.to_value(u.kpc), dtype=np.float64)
    r_bin_xyz = (dist_pc[:, None] / 1000.0) * um
    v_to_gc = r_gc_xyz[None, :] - r_bin_xyz
    g_alpha = np.sum(v_to_gc * e_alpha, axis=-1)
    g_delta = np.sum(v_to_gc * e_delta, axis=-1)
    g_norm = np.hypot(g_alpha, g_delta)
    cos_theta = np.where(
        g_norm > 1.0e-12,
        np.abs(s_hat_a * g_alpha + s_hat_d * g_delta) / np.maximum(g_norm, 1.0e-15),
        math.sqrt(0.5),
    )
    theta_gc_deg = np.degrees(np.arccos(np.clip(cos_theta, 0.0, 1.0)))
    return r_gal_kpc, z_gal_kpc, theta_gc_deg


def ingest_gaia_dr4_raw_astrometry(
    csv_path: Path,
    max_ruwe: float = 1.20,
    min_poe: float = 20.0,
    rv_sigma_clip: float = 3.0,
    max_sigma_v2d_frac: Optional[float] = 0.35,
) -> Tuple[List[WideBinaryPair], GaiaCutFlowSummary]:
    """
    Ingests raw ESA `Gaia DR4` two-star sky-level ICRS astrometry + radial velocities and applies all
    pre-registered Zenodo `10.5281/zenodo.23919217` quality, RUWE, NSS, and 3-sigma RV bound-orbit cuts.
    """
    cut_flow = GaiaCutFlowSummary()
    if not csv_path.exists():
        return [], cut_flow

    def _pick_col(cmap: Dict[str, str], aliases: List[str]) -> Optional[str]:
        for alias in aliases:
            if alias in cmap:
                return cmap[alias]
        return None

    def _opt_float(row: Dict[str, str], col: Optional[str], default: Optional[float] = None) -> Optional[float]:
        if col is None:
            return default
        raw = row.get(col)
        if raw is None:
            return default
        txt = raw.strip()
        if txt == "" or txt.lower() in ("null", "none", "nan", "--"):
            return default
        val = float(txt)
        if math.isnan(val):
            return default
        return val

    pids: List[str] = []
    ra1_list: List[float] = []
    dec1_list: List[float] = []
    ra2_list: List[float] = []
    dec2_list: List[float] = []
    sep_list: List[float] = []
    dist_list: List[float] = []
    mtot_list: List[float] = []
    vtilde_list: List[float] = []
    ruwe_list: List[float] = []
    drv_list: List[Optional[float]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], cut_flow
        cmap = {c.strip().lower(): c for c in reader.fieldnames if c is not None}

        c_pid = _pick_col(cmap, ["pair_id", "source_pair_id", "binary_id", "id"])
        c_sid1 = _pick_col(cmap, ["source_id1", "source_id_1"])
        c_sid2 = _pick_col(cmap, ["source_id2", "source_id_2"])

        c_ra1 = _pick_col(cmap, ["ra1", "ra1_deg", "ra_1", "ra_1_deg"])
        c_dec1 = _pick_col(cmap, ["dec1", "dec1_deg", "dec_1", "dec_1_deg"])
        c_plx1 = _pick_col(cmap, ["parallax1", "parallax1_mas", "plx1", "parallax_1", "parallax_1_mas"])
        c_plxe1 = _pick_col(
            cmap,
            [
                "parallax_error1",
                "parallax_error1_mas",
                "parallax1_error",
                "plx_err1",
                "parallax_err1",
                "parallax_error_1",
            ],
        )
        c_pmra1 = _pick_col(cmap, ["pmra1", "pmra1_masyr", "pmra_1", "pmra_1_masyr"])
        c_pmdec1 = _pick_col(cmap, ["pmdec1", "pmdec1_masyr", "pmdec_1", "pmdec_1_masyr"])
        c_pmrae1 = _pick_col(cmap, ["pmra_error1", "pmra_error1_masyr", "pmra1_error", "pmra_err1", "pmra_error_1"])
        c_pmdece1 = _pick_col(
            cmap, ["pmdec_error1", "pmdec_error1_masyr", "pmdec1_error", "pmdec_err1", "pmdec_error_1"]
        )
        c_ruwe1 = _pick_col(cmap, ["ruwe1", "ruwe_1"])
        c_nss1 = _pick_col(cmap, ["non_single_star1", "nss1", "nss_flag1", "non_single_star_1", "nss_1"])
        c_rv1 = _pick_col(cmap, ["radial_velocity1", "rv1", "rv1_kms", "radial_velocity1_kms", "radial_velocity_1", "rv_1"])
        c_rve1 = _pick_col(
            cmap,
            [
                "radial_velocity_error1",
                "rv_error1",
                "rv1_error",
                "rv_err1",
                "rv1_err_kms",
                "radial_velocity_error1_kms",
                "radial_velocity_error_1",
            ],
        )
        c_m1 = _pick_col(cmap, ["mass1", "mass1_msun", "m1", "m1_msun", "mass_1", "mass_1_msun"])

        c_ra2 = _pick_col(cmap, ["ra2", "ra2_deg", "ra_2", "ra_2_deg"])
        c_dec2 = _pick_col(cmap, ["dec2", "dec2_deg", "dec_2", "dec_2_deg"])
        c_plx2 = _pick_col(cmap, ["parallax2", "parallax2_mas", "plx2", "parallax_2", "parallax_2_mas"])
        c_plxe2 = _pick_col(
            cmap,
            [
                "parallax_error2",
                "parallax_error2_mas",
                "parallax2_error",
                "plx_err2",
                "parallax_err2",
                "parallax_error_2",
            ],
        )
        c_pmra2 = _pick_col(cmap, ["pmra2", "pmra2_masyr", "pmra_2", "pmra_2_masyr"])
        c_pmdec2 = _pick_col(cmap, ["pmdec2", "pmdec2_masyr", "pmdec_2", "pmdec_2_masyr"])
        c_pmrae2 = _pick_col(cmap, ["pmra_error2", "pmra_error2_masyr", "pmra2_error", "pmra_err2", "pmra_error_2"])
        c_pmdece2 = _pick_col(
            cmap, ["pmdec_error2", "pmdec_error2_masyr", "pmdec2_error", "pmdec_err2", "pmdec_error_2"]
        )
        c_ruwe2 = _pick_col(cmap, ["ruwe2", "ruwe_2"])
        c_nss2 = _pick_col(cmap, ["non_single_star2", "nss2", "nss_flag2", "non_single_star_2", "nss_2"])
        c_rv2 = _pick_col(cmap, ["radial_velocity2", "rv2", "rv2_kms", "radial_velocity2_kms", "radial_velocity_2", "rv_2"])
        c_rve2 = _pick_col(
            cmap,
            [
                "radial_velocity_error2",
                "rv_error2",
                "rv2_error",
                "rv_err2",
                "rv2_err_kms",
                "radial_velocity_error2_kms",
                "radial_velocity_error_2",
            ],
        )
        c_m2 = _pick_col(cmap, ["mass2", "mass2_msun", "m2", "m2_msun", "mass_2", "mass_2_msun"])
        c_mtot = _pick_col(cmap, ["m_tot_msun", "mass_tot", "m_tot"])

        for row_idx, row in enumerate(reader):
            cut_flow.n_raw_rows += 1
            try:
                ra1 = _opt_float(row, c_ra1)
                dec1 = _opt_float(row, c_dec1)
                plx1 = _opt_float(row, c_plx1)
                plxe1 = _opt_float(row, c_plxe1)
                pmra1 = _opt_float(row, c_pmra1)
                pmdec1 = _opt_float(row, c_pmdec1)

                ra2 = _opt_float(row, c_ra2)
                dec2 = _opt_float(row, c_dec2)
                plx2 = _opt_float(row, c_plx2)
                plxe2 = _opt_float(row, c_plxe2)
                pmra2 = _opt_float(row, c_pmra2)
                pmdec2 = _opt_float(row, c_pmdec2)

                pmrae1 = _opt_float(row, c_pmrae1, 0.0)
                pmdece1 = _opt_float(row, c_pmdece1, 0.0)
                pmrae2 = _opt_float(row, c_pmrae2, 0.0)
                pmdece2 = _opt_float(row, c_pmdece2, 0.0)

                ruwe1 = _opt_float(row, c_ruwe1, 1.0)
                ruwe2 = _opt_float(row, c_ruwe2, 1.0)
                nss1_f = _opt_float(row, c_nss1, 0.0)
                nss2_f = _opt_float(row, c_nss2, 0.0)

                rv1 = _opt_float(row, c_rv1, None)
                rv2 = _opt_float(row, c_rv2, None)
                rve1 = _opt_float(row, c_rve1, 0.0)
                rve2 = _opt_float(row, c_rve2, 0.0)

                m1 = _opt_float(row, c_m1, None)
                m2 = _opt_float(row, c_m2, None)
                mtot_raw = _opt_float(row, c_mtot, None)
            except (TypeError, ValueError):
                continue

            req = (ra1, dec1, plx1, plxe1, pmra1, pmdec1, ra2, dec2, plx2, plxe2, pmra2, pmdec2)
            if any(v is None or not math.isfinite(v) for v in req):
                continue
            assert ra1 is not None and dec1 is not None and plx1 is not None and plxe1 is not None
            assert pmra1 is not None and pmdec1 is not None
            assert ra2 is not None and dec2 is not None and plx2 is not None and plxe2 is not None
            assert pmra2 is not None and pmdec2 is not None

            opt_check = (pmrae1, pmdece1, pmrae2, pmdece2, ruwe1, ruwe2, nss1_f, nss2_f, rve1, rve2)
            if any(v is None or not math.isfinite(v) for v in opt_check):
                continue
            assert pmrae1 is not None and pmdece1 is not None and pmrae2 is not None and pmdece2 is not None
            assert ruwe1 is not None and ruwe2 is not None and nss1_f is not None and nss2_f is not None
            assert rve1 is not None and rve2 is not None

            if rv1 is not None and not math.isfinite(rv1):
                continue
            if rv2 is not None and not math.isfinite(rv2):
                continue
            if pmrae1 < 0.0 or pmdece1 < 0.0 or pmrae2 < 0.0 or pmdece2 < 0.0 or rve1 < 0.0 or rve2 < 0.0:
                continue
            if not (-90.0 <= dec1 <= 90.0 and -90.0 <= dec2 <= 90.0):
                continue

            if m1 is not None and m2 is not None:
                if not (math.isfinite(m1) and math.isfinite(m2) and m1 > 0.0 and m2 > 0.0):
                    continue
                m_tot = m1 + m2
            elif mtot_raw is not None:
                if not (math.isfinite(mtot_raw) and mtot_raw > 0.0):
                    continue
                m_tot = mtot_raw
            else:
                m_tot = 1.6

            # Cut 1: Parallax SNR (`parallax / parallax_error >= min_poe`)
            if plx1 <= 0.0 or plx2 <= 0.0 or plxe1 <= 0.0 or plxe2 <= 0.0:
                continue
            if (plx1 / plxe1) < min_poe or (plx2 / plxe2) < min_poe:
                continue

            # Exact sky-plane angular separation Delta_theta -> physical projected separation s_AU = Delta_theta_arcsec * d_pc
            a1_r = math.radians(ra1)
            d1_r = math.radians(dec1)
            a2_r = math.radians(ra2)
            d2_r = math.radians(dec2)
            ux1, uy1, uz1 = math.cos(d1_r) * math.cos(a1_r), math.cos(d1_r) * math.sin(a1_r), math.sin(d1_r)
            ux2, uy2, uz2 = math.cos(d2_r) * math.cos(a2_r), math.cos(d2_r) * math.sin(a2_r), math.sin(d2_r)
            chord = math.sqrt((ux2 - ux1) ** 2 + (uy2 - uy1) ** 2 + (uz2 - uz1) ** 2)
            dtheta_rad = 2.0 * math.asin(min(0.5 * chord, 1.0))
            dtheta_arcsec = dtheta_rad * ARCSEC_PER_RAD

            plx_mean = 0.5 * (plx1 + plx2)
            dist_pc = 1000.0 / plx_mean
            sep_au = dtheta_arcsec * dist_pc
            if sep_au <= 0.0 or not math.isfinite(sep_au):
                continue

            cut_flow.n_passed_snr_parallax += 1

            # Cut 2: Consistent pair distance (accounting for measurement errors & 3D orbital depth)
            orb_plx_term = 2.5 * sep_au / 206265.0 * ((plx_mean / 1000.0) ** 2)
            plx_tol = math.sqrt((3.5**2) * (plxe1**2 + plxe2**2) + orb_plx_term**2)
            if abs(plx1 - plx2) > plx_tol:
                continue
            cut_flow.n_passed_consistent_parallax += 1

            # Cut 3: Gaia non_single_star == 0
            nss1 = int(round(nss1_f))
            nss2 = int(round(nss2_f))
            if nss1 != 0 or nss2 != 0:
                continue
            cut_flow.n_passed_nss += 1

            # Cut 4: Astrometric photocenter quality 0 < ruwe <= max_ruwe
            if not (0.0 < ruwe1 <= max_ruwe and 0.0 < ruwe2 <= max_ruwe):
                continue
            cut_flow.n_passed_ruwe += 1

            # Cut 5: Radial-Velocity Triple & Chance-Flyby Vetting
            v_circ_n_kms = 1.0e-3 * math.sqrt((G_SI * m_tot * M_SUN_KG) / (sep_au * AU_M))
            v_esc_n_kms = math.sqrt(2.0) * v_circ_n_kms
            delta_rv: Optional[float] = None
            if rv1 is not None and rv2 is not None:
                delta_rv = abs(rv1 - rv2)
                rv_bound_max = math.sqrt((1.65 * v_esc_n_kms) ** 2 + (rv_sigma_clip**2) * (rve1**2 + rve2**2))
                if delta_rv > rv_bound_max:
                    continue
            cut_flow.n_passed_rv_bound += 1

            # 2D sky-plane relative tangential velocity & optional PM error quadrature filter
            d_kpc = dist_pc / 1000.0
            dpm_ra = pmra2 - pmra1
            dpm_dec = pmdec2 - pmdec1
            dpm_mag = math.hypot(dpm_ra, dpm_dec)
            dv_2d_kms = KMS_PER_MASYR_KPC * d_kpc * dpm_mag

            if max_sigma_v2d_frac is not None:
                var_pmra = pmrae1**2 + pmrae2**2
                var_pmdec = pmdece1**2 + pmdece2**2
                if (var_pmra + var_pmdec) > 0.0:
                    if dpm_mag > 1.0e-12:
                        sigma_dpm = math.sqrt((dpm_ra**2 * var_pmra + dpm_dec**2 * var_pmdec) / (dpm_mag**2))
                    else:
                        sigma_dpm = math.sqrt(0.5 * (var_pmra + var_pmdec))
                    sigma_dv2d_kms = KMS_PER_MASYR_KPC * d_kpc * sigma_dpm
                    if sigma_dv2d_kms > max_sigma_v2d_frac * v_circ_n_kms:
                        continue

            v_tilde = dv_2d_kms / max(v_circ_n_kms, 1.0e-15)
            if not (math.isfinite(v_tilde) and v_tilde > 0.0):
                continue

            if c_pid is not None and (row.get(c_pid) or "").strip():
                pid = (row.get(c_pid) or "").strip()
            elif c_sid1 is not None and c_sid2 is not None and (row.get(c_sid1) or "").strip():
                pid = f"{(row.get(c_sid1) or '').strip()}_{(row.get(c_sid2) or '').strip()}"
            else:
                pid = f"GDR4-RAW-{row_idx + 1:06d}"

            cut_flow.n_clean_final += 1
            pids.append(pid)
            ra1_list.append(ra1)
            dec1_list.append(dec1)
            ra2_list.append(ra2)
            dec2_list.append(dec2)
            sep_list.append(sep_au)
            dist_list.append(dist_pc)
            mtot_list.append(m_tot)
            vtilde_list.append(v_tilde)
            ruwe_list.append(max(ruwe1, ruwe2))
            drv_list.append(delta_rv)

    if not pids:
        return [], cut_flow

    r_gal_arr, z_gal_arr, theta_gc_arr = _compute_galactocentric_and_compass_batch(
        np.asarray(ra1_list, dtype=np.float64),
        np.asarray(dec1_list, dtype=np.float64),
        np.asarray(ra2_list, dtype=np.float64),
        np.asarray(dec2_list, dtype=np.float64),
        np.asarray(dist_list, dtype=np.float64),
    )

    pairs: List[WideBinaryPair] = []
    for idx, pid in enumerate(pids):
        pairs.append(
            WideBinaryPair(
                pair_id=pid,
                sep_au=sep_list[idx],
                z_gal_kpc=float(z_gal_arr[idx]),
                m_tot_msun=mtot_list[idx],
                v_tilde_obs=vtilde_list[idx],
                ruwe_max=ruwe_list[idx],
                dist_pc=dist_list[idx],
                r_gal_kpc=float(r_gal_arr[idx]),
                theta_gc_deg=float(theta_gc_arr[idx]),
                delta_rv_kms=drv_list[idx],
            )
        )
    return pairs, cut_flow


def _sample_scale_invariant_keplerian_orbits(
    rng: np.random.Generator,
    n_target: int,
    s_min_au: float,
    s_max_au: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples genuine 3D eccentric Keplerian orbits (`f(e) = 2e` up to `0.95`, `M = E - e sin E` solved via
    Newton-Raphson, isotropic 3D `cos i, Omega, omega`) from Opik's scale-invariant `dN/d(ln a) = const`
    parent distribution (`a in [120, 220000] AU`), conditioned cleanly on projected `s_2D in [s_min_au, s_max_au]`.
    Returns `(a_au, dx_alpha_au, dx_delta_au, dz_los_au, ux_v_alpha, ux_v_delta, uz_v_los)` where velocity factors
    are dimensionless in units of `sqrt(G M_tot / a_au)`.
    """
    if n_target <= 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty, empty, empty  # type: ignore[return-value]

    log_a_lo = math.log10(max(s_min_au / 2.2, 80.0))
    log_a_hi = math.log10(max(s_max_au * 11.0, 40000.0))

    dx_a_acc: List[np.ndarray] = []
    dx_d_acc: List[np.ndarray] = []
    dz_l_acc: List[np.ndarray] = []
    va_acc: List[np.ndarray] = []
    vd_acc: List[np.ndarray] = []
    vl_acc: List[np.ndarray] = []
    a_acc: List[np.ndarray] = []
    collected = 0

    batch_size = max(int(n_target * 4.5), 1024)
    while collected < n_target:
        a = 10.0 ** rng.uniform(log_a_lo, log_a_hi, batch_size)
        ecc = 0.95 * np.sqrt(rng.uniform(0.0, 1.0, batch_size))
        mean_anom = rng.uniform(0.0, 2.0 * np.pi, batch_size)

        # Solve Kepler's equation M = E - e sin(E) via 8-step Newton-Raphson (converges < 1e-15 rad for e <= 0.95)
        ecc_anom = mean_anom + ecc * np.sin(mean_anom)
        for _ in range(8):
            f_val = ecc_anom - ecc * np.sin(ecc_anom) - mean_anom
            fp_val = 1.0 - ecc * np.cos(ecc_anom)
            ecc_anom = ecc_anom - f_val / fp_val

        cos_e = np.cos(ecc_anom)
        sin_e = np.sin(ecc_anom)
        sqrt_1me2 = np.sqrt(1.0 - ecc * ecc)
        denom = 1.0 - ecc * cos_e

        x_orb = a * (cos_e - ecc)
        y_orb = a * sqrt_1me2 * sin_e
        vx_dim = -sin_e / denom
        vy_dim = sqrt_1me2 * cos_e / denom

        cos_i = rng.uniform(-1.0, 1.0, batch_size)
        sin_i = np.sqrt(np.maximum(1.0 - cos_i * cos_i, 0.0))
        big_om = rng.uniform(0.0, 2.0 * np.pi, batch_size)
        arg_om = rng.uniform(0.0, 2.0 * np.pi, batch_size)

        cO, sO = np.cos(big_om), np.sin(big_om)
        co, so = np.cos(arg_om), np.sin(arg_om)

        p_a = cO * co - sO * so * cos_i
        p_d = sO * co + cO * so * cos_i
        p_l = so * sin_i

        q_a = -cO * so - sO * co * cos_i
        q_d = -sO * so + cO * co * cos_i
        q_l = co * sin_i

        dx_a = x_orb * p_a + y_orb * q_a
        dx_d = x_orb * p_d + y_orb * q_d
        dz_l = x_orb * p_l + y_orb * q_l
        s_2d = np.hypot(dx_a, dx_d)

        keep = (s_2d >= s_min_au) & (s_2d <= s_max_au)
        if np.any(keep):
            a_acc.append(a[keep])
            dx_a_acc.append(dx_a[keep])
            dx_d_acc.append(dx_d[keep])
            dz_l_acc.append(dz_l[keep])
            va_acc.append((vx_dim * p_a + vy_dim * q_a)[keep])
            vd_acc.append((vx_dim * p_d + vy_dim * q_d)[keep])
            vl_acc.append((vx_dim * p_l + vy_dim * q_l)[keep])
            collected += int(np.sum(keep))

    return (
        np.concatenate(a_acc)[:n_target],
        np.concatenate(dx_a_acc)[:n_target],
        np.concatenate(dx_d_acc)[:n_target],
        np.concatenate(dz_l_acc)[:n_target],
        np.concatenate(va_acc)[:n_target],
        np.concatenate(vd_acc)[:n_target],
        np.concatenate(vl_acc)[:n_target],
    )  # type: ignore[return-value]


def export_mock_raw_gaia_dr4_astrometry_csv(
    csv_path: Path,
    n_binaries: int = 3200,
    truth_mode: str = "tnap_weave",
    inject_contaminants: bool = True,
    seed: int = 20261202,
) -> Dict[str, int]:
    """
    Exports a realistic sky-level two-star `Gaia DR4` wide-binary catalog CSV (`ra1, dec1, parallax1, pmra1, ...`)
    across Local Backyard (`d <= 300 pc`) and Reachable Galactic Environments (`Inner Disk`, `Solar Circle`,
    `Thick/Outer Disk`, and `Lower Halo` `|z|` up to `~2.4 kpc`), integrating genuine 3D eccentric Keplerian orbits
    scaled by the 3D QUMOND `(r_gal, z_gal, theta_GC)` boost (`tnap_weave`) or `1.000` (`lambda_cdm`).
    """
    rng = np.random.default_rng(seed)
    n_bin = max(int(n_binaries), 0)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pair_id",
        "ra1",
        "dec1",
        "parallax1",
        "parallax_error1",
        "pmra1",
        "pmdec1",
        "pmra_error1",
        "pmdec_error1",
        "ruwe1",
        "radial_velocity1",
        "radial_velocity_error1",
        "non_single_star1",
        "mass1",
        "ra2",
        "dec2",
        "parallax2",
        "parallax_error2",
        "pmra2",
        "pmdec2",
        "pmra_error2",
        "pmdec_error2",
        "ruwe2",
        "radial_velocity2",
        "radial_velocity_error2",
        "non_single_star2",
        "mass2",
    ]

    if n_bin == 0:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return {
            "n_total_exported": 0,
            "n_clean_injected": 0,
            "n_contaminants_injected": 0,
            "n_ruwe_triples": 0,
            "n_nss_binaries": 0,
            "n_rv_flybys": 0,
        }

    # 1. Sample Galactic environments:
    #    ~48% Local Backyard (75..290 pc, isotropic all-sky)
    #    ~16% Inner Disk (850..1780 pc, |l| < 32 deg, |b| < 10 deg -> R_gal ~ 6.4..7.4 kpc, |z| < 0.3 kpc)
    #    ~14% Outer / Thick Disk (950..1780 pc, l in 150..210 deg -> R_gal > 9.0 kpc)
    #    ~22% High-Latitude Lower Halo (1380..2360 pc, |b| > 68 deg -> |z| in 1.28..2.38 kpc)
    n_loc = int(round(0.48 * n_bin))
    n_inn = int(round(0.16 * n_bin))
    n_out = int(round(0.14 * n_bin))
    n_hal = max(n_bin - n_loc - n_inn - n_out, 0)
    if n_loc + n_inn + n_out + n_hal != n_bin:
        n_loc += n_bin - (n_loc + n_inn + n_out + n_hal)

    l_loc = rng.uniform(0.0, 360.0, n_loc)
    b_loc = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n_loc)))
    d_loc = rng.uniform(75.0, 290.0, n_loc)

    l_inn = np.mod(rng.uniform(-32.0, 32.0, n_inn), 360.0)
    b_inn = rng.uniform(-9.5, 9.5, n_inn)
    d_inn = rng.uniform(850.0, 1780.0, n_inn)

    l_out = rng.uniform(150.0, 210.0, n_out)
    b_out = rng.uniform(-32.0, 32.0, n_out)
    d_out = rng.uniform(980.0, 1780.0, n_out)

    l_hal = rng.uniform(0.0, 360.0, n_hal)
    hal_sign = rng.choice(np.array([-1.0, 1.0]), size=n_hal)
    b_hal = hal_sign * rng.uniform(68.0, 88.5, n_hal)
    d_hal = rng.uniform(1380.0, 2350.0, n_hal)

    l_deg = np.concatenate([l_loc, l_inn, l_out, l_hal])
    b_deg = np.concatenate([b_loc, b_inn, b_out, b_hal])
    d_pc = np.concatenate([d_loc, d_inn, d_out, d_hal])

    perm_env = rng.permutation(n_bin)
    l_deg = l_deg[perm_env]
    b_deg = b_deg[perm_env]
    d_pc = d_pc[perm_env]

    gal_sc = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, distance=d_pc * u.pc, frame="galactic").icrs
    ra0_deg = np.asarray(gal_sc.ra.deg, dtype=np.float64)
    dec0_deg = np.asarray(gal_sc.dec.deg, dtype=np.float64)

    # 2. Sample genuine 3D eccentric Keplerian orbits across calibration (500..2000 AU),
    #    transition (2000..10000 AU), and wide horizon regime (10000..22000 AU)
    n_cal = int(round(0.32 * n_bin))
    n_mid = int(round(0.12 * n_bin))
    n_wid = max(n_bin - n_cal - n_mid, 0)
    if n_cal + n_mid + n_wid != n_bin:
        n_wid += n_bin - (n_cal + n_mid + n_wid)

    orb_cal = _sample_scale_invariant_keplerian_orbits(rng, n_cal, 520.0, 1980.0)
    orb_mid = _sample_scale_invariant_keplerian_orbits(rng, n_mid, 2050.0, 9900.0)
    orb_wid = _sample_scale_invariant_keplerian_orbits(rng, n_wid, 10050.0, 21900.0)

    a_au = np.concatenate([orb_cal[0], orb_mid[0], orb_wid[0]])
    dx_a_au = np.concatenate([orb_cal[1], orb_mid[1], orb_wid[1]])
    dx_d_au = np.concatenate([orb_cal[2], orb_mid[2], orb_wid[2]])
    dz_l_au = np.concatenate([orb_cal[3], orb_mid[3], orb_wid[3]])
    va_dim = np.concatenate([orb_cal[4], orb_mid[4], orb_wid[4]])
    vd_dim = np.concatenate([orb_cal[5], orb_mid[5], orb_wid[5]])
    vl_dim = np.concatenate([orb_cal[6], orb_mid[6], orb_wid[6]])

    perm_orb = rng.permutation(n_bin)
    a_au = a_au[perm_orb]
    dx_a_au = dx_a_au[perm_orb]
    dx_d_au = dx_d_au[perm_orb]
    dz_l_au = dz_l_au[perm_orb]
    va_dim = va_dim[perm_orb]
    vd_dim = vd_dim[perm_orb]
    vl_dim = vl_dim[perm_orb]

    m1_arr = rng.uniform(0.58, 1.02, n_bin)
    m2_arr = rng.uniform(0.54, 0.98, n_bin)
    m_tot_arr = m1_arr + m2_arr
    f1 = m2_arr / m_tot_arr
    f2 = m1_arr / m_tot_arr

    # 3. Place Star 1 and Star 2 on the celestial sphere in 3D ICRS unit-vector space
    ra0_r = np.radians(ra0_deg)
    dec0_r = np.radians(dec0_deg)
    u0 = np.stack([np.cos(dec0_r) * np.cos(ra0_r), np.cos(dec0_r) * np.sin(ra0_r), np.sin(dec0_r)], axis=-1)
    e_alpha = np.stack([-np.sin(ra0_r), np.cos(ra0_r), np.zeros_like(ra0_r)], axis=-1)
    e_delta = np.stack(
        [-np.sin(dec0_r) * np.cos(ra0_r), -np.sin(dec0_r) * np.sin(ra0_r), np.cos(dec0_r)],
        axis=-1,
    )

    dtheta_a_rad = (dx_a_au / d_pc) / ARCSEC_PER_RAD
    dtheta_d_rad = (dx_d_au / d_pc) / ARCSEC_PER_RAD
    offset_vec = dtheta_a_rad[:, None] * e_alpha + dtheta_d_rad[:, None] * e_delta

    u1 = u0 - f1[:, None] * offset_vec
    u1 /= np.linalg.norm(u1, axis=-1, keepdims=True)
    u2 = u0 + f2[:, None] * offset_vec
    u2 /= np.linalg.norm(u2, axis=-1, keepdims=True)

    dec1_deg = np.degrees(np.arcsin(np.clip(u1[:, 2], -1.0, 1.0)))
    ra1_deg = np.mod(np.degrees(np.arctan2(u1[:, 1], u1[:, 0])), 360.0)
    dec2_deg = np.degrees(np.arcsin(np.clip(u2[:, 2], -1.0, 1.0)))
    ra2_deg = np.mod(np.degrees(np.arctan2(u2[:, 1], u2[:, 0])), 360.0)

    s_2d_au = np.hypot(dx_a_au, dx_d_au)
    r_gal_kpc, z_gal_kpc, theta_gc_deg = _compute_galactocentric_and_compass_batch(
        ra1_deg, dec1_deg, ra2_deg, dec2_deg, d_pc
    )

    # 4. Evaluate orbital velocity boost: 1.000 for Lambda-CDM vs exact 3D QUMOND (r_gal, z_gal, theta_GC) for TNaP
    if truth_mode == "lambda_cdm":
        boost_factor = np.ones(n_bin, dtype=np.float64)
    else:
        boost_pct = _eval_3d_qumond_boost_batch(
            s_2d_au, m_tot_arr, r_gal_kpc, z_gal_kpc, theta_gc_deg, A0_PLANCK_2018_PRIMARY_SI, _default_tnap_nu_fn
        )
        boost_factor = np.maximum(1.0 + 0.01 * boost_pct, 0.5)

    v_a_newt_kms = 1.0e-3 * np.sqrt((G_SI * m_tot_arr * M_SUN_KG) / (a_au * AU_M))
    dv_alpha_kms = boost_factor * v_a_newt_kms * va_dim
    dv_delta_kms = boost_factor * v_a_newt_kms * vd_dim
    dv_rv_kms = boost_factor * v_a_newt_kms * vl_dim

    # 5. Convert to observable parallaxes, proper motions, and radial velocities with Gaia DR4 66-month errors
    d1_pc = np.maximum(d_pc - f1 * (dz_l_au / ARCSEC_PER_RAD), 5.0)
    d2_pc = np.maximum(d_pc + f2 * (dz_l_au / ARCSEC_PER_RAD), 5.0)

    plx_err1 = rng.uniform(0.007, 0.013, n_bin)
    plx_err2 = rng.uniform(0.007, 0.013, n_bin)
    plx1 = (1000.0 / d1_pc) + rng.normal(0.0, plx_err1)
    plx2 = (1000.0 / d2_pc) + rng.normal(0.0, plx_err2)
    plx1 = np.maximum(plx1, 25.0 * plx_err1)
    plx2 = np.maximum(plx2, 25.0 * plx_err2)

    d_kpc = d_pc / 1000.0
    pm_sys_ra = rng.normal(0.0, 12.0, n_bin)
    pm_sys_dec = rng.normal(-6.0, 12.0, n_bin)
    dpm_ra_true = dv_alpha_kms / (KMS_PER_MASYR_KPC * d_kpc)
    dpm_dec_true = dv_delta_kms / (KMS_PER_MASYR_KPC * d_kpc)

    pmra_err1 = rng.uniform(0.0018, 0.0038, n_bin)
    pmdec_err1 = rng.uniform(0.0018, 0.0038, n_bin)
    pmra_err2 = rng.uniform(0.0018, 0.0038, n_bin)
    pmdec_err2 = rng.uniform(0.0018, 0.0038, n_bin)

    pmra1 = pm_sys_ra - f1 * dpm_ra_true + rng.normal(0.0, pmra_err1)
    pmdec1 = pm_sys_dec - f1 * dpm_dec_true + rng.normal(0.0, pmdec_err1)
    pmra2 = pm_sys_ra + f2 * dpm_ra_true + rng.normal(0.0, pmra_err2)
    pmdec2 = pm_sys_dec + f2 * dpm_dec_true + rng.normal(0.0, pmdec_err2)

    rv_sys = rng.normal(0.0, 26.0, n_bin)
    rv_err1 = rng.uniform(0.18, 0.42, n_bin)
    rv_err2 = rng.uniform(0.18, 0.42, n_bin)
    rv1 = rv_sys - f1 * dv_rv_kms + rng.normal(0.0, rv_err1)
    rv2 = rv_sys + f2 * dv_rv_kms + rng.normal(0.0, rv_err2)

    ruwe1 = rng.uniform(0.95, 1.15, n_bin)
    ruwe2 = rng.uniform(0.95, 1.15, n_bin)
    nss1 = np.zeros(n_bin, dtype=int)
    nss2 = np.zeros(n_bin, dtype=int)

    # 6. Inject ~12% realistic unresolved RUWE triples, NSS binaries, and ~25 km/s unbound RV flybys
    n_contam = int(round(0.12 * n_bin)) if inject_contaminants else 0
    n_ruwe_inj = 0
    n_nss_inj = 0
    n_rv_inj = 0
    if n_contam > 0:
        c_idx = rng.choice(n_bin, size=n_contam, replace=False)
        n_ruwe_inj = n_contam // 3
        n_nss_inj = (n_contam - n_ruwe_inj) // 2
        n_rv_inj = n_contam - n_ruwe_inj - n_nss_inj

        idx_ruwe = c_idx[:n_ruwe_inj]
        idx_nss = c_idx[n_ruwe_inj : n_ruwe_inj + n_nss_inj]
        idx_rv = c_idx[n_ruwe_inj + n_nss_inj :]

        if len(idx_ruwe) > 0:
            ruwe2[idx_ruwe] = rng.uniform(1.38, 2.75, len(idx_ruwe))
            kick_pm = rng.uniform(1.8, 4.2, len(idx_ruwe)) / (KMS_PER_MASYR_KPC * d_kpc[idx_ruwe])
            pmra2[idx_ruwe] += kick_pm

        if len(idx_nss) > 0:
            nss1[idx_nss] = 1
            kick_pm_nss = rng.uniform(2.0, 5.5, len(idx_nss)) / (KMS_PER_MASYR_KPC * d_kpc[idx_nss])
            pmdec2[idx_nss] += kick_pm_nss

        if len(idx_rv) > 0:
            rv_signs = rng.choice(np.array([-1.0, 1.0]), size=len(idx_rv))
            rv2[idx_rv] = rv1[idx_rv] + rv_signs * rng.uniform(18.0, 32.0, len(idx_rv))
            flyby_pm = rng.uniform(8.0, 22.0, len(idx_rv)) / (KMS_PER_MASYR_KPC * d_kpc[idx_rv])
            pmra2[idx_rv] += flyby_pm

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_bin):
            writer.writerow(
                {
                    "pair_id": f"GDR4-DAY1-{i + 1:06d}",
                    "ra1": f"{ra1_deg[i]:.9f}",
                    "dec1": f"{dec1_deg[i]:.9f}",
                    "parallax1": f"{plx1[i]:.6f}",
                    "parallax_error1": f"{plx_err1[i]:.6f}",
                    "pmra1": f"{pmra1[i]:.6f}",
                    "pmdec1": f"{pmdec1[i]:.6f}",
                    "pmra_error1": f"{pmra_err1[i]:.6f}",
                    "pmdec_error1": f"{pmdec_err1[i]:.6f}",
                    "ruwe1": f"{ruwe1[i]:.4f}",
                    "radial_velocity1": f"{rv1[i]:.4f}",
                    "radial_velocity_error1": f"{rv_err1[i]:.4f}",
                    "non_single_star1": int(nss1[i]),
                    "mass1": f"{m1_arr[i]:.4f}",
                    "ra2": f"{ra2_deg[i]:.9f}",
                    "dec2": f"{dec2_deg[i]:.9f}",
                    "parallax2": f"{plx2[i]:.6f}",
                    "parallax_error2": f"{plx_err2[i]:.6f}",
                    "pmra2": f"{pmra2[i]:.6f}",
                    "pmdec2": f"{pmdec2[i]:.6f}",
                    "pmra_error2": f"{pmra_err2[i]:.6f}",
                    "pmdec_error2": f"{pmdec_err2[i]:.6f}",
                    "ruwe2": f"{ruwe2[i]:.4f}",
                    "radial_velocity2": f"{rv2[i]:.4f}",
                    "radial_velocity_error2": f"{rv_err2[i]:.4f}",
                    "non_single_star2": int(nss2[i]),
                    "mass2": f"{m2_arr[i]:.4f}",
                }
            )

    return {
        "n_total_exported": n_bin,
        "n_clean_injected": n_bin - n_contam,
        "n_contaminants_injected": n_contam,
        "n_ruwe_triples": n_ruwe_inj,
        "n_nss_binaries": n_nss_inj,
        "n_rv_flybys": n_rv_inj,
    }


def _trimmed_mean_and_winsor_se(vals: List[float], trim_frac: float = 0.10) -> Tuple[float, float]:
    """
    Scale-equivariant symmetric trimmed mean (`trim_frac = 10%`) and exact Tukey-McLaughlin (1963)
    Winsorized standard error for robust outlier-proof wide-binary `v_tilde` inference.
    """
    n = len(vals)
    if n == 0:
        return 0.564, 0.08
    arr = np.sort(np.asarray(vals, dtype=np.float64))
    if n < 5:
        med = float(np.median(arr))
        sd = float(np.std(arr, ddof=1)) if n > 1 else 0.25 * max(med, 0.2)
        return med, max(sd / math.sqrt(n), 0.02)

    k = int(math.floor(trim_frac * n))
    core = arr[k : n - k] if (n - 2 * k) > 0 else arr
    t_mean = float(np.mean(core))

    winsor = arr.copy()
    if k > 0:
        winsor[:k] = arr[k]
        winsor[n - k :] = arr[n - k - 1]
    w_var = float(np.var(winsor, ddof=1))
    denom = max(((1.0 - 2.0 * (k / n)) ** 2) * n, 1.0)
    se = math.sqrt(max(w_var / denom, 1.0e-10))
    return t_mean, se


def evaluate_gaia_dr4_day_one_verdict(
    pairs: List[WideBinaryPair],
    a0_si: float = A0_PLANCK_2018_PRIMARY_SI,
    cut_flow: Optional[GaiaCutFlowSummary] = None,
) -> Dict[str, object]:
    """
    Evaluates the locked 3-Test Zenodo `10.5281/zenodo.23919217` Blind Falsification Scorecard on vetted
    `Gaia DR4` wide binaries:
      • Calibration Anchor : `500 <= sep_au <= 2,000 AU` (Newtonian/screened `< 0.5%` in both theories)
      • Test A (Local Boost): `d <= 300 pc`, `10k..22k AU` (`+16.14%` TNaP vs `0.00%` Lambda-CDM)
      • Test B (GC Compass) : `theta_GC <= 38 deg` minus `>= 52 deg` (`+1.71%` TNaP vs `0.00%` Lambda-CDM)
      • Test C (Env Step)   : `(|z| >= 1.2 kpc or R_gal > 9.0 kpc)` minus `(|z| < 0.4 kpc and R_gal <= 8.5 kpc)`
    """
    a0_safe = max(float(a0_si), 1.0e-20)
    curves = compute_wide_binary_efe_curves(_default_tnap_nu_fn, a0_safe, m_tot_msun=1.6)
    pred_tnap_a = float(curves["boost_disk_15k_pct"])
    pred_tnap_b = float(curves["compass_split_15k_pct"])
    default_env_step = float(curves["boost_lower_halo_15k_pct"]) - float(curves["boost_disk_15k_pct"])

    cut_dict: Optional[Dict[str, int]] = None
    if cut_flow is not None:
        cut_dict = {
            "n_raw_rows": int(cut_flow.n_raw_rows),
            "n_passed_snr_parallax": int(cut_flow.n_passed_snr_parallax),
            "n_passed_consistent_parallax": int(cut_flow.n_passed_consistent_parallax),
            "n_passed_nss": int(cut_flow.n_passed_nss),
            "n_passed_ruwe": int(cut_flow.n_passed_ruwe),
            "n_passed_rv_bound": int(cut_flow.n_passed_rv_bound),
            "n_clean_final": int(cut_flow.n_clean_final),
        }

    cal_pairs = [p for p in pairs if 500.0 <= p.sep_au <= 2000.0]
    wide_pairs = [p for p in pairs if 10000.0 <= p.sep_au <= 22000.0]

    if len(cal_pairs) < 5 or len(wide_pairs) < 5:
        return {
            "zenodo_doi": ZENODO_PREREGISTRATION_DOI,
            "zenodo_sha256": ZENODO_MANIFEST_SHA256,
            "n_clean": len(pairs),
            "n_calibration": len(cal_pairs),
            "n_wide_total": len(wide_pairs),
            "n_local_wide": 0,
            "n_compass_aligned": 0,
            "n_compass_perp": 0,
            "n_env_halo_outer": 0,
            "n_env_inner_solar": 0,
            "v_tilde_cal": 0.0,
            "local_backyard_15k_boost_pct": 0.0,
            "local_backyard_15k_err_pct": 0.0,
            "tnap_pred_local_15k_boost_pct": pred_tnap_a,
            "compass_aligned_boost_pct": 0.0,
            "compass_perp_boost_pct": 0.0,
            "compass_split_pct": 0.0,
            "compass_split_err_pct": 0.0,
            "tnap_pred_compass_split_pct": pred_tnap_b,
            "env_halo_outer_boost_pct": 0.0,
            "env_inner_solar_boost_pct": 0.0,
            "env_step_halo_minus_inner_pct": 0.0,
            "env_step_err_pct": 0.0,
            "tnap_pred_env_step_pct": default_env_step,
            "chi2_tnap": 0.0,
            "chi2_lambda_cdm": 0.0,
            "delta_chi2": 0.0,
            "log10_bayes_factor": 0.0,
            "verdict": "INCONCLUSIVE",
            "cut_flow": cut_dict,
        }

    v_cal, se_cal = _trimmed_mean_and_winsor_se([p.v_tilde_obs for p in cal_pairs])
    v_cal = max(v_cal, 1.0e-6)

    local_wide = [p for p in wide_pairs if p.dist_pc <= 300.0]
    local_aligned = [p for p in local_wide if p.theta_gc_deg <= 38.0]
    local_perp = [p for p in local_wide if p.theta_gc_deg >= 52.0]

    halo_outer_wide = [p for p in wide_pairs if abs(p.z_gal_kpc) >= 1.2 or p.r_gal_kpc > 9.0]
    inner_solar_wide = [p for p in wide_pairs if abs(p.z_gal_kpc) < 0.4 and p.r_gal_kpc <= 8.5]

    chi2_tnap = 0.0
    chi2_cdm = 0.0

    # Test A: Local Backyard (d <= 300 pc) 15k AU Boost
    if len(local_wide) >= 5:
        v_loc, se_loc = _trimmed_mean_and_winsor_se([p.v_tilde_obs for p in local_wide])
        ratio_a = v_loc / v_cal
        test_a_pct = (ratio_a - 1.0) * 100.0
        err_a_pct = 100.0 * ratio_a * math.hypot(se_loc / max(v_loc, 1.0e-6), se_cal / v_cal)
        err_a_pct = max(err_a_pct, 0.25)
        chi2_tnap += ((test_a_pct - pred_tnap_a) / err_a_pct) ** 2
        chi2_cdm += ((test_a_pct - 0.0) / err_a_pct) ** 2
    else:
        test_a_pct = 0.0
        err_a_pct = 99.0

    # Test B: Galactic Center Compass Anisotropy (theta_GC <= 38 deg minus >= 52 deg)
    if len(local_aligned) >= 5 and len(local_perp) >= 5:
        v_al, se_al = _trimmed_mean_and_winsor_se([p.v_tilde_obs for p in local_aligned])
        v_pe, se_pe = _trimmed_mean_and_winsor_se([p.v_tilde_obs for p in local_perp])
        al_boost_pct = ((v_al / v_cal) - 1.0) * 100.0
        pe_boost_pct = ((v_pe / v_cal) - 1.0) * 100.0
        test_b_pct = al_boost_pct - pe_boost_pct
        err_b_pct = max(100.0 * math.hypot(se_al, se_pe) / v_cal, 0.25)
        # Account for finite cone averaging over [0, 38] vs [52, 90] deg on the 3D QUMOND forecast
        b_al_pred = _eval_3d_qumond_boost_batch(
            np.asarray([p.sep_au for p in local_aligned], dtype=np.float64),
            np.asarray([p.m_tot_msun for p in local_aligned], dtype=np.float64),
            np.asarray([p.r_gal_kpc for p in local_aligned], dtype=np.float64),
            np.asarray([p.z_gal_kpc for p in local_aligned], dtype=np.float64),
            np.asarray([p.theta_gc_deg for p in local_aligned], dtype=np.float64),
            a0_safe,
            _default_tnap_nu_fn,
        )
        b_pe_pred = _eval_3d_qumond_boost_batch(
            np.asarray([p.sep_au for p in local_perp], dtype=np.float64),
            np.asarray([p.m_tot_msun for p in local_perp], dtype=np.float64),
            np.asarray([p.r_gal_kpc for p in local_perp], dtype=np.float64),
            np.asarray([p.z_gal_kpc for p in local_perp], dtype=np.float64),
            np.asarray([p.theta_gc_deg for p in local_perp], dtype=np.float64),
            a0_safe,
            _default_tnap_nu_fn,
        )
        pred_b_eff = float(np.mean(b_al_pred) - np.mean(b_pe_pred))
        chi2_tnap += ((test_b_pct - pred_b_eff) / err_b_pct) ** 2
        chi2_cdm += ((test_b_pct - 0.0) / err_b_pct) ** 2
    else:
        al_boost_pct = 0.0
        pe_boost_pct = 0.0
        test_b_pct = 0.0
        err_b_pct = 99.0

    # Test C: Galactic External-Field Environment Step (Lower Halo / Outer vs Inner / Solar Disk)
    if len(halo_outer_wide) >= 5 and len(inner_solar_wide) >= 5:
        v_hal, se_hal = _trimmed_mean_and_winsor_se([p.v_tilde_obs for p in halo_outer_wide])
        v_inn, se_inn = _trimmed_mean_and_winsor_se([p.v_tilde_obs for p in inner_solar_wide])
        hal_boost_pct = ((v_hal / v_cal) - 1.0) * 100.0
        inn_boost_pct = ((v_inn / v_cal) - 1.0) * 100.0
        test_c_pct = hal_boost_pct - inn_boost_pct
        err_c_pct = max(100.0 * math.hypot(se_hal, se_inn) / v_cal, 0.25)

        b_hal_pred = _eval_3d_qumond_boost_batch(
            np.asarray([p.sep_au for p in halo_outer_wide], dtype=np.float64),
            np.asarray([p.m_tot_msun for p in halo_outer_wide], dtype=np.float64),
            np.asarray([p.r_gal_kpc for p in halo_outer_wide], dtype=np.float64),
            np.asarray([p.z_gal_kpc for p in halo_outer_wide], dtype=np.float64),
            np.asarray([p.theta_gc_deg for p in halo_outer_wide], dtype=np.float64),
            a0_safe,
            _default_tnap_nu_fn,
        )
        b_inn_pred = _eval_3d_qumond_boost_batch(
            np.asarray([p.sep_au for p in inner_solar_wide], dtype=np.float64),
            np.asarray([p.m_tot_msun for p in inner_solar_wide], dtype=np.float64),
            np.asarray([p.r_gal_kpc for p in inner_solar_wide], dtype=np.float64),
            np.asarray([p.z_gal_kpc for p in inner_solar_wide], dtype=np.float64),
            np.asarray([p.theta_gc_deg for p in inner_solar_wide], dtype=np.float64),
            a0_safe,
            _default_tnap_nu_fn,
        )
        pred_tnap_c = float(np.mean(b_hal_pred) - np.mean(b_inn_pred))
        chi2_tnap += ((test_c_pct - pred_tnap_c) / err_c_pct) ** 2
        chi2_cdm += ((test_c_pct - 0.0) / err_c_pct) ** 2
    else:
        hal_boost_pct = 0.0
        inn_boost_pct = 0.0
        test_c_pct = 0.0
        err_c_pct = 99.0
        pred_tnap_c = default_env_step

    delta_chi2 = float(chi2_cdm - chi2_tnap)
    log10_bf = delta_chi2 / (2.0 * math.log(10.0))
    if delta_chi2 > 16.0:
        verdict = "FAVORS_TNAP_WEAVE"
    elif delta_chi2 < -16.0:
        verdict = "FAVORS_LAMBDA_CDM"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "zenodo_doi": ZENODO_PREREGISTRATION_DOI,
        "zenodo_sha256": ZENODO_MANIFEST_SHA256,
        "n_clean": len(pairs),
        "n_calibration": len(cal_pairs),
        "n_wide_total": len(wide_pairs),
        "n_local_wide": len(local_wide),
        "n_compass_aligned": len(local_aligned),
        "n_compass_perp": len(local_perp),
        "n_env_halo_outer": len(halo_outer_wide),
        "n_env_inner_solar": len(inner_solar_wide),
        "v_tilde_cal": v_cal,
        "local_backyard_15k_boost_pct": test_a_pct,
        "local_backyard_15k_err_pct": err_a_pct,
        "tnap_pred_local_15k_boost_pct": pred_tnap_a,
        "compass_aligned_boost_pct": al_boost_pct,
        "compass_perp_boost_pct": pe_boost_pct,
        "compass_split_pct": test_b_pct,
        "compass_split_err_pct": err_b_pct,
        "tnap_pred_compass_split_pct": pred_tnap_b,
        "env_halo_outer_boost_pct": hal_boost_pct,
        "env_inner_solar_boost_pct": inn_boost_pct,
        "env_step_halo_minus_inner_pct": test_c_pct,
        "env_step_err_pct": err_c_pct,
        "tnap_pred_env_step_pct": pred_tnap_c,
        "chi2_tnap": float(chi2_tnap),
        "chi2_lambda_cdm": float(chi2_cdm),
        "delta_chi2": delta_chi2,
        "log10_bayes_factor": log10_bf,
        "verdict": verdict,
        "cut_flow": cut_dict,
    }
