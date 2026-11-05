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
import hashlib
import io
import json
import math
from pathlib import Path
import threading
from typing import Callable, Dict, List, Optional, Tuple

from astropy import constants as const
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.table import Table
from astropy import units as u
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, RegularGridInterpolator
from scipy.optimize import brentq
from scipy.spatial import cKDTree


ZENODO_PREREGISTRATION_DOI = "10.5281/zenodo.23919217"
ZENODO_MANIFEST_SHA256 = "fa1f5cc50885f1de71fde71201a9320d6c071e44c586708207fb8fc3fe471328"

BUNDLED_REAL_GAIA_DR3_NPZ = Path(__file__).resolve().parent / "gaia_dr3_real_wide_binaries.npz"

GAIA_TAP_SYNC_MIRRORS: Tuple[str, ...] = (
    "https://gaia.ari.uni-heidelberg.de/tap/sync",
    "https://gea.esac.esa.int/tap-server/tap/sync",
    "https://gaia.aip.de/tap/sync",
)

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

PROTOCOL_V2_GAIA_CUTS: Dict[str, object] = {
    "min_parallax_over_error": 20.0,
    "max_ruwe": 1.20,
    "require_nss_zero": True,
    "max_sigma_v_tilde_1d": 0.25,
    "max_bound_v_tilde": 2.60,
    "reject_evolved_giants": True,
    "spherical_perspective_pm_correction": True,
    "wardle_kronberg_ricean_debiasing": True,
    "inner_triple_forward_calibration_sep_au": [500.0, 2000.0],
    "wide_test_sep_au": [10000.0, 22000.0],
    "local_backyard_max_dist_pc": 300.0,
    "environment_max_dist_pc": 2200.0,
}

# Calibrated Pecaut & Mamajek (2013, updated Gaia DR3 EEM dwarf sequence) (M_G, M / M_sun) knots
# Exact solar anchor: M_G,sun = +4.67 -> 1.000 M_sun
_PM_MG_KNOTS = np.array(
    [-1.00, 0.50, 1.50, 2.60, 3.50, 4.40, 4.67, 5.50, 6.50, 7.50, 8.80, 10.20, 11.80, 13.50, 15.50],
    dtype=np.float64,
)
_PM_MSUN_KNOTS = np.array(
    [5.20, 2.90, 2.00, 1.50, 1.25, 1.05, 1.00, 0.88, 0.75, 0.64, 0.50, 0.35, 0.20, 0.12, 0.085],
    dtype=np.float64,
)
_PM_LOGM_SPLINE = PchipInterpolator(_PM_MG_KNOTS, np.log10(_PM_MSUN_KNOTS), extrapolate=True)

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
    v_tilde_persp_debiased: Optional[float] = None
    sigma_v_tilde_1d: float = 0.0
    comp_class: str = "MS+MS"


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


def classify_gaia_stellar_component(
    phot_g_mean_mag: float,
    parallax_mas: float,
    bp_rp: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Empirical Gaia photometric stellar mass & HR-diagram locus solver (`"MS"`, `"WD"`, `"GIANT"`):
      • Absolute magnitude: `M_G = G + 5 * log10(parallax_mas / 100)`.
      • White-dwarf locus (Gentile Fusillo+ 2021 / El-Badry+ 2021):
        `M_G > 9.5` and `M_G > 10.0 + 2.6 * bp_rp` -> field WD mass peak `~0.60 M_sun`.
      • Evolved subgiant / red-giant locus:
        `M_G < 3.5` and `bp_rp > 0.92` -> `1.15 M_sun` (`"GIANT"`, flagged for `Protocol v2`).
      • Main sequence: monotonic `PchipInterpolator` over Pecaut–Mamajek `(M_G, log10 M)` knots.
    """
    g_val = float(phot_g_mean_mag)
    plx_val = float(parallax_mas)
    if not (math.isfinite(g_val) and math.isfinite(plx_val) and plx_val > 0.0):
        return 0.80, "MS"

    m_g = g_val + 5.0 * math.log10(plx_val / 100.0)
    has_color = bp_rp is not None and math.isfinite(float(bp_rp))
    col = float(bp_rp) if has_color else math.nan

    if has_color and m_g > 9.5 and m_g > (10.0 + 2.6 * col):
        m_wd = float(np.clip(0.60 + 0.025 * (m_g - 11.8), 0.54, 0.74))
        return m_wd, "WD"

    if has_color and m_g < 3.5 and col > 0.92:
        return 1.15, "GIANT"

    m_g_clamped = float(np.clip(m_g, -1.5, 16.2))
    m_ms = float(10.0 ** float(_PM_LOGM_SPLINE(m_g_clamped)))
    return float(np.clip(m_ms, 0.075, 6.50)), "MS"


def estimate_gaia_stellar_mass_msun(
    phot_g_mean_mag: float,
    parallax_mas: float,
    bp_rp: Optional[float] = None,
) -> float:
    """Returns empirical stellar mass in `M_sun` from Gaia `(phot_g_mean_mag, parallax_mas, bp_rp)`."""
    m_msun, _ = classify_gaia_stellar_component(phot_g_mean_mag, parallax_mas, bp_rp=bp_rp)
    return m_msun


def radec_to_gaia_healpix_nested(ra_deg: float, dec_deg: float, level: int = 12) -> int:
    """
    Exact Górski+ (2005, ApJ 622, 759) ICRS `(ra_deg, dec_deg) -> NESTED HEALPix` index at `nside = 2**level`.
    Matches ESA Gaia `source_id >> (59 - 2 * level)` (`source_id >> 35` at `level = 12`) bit-identically.
    """
    lvl = int(level)
    if not (0 <= lvl <= 12):
        raise ValueError(f"Gaia HEALPix level must be in [0, 12], got {level}")
    nside = 1 << lvl
    theta = math.radians(90.0 - float(dec_deg))
    phi = math.radians(float(ra_deg) % 360.0)
    z = math.cos(theta)
    za = abs(z)
    tt = (2.0 / math.pi) * phi

    if za <= 2.0 / 3.0:
        temp1 = nside * (0.5 + tt)
        temp2 = nside * (z * 0.75)
        jp = int(math.floor(temp1 - temp2))
        jm = int(math.floor(temp1 + temp2))
        ifp = jp // nside
        ifm = jm // nside
        if ifp == ifm:
            face_num = (ifp % 4) + 4
        elif ifp < ifm:
            face_num = ifp % 4
        else:
            face_num = (ifm % 4) + 8
        ix = jm & (nside - 1)
        iy = nside - (jp & (nside - 1)) - 1
    else:
        ntt = min(int(math.floor(tt)), 3)
        tp = tt - ntt
        tmp = nside * math.sqrt(3.0 * (1.0 - za))
        jp = min(int(math.floor(tp * tmp)), nside - 1)
        jm = min(int(math.floor((1.0 - tp) * tmp)), nside - 1)
        if z >= 0.0:
            face_num = ntt
            ix = nside - jm - 1
            iy = nside - jp - 1
        else:
            face_num = ntt + 8
            ix = jp
            iy = jm

    sub = 0
    for b in range(lvl):
        sub |= (((ix >> b) & 1) << (2 * b)) | (((iy >> b) & 1) << (2 * b + 1))
    return (face_num << (2 * lvl)) | sub


def _spherical_perspective_and_ricean_debias(
    ra1_deg: float,
    dec1_deg: float,
    pmra1: float,
    pmdec1: float,
    pmrae1: float,
    pmdece1: float,
    ra2_deg: float,
    dec2_deg: float,
    pmra2: float,
    pmdec2: float,
    pmrae2: float,
    pmdece2: float,
    dist_pc: float,
    rv1: Optional[float] = None,
    rv2: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Computes:
      1. `dv_2d_raw_kms`: standard sky-plane proper-motion difference (`Protocol v1`).
      2. `dv_2d_persp_debiased_kms`: exact 3D spherical-triad perspective proper-motion subtraction
         (Shaya & Olling 2011; El-Badry 2019) at common pair distance `dist_pc` (zero parallax-difference
         noise injection) followed by 4th-order-accurate Wardle–Kronberg Ricean perpendicular noise debiasing
         `sqrt(max(dv_persp^2 - sigma_v_perp^2, (0.10 * sigma_v_perp)^2))`.
      3. `sigma_v_perp_kms`: 1D proper-motion velocity error perpendicular to `Delta_mu_persp` (`km/s`).
    """
    d_kpc = max(float(dist_pc) / 1000.0, 1.0e-6)
    kd = KMS_PER_MASYR_KPC * d_kpc

    dpm_ra_raw = float(pmra2 - pmra1)
    dpm_dec_raw = float(pmdec2 - pmdec1)
    dv_2d_raw_kms = kd * math.hypot(dpm_ra_raw, dpm_dec_raw)

    a1, d1 = math.radians(ra1_deg), math.radians(dec1_deg)
    a2, d2 = math.radians(ra2_deg), math.radians(dec2_deg)

    ca1, sa1 = math.cos(a1), math.sin(a1)
    cd1, sd1 = math.cos(d1), math.sin(d1)
    ca2, sa2 = math.cos(a2), math.sin(a2)
    cd2, sd2 = math.cos(d2), math.sin(d2)

    r1 = (cd1 * ca1, cd1 * sa1, sd1)
    r2 = (cd2 * ca2, cd2 * sa2, sd2)
    ea1 = (-sa1, ca1, 0.0)
    ed1 = (-sd1 * ca1, -sd1 * sa1, cd1)
    ea2 = (-sa2, ca2, 0.0)
    ed2 = (-sd2 * ca2, -sd2 * sa2, cd2)

    rm_x, rm_y, rm_z = r1[0] + r2[0], r1[1] + r2[1], r1[2] + r2[2]
    rm_norm = max(math.sqrt(rm_x * rm_x + rm_y * rm_y + rm_z * rm_z), 1.0e-15)
    rm = (rm_x / rm_norm, rm_y / rm_norm, rm_z / rm_norm)
    dm = math.asin(max(-1.0, min(1.0, rm[2])))
    am = math.atan2(rm[1], rm[0])
    cam, sam = math.cos(am), math.sin(am)
    cdm, sdm = math.cos(dm), math.sin(dm)
    if abs(cdm) < 1.0e-12:
        eam = (0.0, 1.0, 0.0)
        edm = (-1.0, 0.0, 0.0)
    else:
        eam = (-sam, cam, 0.0)
        edm = (-sdm * cam, -sdm * sam, cdm)

    pmra_sys = 0.5 * (pmra1 + pmra2)
    pmdec_sys = 0.5 * (pmdec1 + pmdec2)
    if rv1 is not None and rv2 is not None and math.isfinite(rv1) and math.isfinite(rv2):
        rv_sys = 0.5 * (rv1 + rv2)
    elif rv1 is not None and math.isfinite(rv1):
        rv_sys = float(rv1)
    elif rv2 is not None and math.isfinite(rv2):
        rv_sys = float(rv2)
    else:
        rv_sys = 0.0

    vx = kd * (pmra_sys * eam[0] + pmdec_sys * edm[0]) + rv_sys * rm[0]
    vy = kd * (pmra_sys * eam[1] + pmdec_sys * edm[1]) + rv_sys * rm[1]
    vz = kd * (pmra_sys * eam[2] + pmdec_sys * edm[2]) + rv_sys * rm[2]

    pmra_persp = ((vx * ea2[0] + vy * ea2[1] + vz * ea2[2]) - (vx * ea1[0] + vy * ea1[1] + vz * ea1[2])) / kd
    pmdec_persp = ((vx * ed2[0] + vy * ed2[1] + vz * ed2[2]) - (vx * ed1[0] + vy * ed1[1] + vz * ed1[2])) / kd

    dpm_ra_corr = dpm_ra_raw - pmra_persp
    dpm_dec_corr = dpm_dec_raw - pmdec_persp
    dpm_corr_mag = math.hypot(dpm_ra_corr, dpm_dec_corr)
    dv_persp_kms = kd * dpm_corr_mag

    var_pmra = pmrae1 * pmrae1 + pmrae2 * pmrae2
    var_pmdec = pmdece1 * pmdece1 + pmdece2 * pmdece2
    if (var_pmra + var_pmdec) > 0.0:
        if dpm_corr_mag > 1.0e-12:
            nx = -dpm_dec_corr / dpm_corr_mag
            ny = dpm_ra_corr / dpm_corr_mag
            sigma_pm_perp = math.sqrt(nx * nx * var_pmra + ny * ny * var_pmdec)
        else:
            sigma_pm_perp = math.sqrt(0.5 * (var_pmra + var_pmdec))
        sigma_v_perp_kms = kd * sigma_pm_perp
        floor_sq = (0.10 * sigma_v_perp_kms) ** 2
        dv_debiased_kms = math.sqrt(max(dv_persp_kms * dv_persp_kms - sigma_v_perp_kms * sigma_v_perp_kms, floor_sq))
    else:
        sigma_v_perp_kms = 0.0
        dv_debiased_kms = dv_persp_kms

    return dv_2d_raw_kms, dv_debiased_kms, sigma_v_perp_kms


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
    vtilde_deb_list: List[float] = []
    sig_vt_list: List[float] = []
    cls_list: List[str] = []
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
        c_g1 = _pick_col(cmap, ["phot_g_mean_mag1", "phot_g_mean_mag_1", "gmag1", "g_mag1"])
        c_bprp1 = _pick_col(cmap, ["bp_rp1", "bp_rp_1", "bprp1"])

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
        c_g2 = _pick_col(cmap, ["phot_g_mean_mag2", "phot_g_mean_mag_2", "gmag2", "g_mag2"])
        c_bprp2 = _pick_col(cmap, ["bp_rp2", "bp_rp_2", "bprp2"])
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
                g1 = _opt_float(row, c_g1, None)
                bprp1 = _opt_float(row, c_bprp1, None)
                g2 = _opt_float(row, c_g2, None)
                bprp2 = _opt_float(row, c_bprp2, None)
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

            # Cut 1: Parallax SNR (`parallax / parallax_error >= min_poe`)
            if plx1 <= 0.0 or plx2 <= 0.0 or plxe1 <= 0.0 or plxe2 <= 0.0:
                continue
            if (plx1 / plxe1) < min_poe or (plx2 / plxe2) < min_poe:
                continue

            comp_cls = "MS+MS"
            if m1 is not None and m2 is not None:
                if not (math.isfinite(m1) and math.isfinite(m2) and m1 > 0.0 and m2 > 0.0):
                    continue
                m_tot = m1 + m2
                if g1 is not None and g2 is not None:
                    _, c1 = classify_gaia_stellar_component(g1, plx1, bprp1)
                    _, c2 = classify_gaia_stellar_component(g2, plx2, bprp2)
                    comp_cls = f"{c1}+{c2}"
            elif mtot_raw is not None:
                if not (math.isfinite(mtot_raw) and mtot_raw > 0.0):
                    continue
                m_tot = mtot_raw
            elif g1 is not None or g2 is not None:
                m1_est, c1 = classify_gaia_stellar_component(g1, plx1, bprp1) if g1 is not None else (0.80, "MS")
                m2_est, c2 = classify_gaia_stellar_component(g2, plx2, bprp2) if g2 is not None else (0.80, "MS")
                m_tot = m1_est + m2_est
                comp_cls = f"{c1}+{c2}"
            else:
                m_tot = 1.6

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

            _, dv_debiased_kms, sigma_v_perp_kms = _spherical_perspective_and_ricean_debias(
                ra1,
                dec1,
                pmra1,
                pmdec1,
                pmrae1,
                pmdece1,
                ra2,
                dec2,
                pmra2,
                pmdec2,
                pmrae2,
                pmdece2,
                dist_pc,
                rv1=rv1,
                rv2=rv2,
            )

            vc_safe = max(v_circ_n_kms, 1.0e-15)
            v_tilde = dv_2d_kms / vc_safe
            v_tilde_deb = dv_debiased_kms / vc_safe
            sig_vt = sigma_v_perp_kms / vc_safe
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
            vtilde_deb_list.append(v_tilde_deb)
            sig_vt_list.append(sig_vt)
            cls_list.append(comp_cls)
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
                v_tilde_persp_debiased=vtilde_deb_list[idx],
                sigma_v_tilde_1d=sig_vt_list[idx],
                comp_class=cls_list[idx],
            )
        )
    return pairs, cut_flow


def _ingest_two_star_column_mapping(
    col_data: Dict[str, np.ndarray],
    max_ruwe: float = 1.20,
    min_poe: float = 20.0,
    rv_sigma_clip: float = 3.0,
    max_sigma_v2d_frac: Optional[float] = 0.35,
) -> Tuple[List[WideBinaryPair], GaiaCutFlowSummary]:
    """
    Featherweight columnar ingestion (`< 25 MB` RAM) for `.npz`, `.fits`, `.vot`, and `.ecsv` two-star Gaia tables.
    Applies identical pre-registered Zenodo quality cuts, Pecaut–Mamajek + WD/Giant photometric mass solving,
    and 3D spherical perspective + Wardle–Kronberg Ricean debiasing.
    """
    cut_flow = GaiaCutFlowSummary()
    cmap = {k.strip().lower(): k for k in col_data.keys() if not k.startswith("catalog_")}

    def _arr(aliases: List[str], default_val: Optional[float] = None) -> Optional[np.ndarray]:
        for al in aliases:
            if al in cmap:
                raw_a = col_data[cmap[al]]
                if hasattr(raw_a, "filled"):
                    raw_a = raw_a.filled(np.nan if default_val is None else default_val)
                return np.asarray(raw_a)
        return None

    a_ra1 = _arr(["ra1", "ra1_deg", "ra_1", "ra_1_deg"])
    a_dec1 = _arr(["dec1", "dec1_deg", "dec_1", "dec_1_deg"])
    a_plx1 = _arr(["parallax1", "parallax1_mas", "plx1", "parallax_1", "parallax_1_mas"])
    a_plxe1 = _arr(["parallax_error1", "parallax_error1_mas", "parallax1_error", "plx_err1", "parallax_error_1"])
    a_pmra1 = _arr(["pmra1", "pmra1_masyr", "pmra_1", "pmra_1_masyr"])
    a_pmdec1 = _arr(["pmdec1", "pmdec1_masyr", "pmdec_1", "pmdec_1_masyr"])

    a_ra2 = _arr(["ra2", "ra2_deg", "ra_2", "ra_2_deg"])
    a_dec2 = _arr(["dec2", "dec2_deg", "dec_2", "dec_2_deg"])
    a_plx2 = _arr(["parallax2", "parallax2_mas", "plx2", "parallax_2", "parallax_2_mas"])
    a_plxe2 = _arr(["parallax_error2", "parallax_error2_mas", "parallax2_error", "plx_err2", "parallax_error_2"])
    a_pmra2 = _arr(["pmra2", "pmra2_masyr", "pmra_2", "pmra_2_masyr"])
    a_pmdec2 = _arr(["pmdec2", "pmdec2_masyr", "pmdec_2", "pmdec_2_masyr"])

    if any(
        x is None
        for x in (a_ra1, a_dec1, a_plx1, a_plxe1, a_pmra1, a_pmdec1, a_ra2, a_dec2, a_plx2, a_plxe2, a_pmra2, a_pmdec2)
    ):
        return [], cut_flow

    assert a_ra1 is not None and a_dec1 is not None and a_plx1 is not None and a_plxe1 is not None
    assert a_pmra1 is not None and a_pmdec1 is not None
    assert a_ra2 is not None and a_dec2 is not None and a_plx2 is not None and a_plxe2 is not None
    assert a_pmra2 is not None and a_pmdec2 is not None

    n_rows = int(len(a_ra1))
    cut_flow.n_raw_rows = n_rows
    if n_rows == 0:
        return [], cut_flow

    ra1_v = np.asarray(a_ra1, dtype=np.float64)
    dec1_v = np.asarray(a_dec1, dtype=np.float64)
    plx1_v = np.asarray(a_plx1, dtype=np.float64)
    plxe1_v = np.asarray(a_plxe1, dtype=np.float64)
    pmra1_v = np.asarray(a_pmra1, dtype=np.float64)
    pmdec1_v = np.asarray(a_pmdec1, dtype=np.float64)

    ra2_v = np.asarray(a_ra2, dtype=np.float64)
    dec2_v = np.asarray(a_dec2, dtype=np.float64)
    plx2_v = np.asarray(a_plx2, dtype=np.float64)
    plxe2_v = np.asarray(a_plxe2, dtype=np.float64)
    pmra2_v = np.asarray(a_pmra2, dtype=np.float64)
    pmdec2_v = np.asarray(a_pmdec2, dtype=np.float64)

    def _f64_or_default(aliases: List[str], fill: float) -> np.ndarray:
        found = _arr(aliases, fill)
        if found is None:
            return np.full(n_rows, fill, dtype=np.float64)
        out = np.asarray(found, dtype=np.float64)
        return np.where(np.isfinite(out), out, fill)

    def _f64_or_nan(aliases: List[str]) -> Optional[np.ndarray]:
        found = _arr(aliases, None)
        if found is None:
            return None
        return np.asarray(found, dtype=np.float64)

    pmrae1_v = _f64_or_default(["pmra_error1", "pmra_error1_masyr", "pmra1_error", "pmra_err1", "pmra_error_1"], 0.0)
    pmdece1_v = _f64_or_default(
        ["pmdec_error1", "pmdec_error1_masyr", "pmdec1_error", "pmdec_err1", "pmdec_error_1"], 0.0
    )
    pmrae2_v = _f64_or_default(["pmra_error2", "pmra_error2_masyr", "pmra2_error", "pmra_err2", "pmra_error_2"], 0.0)
    pmdece2_v = _f64_or_default(
        ["pmdec_error2", "pmdec_error2_masyr", "pmdec2_error", "pmdec_err2", "pmdec_error_2"], 0.0
    )
    ruwe1_v = _f64_or_default(["ruwe1", "ruwe_1"], 1.0)
    ruwe2_v = _f64_or_default(["ruwe2", "ruwe_2"], 1.0)
    nss1_v = _f64_or_default(["non_single_star1", "nss1", "nss_flag1", "non_single_star_1", "nss_1"], 0.0)
    nss2_v = _f64_or_default(["non_single_star2", "nss2", "nss_flag2", "non_single_star_2", "nss_2"], 0.0)
    rve1_v = _f64_or_default(
        ["radial_velocity_error1", "rv_error1", "rv1_error", "rv_err1", "radial_velocity_error_1"], 0.0
    )
    rve2_v = _f64_or_default(
        ["radial_velocity_error2", "rv_error2", "rv2_error", "rv_err2", "radial_velocity_error_2"], 0.0
    )

    rv1_v = _f64_or_nan(["radial_velocity1", "rv1", "rv1_kms", "radial_velocity_1"])
    rv2_v = _f64_or_nan(["radial_velocity2", "rv2", "rv2_kms", "radial_velocity_2"])
    m1_v = _f64_or_nan(["mass1", "mass1_msun", "m1", "m1_msun", "mass_1"])
    m2_v = _f64_or_nan(["mass2", "mass2_msun", "m2", "m2_msun", "mass_2"])
    mtot_v = _f64_or_nan(["m_tot_msun", "mass_tot", "m_tot"])
    g1_v = _f64_or_nan(["phot_g_mean_mag1", "phot_g_mean_mag_1", "gmag1", "g_mag1"])
    bprp1_v = _f64_or_nan(["bp_rp1", "bp_rp_1", "bprp1"])
    g2_v = _f64_or_nan(["phot_g_mean_mag2", "phot_g_mean_mag_2", "gmag2", "g_mag2"])
    bprp2_v = _f64_or_nan(["bp_rp2", "bp_rp_2", "bprp2"])

    a_pid = _arr(["pair_id", "source_pair_id", "binary_id", "id"])
    a_sid1 = _arr(["source_id1", "source_id_1"])
    a_sid2 = _arr(["source_id2", "source_id_2"])

    pids: List[str] = []
    ra1_list: List[float] = []
    dec1_list: List[float] = []
    ra2_list: List[float] = []
    dec2_list: List[float] = []
    sep_list: List[float] = []
    dist_list: List[float] = []
    mtot_list: List[float] = []
    vtilde_list: List[float] = []
    vtilde_deb_list: List[float] = []
    sig_vt_list: List[float] = []
    cls_list: List[str] = []
    ruwe_list: List[float] = []
    drv_list: List[Optional[float]] = []

    for i in range(n_rows):
        ra1, dec1, plx1, plxe1 = float(ra1_v[i]), float(dec1_v[i]), float(plx1_v[i]), float(plxe1_v[i])
        pmra1, pmdec1 = float(pmra1_v[i]), float(pmdec1_v[i])
        ra2, dec2, plx2, plxe2 = float(ra2_v[i]), float(dec2_v[i]), float(plx2_v[i]), float(plxe2_v[i])
        pmra2, pmdec2 = float(pmra2_v[i]), float(pmdec2_v[i])

        if not all(
            math.isfinite(v)
            for v in (ra1, dec1, plx1, plxe1, pmra1, pmdec1, ra2, dec2, plx2, plxe2, pmra2, pmdec2)
        ):
            continue
        if not (-90.0 <= dec1 <= 90.0 and -90.0 <= dec2 <= 90.0):
            continue
        if plx1 <= 0.0 or plx2 <= 0.0 or plxe1 <= 0.0 or plxe2 <= 0.0:
            continue
        if (plx1 / plxe1) < min_poe or (plx2 / plxe2) < min_poe:
            continue

        pmrae1, pmdece1 = max(float(pmrae1_v[i]), 0.0), max(float(pmdece1_v[i]), 0.0)
        pmrae2, pmdece2 = max(float(pmrae2_v[i]), 0.0), max(float(pmdece2_v[i]), 0.0)
        ruwe1, ruwe2 = float(ruwe1_v[i]), float(ruwe2_v[i])
        nss1 = int(round(float(nss1_v[i])))
        nss2 = int(round(float(nss2_v[i])))
        rve1, rve2 = max(float(rve1_v[i]), 0.0), max(float(rve2_v[i]), 0.0)

        rv1 = float(rv1_v[i]) if (rv1_v is not None and math.isfinite(float(rv1_v[i]))) else None
        rv2 = float(rv2_v[i]) if (rv2_v is not None and math.isfinite(float(rv2_v[i]))) else None
        g1 = float(g1_v[i]) if (g1_v is not None and math.isfinite(float(g1_v[i]))) else None
        bprp1 = float(bprp1_v[i]) if (bprp1_v is not None and math.isfinite(float(bprp1_v[i]))) else None
        g2 = float(g2_v[i]) if (g2_v is not None and math.isfinite(float(g2_v[i]))) else None
        bprp2 = float(bprp2_v[i]) if (bprp2_v is not None and math.isfinite(float(bprp2_v[i]))) else None

        m1_in = float(m1_v[i]) if (m1_v is not None and math.isfinite(float(m1_v[i]))) else None
        m2_in = float(m2_v[i]) if (m2_v is not None and math.isfinite(float(m2_v[i]))) else None
        mtot_in = float(mtot_v[i]) if (mtot_v is not None and math.isfinite(float(mtot_v[i]))) else None

        comp_cls = "MS+MS"
        if m1_in is not None and m2_in is not None:
            if m1_in <= 0.0 or m2_in <= 0.0:
                continue
            m_tot = m1_in + m2_in
            if g1 is not None and g2 is not None:
                _, c1 = classify_gaia_stellar_component(g1, plx1, bprp1)
                _, c2 = classify_gaia_stellar_component(g2, plx2, bprp2)
                comp_cls = f"{c1}+{c2}"
        elif mtot_in is not None:
            if mtot_in <= 0.0:
                continue
            m_tot = mtot_in
        elif g1 is not None or g2 is not None:
            m1_est, c1 = classify_gaia_stellar_component(g1, plx1, bprp1) if g1 is not None else (0.80, "MS")
            m2_est, c2 = classify_gaia_stellar_component(g2, plx2, bprp2) if g2 is not None else (0.80, "MS")
            m_tot = m1_est + m2_est
            comp_cls = f"{c1}+{c2}"
        else:
            m_tot = 1.6

        a1_r, d1_r = math.radians(ra1), math.radians(dec1)
        a2_r, d2_r = math.radians(ra2), math.radians(dec2)
        ux1, uy1, uz1 = math.cos(d1_r) * math.cos(a1_r), math.cos(d1_r) * math.sin(a1_r), math.sin(d1_r)
        ux2, uy2, uz2 = math.cos(d2_r) * math.cos(a2_r), math.cos(d2_r) * math.sin(a2_r), math.sin(d2_r)
        chord = math.sqrt((ux2 - ux1) ** 2 + (uy2 - uy1) ** 2 + (uz2 - uz1) ** 2)
        dtheta_arcsec = 2.0 * math.asin(min(0.5 * chord, 1.0)) * ARCSEC_PER_RAD

        plx_mean = 0.5 * (plx1 + plx2)
        dist_pc = 1000.0 / plx_mean
        sep_au = dtheta_arcsec * dist_pc
        if sep_au <= 0.0 or not math.isfinite(sep_au):
            continue
        cut_flow.n_passed_snr_parallax += 1

        orb_plx_term = 2.5 * sep_au / 206265.0 * ((plx_mean / 1000.0) ** 2)
        plx_tol = math.sqrt((3.5**2) * (plxe1**2 + plxe2**2) + orb_plx_term**2)
        if abs(plx1 - plx2) > plx_tol:
            continue
        cut_flow.n_passed_consistent_parallax += 1

        if nss1 != 0 or nss2 != 0:
            continue
        cut_flow.n_passed_nss += 1

        if not (0.0 < ruwe1 <= max_ruwe and 0.0 < ruwe2 <= max_ruwe):
            continue
        cut_flow.n_passed_ruwe += 1

        v_circ_n_kms = 1.0e-3 * math.sqrt((G_SI * m_tot * M_SUN_KG) / (sep_au * AU_M))
        v_esc_n_kms = math.sqrt(2.0) * v_circ_n_kms
        delta_rv: Optional[float] = None
        if rv1 is not None and rv2 is not None:
            delta_rv = abs(rv1 - rv2)
            rv_bound_max = math.sqrt((1.65 * v_esc_n_kms) ** 2 + (rv_sigma_clip**2) * (rve1**2 + rve2**2))
            if delta_rv > rv_bound_max:
                continue
        cut_flow.n_passed_rv_bound += 1

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

        _, dv_debiased_kms, sigma_v_perp_kms = _spherical_perspective_and_ricean_debias(
            ra1,
            dec1,
            pmra1,
            pmdec1,
            pmrae1,
            pmdece1,
            ra2,
            dec2,
            pmra2,
            pmdec2,
            pmrae2,
            pmdece2,
            dist_pc,
            rv1=rv1,
            rv2=rv2,
        )

        vc_safe = max(v_circ_n_kms, 1.0e-15)
        v_tilde = dv_2d_kms / vc_safe
        v_tilde_deb = dv_debiased_kms / vc_safe
        sig_vt = sigma_v_perp_kms / vc_safe
        if not (math.isfinite(v_tilde) and v_tilde > 0.0):
            continue

        if a_pid is not None and str(a_pid[i]).strip():
            pid = str(a_pid[i]).strip()
        elif a_sid1 is not None and a_sid2 is not None:
            pid = f"{int(a_sid1[i])}_{int(a_sid2[i])}"
        else:
            pid = f"GDR-COL-{i + 1:06d}"

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
        vtilde_deb_list.append(v_tilde_deb)
        sig_vt_list.append(sig_vt)
        cls_list.append(comp_cls)
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
                v_tilde_persp_debiased=vtilde_deb_list[idx],
                sigma_v_tilde_1d=sig_vt_list[idx],
                comp_class=cls_list[idx],
            )
        )
    return pairs, cut_flow


def ingest_gaia_wide_binary_table(
    table_path: Path,
    max_ruwe: float = 1.20,
    min_poe: float = 20.0,
    rv_sigma_clip: float = 3.0,
    max_sigma_v2d_frac: Optional[float] = 0.35,
) -> Tuple[List[WideBinaryPair], GaiaCutFlowSummary]:
    """
    Multi-format featherweight (`< 50 MB` RAM) Gaia wide-binary table reader supporting:
      • `.npz` (compressed NumPy columnar snapshot)
      • `.fits` / `.fit` (FITS binary table)
      • `.vot` / `.xml` (IVOA VOTable from ESA/DPAC TAP)
      • `.ecsv` (Astropy Enhanced CSV)
      • `.csv` (raw two-star ICRS astrometry or pre-reduced `v_tilde_obs` CSV)
    """
    p = Path(table_path)
    if not p.exists():
        return [], GaiaCutFlowSummary()

    suf = p.suffix.lower()
    if suf == ".csv":
        if is_raw_gaia_astrometry_csv(p):
            return ingest_gaia_dr4_raw_astrometry(
                p,
                max_ruwe=max_ruwe,
                min_poe=min_poe,
                rv_sigma_clip=rv_sigma_clip,
                max_sigma_v2d_frac=max_sigma_v2d_frac,
            )
        red_pairs = load_gaia_dr4_csv(p, max_ruwe=max_ruwe)
        cf = GaiaCutFlowSummary(n_raw_rows=len(red_pairs), n_clean_final=len(red_pairs))
        return red_pairs, cf

    if suf == ".npz":
        with np.load(p, allow_pickle=False) as data:
            col_dict = {k: data[k] for k in data.files if data[k].ndim >= 1}
        return _ingest_two_star_column_mapping(
            col_dict,
            max_ruwe=max_ruwe,
            min_poe=min_poe,
            rv_sigma_clip=rv_sigma_clip,
            max_sigma_v2d_frac=max_sigma_v2d_frac,
        )

    if suf in (".fits", ".fit"):
        tbl = Table.read(p, format="fits")
    elif suf in (".vot", ".xml", ".votable"):
        tbl = Table.read(p, format="votable")
    elif suf == ".ecsv":
        tbl = Table.read(p, format="ascii.ecsv")
    else:
        tbl = Table.read(p)

    lower_cols = {c.lower(): c for c in tbl.colnames}
    if "v_tilde_obs" in lower_cols and not bool(set(lower_cols).intersection({"ra1", "ra1_deg", "ra_1"})):
        pairs_red: List[WideBinaryPair] = []
        for r_idx, row in enumerate(tbl):
            ruwe_val = float(row[lower_cols["ruwe_max"]]) if "ruwe_max" in lower_cols else 1.0
            if not (0.0 < ruwe_val <= max_ruwe):
                continue
            sep_v = float(row[lower_cols["sep_au"]])
            z_v = float(row[lower_cols["z_gal_kpc"]])
            m_v = float(row[lower_cols["m_tot_msun"]]) if "m_tot_msun" in lower_cols else 1.6
            vt_v = float(row[lower_cols["v_tilde_obs"]])
            pid_v = str(row[lower_cols["pair_id"]]) if "pair_id" in lower_cols else f"TBL-{r_idx + 1:06d}"
            if sep_v > 0.0 and m_v > 0.0 and vt_v > 0.0:
                pairs_red.append(
                    WideBinaryPair(
                        pair_id=pid_v,
                        sep_au=sep_v,
                        z_gal_kpc=z_v,
                        m_tot_msun=m_v,
                        v_tilde_obs=vt_v,
                        ruwe_max=ruwe_val,
                    )
                )
        cf_red = GaiaCutFlowSummary(n_raw_rows=len(tbl), n_clean_final=len(pairs_red))
        return pairs_red, cf_red

    col_map: Dict[str, np.ndarray] = {}
    for cname in tbl.colnames:
        col_obj = tbl[cname]
        if hasattr(col_obj, "filled"):
            fill_v = np.nan if col_obj.dtype.kind == "f" else ("" if col_obj.dtype.kind in ("U", "S") else 0)
            col_map[cname] = np.asarray(col_obj.filled(fill_v))
        else:
            col_map[cname] = np.asarray(col_obj)
    return _ingest_two_star_column_mapping(
        col_map,
        max_ruwe=max_ruwe,
        min_poe=min_poe,
        rv_sigma_clip=rv_sigma_clip,
        max_sigma_v2d_frac=max_sigma_v2d_frac,
    )


_BUNDLED_DR3_CACHE: Dict[Tuple[float, float, Optional[float]], Tuple[List[WideBinaryPair], GaiaCutFlowSummary]] = {}


def load_bundled_real_gaia_dr3_wide_binaries(
    max_ruwe: float = 1.20,
    min_poe: float = 20.0,
    max_sigma_v2d_frac: Optional[float] = 0.35,
    npz_path: Optional[Path] = None,
) -> Tuple[List[WideBinaryPair], GaiaCutFlowSummary]:
    """
    Loads the bundled sky-complete real ESA/DPAC `Gaia DR3` isolated wide-binary calibration snapshot
    (`weaveforge/data/gaia_dr3_real_wide_binaries.npz`, `6,322` isolated comoving pairs, `1.16 MB`).
    """
    target = Path(npz_path) if npz_path is not None else BUNDLED_REAL_GAIA_DR3_NPZ
    use_cache = npz_path is None
    key = (round(float(max_ruwe), 4), round(float(min_poe), 4), max_sigma_v2d_frac)
    if use_cache and key in _BUNDLED_DR3_CACHE:
        cached_pairs, cached_cf = _BUNDLED_DR3_CACHE[key]
        return list(cached_pairs), GaiaCutFlowSummary(**cached_cf.__dict__)

    pairs, cf = ingest_gaia_wide_binary_table(
        target,
        max_ruwe=max_ruwe,
        min_poe=min_poe,
        max_sigma_v2d_frac=max_sigma_v2d_frac,
    )
    if use_cache and pairs:
        _BUNDLED_DR3_CACHE[key] = (list(pairs), GaiaCutFlowSummary(**cf.__dict__))
    return pairs, cf


def pair_isolated_wide_binaries_ckdtree(
    stars: Dict[str, np.ndarray],
    min_sep_au: float = 350.0,
    max_sep_au: float = 25000.0,
    max_dv_2d_kms: float = 5.0,
    max_search_deg: float = 0.65,
) -> Dict[str, np.ndarray]:
    """
    Local 3D unit-sphere `scipy.spatial.cKDTree` comoving wide-binary pair finder with strict
    El-Badry+ (2021) / Banik+ (2024) 3rd-star phase-space neighbor isolation (`partner_count == 1`),
    eliminating slow server-side ADQL spatial self-joins on Dec 2, 2026 (`Gaia DR4` Day One).
    """
    sid_arr = np.asarray(stars["source_id"], dtype=np.int64)
    ra_arr = np.asarray(stars["ra"], dtype=np.float64)
    dec_arr = np.asarray(stars["dec"], dtype=np.float64)
    plx_arr = np.asarray(stars["parallax"], dtype=np.float64)
    plxe_arr = np.asarray(stars["parallax_error"], dtype=np.float64)
    pmra_arr = np.asarray(stars["pmra"], dtype=np.float64)
    pmdec_arr = np.asarray(stars["pmdec"], dtype=np.float64)
    pmrae_arr = np.asarray(stars["pmra_error"], dtype=np.float64)
    pmdece_arr = np.asarray(stars["pmdec_error"], dtype=np.float64)
    ruwe_arr = np.asarray(stars.get("ruwe", np.ones_like(ra_arr)), dtype=np.float64)
    rv_arr = np.asarray(stars.get("radial_velocity", np.full_like(ra_arr, np.nan)), dtype=np.float64)
    rve_arr = np.asarray(stars.get("radial_velocity_error", np.zeros_like(ra_arr)), dtype=np.float64)
    nss_arr = np.asarray(stars.get("non_single_star", np.zeros(len(ra_arr), dtype=np.int16)), dtype=np.int16)
    gmag_arr = np.asarray(stars.get("phot_g_mean_mag", np.full_like(ra_arr, np.nan)), dtype=np.float64)
    bprp_arr = np.asarray(stars.get("bp_rp", np.full_like(ra_arr, np.nan)), dtype=np.float64)

    if len(sid_arr) < 2:
        return {f"{k}1": np.empty(0) for k in ("source_id", "ra", "dec", "parallax")}

    ra_rad = np.radians(ra_arr)
    dec_rad = np.radians(dec_arr)
    u_xyz = np.column_stack(
        [np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)]
    )
    tree = cKDTree(u_xyz)
    max_chord = 2.0 * math.sin(math.radians(float(max_search_deg)) / 2.0)
    raw_pairs = tree.query_pairs(r=max_chord, output_type="ndarray")

    cand_i: List[int] = []
    cand_j: List[int] = []
    partner_count = np.zeros(len(sid_arr), dtype=np.int32)

    for i_raw, j_raw in raw_pairs:
        g_i = gmag_arr[i_raw] if math.isfinite(float(gmag_arr[i_raw])) else 99.0
        g_j = gmag_arr[j_raw] if math.isfinite(float(gmag_arr[j_raw])) else 99.0
        i, j = (int(i_raw), int(j_raw)) if g_i <= g_j else (int(j_raw), int(i_raw))
        plx_m = 0.5 * (plx_arr[i] + plx_arr[j])
        if plx_m <= 0.0:
            continue
        dist_pc = 1000.0 / plx_m
        chord = float(np.linalg.norm(u_xyz[i] - u_xyz[j]))
        dtheta_arcsec = 2.0 * math.asin(min(0.5 * chord, 1.0)) * ARCSEC_PER_RAD
        sep_au = dtheta_arcsec * dist_pc
        if not (min_sep_au <= sep_au <= max_sep_au):
            continue
        orb_plx = 2.5 * sep_au / 206265.0 * ((plx_m / 1000.0) ** 2)
        plx_tol = math.sqrt(16.0 * (plxe_arr[i] ** 2 + plxe_arr[j] ** 2) + orb_plx**2)
        if abs(plx_arr[i] - plx_arr[j]) > plx_tol:
            continue
        dpm = math.hypot(pmra_arr[j] - pmra_arr[i], pmdec_arr[j] - pmdec_arr[i])
        dv_2d = KMS_PER_MASYR_KPC * (dist_pc / 1000.0) * dpm
        if dv_2d > max_dv_2d_kms:
            continue
        partner_count[i] += 1
        partner_count[j] += 1
        cand_i.append(i)
        cand_j.append(j)

    iso_mask = np.array(
        [(partner_count[i] == 1) and (partner_count[j] == 1) for i, j in zip(cand_i, cand_j)],
        dtype=bool,
    )
    idx_i = np.asarray(cand_i, dtype=np.int64)[iso_mask]
    idx_j = np.asarray(cand_j, dtype=np.int64)[iso_mask]

    return {
        "source_id1": sid_arr[idx_i],
        "ra1": ra_arr[idx_i],
        "dec1": dec_arr[idx_i],
        "parallax1": plx_arr[idx_i],
        "parallax_error1": plxe_arr[idx_i],
        "pmra1": pmra_arr[idx_i],
        "pmdec1": pmdec_arr[idx_i],
        "pmra_error1": pmrae_arr[idx_i],
        "pmdec_error1": pmdece_arr[idx_i],
        "ruwe1": ruwe_arr[idx_i],
        "radial_velocity1": rv_arr[idx_i],
        "radial_velocity_error1": rve_arr[idx_i],
        "non_single_star1": nss_arr[idx_i],
        "phot_g_mean_mag1": gmag_arr[idx_i],
        "bp_rp1": bprp_arr[idx_i],
        "source_id2": sid_arr[idx_j],
        "ra2": ra_arr[idx_j],
        "dec2": dec_arr[idx_j],
        "parallax2": plx_arr[idx_j],
        "parallax_error2": plxe_arr[idx_j],
        "pmra2": pmra_arr[idx_j],
        "pmdec2": pmdec_arr[idx_j],
        "pmra_error2": pmrae_arr[idx_j],
        "pmdec_error2": pmdece_arr[idx_j],
        "ruwe2": ruwe_arr[idx_j],
        "radial_velocity2": rv_arr[idx_j],
        "radial_velocity_error2": rve_arr[idx_j],
        "non_single_star2": nss_arr[idx_j],
        "phot_g_mean_mag2": gmag_arr[idx_j],
        "bp_rp2": bprp_arr[idx_j],
    }


def fetch_esa_gaia_wide_binaries_tap(
    ra_deg: float = 192.86,
    dec_deg: float = 27.13,
    healpix_level: int = 2,
    min_parallax_mas: float = 0.65,
    min_poe: float = 20.0,
    max_ruwe: float = 1.20,
    top_rows: int = 12000,
    release: str = "dr3",
    timeout_s: float = 20.0,
    mirrors: Tuple[str, ...] = GAIA_TAP_SYNC_MIRRORS,
    save_npz_path: Optional[Path] = None,
    http_client: Optional[object] = None,
) -> Tuple[List[WideBinaryPair], GaiaCutFlowSummary, Dict[str, object]]:
    """
    Live Day-One ESA / DPAC Gaia TAP client (`release='dr3'` -> `gaiadr3.gaia_source`,
    `release='dr4'` -> `gaiadr4.gaia_source` on Dec 2, 2026):
      • Uses exact nested-HEALPix `PRIMARY KEY (source_id BETWEEN lo AND hi)` single-table B-tree range
        queries (zero server-side spatial self-joins) + multi-mirror failover (`ARI Heidelberg -> ESA ESAC -> AIP`).
      • Pairs strictly isolated wide binaries locally in `< 40 ms` via 3D `cKDTree`.
    """
    import httpx

    rel = release.strip().lower()
    table_name = "gaiadr4.gaia_source" if rel in ("dr4", "gaiadr4") else "gaiadr3.gaia_source"
    lvl = int(np.clip(healpix_level, 0, 8))
    hp_idx = radec_to_gaia_healpix_nested(ra_deg, dec_deg, level=lvl)
    span = 1 << (59 - 2 * lvl)
    sid_lo = hp_idx * span
    sid_hi = (hp_idx + 1) * span - 1

    cols = (
        "source_id, ra, dec, parallax, parallax_error, "
        "pmra, pmdec, pmra_error, pmdec_error, "
        "ruwe, radial_velocity, radial_velocity_error, "
        "non_single_star, phot_g_mean_mag, bp_rp"
    )
    adql = (
        f"SELECT TOP {int(top_rows)} {cols} FROM {table_name} "
        f"WHERE source_id BETWEEN {sid_lo} AND {sid_hi} "
        f"AND parallax >= {float(min_parallax_mas):.4f} "
        f"AND parallax_over_error >= {float(min_poe):.2f}"
    )

    last_err = ""
    used_mirror = ""
    csv_text = ""

    def _query_one(cli: object, url: str) -> str:
        resp = cli.post(  # type: ignore[attr-defined]
            url,
            data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
        )
        if resp.status_code == 200 and "source_id" in resp.text[:160]:
            return str(resp.text)
        raise RuntimeError(f"HTTP {resp.status_code} from {url}")

    if http_client is not None:
        for m_url in mirrors:
            try:
                csv_text = _query_one(http_client, m_url)
                used_mirror = m_url
                break
            except Exception as exc:
                last_err = f"{m_url}: {exc}"
    else:
        with httpx.Client(timeout=float(timeout_s), follow_redirects=True) as cli:
            for m_url in mirrors:
                try:
                    csv_text = _query_one(cli, m_url)
                    used_mirror = m_url
                    break
                except Exception as exc:
                    last_err = f"{m_url}: {exc}"

    if not csv_text:
        raise RuntimeError(f"All Gaia TAP mirrors failed ({last_err})")

    sids, ras, decs, plxs, plxes = [], [], [], [], []
    pmras, pmdecs, pmraes, pmdeces = [], [], [], []
    ruwes, rvs, rves, nsss, gmags, bprps = [], [], [], [], [], []

    rdr = csv.DictReader(io.StringIO(csv_text))
    for row in rdr:
        try:
            sids.append(int(row["source_id"]))
            ras.append(float(row["ra"]))
            decs.append(float(row["dec"]))
            plxs.append(float(row["parallax"]))
            plxes.append(float(row["parallax_error"]))
            pmras.append(float(row["pmra"]))
            pmdecs.append(float(row["pmdec"]))
            pmraes.append(float(row["pmra_error"]))
            pmdeces.append(float(row["pmdec_error"]))
            ruwes.append(float(row["ruwe"]) if row.get("ruwe") else 1.0)
            rvs.append(float(row["radial_velocity"]) if row.get("radial_velocity") else math.nan)
            rves.append(float(row["radial_velocity_error"]) if row.get("radial_velocity_error") else 0.0)
            nsss.append(int(float(row["non_single_star"])) if row.get("non_single_star") else 0)
            gmags.append(float(row["phot_g_mean_mag"]) if row.get("phot_g_mean_mag") else math.nan)
            bprps.append(float(row["bp_rp"]) if row.get("bp_rp") else math.nan)
        except (KeyError, ValueError, TypeError):
            continue

    star_dict = {
        "source_id": np.asarray(sids, dtype=np.int64),
        "ra": np.asarray(ras, dtype=np.float64),
        "dec": np.asarray(decs, dtype=np.float64),
        "parallax": np.asarray(plxs, dtype=np.float64),
        "parallax_error": np.asarray(plxes, dtype=np.float64),
        "pmra": np.asarray(pmras, dtype=np.float64),
        "pmdec": np.asarray(pmdecs, dtype=np.float64),
        "pmra_error": np.asarray(pmraes, dtype=np.float64),
        "pmdec_error": np.asarray(pmdeces, dtype=np.float64),
        "ruwe": np.asarray(ruwes, dtype=np.float64),
        "radial_velocity": np.asarray(rvs, dtype=np.float64),
        "radial_velocity_error": np.asarray(rves, dtype=np.float64),
        "non_single_star": np.asarray(nsss, dtype=np.int16),
        "phot_g_mean_mag": np.asarray(gmags, dtype=np.float64),
        "bp_rp": np.asarray(bprps, dtype=np.float64),
    }

    paired_cols = pair_isolated_wide_binaries_ckdtree(star_dict)
    if save_npz_path is not None and len(paired_cols.get("source_id1", ())) > 0:
        out_p = Path(save_npz_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_p, **paired_cols)

    pairs, cf = _ingest_two_star_column_mapping(paired_cols, max_ruwe=max_ruwe, min_poe=min_poe)
    meta: Dict[str, object] = {
        "mirror_url": used_mirror,
        "table_name": table_name,
        "healpix_level": lvl,
        "healpix_index": hp_idx,
        "n_stars_fetched": len(sids),
        "n_isolated_pairs": int(len(paired_cols.get("source_id1", ()))),
        "n_clean_pairs": len(pairs),
    }
    return pairs, cf, meta


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


_MC_TRIPLE_TEMPLATE: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None


def _get_mc_kepler_triple_template() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic `N = 4096` Opik scale-invariant Keplerian + Clarke (2020) / Pittordis & Sutherland (2023)
    unresolved hierarchical-triple photocenter drift template (`< 240 KB` RAM).
    """
    global _MC_TRIPLE_TEMPLATE
    with _QUMOND_KERNEL_LOCK:
        if _MC_TRIPLE_TEMPLATE is not None:
            return _MC_TRIPLE_TEMPLATE
        rng = np.random.default_rng(20261105)
        a_au, dx_a, dx_d, _, va_dim, vd_dim, _ = _sample_scale_invariant_keplerian_orbits(rng, 4096, 500.0, 2000.0)
        s_2d = np.hypot(dx_a, dx_d)
        scale_v = np.sqrt(np.maximum(s_2d / np.maximum(a_au, 1.0), 0.0))
        vtx = (scale_v * va_dim).astype(np.float64)
        vty = (scale_v * vd_dim).astype(np.float64)
        u_ain = rng.uniform(0.0, 1.0, 4096).astype(np.float64)
        eta_pb = rng.uniform(0.08, 0.36, 4096) * np.sqrt(1.0 - rng.uniform(-1.0, 1.0, 4096) ** 2)
        phi_k = rng.uniform(0.0, 2.0 * np.pi, 4096)
        kx_dir = (eta_pb * np.cos(phi_k)).astype(np.float64)
        ky_dir = (eta_pb * np.sin(phi_k)).astype(np.float64)
        u_mix = rng.uniform(0.0, 1.0, 4096).astype(np.float64)
        _MC_TRIPLE_TEMPLATE = (vtx, vty, u_ain, kx_dir, ky_dir, u_mix)
        return _MC_TRIPLE_TEMPLATE


def calibrate_inner_triple_forward_bias(
    cal_all_pairs: List[WideBinaryPair],
    wide_group: List[WideBinaryPair],
    max_bound_v_tilde: float = 2.60,
) -> Dict[str, float]:
    """
    Zero-target-bias inner-anchored (`500 <= s <= 2,000 AU`) unresolved-triple forward mixture calibration:
      1. Strictly inside the Newtonian inner bin (`r < 2,000 AU << r_crit = 7,022 AU`), counts pairs
         exceeding the noise-broadened Newtonian escape ceiling `sqrt(2 + (2.5 * sigma_v_tilde)^2)`.
      2. Models `RUWE <= 1.2` surviving hierarchical inner companions via Opik's scale-invariant
         log-uniform semi-major axis `a_in in [6 AU, min(0.30 * s_outer, 0.75" * d_pc)]` (bounded below by
         `RUWE <= 1.2` photocenter curvature and above by Mardling–Aarseth stability & Gaia's `0.75"`
         angular source resolution + `cKDTree` 3rd-star isolation).
      3. Infers `f_triple_hat` solely from `[500, 2,000] AU` and propagates the frozen physical photocenter
         velocity drift (`s^{1/2}` dimensionless scaling) to `wide_group` under identical `v_tilde <= 2.60`
         selection and `10%` trimmed-mean inference.
    """
    if len(cal_all_pairs) < 12 or len(wide_group) < 5:
        return {
            "f_triple_hat": 0.0,
            "f_triple_err": 0.0,
            "sigma_kick_kms": 0.0,
            "triple_bias_pct": 0.0,
            "triple_bias_err_pct": 0.0,
            "n_cal_super_keplerian": 0.0,
        }

    def _vt(p: WideBinaryPair) -> float:
        return float(p.v_tilde_persp_debiased if p.v_tilde_persp_debiased is not None else p.v_tilde_obs)

    def _vc_kms(p: WideBinaryPair) -> float:
        return 1.0e-3 * math.sqrt((G_SI * max(p.m_tot_msun, 0.08) * M_SUN_KG) / (max(p.sep_au, 1.0) * AU_M))

    cal_45 = [p for p in cal_all_pairs if 0.0 < _vt(p) <= 4.50]
    if len(cal_45) < 12:
        return {
            "f_triple_hat": 0.0,
            "f_triple_err": 0.0,
            "sigma_kick_kms": 0.0,
            "triple_bias_pct": 0.0,
            "triple_bias_err_pct": 0.0,
            "n_cal_super_keplerian": 0.0,
        }

    n_tail = 0
    for p in cal_45:
        vt_p = _vt(p)
        ceil_p = math.sqrt(2.0 + (2.5 * max(p.sigma_v_tilde_1d, 0.0)) ** 2)
        if vt_p > ceil_p:
            n_tail += 1

    if n_tail == 0:
        return {
            "f_triple_hat": 0.0,
            "f_triple_err": 0.0,
            "sigma_kick_kms": 0.0,
            "triple_bias_pct": 0.0,
            "triple_bias_err_pct": 0.0,
            "n_cal_super_keplerian": 0.0,
        }

    vc_cal_med = float(np.median([_vc_kms(p) for p in cal_45]))
    s_cal_med = float(np.median([p.sep_au for p in cal_45]))
    d_cal_med = float(np.median([p.dist_pc for p in cal_45]))
    m_cal_med = float(np.median([p.m_tot_msun for p in cal_45]))
    sig_cal_med = float(np.median([max(p.sigma_v_tilde_1d, 0.0) for p in cal_45]))

    vc_wide_med = float(np.median([_vc_kms(p) for p in wide_group]))
    s_wide_med = float(np.median([p.sep_au for p in wide_group]))
    d_wide_med = float(np.median([p.dist_pc for p in wide_group]))
    m_wide_med = float(np.median([p.m_tot_msun for p in wide_group]))

    vtx, vty, u_ain, kx_dir, ky_dir, u_mix = _get_mc_kepler_triple_template()
    vt_pure = np.hypot(vtx, vty)

    ain_min = 6.0
    ain_max_cal = max(min(0.30 * s_cal_med, 0.75 * d_cal_med), 24.0)
    ain_cal = ain_min * ((ain_max_cal / ain_min) ** u_ain)
    vorb_in_cal = 1.0e-3 * np.sqrt((G_SI * (0.65 * max(m_cal_med, 0.2)) * M_SUN_KG) / (ain_cal * AU_M))
    dv_kick_cal_kms = vorb_in_cal * np.hypot(kx_dir, ky_dir)
    sigma_kick_kms = float(np.median(dv_kick_cal_kms))

    vt_trip_cal = np.hypot(
        vtx + (vorb_in_cal * kx_dir) / max(vc_cal_med, 1.0e-4),
        vty + (vorb_in_cal * ky_dir) / max(vc_cal_med, 1.0e-4),
    )
    ceil_med = math.sqrt(2.0 + (2.5 * sig_cal_med) ** 2)
    p_tail = float(np.mean((vt_trip_cal > ceil_med) & (vt_trip_cal <= 4.50)))
    p_tail_eff = max(p_tail, 0.08)

    f_raw = (float(n_tail) / float(len(cal_45))) / p_tail_eff
    f_hat = float(np.clip(f_raw, 0.0, 0.48))
    f_err = float(min(math.sqrt(float(n_tail) + 0.5) / (float(len(cal_45)) * p_tail_eff), 0.25))

    ain_max_wide = max(min(0.30 * s_wide_med, 0.75 * d_wide_med), 24.0)
    ain_wide = ain_min * ((ain_max_wide / ain_min) ** u_ain)
    vorb_in_wide = 1.0e-3 * np.sqrt((G_SI * (0.65 * max(m_wide_med, 0.2)) * M_SUN_KG) / (ain_wide * AU_M))
    vt_trip_wide = np.hypot(
        vtx + (vorb_in_wide * kx_dir) / max(vc_wide_med, 1.0e-4),
        vty + (vorb_in_wide * ky_dir) / max(vc_wide_med, 1.0e-4),
    )

    def _eval_ratio_bias_pct(f_val: float) -> float:
        if f_val <= 1.0e-6:
            return 0.0
        is_trip = u_mix < f_val
        mc_cal = np.where(is_trip, vt_trip_cal, vt_pure)
        mc_wid = np.where(is_trip, vt_trip_wide, vt_pure)
        cal_b = mc_cal[(mc_cal > 0.0) & (mc_cal <= max_bound_v_tilde)]
        wid_b = mc_wid[(mc_wid > 0.0) & (mc_wid <= max_bound_v_tilde)]
        if len(cal_b) < 20 or len(wid_b) < 20:
            return 0.0
        m_c, _ = _trimmed_mean_and_winsor_se(cal_b.tolist())
        m_w, _ = _trimmed_mean_and_winsor_se(wid_b.tolist())
        return ((m_w / max(m_c, 1.0e-6)) - 1.0) * 100.0

    bias_pct = _eval_ratio_bias_pct(f_hat)
    bias_hi = _eval_ratio_bias_pct(min(f_hat + f_err, 0.50))
    bias_lo = _eval_ratio_bias_pct(max(f_hat - f_err, 0.0))
    bias_err_pct = 0.5 * abs(bias_hi - bias_lo)

    return {
        "f_triple_hat": f_hat,
        "f_triple_err": f_err,
        "sigma_kick_kms": sigma_kick_kms,
        "triple_bias_pct": bias_pct,
        "triple_bias_err_pct": bias_err_pct,
        "n_cal_super_keplerian": float(n_tail),
    }


ZENODO_PROTOCOL_V2_VERSION = "2.0.0-pre-dr4"


def build_zenodo_v2_preregistration_manifest() -> Dict[str, object]:
    """
    Builds the canonical deterministic `Protocol v2` Pre-Registration Amendment Manifest & SHA-256 hash
    for minting a Zenodo `New Version` ahead of ESA `Gaia DR4` (December 2, 2026), keeping `Protocol v1`
    (`10.5281/zenodo.23919217`, SHA-256 `fa1f5cc5...`) archived untouched for full transparency.
    """
    payload: Dict[str, object] = {
        "protocol_version": ZENODO_PROTOCOL_V2_VERSION,
        "parent_zenodo_doi": ZENODO_PREREGISTRATION_DOI,
        "parent_v1_manifest_sha256": ZENODO_MANIFEST_SHA256,
        "target_release": "ESA Gaia Data Release 4 (2026-12-02)",
        "zero_free_wide_bin_knobs": True,
        "fundamental_acceleration_floor_si": round(A0_PLANCK_2018_PRIMARY_SI, 18),
        "h0_planck_2018_kms_mpc": H0_PLANCK_PAPER_VI_KMS_MPC,
        "cuts_v2": PROTOCOL_V2_GAIA_CUTS,
        "photometric_mass_solver": {
            "method": "Pecaut-Mamajek monotonic PCHIP M_G spline + Gentile Fusillo WD locus + RGB giant gate",
            "mg_knots": [round(float(x), 3) for x in _PM_MG_KNOTS.tolist()],
            "msun_knots": [round(float(x), 4) for x in _PM_MSUN_KNOTS.tolist()],
            "wd_locus_rule": "M_G > 9.5 and M_G > 10.0 + 2.6 * bp_rp -> 0.60 M_sun",
            "giant_gate_rule": "M_G < 3.5 and bp_rp > 0.92 -> flagged GIANT (excluded in v2)",
        },
        "astrometric_debiasing": {
            "perspective_pm": "Exact 3D ICRS tangent-triad rotation at common pair distance d_kpc",
            "ricean_pm_noise": "Wardle-Kronberg 4th-order perpendicular variance subtraction sqrt(max(dv^2 - sigma_v_perp^2, (0.10*sigma_v_perp)^2))",
            "inner_triple_calibration": "Strictly inner [500, 2000] AU super-Keplerian tail (> sqrt(2 + (2.5*sigma)^2)) forward Opik+Rayleigh propagation",
        },
        "decision_thresholds": {
            "favors_tnap_weave_delta_chi2_gt": 16.0,
            "favors_lambda_cdm_delta_chi2_lt": -16.0,
        },
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return {
        "manifest": payload,
        "canonical_json": canonical_json,
        "sha256": digest,
    }


def evaluate_gaia_dr4_protocol_v2_verdict(
    pairs: List[WideBinaryPair],
    a0_si: float = A0_PLANCK_2018_PRIMARY_SI,
    cut_flow: Optional[GaiaCutFlowSummary] = None,
    max_sigma_v_tilde: float = 0.25,
    max_bound_v_tilde: float = 2.60,
    reject_giants: bool = True,
) -> Dict[str, object]:
    """
    Evaluates **Protocol `v2`** (Systematics-Hardened Pre-Dec-2 Amendment to Zenodo `10.5281/zenodo.23919217`)
    with **zero free wide-bin parameters**:
      1. Rejects evolved RGB/clump giants (`reject_giants=True`) and enforces `sigma_v_tilde_1d <= 0.25`
         so distant noisy halo pairs (`1.4–2.2 kpc`) cannot inflate Test C via positive-definite PM noise.
      2. Uses 3D spherical-perspective-corrected + Wardle–Kronberg Ricean-debiased `v_tilde_persp_debiased`.
      3. Enforces scale-invariant bound selection `0 < v_tilde <= 2.60` identically across `[500, 2,000] AU`
         and `[10,000, 22,000] AU` (preventing missing-RV unbound flyby asymmetry at wide separations).
      4. Calibrates surviving `RUWE <= 1.2` inner hierarchical triples strictly inside Newtonian `[500, 2,000] AU`
         and subtracts the frozen forward-propagated `s^{1/2}` ratio bias.
      5. Evaluates 3D QUMOND per-pair `(s_AU, M_tot, R_gal, z_gal, theta_GC)` predictions across all 3 tests!
    """
    a0_safe = max(float(a0_si), 1.0e-20)
    curves = compute_wide_binary_efe_curves(_default_tnap_nu_fn, a0_safe, m_tot_msun=1.6)
    default_pred_a = float(curves["boost_disk_15k_pct"])
    default_pred_b = float(curves["compass_split_15k_pct"])
    default_pred_c = float(curves["boost_lower_halo_15k_pct"]) - float(curves["boost_disk_15k_pct"])
    v2_man = build_zenodo_v2_preregistration_manifest()
    v2_sha = str(v2_man["sha256"])

    def _vt(p: WideBinaryPair) -> float:
        return float(p.v_tilde_persp_debiased if p.v_tilde_persp_debiased is not None else p.v_tilde_obs)

    snr_clean: List[WideBinaryPair] = []
    n_rejected_giants = 0
    n_rejected_snr = 0
    for p in pairs:
        if reject_giants and "GIANT" in p.comp_class:
            n_rejected_giants += 1
            continue
        if p.sigma_v_tilde_1d > max_sigma_v_tilde:
            n_rejected_snr += 1
            continue
        vt_val = _vt(p)
        if math.isfinite(vt_val) and vt_val > 0.0:
            snr_clean.append(p)

    cal_all = [p for p in snr_clean if 500.0 <= p.sep_au <= 2000.0]
    cal_pairs = [p for p in cal_all if _vt(p) <= max_bound_v_tilde]
    wide_pairs = [p for p in snr_clean if 10000.0 <= p.sep_au <= 22000.0 and _vt(p) <= max_bound_v_tilde]

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
            "n_v2_snr_clean": len(snr_clean),
            "n_rejected_giants": n_rejected_giants,
            "n_rejected_v2_pm_snr": n_rejected_snr,
        }

    if len(cal_pairs) < 5 or len(wide_pairs) < 5:
        return {
            "protocol": "v2",
            "zenodo_doi": ZENODO_PREREGISTRATION_DOI,
            "zenodo_v2_sha256": v2_sha,
            "n_clean": len(snr_clean),
            "n_calibration": len(cal_pairs),
            "n_wide_total": len(wide_pairs),
            "n_local_wide": 0,
            "n_compass_aligned": 0,
            "n_compass_perp": 0,
            "n_env_halo_outer": 0,
            "n_env_inner_solar": 0,
            "v_tilde_cal": 0.0,
            "f_triple_inner_hat": 0.0,
            "sigma_kick_triple_kms": 0.0,
            "triple_forward_bias_local_pct": 0.0,
            "raw_debiased_local_boost_pct": 0.0,
            "local_backyard_15k_boost_pct": 0.0,
            "local_backyard_15k_err_pct": 0.0,
            "tnap_pred_local_15k_boost_pct": default_pred_a,
            "compass_aligned_boost_pct": 0.0,
            "compass_perp_boost_pct": 0.0,
            "compass_split_pct": 0.0,
            "compass_split_err_pct": 0.0,
            "tnap_pred_compass_split_pct": default_pred_b,
            "env_halo_outer_boost_pct": 0.0,
            "env_inner_solar_boost_pct": 0.0,
            "env_step_halo_minus_inner_pct": 0.0,
            "env_step_err_pct": 0.0,
            "tnap_pred_env_step_pct": default_pred_c,
            "chi2_tnap": 0.0,
            "chi2_lambda_cdm": 0.0,
            "delta_chi2": 0.0,
            "log10_bayes_factor": 0.0,
            "verdict": "INCONCLUSIVE",
            "cut_flow": cut_dict,
        }

    v_cal, se_cal = _trimmed_mean_and_winsor_se([_vt(p) for p in cal_pairs])
    v_cal = max(v_cal, 1.0e-6)

    local_wide = [p for p in wide_pairs if p.dist_pc <= 300.0]
    local_aligned = [p for p in local_wide if p.theta_gc_deg <= 38.0]
    local_perp = [p for p in local_wide if p.theta_gc_deg >= 52.0]

    halo_outer_wide = [p for p in wide_pairs if abs(p.z_gal_kpc) >= 1.2 or p.r_gal_kpc > 9.0]
    inner_solar_wide = [p for p in wide_pairs if abs(p.z_gal_kpc) < 0.4 and p.r_gal_kpc <= 8.5]

    chi2_tnap = 0.0
    chi2_cdm = 0.0

    def _pred_mean_pct(grp: List[WideBinaryPair]) -> float:
        arr = _eval_3d_qumond_boost_batch(
            np.asarray([p.sep_au for p in grp], dtype=np.float64),
            np.asarray([p.m_tot_msun for p in grp], dtype=np.float64),
            np.asarray([p.r_gal_kpc for p in grp], dtype=np.float64),
            np.asarray([p.z_gal_kpc for p in grp], dtype=np.float64),
            np.asarray([p.theta_gc_deg for p in grp], dtype=np.float64),
            a0_safe,
            _default_tnap_nu_fn,
        )
        return float(np.mean(arr))

    # Test A: Local Backyard (d <= 300 pc) 10k..22k AU Boost (with inner-anchored triple forward calibration)
    if len(local_wide) >= 5:
        v_loc, se_loc = _trimmed_mean_and_winsor_se([_vt(p) for p in local_wide])
        ratio_a = v_loc / v_cal
        raw_loc_pct = (ratio_a - 1.0) * 100.0
        stat_err_a = 100.0 * ratio_a * math.hypot(se_loc / max(v_loc, 1.0e-6), se_cal / v_cal)

        trip_cal = calibrate_inner_triple_forward_bias(cal_all, local_wide, max_bound_v_tilde=max_bound_v_tilde)
        f_trip_hat = float(trip_cal["f_triple_hat"])
        sig_kick_kms = float(trip_cal["sigma_kick_kms"])
        trip_bias_loc = float(trip_cal["triple_bias_pct"])
        trip_err_loc = float(trip_cal["triple_bias_err_pct"])

        test_a_pct = raw_loc_pct - trip_bias_loc
        err_a_pct = max(math.hypot(stat_err_a, trip_err_loc), 0.25)
        pred_tnap_a = _pred_mean_pct(local_wide)
        chi2_tnap += ((test_a_pct - pred_tnap_a) / err_a_pct) ** 2
        chi2_cdm += ((test_a_pct - 0.0) / err_a_pct) ** 2
    else:
        raw_loc_pct = 0.0
        f_trip_hat = 0.0
        sig_kick_kms = 0.0
        trip_bias_loc = 0.0
        test_a_pct = 0.0
        err_a_pct = 99.0
        pred_tnap_a = default_pred_a

    # Test B: Galactic Center Compass Anisotropy (theta_GC <= 38 deg minus >= 52 deg)
    if len(local_aligned) >= 5 and len(local_perp) >= 5:
        v_al, se_al = _trimmed_mean_and_winsor_se([_vt(p) for p in local_aligned])
        v_pe, se_pe = _trimmed_mean_and_winsor_se([_vt(p) for p in local_perp])
        al_boost_pct = ((v_al / v_cal) - 1.0) * 100.0 - trip_bias_loc
        pe_boost_pct = ((v_pe / v_cal) - 1.0) * 100.0 - trip_bias_loc
        test_b_pct = al_boost_pct - pe_boost_pct
        err_b_pct = max(100.0 * math.hypot(se_al, se_pe) / v_cal, 0.25)
        pred_b_eff = _pred_mean_pct(local_aligned) - _pred_mean_pct(local_perp)
        chi2_tnap += ((test_b_pct - pred_b_eff) / err_b_pct) ** 2
        chi2_cdm += ((test_b_pct - 0.0) / err_b_pct) ** 2
    else:
        al_boost_pct = 0.0
        pe_boost_pct = 0.0
        test_b_pct = 0.0
        err_b_pct = 99.0
        pred_b_eff = default_pred_b

    # Test C: Galactic External-Field Environment Step (Halo/Outer vs Inner/Solar Disk)
    if len(halo_outer_wide) >= 5 and len(inner_solar_wide) >= 5:
        v_hal, se_hal = _trimmed_mean_and_winsor_se([_vt(p) for p in halo_outer_wide])
        v_inn, se_inn = _trimmed_mean_and_winsor_se([_vt(p) for p in inner_solar_wide])
        trip_hal = calibrate_inner_triple_forward_bias(cal_all, halo_outer_wide, max_bound_v_tilde=max_bound_v_tilde)
        trip_inn = calibrate_inner_triple_forward_bias(cal_all, inner_solar_wide, max_bound_v_tilde=max_bound_v_tilde)
        hal_boost_pct = ((v_hal / v_cal) - 1.0) * 100.0 - float(trip_hal["triple_bias_pct"])
        inn_boost_pct = ((v_inn / v_cal) - 1.0) * 100.0 - float(trip_inn["triple_bias_pct"])
        test_c_pct = hal_boost_pct - inn_boost_pct
        diff_trip_err = abs(float(trip_hal["triple_bias_err_pct"]) - float(trip_inn["triple_bias_err_pct"]))
        err_c_pct = max(math.hypot(100.0 * math.hypot(se_hal, se_inn) / v_cal, diff_trip_err), 0.25)
        pred_tnap_c = _pred_mean_pct(halo_outer_wide) - _pred_mean_pct(inner_solar_wide)
        chi2_tnap += ((test_c_pct - pred_tnap_c) / err_c_pct) ** 2
        chi2_cdm += ((test_c_pct - 0.0) / err_c_pct) ** 2
    else:
        hal_boost_pct = 0.0
        inn_boost_pct = 0.0
        test_c_pct = 0.0
        err_c_pct = 99.0
        pred_tnap_c = default_pred_c

    delta_chi2 = float(chi2_cdm - chi2_tnap)
    log10_bf = delta_chi2 / (2.0 * math.log(10.0))
    if delta_chi2 > 16.0:
        verdict = "FAVORS_TNAP_WEAVE"
    elif delta_chi2 < -16.0:
        verdict = "FAVORS_LAMBDA_CDM"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "protocol": "v2",
        "zenodo_doi": ZENODO_PREREGISTRATION_DOI,
        "zenodo_v2_sha256": v2_sha,
        "n_clean": len(snr_clean),
        "n_calibration": len(cal_pairs),
        "n_wide_total": len(wide_pairs),
        "n_local_wide": len(local_wide),
        "n_compass_aligned": len(local_aligned),
        "n_compass_perp": len(local_perp),
        "n_env_halo_outer": len(halo_outer_wide),
        "n_env_inner_solar": len(inner_solar_wide),
        "v_tilde_cal": v_cal,
        "f_triple_inner_hat": f_trip_hat,
        "sigma_kick_triple_kms": sig_kick_kms,
        "triple_forward_bias_local_pct": trip_bias_loc,
        "raw_debiased_local_boost_pct": raw_loc_pct,
        "local_backyard_15k_boost_pct": test_a_pct,
        "local_backyard_15k_err_pct": err_a_pct,
        "tnap_pred_local_15k_boost_pct": pred_tnap_a,
        "compass_aligned_boost_pct": al_boost_pct,
        "compass_perp_boost_pct": pe_boost_pct,
        "compass_split_pct": test_b_pct,
        "compass_split_err_pct": err_b_pct,
        "tnap_pred_compass_split_pct": pred_b_eff,
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
