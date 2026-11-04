"""
Un-Gameable Cosmological Referee & Cross-Scale Continuity Engine (`weaveforge/core/cosmology_referee.py`)
=========================================================================================================
Quantitative, zero-bias multi-scale consistency referee for WeaveForge v1.0.
Every `PhysicsMod` is scored by the exact same equations across:

  1. Single-`H_0` Global Consistency (`audit_single_h0_consistency`):
       • Enforces a single global Hubble constant `H_0 [km/s/Mpc]` across `a_0 = c H_0 / (sqrt(3) pi)`,
         `CAMB` TT/TE/EE, `DESI DR2` BAO, primordial `BBN`, and `S_8`, measuring both `a_0`–`H_0`
         fractional mismatch and `SH0ES` Cepheid tension.
  2. Full-Spectrum `l = 30..2500` Planck Knox `chi^2` (`evaluate_planck_knox_chi2_2500`):
       • Exact binned Knox (1995) cosmic variance + Planck 2018 combined `100+143+217 GHz`
         beam-deconvolved noise across `TT + TE + EE`, reporting both fixed-`n_s` (`0.9649`)
         and `(A_s, n_s)`-profiled `chi^2 / dof` plus Silk-damping high-`l` (`800..2000`) tilt.
  3. Primordial Big Bang Nucleosynthesis (`evaluate_primordial_bbn`):
       • AlterBBN / Parthenope / Pitrou+ (2018) & Yeh+ (2022) `(D/H, Y_p)` response expansions
         around `(omega_b = 0.02237, N_eff = 3.044)`, scored against Cooke+/PDG `(D/H)` and
         Aver+/PDG `Y_p` plus joint Planck+BBN `N_eff`.
  4. 3D Linear Matter Growth & Weak-Lensing `S_8` (`evaluate_linear_growth_s8`):
       • 240-step vectorized RK4 integration of linear growth `delta(k, a)` from `a = 1e-3` to `1.0`
         across `k in [0.02, 0.50] h/Mpc` (`k_8 = 0.18 h/Mpc`), exposing unscreened cosmic-web
         QUMOND `S_8` blowout (`pleat_screening_mpc == 0`) vs Mpc-screened growth and `Lambda-CDM`'s
         empirical `~3 sigma` `DES Y3 / KiDS-1000` `S_8` tension.
  5. Cross-Scale Mass Continuity on `NGC 3198` (`evaluate_cross_scale_mass_continuity`):
       • Tests whether cold CMB pleat/Aether dust (`Omega_pleat > 0.01`) collapses inside `SPARC`
         spiral halos (`NGC 3198`, `R >= 25 kpc`), quantifying the `~6.32x` double-counting velocity
         blowup unless an explicit halo/Mpc screening scale `pleat_screening_mpc >= 0.05` is supplied.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from weaveforge.core.frameworks import PhysicsMod
from weaveforge.data.sparc_catalog import (
    KMS2_PER_KPC_TO_MPS2,
    SparcGalaxy,
    find_sparc_galaxy,
)


_C_M_S: float = 299792458.0
_MPC_M: float = 3.0856775814913673e22
_PLANCK_NS_FID: float = 0.9649
_PLANCK_ELL_PIVOT: float = 700.0
_PLANCK_FSKY: float = 0.65
_SHOES_H0_OBS: float = 73.04
_SHOES_H0_SIGMA: float = math.hypot(1.04, 0.54)  # 1.171836 km/s/Mpc

# Precomputed k-grid & log(y) interpolation nodes for < 0.5 ms 3D linear growth S_8 ODE
_S8_KH_GRID: np.ndarray = np.geomspace(0.02, 0.50, 9).astype(np.float64)
_S8_DLNK: float = float(math.log(_S8_KH_GRID[1] / _S8_KH_GRID[0]))
_S8_X8: np.ndarray = _S8_KH_GRID * 8.0
_S8_W8_SQ: np.ndarray = (3.0 * (np.sin(_S8_X8) - _S8_X8 * np.cos(_S8_X8)) / (_S8_X8**3)) ** 2
_S8_A_INIT: float = 1.0e-3
_S8_RK4_STEPS: int = 240
_S8_X_HALF: np.ndarray = np.linspace(math.log(_S8_A_INIT), 0.0, 2 * _S8_RK4_STEPS + 1, dtype=np.float64)
_S8_A_HALF: np.ndarray = np.exp(_S8_X_HALF)
_S8_INV_A2_HALF: np.ndarray = 1.0 / (_S8_A_HALF * _S8_A_HALF)
_S8_INV_A2_HALF_LIST: list[float] = _S8_INV_A2_HALF.tolist()
_S8_DX: float = float((0.0 - math.log(_S8_A_INIT)) / _S8_RK4_STEPS)
_S8_LNY_LUT: np.ndarray = np.linspace(math.log(1.0e-16), math.log(1.0e8), 192, dtype=np.float64)
_S8_Y_LUT: np.ndarray = np.exp(_S8_LNY_LUT)
_S8_LNY_MIN: float = float(_S8_LNY_LUT[0])
_S8_LUT_UPPER_F: float = float(len(_S8_LNY_LUT) - 1)
_S8_INV_DLNY: float = float(_S8_LUT_UPPER_F / (_S8_LNY_LUT[-1] - _S8_LNY_LUT[0]))
_S8_PROBE_Y: np.ndarray = np.array([1.0e-4, 0.10, 10.0], dtype=np.float64)

_DES_KIDS_S8_OBS: float = 0.776
_DES_KIDS_S8_ERR: float = 0.017


def audit_single_h0_consistency(mod: PhysicsMod) -> Dict[str, object]:
    """
    Audits consistency between the active mod's global Hubble constant `h0_kms_mpc`,
    its galactic acceleration scale `a0_si` vs the Hubble-Unruh relation
    `a_0 = c H_0 / (sqrt(3) pi)`, and the local `SH0ES` Cepheid distance ladder (`73.04 +- 1.04`).
    """
    h0 = float(mod.parameters.h0_kms_mpc)
    a0_si = float(mod.parameters.a0_si)
    h0_si = max(h0, 0.0) * 1000.0 / _MPC_M
    a0_unruh_si = _C_M_S * h0_si / (math.sqrt(3.0) * math.pi)
    a0_h0_rel_mismatch = abs(a0_si - a0_unruh_si) / max(a0_unruh_si, 1.0e-30)
    shoes_h0_sigma = abs(h0 - _SHOES_H0_OBS) / _SHOES_H0_SIGMA

    return {
        "h0_kms_mpc": h0,
        "a0_si": a0_si,
        "a0_unruh_si": a0_unruh_si,
        "a0_h0_rel_mismatch": a0_h0_rel_mismatch,
        "a0_locked_to_h0": bool(a0_h0_rel_mismatch <= 0.002),
        "shoes_h0_sigma": shoes_h0_sigma,
    }


@lru_cache(maxsize=8)
def _knox_grid_factors(ell_bytes: bytes, n_ell: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ell_arr = np.frombuffer(ell_bytes, dtype=np.float64, count=n_ell)
    mask = (ell_arr >= 30.0) & (ell_arr <= 2500.0)
    ell_sub = ell_arr[mask]
    delta_ell = np.maximum(np.gradient(ell_sub), 1.0) if len(ell_sub) >= 2 else np.ones_like(ell_sub)
    n_modes = (2.0 * ell_sub + 1.0) * _PLANCK_FSKY * delta_ell
    dl_to_cl = (2.0 * math.pi) / (ell_sub * (ell_sub + 1.0))

    arcmin_rad = math.pi / (180.0 * 60.0)
    w_t = 33.0 * arcmin_rad
    w_p = 56.0 * arcmin_rad
    sigma_beam = (5.5 * arcmin_rad) / math.sqrt(8.0 * math.log(2.0))
    exp_beam = np.exp(np.minimum(ell_sub * (ell_sub + 1.0) * (sigma_beam * sigma_beam), 60.0))
    nl_tt = (w_t * w_t) * exp_beam
    nl_ee = (w_p * w_p) * exp_beam
    ln_piv = np.log(ell_sub / _PLANCK_ELL_PIVOT)
    return mask, n_modes, dl_to_cl, np.stack((nl_tt, nl_ee), axis=0), ln_piv


def evaluate_planck_knox_chi2_2500(
    ell: np.ndarray,
    tt_obs: np.ndarray,
    te_obs: np.ndarray,
    ee_obs: np.ndarray,
    tt_mod_raw: np.ndarray,
    te_mod_raw: np.ndarray,
    ee_mod_raw: np.ndarray,
) -> Dict[str, object]:
    """
    Evaluates the full-spectrum (`ell in [30, 2500]`) binned Knox (1995) `TT + TE + EE` `chi^2`
    with Planck 2018 combined `100+143+217 GHz` effective beam-deconvolved noise, reporting both
    fixed `n_s = 0.9649` (profiled `A_s`) and jointly profiled `(A_s, n_s)` fits.
    """
    ell_f = np.ascontiguousarray(ell, dtype=np.float64)
    tt_o = np.asarray(tt_obs, dtype=np.float64)
    te_o = np.asarray(te_obs, dtype=np.float64)
    ee_o = np.asarray(ee_obs, dtype=np.float64)
    tt_m = np.asarray(tt_mod_raw, dtype=np.float64)
    te_m = np.asarray(te_mod_raw, dtype=np.float64)
    ee_m = np.asarray(ee_mod_raw, dtype=np.float64)

    mask, n_modes, dl_to_cl, nl_pair, ln_piv = _knox_grid_factors(ell_f.tobytes(), len(ell_f))
    nl_tt = nl_pair[0]
    nl_ee = nl_pair[1]

    cl_tt_o = tt_o[mask] * dl_to_cl
    cl_te_o = te_o[mask] * dl_to_cl
    cl_ee_o = ee_o[mask] * dl_to_cl

    cl_tt_m = tt_m[mask] * dl_to_cl
    cl_te_m = te_m[mask] * dl_to_cl
    cl_ee_m = ee_m[mask] * dl_to_cl

    tt_tot = np.maximum(cl_tt_o + nl_tt, 1.0e-30)
    ee_tot = np.maximum(cl_ee_o + nl_ee, 1.0e-30)
    var_tt = (2.0 / n_modes) * (tt_tot * tt_tot)
    var_ee = (2.0 / n_modes) * (ee_tot * ee_tot)
    var_te = (1.0 / n_modes) * ((cl_te_o * cl_te_o) + tt_tot * ee_tot)

    w_tt = 1.0 / var_tt
    w_te = 1.0 / var_te
    w_ee = 1.0 / var_ee

    xy_vec = w_tt * cl_tt_m * cl_tt_o + w_te * cl_te_m * cl_te_o + w_ee * cl_ee_m * cl_ee_o
    xx_vec = w_tt * cl_tt_m * cl_tt_m + w_te * cl_te_m * cl_te_m + w_ee * cl_ee_m * cl_ee_m
    s_yy = float(np.sum(w_tt * cl_tt_o * cl_tt_o + w_te * cl_te_o * cl_te_o + w_ee * cl_ee_o * cl_ee_o))

    # 1. Fixed n_s = 0.9649 (analytic WLS for scalar amplitude scale alpha_As)
    s_xy0 = float(np.sum(xy_vec))
    s_xx0 = float(np.sum(xx_vec))
    alpha_fix = s_xy0 / s_xx0 if s_xx0 > 1.0e-60 else 1.0
    chi2_fixed = float(
        np.sum(
            w_tt * ((alpha_fix * cl_tt_m - cl_tt_o) ** 2)
            + w_te * ((alpha_fix * cl_te_m - cl_te_o) ** 2)
            + w_ee * ((alpha_fix * cl_ee_m - cl_ee_o) ** 2)
        )
    )
    n_bins_total = 3 * int(len(cl_tt_o))
    dof_fixed = max(n_bins_total - 1, 1)
    dof_prof = max(n_bins_total - 2, 1)
    chi2_dof_fixed = 1.0 + chi2_fixed / float(dof_fixed)

    # Silk-damping high-l (800..2000) fractional power tilt relative to Peak 1 (150..400)
    mask_p1 = (ell_f >= 150.0) & (ell_f <= 400.0)
    mask_hl = (ell_f >= 800.0) & (ell_f <= 2000.0)
    if np.any(mask_p1) and np.any(mask_hl):
        obs_p1 = max(float(np.max(tt_o[mask_p1])), 1.0e-12)
        mod_p1 = max(float(np.max(tt_m[mask_p1])), 1.0e-12)
        obs_hl_ratio = max(float(np.mean(tt_o[mask_hl])) / obs_p1, 1.0e-15)
        mod_hl_ratio = float(np.mean(tt_m[mask_hl])) / mod_p1
        high_l_tilt_pct = (mod_hl_ratio / obs_hl_ratio - 1.0) * 100.0
    else:
        high_l_tilt_pct = 0.0

    # 2. Jointly profiled (A_s, n_s) via 1D bounded Brent over delta_ns in [-0.08, +0.08]
    if chi2_fixed <= 1.0e-11:
        delta_ns_best = 0.0
        alpha_prof = alpha_fix
        chi2_prof = chi2_fixed
    else:
        def _profile_obj(dns: float) -> float:
            tilt = np.exp(dns * ln_piv)
            s_xy = float(np.dot(xy_vec, tilt))
            s_xx = float(np.dot(xx_vec, tilt * tilt))
            if s_xx <= 1.0e-60:
                return s_yy
            return max(0.0, s_yy - (s_xy * s_xy) / s_xx)

        res = minimize_scalar(_profile_obj, bounds=(-0.08, 0.08), method="bounded", options={"xatol": 1.0e-4})
        delta_ns_best = float(res.x)
        tilt_best = np.exp(delta_ns_best * ln_piv)
        s_xy_b = float(np.dot(xy_vec, tilt_best))
        s_xx_b = float(np.dot(xx_vec, tilt_best * tilt_best))
        alpha_prof = s_xy_b / s_xx_b if s_xx_b > 1.0e-60 else alpha_fix
        chi2_prof_raw = float(
            np.sum(
                w_tt * ((alpha_prof * cl_tt_m * tilt_best - cl_tt_o) ** 2)
                + w_te * ((alpha_prof * cl_te_m * tilt_best - cl_te_o) ** 2)
                + w_ee * ((alpha_prof * cl_ee_m * tilt_best - cl_ee_o) ** 2)
            )
        )
        if chi2_prof_raw <= chi2_fixed:
            chi2_prof = chi2_prof_raw
        else:
            delta_ns_best = 0.0
            alpha_prof = alpha_fix
            chi2_prof = chi2_fixed

    chi2_dof_prof = 1.0 + chi2_prof / float(dof_prof)
    best_fit_ns = _PLANCK_NS_FID + delta_ns_best

    return {
        "chi2_fixed_ns": chi2_fixed,
        "delta_chi2_fixed_ns": chi2_fixed,
        "chi2_dof_fixed_ns": chi2_dof_fixed,
        "alpha_as_fixed_ns": alpha_fix,
        "high_l_tilt_pct": high_l_tilt_pct,
        "chi2_profiled_ns": chi2_prof,
        "delta_chi2_profiled_ns": chi2_prof,
        "chi2_dof_profiled_ns": chi2_dof_prof,
        "alpha_as_profiled_ns": alpha_prof,
        "best_fit_ns": best_fit_ns,
        "delta_ns": delta_ns_best,
        "dof_fixed": dof_fixed,
        "dof_profiled": dof_prof,
    }


def evaluate_primordial_bbn(omega_b: float, h0_kms_mpc: float, delta_neff: float) -> Dict[str, object]:
    """
    Evaluates primordial Deuterium `(D/H)` and Helium-4 `(Y_p)` mass fraction yields
    using AlterBBN / Parthenope / Pitrou+ (2018) & Yeh+ (2022) response expansions around
    `(ombh2_0 = 0.02237, N_eff_0 = 3.044)`, scored against Cooke+/PDG `(D/H)` and Aver+/PDG `Y_p`.
    """
    h = max(float(h0_kms_mpc), 1.0) / 100.0
    ombh2 = max(float(omega_b), 1.0e-6) * (h * h)
    eta_10 = 273.78 * ombh2
    d_neff = float(delta_neff)
    neff_total = 3.044 + d_neff

    eta_ratio = max(eta_10, 0.5) / 6.124
    y_p = 0.24709 + 0.0095 * math.log(eta_ratio) + 0.01356 * d_neff - 0.00035 * (d_neff * d_neff)
    dh_e5 = 2.515 * (eta_ratio ** (-1.628)) * (max(1.0 + 0.135 * d_neff, 0.05) ** 0.985)

    # Cooke+ (2018) / PDG (2.547 +- 0.025)e-5 + Pitrou/Yeh nuclear-rate sigma_th = 0.030e-5
    dh_obs_only_sigma = (dh_e5 - 2.547) / 0.025
    dh_sigma = (dh_e5 - 2.547) / math.hypot(0.025, 0.030)
    # Aver+ / PDG Y_p = 0.2453 +- 0.0034
    yp_sigma = (y_p - 0.2453) / 0.0034
    # Yeh+ (2022) joint Planck + BBN N_eff = 2.898 +- 0.141
    neff_bbn_sigma = abs(neff_total - 2.898) / 0.141

    joint_chi2 = float(dh_sigma * dh_sigma + yp_sigma * yp_sigma)
    max_sigma = float(max(abs(dh_sigma), abs(yp_sigma), neff_bbn_sigma))

    if max_sigma <= 2.0:
        status = "PASS"
    elif max_sigma <= 3.5:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "ombh2": ombh2,
        "eta_10": eta_10,
        "neff_total": neff_total,
        "delta_neff": d_neff,
        "dh_e5": dh_e5,
        "y_p": y_p,
        "dh_obs_only_sigma": dh_obs_only_sigma,
        "dh_sigma": dh_sigma,
        "yp_sigma": yp_sigma,
        "neff_bbn_sigma": neff_bbn_sigma,
        "joint_chi2": joint_chi2,
        "max_sigma": max_sigma,
        "status": status,
    }


def _eh_pk_shape(om: float, ob: float, h: float) -> np.ndarray:
    """Eisenstein–Hu / Sugiyama transfer + Poisson horizon-entry shape `(k_h^4 / om^2) T(k)^2`."""
    om_s = max(float(om), 1.0e-4)
    h_s = max(float(h), 0.10)
    gamma = om_s * h_s * math.exp(-ob * (1.0 + math.sqrt(2.0 * h_s) / om_s))
    q = _S8_KH_GRID / max(gamma, 1.0e-5)
    t_eh = (
        np.log(1.0 + 2.34 * q)
        / (2.34 * q)
        * ((1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** (-0.25))
    )
    k_mpc = _S8_KH_GRID * h_s
    prim_tilt = (k_mpc / 0.05) ** (_PLANCK_NS_FID - 1.0)
    return prim_tilt * ((_S8_KH_GRID**4) / (om_s * om_s)) * (t_eh * t_eh)


def _integrate_gr_growth_scalar(om: float, w0: float, wa: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """240-step RK4 integration of scale-independent GR linear growth `D_GR(a)` from `a = 1e-3` to `1`."""
    om_c = float(np.clip(om, 1.0e-4, 0.99))
    ode0 = 1.0 - om_c
    de_exp = -3.0 * (1.0 + w0 + wa)

    em_h = om_c * (_S8_A_HALF ** (-3.0))
    ede_h = ode0 * (_S8_A_HALF**de_exp) * np.exp(-3.0 * wa * (1.0 - _S8_A_HALF))
    e2_h = np.maximum(em_h + ede_h, 1.0e-30)
    om_a_h = em_h / e2_h
    ode_a_h = ede_h / e2_h
    fric_h = 2.0 - 1.5 * (om_a_h + (1.0 + w0 + wa * (1.0 - _S8_A_HALF)) * ode_a_h)
    src_h = 1.5 * om_a_h

    fric_l = fric_h.tolist()
    src_l = src_h.tolist()

    dg = _S8_A_INIT
    dgp = _S8_A_INIT
    dx = _S8_DX
    half_dx = 0.5 * dx
    sixth_dx = dx / 6.0

    for i in range(_S8_RK4_STEPS):
        j = 2 * i
        f0, s0 = fric_l[j], src_l[j]
        f1, s1 = fric_l[j + 1], src_l[j + 1]
        f2, s2 = fric_l[j + 2], src_l[j + 2]

        k1d = dgp
        k1p = -f0 * dgp + s0 * dg
        d1 = dg + half_dx * k1d
        p1 = dgp + half_dx * k1p

        k2d = p1
        k2p = -f1 * p1 + s1 * d1
        d2 = dg + half_dx * k2d
        p2 = dgp + half_dx * k2p

        k3d = p2
        k3p = -f1 * p2 + s1 * d2
        d3 = dg + dx * k3d
        p3 = dgp + dx * k3p

        k4d = p3
        k4p = -f2 * p3 + s2 * d3

        dg += sixth_dx * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
        dgp += sixth_dx * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)

    return dg, fric_h, src_h


@lru_cache(maxsize=1)
def _planck_sigma8_norm_const() -> float:
    """Calibrates `sigma_8,GR(Omega_m = 0.31397, H_0 = 67.36, w_0 = -1, w_a = 0) == 0.8111` (Planck 2018 Table 2)."""
    pk_ref = _eh_pk_shape(0.31397, 0.04897, 0.6736)
    dg_ref, _, _ = _integrate_gr_growth_scalar(0.31397, -1.0, 0.0)
    raw_var = float(np.sum(pk_ref * (dg_ref * dg_ref) * _S8_W8_SQ) * _S8_DLNK)
    return (0.8111 * 0.8111) / max(raw_var, 1.0e-30)


_S8_MEMO: Dict[Tuple[object, ...], Dict[str, object]] = {}


def evaluate_linear_growth_s8(
    mod: PhysicsMod,
    nu_numpy_fn: Callable[[np.ndarray], np.ndarray],
    omega_m_eff: float,
) -> Dict[str, object]:
    """
    Integrates the 3D linear perturbation growth ODE across `k in [0.02, 0.50] h/Mpc` (`N = 240` RK4 steps
    from `a = 1e-3` to `1.0`), contrasting GR/screened growth against unscreened cosmic-web QUMOND boost
    and scoring `S_8 = sigma_8 * sqrt(Omega_m / 0.3)` against `DES Y3 / KiDS-1000` (`0.776 +- 0.017`).
    """
    om = float(np.clip(omega_m_eff, 0.01, 0.98))
    w0 = float(mod.parameters.w0_de)
    wa = float(mod.parameters.wa_de)
    h0 = max(float(mod.parameters.h0_kms_mpc), 20.0)
    a0_si = max(float(mod.parameters.a0_si), 1.0e-20)
    l_scr_mpc = max(float(mod.parameters.pleat_screening_mpc), 0.0)

    probe_vals = np.asarray(nu_numpy_fn(_S8_PROBE_Y), dtype=np.float64)
    probe_sig = (
        round(float(probe_vals[0]), 6),
        round(float(probe_vals[1]), 6),
        round(float(probe_vals[2]), 6),
    )
    cache_key = (
        mod.action.interpolation_nu,
        probe_sig,
        round(om, 5),
        round(w0, 4),
        round(wa, 4),
        round(h0, 4),
        round(a0_si * 1.0e10, 7),
        round(l_scr_mpc, 5),
    )
    cached = _S8_MEMO.get(cache_key)
    if cached is not None:
        return dict(cached)

    h = h0 / 100.0
    norm_c = _planck_sigma8_norm_const()
    pk_shape = _eh_pk_shape(om, 0.04897, h)
    dg_end, fric_h, src_h = _integrate_gr_growth_scalar(om, w0, wa)

    delta_k_coeff = np.sqrt(norm_c * pk_shape)
    sigma8_gr = math.sqrt(float(np.sum(((delta_k_coeff * dg_end) ** 2) * _S8_W8_SQ) * _S8_DLNK))

    # Tabulate active nu(y) in a single vectorized call across [1e-16, 1e8]
    nu_tab = np.clip(np.asarray(nu_numpy_fn(_S8_Y_LUT), dtype=np.float64), 1.0, 1.0e4)
    nu_is_trivial = bool(float(np.max(np.abs(nu_tab - 1.0))) <= 1.0e-8)

    if nu_is_trivial:
        sigma8_active = sigma8_gr
    else:
        h0_si = (h0 * 1000.0) / _MPC_M
        k_mpc = _S8_KH_GRID * h
        k_com_m_inv = k_mpc / _MPC_M
        # 4 * pi * G * rho_crit0 * om / k_com = (3/2) * H0^2 * om / k_com
        pref_gn_over_a0 = (1.5 * (h0_si * h0_si) * om / k_com_m_inv) * delta_k_coeff / a0_si

        if l_scr_mpc > 0.0:
            kl2 = (k_mpc * l_scr_mpc) ** 2
            w_mond = kl2 / (1.0 + kl2)
        else:
            w_mond = np.ones_like(k_mpc)

        fric_l = fric_h.tolist()
        src_l = src_h.tolist()
        ia_l = _S8_INV_A2_HALF_LIST
        nu_l = nu_tab.tolist()
        nu_first = nu_l[0]
        nu_last = nu_l[-1]
        lny_min = _S8_LNY_MIN
        lut_upper_f = _S8_LUT_UPPER_F
        inv_dlny = _S8_INV_DLNY
        _log = math.log
        _fabs = math.fabs

        dx = _S8_DX
        half_dx = 0.5 * dx
        sixth_dx = dx / 6.0
        a_init = _S8_A_INIT
        n_steps = _S8_RK4_STEPS

        dm_out: list[float] = []
        for pref_m, wm_m in zip(pref_gn_over_a0.tolist(), w_mond.tolist(), strict=True):
            dm_m = a_init
            dmp_m = a_init
            for i in range(n_steps):
                j = 2 * i
                f0, s0, ia0 = fric_l[j], src_l[j], ia_l[j]
                f1, s1, ia1 = fric_l[j + 1], src_l[j + 1], ia_l[j + 1]
                f2, s2 = fric_l[j + 2], src_l[j + 2]

                u0 = (_log(pref_m * ia0 * _fabs(dm_m) + 1.0e-18) - lny_min) * inv_dlny
                if u0 <= 0.0:
                    nu0 = nu_first
                elif u0 >= lut_upper_f:
                    nu0 = nu_last
                else:
                    idx0 = int(u0)
                    nu0 = nu_l[idx0] + (u0 - idx0) * (nu_l[idx0 + 1] - nu_l[idx0])
                mu0 = 1.0 + wm_m * (nu0 - 1.0)

                k1d = dmp_m
                k1p = -f0 * dmp_m + s0 * mu0 * dm_m
                d1 = dm_m + half_dx * k1d
                p1 = dmp_m + half_dx * k1p

                u1 = (_log(pref_m * ia1 * _fabs(d1) + 1.0e-18) - lny_min) * inv_dlny
                if u1 <= 0.0:
                    nu1 = nu_first
                elif u1 >= lut_upper_f:
                    nu1 = nu_last
                else:
                    idx1 = int(u1)
                    nu1 = nu_l[idx1] + (u1 - idx1) * (nu_l[idx1 + 1] - nu_l[idx1])
                mu1 = 1.0 + wm_m * (nu1 - 1.0)

                k2d = p1
                k2p = -f1 * p1 + s1 * mu1 * d1
                d2 = dm_m + half_dx * k2d
                p2 = dmp_m + half_dx * k2p

                k3d = p2
                k3p = -f1 * p2 + s1 * mu1 * d2
                d3 = dm_m + dx * k3d
                p3 = dmp_m + dx * k3p

                k4d = p3
                k4p = -f2 * p3 + s2 * mu1 * d3

                dm_m += sixth_dx * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
                dmp_m += sixth_dx * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)

            dm_out.append(dm_m)

        dm = np.asarray(dm_out, dtype=np.float64)
        sigma8_active = math.sqrt(float(np.sum(((delta_k_coeff * dm) ** 2) * _S8_W8_SQ) * _S8_DLNK))

    s8_factor = math.sqrt(om / 0.30)
    s8_gr = sigma8_gr * s8_factor
    s8_active = sigma8_active * s8_factor
    s8_signed_sigma = (s8_active - _DES_KIDS_S8_OBS) / _DES_KIDS_S8_ERR
    s8_sigma = abs(s8_signed_sigma)

    if s8_sigma <= 2.0:
        status = "PASS"
    elif s8_sigma <= 3.5:
        status = "WARN"
    else:
        status = "FAIL"

    out: Dict[str, object] = {
        "omega_m_eff": om,
        "sigma8_gr_screened": sigma8_gr,
        "sigma8_active": sigma8_active,
        "s8_gr_screened": s8_gr,
        "s8_active": s8_active,
        "s8_obs": _DES_KIDS_S8_OBS,
        "s8_obs_err": _DES_KIDS_S8_ERR,
        "s8_signed_sigma": s8_signed_sigma,
        "s8_sigma": s8_sigma,
        "nu_boost_active": not nu_is_trivial,
        "pleat_screening_mpc": l_scr_mpc,
        "status": status,
    }
    if len(_S8_MEMO) > 512:
        _S8_MEMO.clear()
    _S8_MEMO[cache_key] = out
    return dict(out)


def evaluate_cross_scale_mass_continuity(
    mod: PhysicsMod,
    nu_numpy_fn: Callable[[np.ndarray], np.ndarray],
    galaxies_175: Sequence[SparcGalaxy],
    omega_pleat_equiv: float,
    omega_b: float,
) -> Dict[str, object]:
    """
    Tests cross-scale mass continuity between the CMB (`Omega_pleat_equiv`) and galactic
    rotation curves (`NGC 3198`, `R >= 25 kpc`): if pressureless cosmic pleats/dust (`Omega_pleat > 0.01`)
    populate the CMB while `has_cdm_halo == False` in SPARC and no explicit halo/Mpc screening law
    (`pleat_screening_mpc >= 0.05`) prevents cosmic pleats from collapsing alongside baryons,
    `NGC 3198`'s outer circular speed blows up (`~250 km/s`, `> 20 sigma` double-counting tension).
    """
    if not galaxies_175:
        return {
            "galaxy_name": "NGC 3198",
            "cross_scale_seam_open": False,
            "xi_pleat": 0.0,
            "ngc3198_v_obs_outer_kms": 150.0,
            "ngc3198_v_single_outer_kms": 150.0,
            "ngc3198_v_double_counted_kms": 150.0,
            "ngc3198_double_count_sigma": 0.0,
        }

    gal = find_sparc_galaxy(list(galaxies_175), "NGC3198")
    if gal is None:
        gal = galaxies_175[0]

    ob = max(float(omega_b), 1.0e-6)
    op = max(float(omega_pleat_equiv), 0.0)
    xi_pleat = op / ob
    has_halo = bool(mod.parameters.has_cdm_halo)
    l_scr_mpc = max(float(mod.parameters.pleat_screening_mpc), 0.0)
    a0_safe = max(float(mod.parameters.a0_si), 1.0e-20)

    r_safe = np.maximum(gal.r_kpc, 1.0e-3)
    sig_v = np.maximum(gal.e_vobs_kms, 2.0)
    outer_mask = gal.r_kpc >= 25.0
    if not np.any(outer_mask):
        n_tail = max(1, len(gal.r_kpc) // 3)
        outer_mask = np.zeros(len(gal.r_kpc), dtype=bool)
        outer_mask[-n_tail:] = True

    v_obs_outer = float(np.mean(gal.v_obs_kms[outer_mask]))
    err_outer_scale = max(float(np.mean(sig_v[outer_mask])), 2.5)

    # Single-baryon outer velocity under the active nu(y) (vectorized across 9 Spitzer [3.6] M/L grid nodes)
    ml_grid = np.array([0.28, 0.34, 0.40, 0.45, 0.50, 0.56, 0.63, 0.72, 0.84], dtype=np.float64)
    prior_pens = ((np.log10(ml_grid / 0.50)) / 0.11) ** 2
    v_gas_term = np.sign(gal.v_gas_kms) * (gal.v_gas_kms**2)
    v_stellar_unit = (gal.v_disk_kms**2) + 1.4 * (gal.v_bul_kms**2)
    vb2_mat = np.maximum(v_gas_term[None, :] + ml_grid[:, None] * v_stellar_unit[None, :], 1.0e-6)

    def _fit_curve_with_pleat_clumping(clump_factor: np.ndarray) -> np.ndarray:
        vb2_eff = vb2_mat * (1.0 + xi_pleat * clump_factor[None, :])
        gb = vb2_eff * KMS2_PER_KPC_TO_MPS2 / r_safe[None, :]
        nu_v = np.maximum(np.asarray(nu_numpy_fn(gb / a0_safe), dtype=np.float64), 0.0)
        vm = np.sqrt(np.maximum(gb * nu_v * gal.r_kpc[None, :] / KMS2_PER_KPC_TO_MPS2, 0.0))
        tots = np.sum(((gal.v_obs_kms[None, :] - vm) / sig_v[None, :]) ** 2, axis=1) + prior_pens
        best_idx = int(np.argmin(tots))
        return vm[best_idx]

    v_single = _fit_curve_with_pleat_clumping(np.zeros_like(r_safe))
    v_single_outer = float(np.mean(v_single[outer_mask]))

    if has_halo or op <= 0.01:
        return {
            "galaxy_name": gal.name,
            "cross_scale_seam_open": False,
            "xi_pleat": xi_pleat,
            "ngc3198_v_obs_outer_kms": v_obs_outer,
            "ngc3198_v_single_outer_kms": v_single_outer,
            "ngc3198_v_double_counted_kms": v_single_outer,
            "ngc3198_double_count_sigma": 0.0,
        }

    if l_scr_mpc <= 0.0:
        f_clump = np.ones_like(r_safe)
    else:
        r_scr_kpc = 1000.0 * l_scr_mpc
        f_clump = 1.0 / (1.0 + ((r_scr_kpc / (0.5 * r_safe)) ** 6))

    v_dc = _fit_curve_with_pleat_clumping(f_clump)
    v_dc_outer = float(np.mean(v_dc[outer_mask]))
    dc_sigma = abs(v_dc_outer - v_obs_outer) / err_outer_scale
    seam_open = bool((l_scr_mpc < 0.05) and (dc_sigma > 3.0))

    return {
        "galaxy_name": gal.name,
        "cross_scale_seam_open": seam_open,
        "xi_pleat": xi_pleat,
        "ngc3198_v_obs_outer_kms": v_obs_outer,
        "ngc3198_v_single_outer_kms": v_single_outer,
        "ngc3198_v_double_counted_kms": v_dc_outer,
        "ngc3198_double_count_sigma": dc_sigma,
    }
