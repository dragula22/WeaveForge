"""
Phase 4 Milestone 4B Automated Proof Suite (`tests/test_phase4_skyrme_cmb_tt_te_ee.py`)
=======================================================================================
Verifies:
  1. Symbolic (`SymPy`) covariant Skyrme-pleat perturbation hierarchy `nabla_mu T^mu_nu = 0`
     matching Ma & Bertschinger (1995, ApJ 455, 7) Eqs. (29)-(30), `sigma_p == 0`,
     `c_{s,p}^2(k, a) = c_w^2 k^2 / (4 m_pleat^2 a^2 + c_w^2 k^2)`, and Euclidean catenoid
     waist adiabaticity `beta_iso < 0.02` vs pathological large-scale cutoff failure (`l_cut = 200`).
  2. Stiff numerical (`SciPy` Radau) 7-variable Einstein-Boltzmann integration proving
     `|phi_well_ratio - 1| < 5e-4` at `k = 0.18 Mpc^-1` (`l ~ 2500`) for pinned `SO(3)` pleats
     (`m_pleat >= 1e-18 eV`) vs Jeans suppression `< 0.45` for ultralight `m_pleat = 1e-25 eV`.
  3. Real `CAMB` `l = 2..2500` `[TT, EE, BB, TE]` column fidelity, catenoid waist clamping across
     `TT/TE/EE`, `cmb_te_corr_2500 > 0.995` and `0.92 <= cmb_ee_peak_ratio <= 1.08` for
     `tnap_weave` and `lambda_cdm`, and polarization failure for `classic_mond`.
  4. DDR5 `camb_te_ee_2500_lut_cache.npz` fast load, `< 15 ms` warm slider evaluation, and 100%
     backward compatibility with `camb_tt_lut_cache.npz` and `CrucibleReport` `l = 2..1000` invariants.
"""

from __future__ import annotations

import copy
import gc
import time

import camb
import numpy as np
import sympy as sp

from weaveforge.arenas import arena4_cosmic_crucible as a4m
from weaveforge.arenas.arena4_cosmic_crucible import (
    CrucibleReport,
    evaluate_observational_crucible,
)
from weaveforge.core.frameworks import get_mods_by_id
from weaveforge.core.skyrme_perturbation import (
    derive_covariant_pleat_perturbation_sympy,
    evaluate_catenoid_waist_adiabaticity,
    integrate_skyrme_pleat_mode_hierarchy,
    skyrme_pleat_jeans_damping_envelope,
)
from weaveforge.core.symbolic_compiler import compile_master_action


def test_sympy_covariant_pleat_hierarchy_and_adiabatic_waist_bound() -> None:
    deriv = derive_covariant_pleat_perturbation_sympy()

    # Exact Ma & Bertschinger (1995) Eqs. (29)-(30) symbolic verification
    assert deriv["ma_bertschinger_eq29_eq30_verified"] is True
    assert sp.simplify(deriv["continuity_residual"]) == 0
    assert sp.simplify(deriv["euler_residual"]) == 0
    assert deriv["four_velocity_norm"] == -1

    # Vanishing linear scalar anisotropic stress sigma_p == 0
    assert deriv["anisotropic_stress_zero"] is True
    assert deriv["sigma_p"] == 0

    # Relativistic WKB dispersion sound speed c_{s,p}^2(k, a) and heavy-pleat dust limit -> 0
    assert sp.simplify(deriv["cs2_residual"]) == 0
    assert deriv["cs2_heavy_pleat_limit"] == 0

    # Euclidean catenoid waist adiabaticity: TNaP (l_cut = 3.21) passes < 0.02, pathological l_cut > 160 fails > 0.038
    ad_tnap = evaluate_catenoid_waist_adiabaticity(waist_clamping_ell=3.21, r0_over_eta0=0.31)
    assert np.isfinite(ad_tnap["kretschmann_max"])
    assert 0.0 < ad_tnap["beta_iso"] < 0.02
    assert ad_tnap["passed_planck_95cl"] == 1.0

    ad_bad_165 = evaluate_catenoid_waist_adiabaticity(waist_clamping_ell=165.0, r0_over_eta0=0.31)
    ad_bad_200 = evaluate_catenoid_waist_adiabaticity(waist_clamping_ell=200.0, r0_over_eta0=0.31)
    assert ad_bad_165["beta_iso"] > 0.038
    assert ad_bad_165["passed_planck_95cl"] == 0.0
    assert ad_bad_200["beta_iso"] > ad_bad_165["beta_iso"] > 0.038
    assert ad_bad_200["passed_planck_95cl"] == 0.0

    # Injecting pathological l_cut = 200 into TNaP must trigger FAIL on planck_cmb badge
    bad_tnap = get_mods_by_id()["tnap_weave"].clone(new_id="bad_waist_200")
    bad_tnap.parameters.waist_clamping_ell = 200.0
    bad_rep = evaluate_observational_crucible(bad_tnap, compile_master_action(bad_tnap))
    b_map = {b.badge_id: b for b in bad_rep.badges}
    assert bad_rep.cmb_beta_iso > 0.038
    assert b_map["planck_cmb"].status == "FAIL"


def test_scipy_boltzmann_pleat_mode_integrator_dust_limit_vs_ultralight_jeans() -> None:
    # High-l CMB scale k = 0.18 Mpc^-1 (ell ~ 2500)
    res_heavy = integrate_skyrme_pleat_mode_hierarchy(k_mpc_inv=0.18, m_pleat_ev=1.0e-10)
    res_pinned = integrate_skyrme_pleat_mode_hierarchy(k_mpc_inv=0.18, m_pleat_ev=1.0e-18)
    assert abs(res_heavy["phi_well_ratio"] - 1.0) < 5.0e-4
    assert abs(res_pinned["phi_well_ratio"] - 1.0) < 5.0e-4
    assert abs(res_pinned["transfer_ratio_pleat_vs_cdm"] - 1.0) < 5.0e-4
    assert res_pinned["c_s2_rec"] < 1.0e-12

    # Ultralight unpinned scalar stress test (m_pleat = 1e-25 eV): Jeans oscillations suppress Phi well < 0.45
    res_ula = integrate_skyrme_pleat_mode_hierarchy(k_mpc_inv=0.18, m_pleat_ev=1.0e-25)
    assert res_ula["phi_well_ratio"] < 0.45
    assert res_ula["transfer_ratio_pleat_vs_cdm"] < 0.45

    # Fast vectorized Jeans damping envelope (< 0.05 ms)
    ell_grid = np.arange(2, 2501, dtype=np.float64)
    _ = skyrme_pleat_jeans_damping_envelope(ell_grid, m_pleat_ev=1.0)
    timings_ms: list[float] = []
    for _ in range(25):
        t0 = time.perf_counter()
        env_pin = skyrme_pleat_jeans_damping_envelope(ell_grid, m_pleat_ev=1.0e-16)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    assert float(np.median(timings_ms)) < 0.05
    assert float(np.max(np.abs(env_pin - 1.0))) < 1.0e-12

    env_ula = skyrme_pleat_jeans_damping_envelope(ell_grid, m_pleat_ev=1.0e-25)
    assert env_ula[0] > 0.999
    assert env_ula[-1] < 0.45


def test_real_camb_2500_multipole_tt_te_ee_polarization_crucible() -> None:
    assert a4m._HAS_CAMB is True

    # 1. Direct verification against raw CAMB [0=TT, 1=EE, 2=BB, 3=TE] columns out to l = 2500
    h = a4m._PLANCK_H0 / 100.0
    om_test = 0.3140
    pars = camb.set_params(
        H0=a4m._PLANCK_H0,
        ombh2=0.02237,
        omch2=(om_test - a4m._PLANCK_OMEGA_B) * (h * h),
        tau=0.0544,
        As=2.1005e-9,
        ns=0.9649,
        nnu=3.044,
        lmax=2500,
    )
    tot_raw = camb.get_results(pars).get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=False)["total"][:2501]
    pol_lut = a4m._get_camb_pol_2500_fast(om_test, 3.044)
    ell_idx = a4m._CAMB_POL_ELL_GRID

    assert np.max(np.abs(pol_lut[0] - tot_raw[ell_idx, 0])) < 0.05  # TT == col 0
    assert np.max(np.abs(pol_lut[1] - tot_raw[ell_idx, 3])) < 0.05  # TE == col 3
    assert np.max(np.abs(pol_lut[2] - tot_raw[ell_idx, 1])) < 0.05  # EE == col 1
    # Confirm EE (col 1) is genuinely distinct from BB lensing (col 2)
    assert np.max(np.abs(pol_lut[2] - tot_raw[ell_idx, 2])) > 30.0

    # 2. Crucible polarization evaluation across TNaP, Lambda-CDM, and Classic MOND
    mods = get_mods_by_id()
    tnap_rep = evaluate_observational_crucible(mods["tnap_weave"], compile_master_action(mods["tnap_weave"]))
    cdm_rep = evaluate_observational_crucible(mods["lambda_cdm"], compile_master_action(mods["lambda_cdm"]))
    mond_rep = evaluate_observational_crucible(mods["classic_mond"], compile_master_action(mods["classic_mond"]))

    for rep in (tnap_rep, cdm_rep, mond_rep):
        assert rep.cmb_ell_2500[0] == 2.0
        assert rep.cmb_ell_2500[-1] == 2500.0
        assert rep.cmb_tt_mod_2500.shape == rep.cmb_ell_2500.shape
        assert rep.cmb_te_mod_2500.shape == rep.cmb_ell_2500.shape
        assert rep.cmb_ee_mod_2500.shape == rep.cmb_ell_2500.shape

    assert tnap_rep.cmb_te_corr_2500 > 0.995
    assert cdm_rep.cmb_te_corr_2500 > 0.995
    assert 0.92 <= tnap_rep.cmb_ee_peak_ratio <= 1.08
    assert 0.92 <= cdm_rep.cmb_ee_peak_ratio <= 1.08
    assert 0.0 < tnap_rep.cmb_beta_iso < 0.02

    # Infrared catenoid waist clamping suppresses l=2 across TT, TE, and EE simultaneously
    assert tnap_rep.cmb_tt_mod_2500[0] < 400.0
    assert tnap_rep.cmb_ee_mod_2500[0] < cdm_rep.cmb_ee_mod_2500[0] * 0.5

    # Classic MOND fails TE/EE acoustic phase & amplitude out to l = 2500
    assert mond_rep.cmb_te_corr_2500 < 0.75
    assert mond_rep.cmb_ee_peak_ratio < 0.90

    # Honest TNaP planck_cmb WARN badge includes 2500-multipole TE/EE + beta_iso + open early-waist item
    b_tnap = {b.badge_id: b for b in tnap_rep.badges}
    assert b_tnap["planck_cmb"].status == "WARN"
    detail = b_tnap["planck_cmb"].detail_honest
    assert f"TE r={tnap_rep.cmb_te_corr_2500:.3f}" in detail
    assert f"EE={tnap_rep.cmb_ee_peak_ratio:.2f}x (l<=2500)" in detail
    assert f"beta_iso={tnap_rep.cmb_beta_iso:.4f}<0.02; open seams:" in detail


def test_camb_2500_lut_latency_and_backward_compatibility() -> None:
    gc.collect()
    # 1. Existing camb_tt_lut_cache.npz invariants must remain 100% intact
    tt_npz = a4m._CAMB_LUT_DISK_CACHE_PATH
    pol_npz = a4m._CAMB_POL_LUT_DISK_CACHE_PATH
    assert tt_npz.is_file() and tt_npz.stat().st_size < 250 * 1024
    assert pol_npz.is_file() and pol_npz.stat().st_size < 450 * 1024

    t_tt = time.perf_counter()
    assert a4m.load_camb_lut_disk_cache(force_reload=True) is True
    assert (time.perf_counter() - t_tt) * 1000.0 < 50.0

    t_pol = time.perf_counter()
    assert a4m.load_camb_pol_lut_disk_cache(force_reload=True) is True
    assert (time.perf_counter() - t_pol) * 1000.0 < 50.0
    assert a4m.is_camb_pol_ddr5_cache_ready() is True

    # 2. Legacy CrucibleReport constructor compatibility + len(cmb_ell) == 999 invariant
    legacy_rep = CrucibleReport(
        mod_id="legacy_check",
        mod_name="Legacy Check",
        omega_b=0.049,
        omega_m_effective=0.30941,
        cmb_r32_ratio=0.98,
        cmb_d2_quadrupole_uK2=334.0,
        sparc_mean_chi2=1.63,
        sparc_rar_scatter_dex=0.13,
        wide_binary_disk_boost_pct=15.8,
        wide_binary_halo_boost_pct=31.2,
    )
    assert legacy_rep.cmb_te_corr_2500 == 1.0
    assert legacy_rep.cmb_ee_peak_ratio == 1.0
    assert legacy_rep.cmb_beta_iso == 0.0

    base_mod = get_mods_by_id()["tnap_weave"]
    comp = compile_master_action(base_mod)
    warm_rep = evaluate_observational_crucible(base_mod, comp)
    assert len(warm_rep.cmb_ell) == 999
    assert len(warm_rep.cmb_dl_obs) == 999
    assert len(warm_rep.cmb_dl_mod) == 999

    # 3. Warm slider evaluation across pinned_pleat_coupling must stay well below < 15 ms (< 5 ms target)
    variants = []
    for ppc in (0.0, 0.25, 0.50, 0.75, 1.00, 1.35):
        m_var = copy.deepcopy(base_mod)
        m_var.parameters.pinned_pleat_coupling = ppc
        variants.append(m_var)

    gc.collect()
    eval_ms: list[float] = []
    for m_var in variants:
        t0 = time.perf_counter()
        rep_v = evaluate_observational_crucible(m_var, comp)
        eval_ms.append((time.perf_counter() - t0) * 1000.0)
        assert np.all(np.isfinite(rep_v.cmb_te_mod_2500))
        assert np.all(np.isfinite(rep_v.cmb_ee_mod_2500))

    assert max(eval_ms) < 15.0, f"Expected warm slider crucible evaluation < 15 ms, got {eval_ms}"
    assert float(np.median(eval_ms)) < 5.0, f"Expected median warm crucible evaluation < 5 ms, got {eval_ms}"
