"""
Phase 6 Track 6A Automated Proof Suite (`tests/test_phase6_ungameable_crucible.py`)
===================================================================================
Verifies the Un-Gameable Cosmological Referee, Cross-Scale Continuity & AST Knob Counter:
  1. Single global `H_0 = 67.36 km/s/Mpc` consistency across `a_0 = c H_0 / (sqrt(3) pi)`,
     `CAMB` TT/TE/EE (`l <= 2500`), `DESI DR2` sound-horizon anchored BAO, primordial `BBN`,
     and `S_8` — with zero `_h0_for_pol_nnu_branch` `69.0` split.
  2. Full-spectrum `l = 30..2500` Planck Knox `chi^2 / dof` across `TT + TE + EE`, separating
     fixed-`n_s` (`0.9649`) Silk-damping tilt from `(A_s, n_s)`-profiled fits and post-BBN `N_eff = 3.044`.
  3. Primordial `BBN` `(D/H, Y_p)` response (`delta_neff_pleat = 0.259 -> WARN ~2.9 sigma`,
     `0.0 -> PASS`, `1.5 -> FAIL`).
  4. 3D Linear Growth `S_8` (`N = 240` RK4 steps) contrasting unscreened cosmic-web QUMOND blowout
     (`pleat_screening_mpc = 0.0 -> FAIL`, `S_8 > 5.0`) vs Mpc-screened (`pleat_screening_mpc = 0.25`)
     vs `Lambda-CDM`'s empirical `~3 sigma` `DES Y3 / KiDS-1000` tension (`WARN`).
  5. Cross-Scale Mass Continuity on `NGC 3198` (`R >= 25 kpc`): unscreened `1 + xi_pleat ~ 6.32x`
     halo collapse blows circular speed past `200 km/s` (`> 15 sigma`), gating `sparc_175` to `WARN`
     unless `pleat_screening_mpc >= 0.05` (`+1` free knob) is supplied.
  6. Live AST `count_mod_free_knobs` transparency across `[action]` and `[parameters]`, plus `< 2 ms`
     warm referee latency.
"""

from __future__ import annotations

import gc
import math
import time

import camb
import numpy as np

from weaveforge.arenas import arena4_cosmic_crucible as a4m
from weaveforge.arenas.arena4_cosmic_crucible import evaluate_observational_crucible
from weaveforge.core import skyrme_perturbation as spm
from weaveforge.core.cosmology_referee import (
    audit_single_h0_consistency,
    evaluate_cross_scale_mass_continuity,
    evaluate_linear_growth_s8,
    evaluate_planck_knox_chi2_2500,
    evaluate_primordial_bbn,
)
from weaveforge.core.frameworks import PhysicsMod, count_mod_free_knobs, get_mods_by_id
from weaveforge.core.symbolic_compiler import compile_master_action


def test_single_h0_6736_unification_and_toml_roundtrip() -> None:
    # 1. Zero secret H0=69 polarization branch anywhere in Arena 4 or Skyrme perturbations
    assert a4m._PLANCK_H0 == 67.36
    assert spm._PLANCK_H0_KMS_MPC == 67.36
    assert not hasattr(a4m, "_PLANCK_H0_NNU_3303")
    assert not hasattr(a4m, "_h0_for_pol_nnu_branch")

    # 2. All 6 mods share h0_kms_mpc = 67.36 and pleat_screening_mpc = 0.0
    mods = get_mods_by_id()
    assert len(mods) >= 6
    for m in mods.values():
        assert math.isclose(m.parameters.h0_kms_mpc, 67.36, rel_tol=0.0, abs_tol=1e-9)
        assert math.isclose(m.parameters.pleat_screening_mpc, 0.0, rel_tol=0.0, abs_tol=1e-12)
        if m.id == "tnap_weave":
            assert math.isclose(m.parameters.delta_neff_pleat, 0.259, rel_tol=0.0, abs_tol=1e-9)
        else:
            assert math.isclose(m.parameters.delta_neff_pleat, 0.0, rel_tol=0.0, abs_tol=1e-12)

    # 3. TOML serialization & deserialization preserves all Track 6A parameters
    tnap = mods["tnap_weave"].clone(new_id="roundtrip_check")
    tnap.parameters.h0_kms_mpc = 68.15
    tnap.parameters.delta_neff_pleat = 0.125
    tnap.parameters.pleat_screening_mpc = 0.22
    reloaded = PhysicsMod.from_toml_bytes(tnap.to_toml_string().encode("utf-8"))
    assert math.isclose(reloaded.parameters.h0_kms_mpc, 68.15, abs_tol=1e-6)
    assert math.isclose(reloaded.parameters.delta_neff_pleat, 0.125, abs_tol=1e-6)
    assert math.isclose(reloaded.parameters.pleat_screening_mpc, 0.22, abs_tol=1e-6)

    # 4. Direct CAMB verification on N_eff = 3.303 proves LUT uses H0 = 67.36 (NOT 69.0!)
    h67 = 67.36 / 100.0
    om_tnap = 0.3094
    pars_6736 = camb.set_params(
        H0=67.36,
        ombh2=0.02237,
        omch2=(om_tnap - a4m._PLANCK_OMEGA_B) * (h67 * h67),
        tau=0.0544,
        As=2.1005e-9,
        ns=0.9649,
        nnu=3.303,
        lmax=2500,
    )
    tot_6736 = camb.get_results(pars_6736).get_cmb_power_spectra(pars_6736, CMB_unit="muK", raw_cl=False)["total"][:2501]
    pol_lut_3303 = a4m._get_camb_pol_2500_fast(om_tnap, 3.303)
    ell_idx = a4m._CAMB_POL_ELL_GRID
    assert np.max(np.abs(pol_lut_3303[0] - tot_6736[ell_idx, 0])) < 0.05
    assert np.max(np.abs(pol_lut_3303[1] - tot_6736[ell_idx, 3])) < 0.05
    assert np.max(np.abs(pol_lut_3303[2] - tot_6736[ell_idx, 1])) < 0.05

    # 5. Single-H0 audit & un-gameable SH0ES H0 = 73.04 sound-horizon BAO collision
    h0_tnap = audit_single_h0_consistency(mods["tnap_weave"])
    assert h0_tnap["a0_locked_to_h0"] is True
    assert float(h0_tnap["a0_h0_rel_mismatch"]) < 1.0e-4
    assert 4.5 <= float(h0_tnap["shoes_h0_sigma"]) <= 5.2

    tnap_shoes = mods["tnap_weave"].clone(new_id="tnap_shoes_drag")
    tnap_shoes.parameters.h0_kms_mpc = 73.04
    rep_shoes = evaluate_observational_crucible(tnap_shoes, compile_master_action(tnap_shoes))
    b_shoes = {b.badge_id: b.status for b in rep_shoes.badges}
    assert rep_shoes.a0_h0_rel_mismatch > 0.07
    assert rep_shoes.desi_bao_anchored_chi2_dof > 3.5
    assert b_shoes["desi_h0"] == "FAIL"


def test_planck_knox_chi2_2500_fixed_vs_profiled_ns() -> None:
    mods = get_mods_by_id()
    cdm_rep = evaluate_observational_crucible(mods["lambda_cdm"], compile_master_action(mods["lambda_cdm"]))
    tnap_rep = evaluate_observational_crucible(mods["tnap_weave"], compile_master_action(mods["tnap_weave"]))
    mond_rep = evaluate_observational_crucible(mods["classic_mond"], compile_master_action(mods["classic_mond"]))

    # 1. Exact self-calibration: mod == obs gives zero excess chi2 (chi2_dof == 1.0)
    exact_self = evaluate_planck_knox_chi2_2500(
        cdm_rep.cmb_ell_2500,
        cdm_rep.cmb_tt_obs_2500,
        cdm_rep.cmb_te_obs_2500,
        cdm_rep.cmb_ee_obs_2500,
        cdm_rep.cmb_tt_obs_2500,
        cdm_rep.cmb_te_obs_2500,
        cdm_rep.cmb_ee_obs_2500,
    )
    assert float(exact_self["chi2_fixed_ns"]) < 1.0e-12
    assert math.isclose(float(exact_self["chi2_dof_fixed_ns"]), 1.0, abs_tol=1.0e-12)
    assert math.isclose(float(exact_self["best_fit_ns"]), 0.9649, abs_tol=1.0e-6)

    # 2. Pure spectral-tilt recovery check: +0.030 tilt triggers fixed-ns penalty, recovered by (As, ns) profile
    tilt_factor = (cdm_rep.cmb_ell_2500 / 700.0) ** 0.030
    tilted = evaluate_planck_knox_chi2_2500(
        cdm_rep.cmb_ell_2500,
        cdm_rep.cmb_tt_obs_2500,
        cdm_rep.cmb_te_obs_2500,
        cdm_rep.cmb_ee_obs_2500,
        1.12 * cdm_rep.cmb_tt_obs_2500 * tilt_factor,
        1.12 * cdm_rep.cmb_te_obs_2500 * tilt_factor,
        1.12 * cdm_rep.cmb_ee_obs_2500 * tilt_factor,
    )
    assert float(tilted["delta_chi2_fixed_ns"]) > 200.0
    assert float(tilted["chi2_dof_profiled_ns"]) < 1.001
    assert math.isclose(float(tilted["best_fit_ns"]), 0.9649 - 0.030, abs_tol=5.0e-4)

    # 3. Lambda-CDM passes cleanly; TNaP N_eff = 3.303 triggers fixed-ns WARN (> 1.25); post-BBN N_eff = 3.044 drops < 1.10
    assert cdm_rep.cmb_chi2_dof_fixed_ns < 1.01
    assert {b.badge_id: b.status for b in cdm_rep.badges}["planck_cmb"] == "PASS"

    assert tnap_rep.cmb_chi2_dof_fixed_ns > 1.25
    assert tnap_rep.cmb_chi2_dof_profiled_ns <= tnap_rep.cmb_chi2_dof_fixed_ns
    assert tnap_rep.cmb_chi2_dof_profiled_ns <= 2.50
    assert tnap_rep.cmb_high_l_tilt_pct < -0.3
    assert {b.badge_id: b.status for b in tnap_rep.badges}["planck_cmb"] == "WARN"

    tnap_post = mods["tnap_weave"].clone(new_id="tnap_post_bbn_knox")
    tnap_post.parameters.delta_neff_pleat = 0.0
    rep_post = evaluate_observational_crucible(tnap_post, compile_master_action(tnap_post))
    assert rep_post.cmb_chi2_dof_fixed_ns < 1.10
    assert tnap_rep.cmb_high_l_tilt_pct < rep_post.cmb_high_l_tilt_pct

    # 4. Classic MOND fails Knox chi2/dof even after (As, ns) profiling
    assert mond_rep.cmb_chi2_dof_profiled_ns > 10.0
    assert {b.badge_id: b.status for b in mond_rep.badges}["planck_cmb"] == "FAIL"


def test_primordial_bbn_abundances_gate() -> None:
    omega_b_planck = 0.02237 / ((67.36 / 100.0) ** 2)

    # 1. Standard N_eff = 3.044 (delta_neff = 0.0) -> PASS
    b0 = evaluate_primordial_bbn(omega_b=omega_b_planck, h0_kms_mpc=67.36, delta_neff=0.0)
    assert math.isclose(float(b0["dh_e5"]), 2.515, abs_tol=0.005)
    assert math.isclose(float(b0["y_p"]), 0.24709, abs_tol=0.0005)
    assert float(b0["max_sigma"]) <= 2.0
    assert b0["status"] == "PASS"

    # 2. Pre-BBN relativistic pleats (delta_neff = +0.259, N_eff = 3.303) -> ~2.87 sigma WARN
    b_tnap = evaluate_primordial_bbn(omega_b=omega_b_planck, h0_kms_mpc=67.36, delta_neff=0.259)
    assert math.isclose(float(b_tnap["dh_e5"]), 2.601, abs_tol=0.01)
    assert math.isclose(float(b_tnap["y_p"]), 0.2506, abs_tol=0.001)
    assert 2.5 <= float(b_tnap["max_sigma"]) <= 3.2
    assert b_tnap["status"] == "WARN"

    # 3. Extreme dark radiation (delta_neff = +1.50) -> FAIL
    b_ext = evaluate_primordial_bbn(omega_b=omega_b_planck, h0_kms_mpc=67.36, delta_neff=1.50)
    assert float(b_ext["max_sigma"]) > 8.0
    assert b_ext["status"] == "FAIL"

    # 4. End-to-end Crucible badge check
    mods = get_mods_by_id()
    tnap_rep = evaluate_observational_crucible(mods["tnap_weave"], compile_master_action(mods["tnap_weave"]))
    aest_rep = evaluate_observational_crucible(mods["aest"], compile_master_action(mods["aest"]))
    assert {b.badge_id: b.status for b in tnap_rep.badges}["bbn_abundances"] == "WARN"
    assert {b.badge_id: b.status for b in aest_rep.badges}["bbn_abundances"] == "PASS"


def test_linear_growth_s8_unscreened_qumond_vs_screened_vs_lcdm() -> None:
    mods = get_mods_by_id()

    # 1. Lambda-CDM captures the real ~2.8-3.2 sigma DES Y3 / KiDS-1000 S_8 tension -> WARN
    cdm_rep = evaluate_observational_crucible(mods["lambda_cdm"], compile_master_action(mods["lambda_cdm"]))
    b_cdm = {b.badge_id: b.status for b in cdm_rep.badges}
    assert 0.820 <= cdm_rep.lensing_s8_active <= 0.840
    assert 2.5 <= cdm_rep.lensing_s8_sigma <= 3.5
    assert b_cdm["lensing_s8"] == "WARN"

    # 2. Unscreened QUMOND / RAR on cosmic-web filaments (pleat_screening_mpc = 0.0) blows out S_8 >> 5.0 -> FAIL
    tnap = mods["tnap_weave"]
    tnap_comp = compile_master_action(tnap)
    tnap_rep = evaluate_observational_crucible(tnap, tnap_comp)
    b_tnap = {b.badge_id: b.status for b in tnap_rep.badges}
    assert tnap_rep.lensing_s8_active > 5.0
    assert 0.79 <= tnap_rep.lensing_s8_gr_screened <= 0.83
    assert b_tnap["lensing_s8"] == "FAIL"

    # 3. Supplying explicit Mpc screening L_scr = 0.25 Mpc suppresses Mpc MOND boost near GR (~0.81) while adding +1 knob
    tnap_scr = tnap.clone(new_id="tnap_mpc_screened")
    tnap_scr.parameters.pleat_screening_mpc = 0.25
    rep_scr = evaluate_observational_crucible(tnap_scr, tnap_comp)
    b_scr = {b.badge_id: b.status for b in rep_scr.badges}
    assert abs(rep_scr.lensing_s8_active - rep_scr.lensing_s8_gr_screened) < 0.03
    assert b_scr["lensing_s8"] in ("PASS", "WARN")
    assert rep_scr.free_global_knobs == tnap_rep.free_global_knobs + 1


def test_cross_scale_mass_continuity_ngc3198_double_counting_seam() -> None:
    mods = get_mods_by_id()

    # 1. TNaP & AeST populate Omega_pleat ~ 0.26 in CMB with pleat_screening_mpc = 0.0 -> NGC 3198 > 200 km/s (> 15 sigma)
    for mod_id in ("tnap_weave", "aest"):
        m = mods[mod_id]
        rep = evaluate_observational_crucible(m, compile_master_action(m))
        b_map = {b.badge_id: b for b in rep.badges}
        assert rep.cross_scale_seam_open is True
        assert rep.ngc3198_v_double_counted_kms > 200.0
        assert rep.ngc3198_double_count_sigma > 15.0
        assert b_map["sparc_175"].status == "WARN"
        assert "NGC 3198" in b_map["sparc_175"].detail_honest

    # 2. Classic MOND has Omega_pleat = 0 in CMB -> no cross-scale double counting on SPARC (passes SPARC, fails CMB)
    mond_rep = evaluate_observational_crucible(mods["classic_mond"], compile_master_action(mods["classic_mond"]))
    assert mond_rep.cross_scale_seam_open is False
    assert {b.badge_id: b.status for b in mond_rep.badges}["sparc_175"] == "PASS"

    # 3. Supplying pleat_screening_mpc = 0.15 (>= 50 kpc) closes the NGC 3198 seam -> sparc_175 == PASS (+1 free knob)
    tnap_scr = mods["tnap_weave"].clone(new_id="tnap_halo_screened")
    tnap_scr.parameters.pleat_screening_mpc = 0.15
    rep_scr = evaluate_observational_crucible(tnap_scr, compile_master_action(tnap_scr))
    assert rep_scr.cross_scale_seam_open is False
    assert {b.badge_id: b.status for b in rep_scr.badges}["sparc_175"] == "PASS"


def test_ast_knob_counter_7row_scoreboard_and_referee_latency() -> None:
    mods = get_mods_by_id()

    # 1. Exact 7-row scoreboard order & preserved badge IDs
    tnap_comp = compile_master_action(mods["tnap_weave"])
    tnap_rep = evaluate_observational_crucible(mods["tnap_weave"], tnap_comp)
    badge_ids = [b.badge_id for b in tnap_rep.badges]
    assert badge_ids == [
        "sparc_175",
        "planck_cmb",
        "bbn_abundances",
        "lensing_s8",
        "solar_gw",
        "desi_h0",
        "gaia_dr4",
    ]

    # 2. Transparent AST knob audit on Lambda-CDM vs TNaP Spacetime Weave
    k_cdm = count_mod_free_knobs(mods["lambda_cdm"])
    assert k_cdm["action_literal_count"] == 0
    assert k_cdm["action_tuning_literals"] == []
    assert k_cdm["per_galaxy_halo_knobs"] == 350
    assert k_cdm["a0_locked_to_h0"] is False

    k_tnap = count_mod_free_knobs(mods["tnap_weave"])
    assert k_tnap["action_literal_count"] == 9
    expected_lits = [0.7098, 0.8425, 1.7, 80.0, 120.0, 12000.0, 9.0, 0.7225, 0.0014996858]
    for exp_v, got_v in zip(expected_lits, k_tnap["action_tuning_literals"], strict=True):
        assert math.isclose(float(got_v), exp_v, rel_tol=1e-9, abs_tol=1e-12)
    assert k_tnap["per_galaxy_halo_knobs"] == 0
    assert k_tnap["a0_locked_to_h0"] is True

    # 3. Live dynamic response when editing [action] or [parameters]
    edited = mods["tnap_weave"].clone(new_id="edited_knobs")
    edited.action.optical_metric_n = "exp(2 * U) + 0.137 * U**3"
    k_ed1 = count_mod_free_knobs(edited)
    assert k_ed1["action_literal_count"] == k_tnap["action_literal_count"] + 1
    assert k_ed1["global_knob_total"] == k_tnap["global_knob_total"] + 1

    # Detuning a0_si away from c * H_0 / (sqrt(3) * pi) unlocks a0_si as an extra free knob
    edited.parameters.a0_si = 1.35e-10
    k_ed2 = count_mod_free_knobs(edited)
    assert k_ed2["a0_locked_to_h0"] is False
    assert k_ed2["cosmo_knob_count"] == k_ed1["cosmo_knob_count"] + 1

    # 4. Warm cosmology_referee suite executes in < 2.0 ms
    gals = a4m.get_sparc_175_cached()
    assert tnap_comp.nu_numpy_fn is not None
    _ = evaluate_linear_growth_s8(mods["tnap_weave"], tnap_comp.nu_numpy_fn, 0.30941)
    gc.collect()
    ref_ms: list[float] = []
    for _ in range(25):
        t0 = time.perf_counter()
        _ = audit_single_h0_consistency(mods["tnap_weave"])
        _ = evaluate_planck_knox_chi2_2500(
            tnap_rep.cmb_ell_2500,
            tnap_rep.cmb_tt_obs_2500,
            tnap_rep.cmb_te_obs_2500,
            tnap_rep.cmb_ee_obs_2500,
            tnap_rep.cmb_tt_mod_2500,
            tnap_rep.cmb_te_mod_2500,
            tnap_rep.cmb_ee_mod_2500,
        )
        _ = evaluate_primordial_bbn(0.049302, 67.36, 0.259)
        _ = evaluate_linear_growth_s8(mods["tnap_weave"], tnap_comp.nu_numpy_fn, 0.30941)
        _ = evaluate_cross_scale_mass_continuity(
            mods["tnap_weave"], tnap_comp.nu_numpy_fn, gals, omega_pleat_equiv=0.26044, omega_b=0.04897
        )
        ref_ms.append((time.perf_counter() - t0) * 1000.0)
    assert float(np.median(ref_ms)) < 2.0, f"Expected warm cosmology_referee suite < 2.0 ms, got {ref_ms}"


def test_studio_v1_wind_tunnel_ui_and_parameter_reset_integration() -> None:
    """
    Verifies WeaveForge v1.0.0 Studio UI integration in headless OpenGL 4.6 mode:
      • `weaveforge.__version__ == "1.0.0"`,
      • 7-row Crucible + AST knob summary rendering cleanly across `tnap_weave` & `lambda_cdm`,
      • `h0_kms_mpc`, `delta_neff_pleat`, and `pleat_screening_mpc` registered in `_COUPLING_PARAM_FIELDS`
        with dirty-state tracking and `reset_single_parameter`,
      • Arena 2 Submode 3 spinning `GW150914` (`spin_a = 0.67`) live ringdown & LVK O4 audit rendering.
    """
    import weaveforge
    from weaveforge.studio.app import WeaveForgeStudioApp

    assert weaveforge.__version__ == "1.0.0"

    app = WeaveForgeStudioApp(width=1280, height=760, visible=False, initial_arena=4)
    try:
        app.render_single_frame()
        assert app.crucible is not None
        assert len(app.crucible.badges) == 7

        # Modify h0_kms_mpc, delta_neff_pleat, and pleat_screening_mpc and verify dirty tracking + reset
        app.working_mod.parameters.h0_kms_mpc = 73.04
        app.working_mod.parameters.delta_neff_pleat = 0.0
        app.working_mod.parameters.pleat_screening_mpc = 0.25
        app._refresh_crucible_fast()
        assert app.is_parameter_modified("h0_kms_mpc") is True
        assert app.is_parameter_modified("delta_neff_pleat") is True
        assert app.is_parameter_modified("pleat_screening_mpc") is True
        assert app.count_modified_couplings() == 3

        app.select_left_tab(1)
        app.render_single_frame()

        app.reset_single_parameter("h0_kms_mpc")
        app.reset_single_parameter("delta_neff_pleat")
        app.reset_single_parameter("pleat_screening_mpc")
        assert app.count_modified_couplings() == 0
        assert math.isclose(app.working_mod.parameters.h0_kms_mpc, 67.36, abs_tol=1e-4)

        # Switch to Arena 2 Submode 3 at GW150914 spin a = 0.67 and render headless UI + 3D FBO
        app._switch_active_arena(1)
        app.arena2.submode = 3
        app.arena2.spin_a = 0.67
        app.arena2.spool_reflectivity = 0.65
        app.render_single_frame()
        assert 0.70 <= app.arena2.qnm_freq_ratio <= 0.90
    finally:
        app.shutdown()

