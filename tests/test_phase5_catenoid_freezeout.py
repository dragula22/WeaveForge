"""
Phase 5 Milestone 5B Automated Proof Suite (`tests/test_phase5_catenoid_freezeout.py`)
======================================================================================
Verifies:
  1. Symbolic (`SymPy`) catenoid bounce throat geometry (`a(tau) = sqrt(r_0^2 + tau^2)`, `H(0) = 0`,
     `H_dot(0) = r_0^-2`, finite Kretschmann invariants `K_{S^3}(0) = 24 / r_0^4` and
     `K_flat(0) = 12 / r_0^4`), exact covariant 3-component Skyrme-Faddeev energy balance (`residual == 0`),
     and Derrick virial late-time dilution exponents (`exp_pleat == -3`, `exp_knot == -3`, `exp_rad == -4`).
  2. Zero-leak `WGSL` Vulkan/D3D12 GPU vs pure-`NumPy` CPU geodesic Störmer-Verlet parity
     (`max|n_gpu - n_cpu| < 1e-4`), isolated `Q_H = 1` Hopfion survival across catenoid expansion
     (`0.88 <= q_net <= 1.08`), and pure `Q_H = 0` long-wavelength pleat standing-wave separation
     (`|q_net| < 0.03`, `f_pleat > 0.85`).
  3. Unbiased 3D Kibble-Zurek throat-scale sweep (`r_0 / ell_S in (1.0, 1.8, 2.578, 4.2)`), proving
     core resolution `min_core_voxels >= 2.5`, defect suppression `Q_eff(4.2) < Q_eff(1.0)`, systematic
     dependence of `xi = sum(E_pleat) / sum(E_knot)` on `r_0 / ell_S` with zero hardcoded `5 + 1/pi`
     target bias, and `< 0.5 ms` warm `get_catenoid_freezeout_crucible_summary` latency.
  4. Topological baryogenesis audit: parity-symmetric CPT bounce (`eta = 0`) yields
     `|<Q_H^net>| << <Q_H^abs>`, whereas CP-violating `SU(2)` Weyl helicity twists (`eta = +0.5` vs `-0.5`)
     generate statistically significant opposite-sign net baryon/Hopf charge `<Q_H^net> != 0`.
"""

from __future__ import annotations

import copy
import gc
import math
import time

import numpy as np
import sympy as sp

from weaveforge.core.catenoid_freezeout import (
    _get_gpu_solver,
    _get_shared_wgpu_device,
    build_localized_hopfion_s2,
    build_pleat_standing_wave_s2,
    compute_forces_s2_plaquette,
    compute_hopf_diagnostics_3d,
    derive_catenoid_throat_freezeout_sympy,
    evolve_catenoid_quench_3d,
    evolve_skyrme_leapfrog_3d,
    get_catenoid_freezeout_crucible_summary,
    run_catenoid_kibble_zurek_ensemble,
)
from weaveforge.core.frameworks import get_mods_by_id


def test_sympy_catenoid_throat_and_adiabatic_freezeout_scaling() -> None:
    deriv = derive_catenoid_throat_freezeout_sympy()
    syms = deriv["symbols"]
    tau = syms["tau"]
    r0 = syms["r0"]
    e_g, e_s, e_v = syms["E_G"], syms["E_S"], syms["E_V"]

    # 1. Non-singular catenoid waist geometry
    assert sp.simplify(deriv["a_tau"] - sp.sqrt(r0**2 + tau**2)) == 0
    assert sp.simplify(deriv["H_tau"] - tau / (r0**2 + tau**2)) == 0
    assert deriv["H_at_throat"] == 0
    assert sp.simplify(deriv["Hdot_at_throat"] - r0 ** (-2)) == 0
    assert sp.simplify(deriv["ricci_flat_at_throat"] - 6 / (r0**2)) == 0
    assert sp.simplify(deriv["kretschmann_flat_at_throat"] - 12 / (r0**4)) == 0
    assert sp.simplify(deriv["kretschmann_s3_at_throat"] - 24 / (r0**4)) == 0
    assert deriv["throat_nonsingular"] is True

    # 2. Exact covariant 3-component Skyrme-Faddeev stress-energy conservation
    assert sp.simplify(deriv["energy_balance_residual"]) == 0
    assert deriv["covariant_energy_conservation_verified"] is True
    assert len(deriv["euler_lagrange_eqs"]) == 3

    # 3. Derrick 3D virial theorem & late-time adiabatic freeze-out dilution exponents
    assert sp.simplify(deriv["derrick_virial_relation"] - (-e_g + e_s - 3 * e_v)) == 0
    assert deriv["w_pleat"] == 0
    assert deriv["w_knot"] == 0
    assert deriv["w_rad"] == sp.Rational(1, 3)
    assert deriv["exp_pleat"] == -3
    assert deriv["exp_knot"] == -3
    assert deriv["exp_rad"] == -4
    assert deriv["late_time_xi_adiabatic_invariant"] is True


def test_gpu_cpu_parity_and_hopfion_vs_pleat_separation() -> None:
    # 1. Variational gradient check on parity-symmetric octant-plaquette Skyrme operator
    n32 = build_localized_hopfion_s2(grid_n=32, dx=0.22, a0=1.10, winding_m=1)
    a_perp, eg0, es0, ep0 = compute_forces_s2_plaquette(
        n32, dx=0.22, a_rel=1.25, c_w=1.0, k_skyrme=1.55, m_eff=0.72, prs_core_preserving=False
    )
    rng = np.random.default_rng(42)
    dn_raw = rng.standard_normal(n32.shape)
    dn_tan = dn_raw - np.sum(dn_raw * n32, axis=0, keepdims=True) * n32
    eps = 1.0e-7
    n_plus = n32 + eps * dn_tan
    n_plus /= np.sqrt(np.sum(n_plus * n_plus, axis=0, keepdims=True))
    n_minus = n32 - eps * dn_tan
    n_minus /= np.sqrt(np.sum(n_minus * n_minus, axis=0, keepdims=True))
    _, egp, esp, epp = compute_forces_s2_plaquette(
        n_plus, dx=0.22, a_rel=1.25, c_w=1.0, k_skyrme=1.55, m_eff=0.72, prs_core_preserving=False
    )
    _, egm, esm, epm = compute_forces_s2_plaquette(
        n_minus, dx=0.22, a_rel=1.25, c_w=1.0, k_skyrme=1.55, m_eff=0.72, prs_core_preserving=False
    )
    de_num = ((egp + esp + epp) - (egm + esm + epm)) / (2.0 * eps)
    de_ana = -float(np.sum(a_perp * dn_tan)) * ((0.22 * 1.25) ** 3)
    assert abs(de_num - de_ana) / max(abs(de_ana), 1.0e-12) < 1.0e-6

    # 2. GPU vs CPU geodesic Störmer-Verlet parity & zero-leak fixed 128-stage bind-group reuse
    v32 = np.zeros_like(n32)
    nc, vc, back_c = evolve_skyrme_leapfrog_3d(
        n32, v32, dx=0.22, dt=0.005, n_steps=35, r0=2.0, c_w=1.0, k_skyrme=1.55, m_eff=0.72,
        prs_core_preserving=True, backend="cpu"
    )
    assert back_c == "cpu"
    assert float(np.max(np.abs(np.sqrt(np.sum(nc * nc, axis=0)) - 1.0))) < 1.0e-6

    if _get_shared_wgpu_device() is not None:
        solver32 = _get_gpu_solver(32)
        assert solver32 is not None
        assert len(solver32.bgs_flux) == 128
        assert len(solver32.bgs_kd) == 128
        bg_ids_before = [id(bg) for bg in solver32.bgs_flux]
        buf_u_id_before = id(solver32.buf_u)

        ng, vg, back_g = evolve_skyrme_leapfrog_3d(
            n32, v32, dx=0.22, dt=0.005, n_steps=35, r0=2.0, c_w=1.0, k_skyrme=1.55, m_eff=0.72,
            prs_core_preserving=True, backend="gpu"
        )
        assert back_g == "gpu"
        assert float(np.max(np.abs(ng - nc))) < 1.0e-4
        assert float(np.max(np.abs(vg - vc))) < 1.0e-4
        assert float(np.max(np.abs(np.sqrt(np.sum(ng * ng, axis=0)) - 1.0))) < 1.0e-5

        # Multi-chunk run (> 128 stages) must reuse the exact same preallocated uniform buffer & bind groups
        _, _, _ = evolve_skyrme_leapfrog_3d(
            n32, v32, dx=0.22, dt=0.004, n_steps=160, r0=2.0, c_w=1.0, k_skyrme=1.55, m_eff=0.72,
            prs_core_preserving=True, backend="gpu"
        )
        assert id(solver32.buf_u) == buf_u_id_before
        assert [id(bg) for bg in solver32.bgs_flux] == bg_ids_before

    # 3. Isolated Q_H = 1 Hopfion survival across catenoid expansion
    n_hopf48 = build_localized_hopfion_s2(grid_n=48, dx=0.135, a0=0.98, winding_m=1)
    hd_init = compute_hopf_diagnostics_3d(n_hopf48, dx_phys=0.135)
    assert 0.95 <= hd_init["q_net"] <= 1.05
    assert hd_init["div_b_rms"] < 0.02

    res_hopf = evolve_catenoid_quench_3d(
        r0_over_ls=2.578,
        grid_n=48,
        dx_com=0.135,
        a_end=1.60,
        c_w=1.0,
        m_eff=0.72,
        k_skyrme=1.55,
        prs_core_preserving=True,
        n_init_override=n_hopf48,
    )
    assert 0.88 <= float(res_hopf["q_net"]) <= 1.08
    assert 0.88 <= float(res_hopf["q_abs"]) <= 1.12
    assert int(res_hopf["n_clusters"]) >= 1
    assert float(res_hopf["div_b_rms"]) < 0.05

    # 4. Pure Q_H = 0 long-wavelength pleat standing wave separation
    n_pleat48 = build_pleat_standing_wave_s2(grid_n=48, dx=0.135, amplitude_rad=0.42, mode_index=1)
    res_pleat = evolve_catenoid_quench_3d(
        r0_over_ls=2.578,
        grid_n=48,
        dx_com=0.135,
        a_end=1.60,
        c_w=1.0,
        m_eff=0.72,
        k_skyrme=1.55,
        prs_core_preserving=True,
        n_init_override=n_pleat48,
    )
    assert abs(float(res_pleat["q_net"])) < 0.03
    assert float(res_pleat["q_abs"]) < 0.03
    assert float(res_pleat["f_pleat"]) > 0.85
    assert float(res_pleat["e_pleat"]) > 5.0 * float(res_pleat["e_rad"])


def test_unbiased_3d_kibble_zurek_ensemble_and_throat_scaling() -> None:
    ens = run_catenoid_kibble_zurek_ensemble(
        r0_over_ls_values=(1.0, 1.8, 2.578, 4.2),
        n_seeds=6,
        grid_n=48,
    )

    # 1. Core resolution & Kibble-Zurek defect suppression
    assert float(ens["min_core_voxels"]) >= 2.5
    q_effs = list(ens["q_eff_by_r0"])
    xis = list(ens["xi_ensemble_by_r0"])
    assert len(q_effs) == 4 and len(xis) == 4
    assert q_effs[-1] < q_effs[0]
    assert float(ens["kz_defect_slope"]) < 0.0

    # 2. Finite positive xi & candid throat-scale dependence (zero hardcoded 5 + 1/pi bias)
    for xi_val in xis:
        assert math.isfinite(xi_val) and xi_val > 0.0
    assert ens["xi_depends_on_throat_scale"] is True
    assert float(ens["kz_xi_slope"]) > 0.25

    xi_target = 5.0 + 1.0 / math.pi
    xi_2578 = float(ens["xi_at_tnap_waist"])
    expected_match = abs(xi_2578 - xi_target) / xi_target <= 0.10
    assert ens["matches_5_plus_inv_pi_within_10pct"] is expected_match
    assert ens["symmetric_bounce_net_baryon_zero"] is True
    assert ens["chiral_twist_generates_net_qh"] is True

    # 3. Crucible summary consistency & < 0.5 ms warm memoization latency
    mods = get_mods_by_id()
    tnap_mod = mods["tnap_weave"]
    summary = get_catenoid_freezeout_crucible_summary(tnap_mod)
    required_keys = {
        "xi_kz_at_mod_waist",
        "xi_target_5_plus_inv_pi",
        "xi_min_sweep",
        "xi_max_sweep",
        "kz_power_index",
        "qh_net_symmetric",
        "qh_net_chiral_pos",
        "matches_5_plus_inv_pi_within_10pct",
        "xi_depends_on_throat_scale",
        "symmetric_bounce_net_baryon_zero",
        "honest_verdict_short",
    }
    assert required_keys.issubset(summary.keys())
    assert abs(float(summary["xi_target_5_plus_inv_pi"]) - xi_target) < 1.0e-12
    assert abs(float(summary["xi_kz_at_mod_waist"]) - xi_2578) / max(xi_2578, 1.0e-6) < 0.02
    assert float(summary["xi_min_sweep"]) <= float(summary["xi_kz_at_mod_waist"]) <= float(summary["xi_max_sweep"])
    assert isinstance(summary["honest_verdict_short"], str) and len(summary["honest_verdict_short"]) > 20

    gc.collect()
    warm_ms: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = get_catenoid_freezeout_crucible_summary(tnap_mod)
        warm_ms.append((time.perf_counter() - t0) * 1000.0)
    assert float(np.median(warm_ms)) < 0.5
    assert max(warm_ms) < 0.5

    # Varying waist_clamping_ell slider reuses the cached 3D action curve instantaneously (< 0.5 ms)
    mod_narrow = copy.deepcopy(tnap_mod)
    mod_narrow.parameters.waist_clamping_ell = 1.60
    t_slider = time.perf_counter()
    sum_narrow = get_catenoid_freezeout_crucible_summary(mod_narrow)
    slider_ms = (time.perf_counter() - t_slider) * 1000.0
    assert slider_ms < 0.5
    assert float(sum_narrow["xi_kz_at_mod_waist"]) < float(summary["xi_kz_at_mod_waist"])

    # Non-Skyrme mod (Lambda-CDM, k_skyrme = 0) reports 0 cleanly without errors
    cdm_sum = get_catenoid_freezeout_crucible_summary(mods["lambda_cdm"])
    assert float(cdm_sum["xi_kz_at_mod_waist"]) == 0.0
    assert cdm_sum["matches_5_plus_inv_pi_within_10pct"] is False


def test_symmetric_vs_chiral_weyl_baryogenesis() -> None:
    n_seeds = 8
    q_net_sym, q_abs_sym = [], []
    q_net_pos, q_net_neg = [], []

    for s_idx in range(n_seeds):
        seed = 3000 + s_idx
        r0 = evolve_catenoid_quench_3d(
            r0_over_ls=1.0, seed=seed, grid_n=48, dx_com=0.135, a_end=1.60, chiral_twist_eta=0.0
        )
        rp = evolve_catenoid_quench_3d(
            r0_over_ls=1.0, seed=seed, grid_n=48, dx_com=0.135, a_end=1.60, chiral_twist_eta=+0.50
        )
        rn = evolve_catenoid_quench_3d(
            r0_over_ls=1.0, seed=seed, grid_n=48, dx_com=0.135, a_end=1.60, chiral_twist_eta=-0.50
        )
        q_net_sym.append(float(r0["q_net"]))
        q_abs_sym.append(float(r0["q_abs"]))
        q_net_pos.append(float(rp["q_net"]))
        q_net_neg.append(float(rn["q_net"]))

    mean_sym = float(np.mean(q_net_sym))
    sem_sym = float(np.std(q_net_sym, ddof=1) / math.sqrt(n_seeds))
    mean_abs_sym = float(np.mean(q_abs_sym))

    mean_pos = float(np.mean(q_net_pos))
    sem_pos = float(np.std(q_net_pos, ddof=1) / math.sqrt(n_seeds))
    mean_neg = float(np.mean(q_net_neg))
    sem_neg = float(np.std(q_net_neg, ddof=1) / math.sqrt(n_seeds))
    t_stat = (mean_pos - mean_neg) / max(math.hypot(sem_pos, sem_neg), 1.0e-9)

    # Parity-symmetric CPT bounce produces |<Q_H^net>| << <Q_H^abs>, statistically consistent with zero
    assert mean_abs_sym > 0.5
    assert abs(mean_sym) < 0.25 * mean_abs_sym
    assert abs(mean_sym) <= 1.5 * sem_sym

    # Opposite-sign chiral SU(2) Weyl helicity twists generate statistically significant opposite-sign <Q_H^net>
    assert mean_pos > 0.25
    assert mean_neg < -0.25
    assert mean_pos * mean_neg < 0.0
    assert t_stat > 3.0


def test_arena4_crucible_kibble_zurek_and_redteam_candor() -> None:
    from weaveforge.arenas.arena4_cosmic_crucible import evaluate_observational_crucible
    from weaveforge.core.symbolic_compiler import compile_master_action

    mods = get_mods_by_id()
    tnap_rep = evaluate_observational_crucible(mods["tnap_weave"], compile_master_action(mods["tnap_weave"]))
    aest_rep = evaluate_observational_crucible(mods["aest"], compile_master_action(mods["aest"]))
    cdm_rep = evaluate_observational_crucible(mods["lambda_cdm"], compile_master_action(mods["lambda_cdm"]))

    assert tnap_rep.cmb_xi_kz_at_waist > 5.5
    assert tnap_rep.cmb_xi_kz_min < 2.0
    assert tnap_rep.cmb_xi_kz_max > 7.0
    assert tnap_rep.cmb_kz_depends_on_r0 is True
    assert tnap_rep.cmb_kz_matches_target_10pct is False
    assert abs(tnap_rep.cmb_qh_net_symmetric) < 0.35
    assert tnap_rep.cmb_qh_net_chiral_pos > 0.40

    b_tnap = {b.badge_id: b for b in tnap_rep.badges}
    assert b_tnap["planck_cmb"].status == "WARN"
    assert "3D KZ quench gives xi=" in b_tnap["planck_cmb"].detail_honest
    assert "not locked to 5+1/pi=5.32" in b_tnap["planck_cmb"].detail_honest
    assert "N_eff=3.303 Silk tilt" in b_tnap["planck_cmb"].detail_honest
    assert "Mpc S_8 & halo-pleat screening" in b_tnap["planck_cmb"].detail_honest
    assert "Open seam: assumes cosmic pleats/dust" in b_tnap["sparc_175"].detail_honest

    b_aest = {b.badge_id: b for b in aest_rep.badges}
    assert b_aest["planck_cmb"].status == "WARN"
    assert "dust-to-MOND nonlinear galaxy screening unclosed" in b_aest["planck_cmb"].detail_honest
    assert cdm_rep.cmb_xi_kz_at_waist == 0.0

