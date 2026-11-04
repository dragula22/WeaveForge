"""
Phase 5 Milestone 5A Automated Proof Suite (`tests/test_phase5_bssn_backreaction.py`)
=====================================================================================
Verifies:
  1. `test_sympy_adm_lichnerowicz_and_weave_vs_gr_throat`:
       • 3+1D ADM Gauss–Codazzi Hamiltonian & Momentum constraint identities (`residual == 0`),
       • Conformal Ricci identity `R == -8 * psi^-5 * nabla^2 psi` & non-linear Lichnerowicz PDE,
       • Isotropic vacuum Schwarzschild `psi_GR = 1 + M / (2r)` zero residual & `R_areal(M/2) = 2M`,
       • Exponential Weave `psi_weave = exp(M / (2r))` 1PN match, `+M^2 / (8 r^2)` 2PN shift,
         horizonless minimal throat `R_min = e * M` at `r = M`, and BSSNok unimodular/traceless splits.
  2. `test_3d_nonlinear_lichnerowicz_solver_and_binding_energy`:
       • `||H_ADM||_L2 < 1e-5`, positive binding energy `M_ADM < M_proper`, non-linear `psi > psi_linear`,
         and self-consistent monopole outer boundary `psi|_{dOmega} = 1 + G * M_ADM / (2r)`.
  3. `test_3d_bssn_minkowski_stability_and_gpu_cpu_parity`:
       • Flat-space `amplitude = 0` Minkowski stability (`|alpha - 1|, |chi - 1|, |K| < 1e-6`),
       • `WGSL` Vulkan/D3D12 GPU vs `NumPy` CPU 3+1D BSSNok parity (`max diff < 5e-4`),
       • Algebraic constraints (`|det(gamma_tilde) - 1| < 1e-5`, `|tr(A_tilde)| < 1e-5`),
       • Zero `create_buffer` / `_create_buffer` / `create_bind_group` allocations across repeated `evolve` calls.
  4. `test_strong_field_bssn_lapse_collapse_and_l2_gw_extraction`:
       • Strong-field quadrupolar 3D soliton (`U_max > 0.20`), genuine dynamical `1+log` lapse collapse
         (`alpha_min < 0.72`), non-perturbative deviation from linearized gravity, bounded Hamiltonian
         constraint norm, non-zero `l = 2` tensor GW strain `h_+^{l=2}(t)`, and `< 0.5 ms` warm summary.
"""

from __future__ import annotations

import gc
import math
import time
from unittest.mock import patch

import numpy as np
import sympy as sp

from weaveforge.core.bssn_solver import (
    _get_gpu_solver,
    _get_shared_wgpu_device,
    build_3d_soliton_matter_state,
    derive_adm_bssn_constraints_sympy,
    evolve_3d_bssn_matter_backreaction,
    get_bssn_strong_field_summary,
    solve_lichnerowicz_initial_data_3d,
)


def test_sympy_adm_lichnerowicz_and_weave_vs_gr_throat() -> None:
    deriv = derive_adm_bssn_constraints_sympy()
    syms = deriv["symbols"]
    r = syms["r"]
    M = syms["M"]
    G = syms["G"]
    psi_s = syms["psi_s"]

    # 1. Exact 3+1D Gauss-Codazzi ADM Hamiltonian & Momentum identities
    assert deriv["gauss_hamiltonian_identity_residual"] == 0
    assert deriv["codazzi_momentum_identity_residual"] == 0
    assert deriv["ricci_conformal_identity_residual"] == 0
    assert deriv["lichnerowicz_from_adm_residual"] == 0

    # 2. Isotropic vacuum Schwarzschild psi_GR = 1 + M / (2r) & R_areal(M/2) = 2M
    assert sp.simplify(deriv["psi_gr"] - (1 + M / (2 * r))) == 0
    assert deriv["schwarzschild_laplacian_radial_residual"] == 0
    assert deriv["schwarzschild_laplacian_cartesian_residual"] == 0
    assert deriv["schwarzschild_hamiltonian_residual"] == 0
    assert sp.simplify(deriv["schwarzschild_horizon_isotropic_r"] - M / 2) == 0
    assert sp.simplify(deriv["schwarzschild_horizon_areal_radius"] - 2 * M) == 0
    assert deriv["schwarzschild_horizon_dR_dr"] == 0
    assert deriv["schwarzschild_horizon_lapse"] == 0

    # 3. Exponential Weave psi_weave = exp(M / (2r)): 1PN match, +M^2/(8r^2) 2PN shift, R_min = e*M throat
    assert sp.simplify(deriv["psi_weave"] - sp.exp(M / (2 * r))) == 0
    assert deriv["weave_minus_gr_1pn_coeff"] == 0
    assert deriv["weave_minus_gr_2pn_coeff"] == sp.Rational(1, 8)
    assert sp.simplify(deriv["weave_minus_gr_2pn_term"] - M**2 / (8 * r**2)) == 0
    assert sp.simplify(deriv["weave_throat_isotropic_r"] - M) == 0
    assert sp.simplify(deriv["weave_throat_areal_radius"] - sp.E * M) == 0
    assert sp.simplify(deriv["weave_throat_lapse"] - sp.exp(-1)) == 0
    assert deriv["weave_horizonless"] is True

    expected_rho_eff = -(M**2) * sp.exp(-2 * M / r) / (8 * sp.pi * G * r**4)
    assert sp.simplify(deriv["rho_eff_weave"] - expected_rho_eff) == 0

    # 4. BSSNok unimodular, traceless, full Ricci, 1+log Bona-Masso, and Gamma-driver identities
    assert sp.simplify(deriv["chi_from_psi"] - psi_s ** (-4)) == 0
    assert deriv["det_gamma_tilde_residual"] == 0
    assert deriv["trace_a_tilde_residual"] == 0
    assert deriv["bssn_ricci_00_residual"] == 0


def test_3d_nonlinear_lichnerowicz_solver_and_binding_energy() -> None:
    sol = build_3d_soliton_matter_state(
        grid_n=32, dx=0.35, amplitude=0.85, core_radius=1.35, m_scalar=0.8, l2_quadrupole_eps=0.18
    )
    lich = solve_lichnerowicz_initial_data_3d(
        sol["rho_matter"], dx=0.35, g_newton=1.0, tol=1.0e-8, max_iters=100
    )

    assert float(lich["hamiltonian_l2_residual"]) < 1.0e-5
    assert float(lich["adm_hamiltonian_l2_norm"]) < 1.0e-5
    assert float(lich["m_adm"]) > 0.0
    assert float(lich["m_proper"]) > float(lich["m_adm"])
    assert float(lich["binding_energy"]) > 1.0e-3
    assert float(lich["nonlinear_psi_excess"]) > 1.0e-2
    assert float(lich["u_max"]) > 0.25

    psi = np.asarray(lich["psi"], dtype=np.float64)
    psi_lin = np.asarray(lich["psi_linear"], dtype=np.float64)
    rr = sol["r"]
    m_adm = float(lich["m_adm"])

    # Peak conformal factor in the strong-field core strictly exceeds linearized Poisson gravity
    assert float(np.max(psi)) > float(np.max(psi_lin)) + 1.0e-2

    # Self-consistent asymptotic monopole boundary psi|_{dOmega} = 1 + G * M_ADM / (2 * r)
    bdry_expected = 1.0 + 0.5 * m_adm / rr[0, :, :]
    assert float(np.max(np.abs(psi[0, :, :] - bdry_expected))) < 1.0e-12


def test_3d_bssn_minkowski_stability_and_gpu_cpu_parity() -> None:
    # 1. Flat-space Minkowski vacuum stability (amplitude = 0)
    mink_cpu = evolve_3d_bssn_matter_backreaction(
        grid_n=24, dx=0.35, n_steps=15, dt=0.045, amplitude=0.0, backend="cpu"
    )
    assert mink_cpu["backend_used"] == "cpu"
    assert float(np.max(np.abs(np.asarray(mink_cpu["alpha"]) - 1.0))) < 1.0e-6
    assert float(np.max(np.abs(np.asarray(mink_cpu["chi"]) - 1.0))) < 1.0e-6
    assert float(np.max(np.abs(np.asarray(mink_cpu["K"])))) < 1.0e-6
    assert float(mink_cpu["det_gamma_max_err"]) < 1.0e-5
    assert float(mink_cpu["tr_a_max_err"]) < 1.0e-5
    assert float(mink_cpu["gw_h_plus_l2_peak"]) < 1.0e-12

    # 2. Non-trivial curved-space 3+1D BSSNok GPU vs CPU parity & zero descriptor allocation
    ev_cpu = evolve_3d_bssn_matter_backreaction(
        grid_n=28, dx=0.35, n_steps=20, dt=0.045, amplitude=0.85, l2_quadrupole_eps=0.18, backend="cpu"
    )
    assert float(ev_cpu["det_gamma_max_err"]) < 1.0e-5
    assert float(ev_cpu["tr_a_max_err"]) < 1.0e-5

    dev = _get_shared_wgpu_device()
    if dev is not None:
        mink_gpu = evolve_3d_bssn_matter_backreaction(
            grid_n=24, dx=0.35, n_steps=15, dt=0.045, amplitude=0.0, backend="gpu"
        )
        assert mink_gpu["backend_used"] == "gpu"
        assert float(np.max(np.abs(np.asarray(mink_gpu["alpha"]) - 1.0))) < 1.0e-6
        assert float(np.max(np.abs(np.asarray(mink_gpu["chi"]) - 1.0))) < 1.0e-6
        assert float(np.max(np.abs(np.asarray(mink_gpu["K"])))) < 1.0e-6

        solver28 = _get_gpu_solver(28)
        assert solver28 is not None
        bg_rhs_ids = [id(bg) for bg in solver28.bgs_rhs]
        bg_rk4_ids = [id(bg) for bg in solver28.bgs_rk4]
        buf_ids = (
            id(solver28.buf_y0),
            id(solver28.buf_ys),
            id(solver28.buf_yacc),
            id(solver28.buf_rhs),
            id(solver28.buf_ham),
            id(solver28.buf_u),
            id(solver28.buf_rb_state),
            id(solver28.buf_rb_ham),
        )

        with (
            patch.object(dev, "create_buffer", wraps=dev.create_buffer) as spy_cb,
            patch.object(dev, "_create_buffer", wraps=dev._create_buffer) as spy_icb,
            patch.object(dev, "create_bind_group", wraps=dev.create_bind_group) as spy_cbg,
        ):
            ev_gpu = evolve_3d_bssn_matter_backreaction(
                grid_n=28, dx=0.35, n_steps=20, dt=0.045, amplitude=0.85, l2_quadrupole_eps=0.18, backend="gpu"
            )
            ev_gpu_repeat = evolve_3d_bssn_matter_backreaction(
                grid_n=28, dx=0.35, n_steps=25, dt=0.040, amplitude=0.75, l2_quadrupole_eps=0.12, backend="gpu"
            )
            assert spy_cb.call_count == 0
            assert spy_icb.call_count == 0
            assert spy_cbg.call_count == 0

        assert ev_gpu["backend_used"] == "gpu"
        assert ev_gpu_repeat["backend_used"] == "gpu"
        assert [id(bg) for bg in solver28.bgs_rhs] == bg_rhs_ids
        assert [id(bg) for bg in solver28.bgs_rk4] == bg_rk4_ids
        assert (
            id(solver28.buf_y0),
            id(solver28.buf_ys),
            id(solver28.buf_yacc),
            id(solver28.buf_rhs),
            id(solver28.buf_ham),
            id(solver28.buf_u),
            id(solver28.buf_rb_state),
            id(solver28.buf_rb_ham),
        ) == buf_ids

        assert float(ev_gpu["det_gamma_max_err"]) < 1.0e-5
        assert float(ev_gpu["tr_a_max_err"]) < 1.0e-5
        assert float(np.max(np.abs(np.asarray(ev_gpu["alpha"]) - np.asarray(ev_cpu["alpha"])))) < 5.0e-4
        assert float(np.max(np.abs(np.asarray(ev_gpu["chi"]) - np.asarray(ev_cpu["chi"])))) < 5.0e-4
        assert float(np.max(np.abs(np.asarray(ev_gpu["A_xx"]) - np.asarray(ev_cpu["A_xx"])))) < 5.0e-4


def test_strong_field_bssn_lapse_collapse_and_l2_gw_extraction() -> None:
    ev = evolve_3d_bssn_matter_backreaction(
        grid_n=32,
        dx=0.35,
        n_steps=40,
        dt=0.045,
        g_newton=1.0,
        amplitude=0.85,
        m_scalar=0.8,
        l2_quadrupole_eps=0.18,
        backend="auto",
    )

    # 1. Strong initial compactness U_max > 0.20 & genuine dynamical 1+log lapse collapse < 0.72
    assert float(ev["u_max_init"]) > 0.20
    alpha_init_min = float(np.min(np.asarray(ev["initial_data"]["alpha_init"])))  # type: ignore[index]
    alpha_min = float(ev["alpha_min"])
    assert alpha_min < 0.72
    assert alpha_min < alpha_init_min - 0.15

    # 2. Measurable non-perturbative deviation from linearized gravity & bounded constraints
    assert float(ev["nonlinear_lapse_deviation"]) > 0.05
    ham_hist = [float(v) for v in ev["hamiltonian_rms_history"]]  # type: ignore[union-attr]
    assert len(ham_hist) == 40
    assert all(math.isfinite(v) and v < 0.05 for v in ham_hist)
    assert float(ev["det_gamma_max_err"]) < 1.0e-5
    assert float(ev["tr_a_max_err"]) < 1.0e-5

    # 3. Non-zero l = 2 transverse-traceless tensor GW strain vs spherical Birkhoff control (eps2 = 0)
    gw_hist = [float(v) for v in ev["gw_h_plus_l2_history"]]  # type: ignore[union-attr]
    assert len(gw_hist) == 40
    assert float(ev["gw_h_plus_l2_peak"]) > 5.0e-6

    ev_spherical = evolve_3d_bssn_matter_backreaction(
        grid_n=32,
        dx=0.35,
        n_steps=40,
        dt=0.045,
        g_newton=1.0,
        amplitude=0.85,
        m_scalar=0.8,
        l2_quadrupole_eps=0.0,
        backend="auto",
    )
    assert float(ev_spherical["gw_h_plus_l2_peak"]) < 1.0e-7
    assert float(ev["gw_h_plus_l2_peak"]) > 50.0 * max(float(ev_spherical["gw_h_plus_l2_peak"]), 1.0e-12)

    # 4. Memoized strong-field summary (< 0.5 ms warm)
    summary = get_bssn_strong_field_summary()
    assert float(summary["binding_energy"]) > 0.0
    assert float(summary["alpha_collapsed_min"]) < 0.72
    assert abs(float(summary["weave_throat_areal_radius_over_m"]) - math.e) < 1.0e-9
    assert abs(float(summary["gr_horizon_areal_radius_over_m"]) - 2.0) < 1.0e-12

    gc.collect()
    warm_ms: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = get_bssn_strong_field_summary()
        warm_ms.append((time.perf_counter() - t0) * 1000.0)
    assert float(np.median(warm_ms)) < 0.5
    assert max(warm_ms) < 0.5
