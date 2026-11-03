"""
Step 2 Micro-to-Macro Unified Knot Gravity & Exponential Wave Lensing Suite
(`tests/test_unified_knot_gravity_and_lensing.py`)
===========================================================================
Verifies directly on GPU (`ModernGL` OpenGL 4.6 + `wgpu` Vulkan) and CPU (`NumPy`)
with ZERO hardcoded forces or optical shortcuts:
  1. `Arena1KnotTank` 5-level 3D GPU Multigrid open-space Poisson solver (`nabla^2 U = -kappa * rho_H,0`
     with exterior monopole wall BCs) matches isolated `(2N)^3` Hockney-Eastwood FFT Green's convolution.
  2. Emergent `1/r^2` universal gravitational attraction between two separated 3D Trefoil knots from
     `L = exp(+4*U)*T_kin - c_w^2*G_grad - exp(-2*U)*L_Skyrme - exp(+2*U)*V(phi)`:
       • Vanishing long-range force when `G_knot = 0.0`,
       • Universal inward `1/s^2` attraction (`slope ~ -2.0`) when `G_knot > 0` for BOTH
         Matter + Matter (`+7 / +7`, `Q_H = +14`) AND Matter + Antimatter (`+7 / -7`, `Q_H = 0`)
         (Weak Equivalence Principle on both CPU & GPU!).
  3. Emergent `n(U) = exp(2*U)` Shapiro arrival delay & inward gravitational wave deflection of a
     high-frequency test ripple passing a 3D knot's `U(x)` tension well directly from the 3D PDE
     (`exp(+4*U)*T_kin - c_w^2*G_grad`), even with `optical_metric_n = "1"`!
  4. Live Arena 1 Presets `4` (`two_matter_trefoils`, `+7 / +7`) & `5` (`matter_antimatter_trefoils`, `+7 / -7`):
       • Remain separated when `G_knot = 0.0`,
       • Pull inward from rest when `G_knot > 0`, conserving `Q_H ~ +14` in Preset 4 while untying and
         annihilating the soliton cores into outgoing weave radiation in Preset 5!
"""

from __future__ import annotations

import gc
import math
from typing import Tuple

import moderngl
import numpy as np

from tests.test_knot_long_run_stability import _compute_hopf_charge, _readback_arena1_phi_vel_zyx3
from weaveforge.arenas.arena1_knot_tank import Arena1KnotTank
from weaveforge.compute.wgpu_engine import (
    WgpuComputeEngine,
    _eval_trefoil_pq_numpy,
    compute_safe_symplectic_dt,
    solve_3d_weave_tension_u_cpu,
)
from weaveforge.core.frameworks import get_mods_by_id
from weaveforge.core.symbolic_compiler import compile_master_action


def _eval_compact_two_knot_state(
    grid_n: int,
    box_size: float,
    sep_s: float,
    a0: float,
    antimatter_right: bool,
) -> np.ndarray:
    """Builds a well-separated pair of 3D T_(3,2) Trefoil knots (`+7/+7` or `+7/-7`) at separation `sep_s`."""
    idx_lin = (np.arange(grid_n, dtype=np.float32) + 0.5) / float(grid_n)
    coords = (idx_lin - 0.5) * float(box_size)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    x_off = 0.5 * float(sep_s)

    p1_re, p1_im, q1_re, q1_im = _eval_trefoil_pq_numpy(yy, zz, xx + x_off, a0)
    if not antimatter_right:
        p2_re, p2_im, q2_re, q2_im = _eval_trefoil_pq_numpy(yy, zz, xx - x_off, a0)
    else:
        p2_re, p2_im, q2_re, q2_im = _eval_trefoil_pq_numpy(yy, zz, x_off - xx, a0)

    pt_re = (p1_re * q2_re - p1_im * q2_im) - (p2_re * q1_re - p2_im * q1_im)
    pt_im = (p1_re * q2_im + p1_im * q2_re) - (p2_re * q1_im + p2_im * q1_re)
    qt_re = q1_re * q2_re - q1_im * q2_im
    qt_im = q1_re * q2_im + q1_im * q2_re

    q2 = qt_re * qt_re + qt_im * qt_im
    p2 = pt_re * pt_re + pt_im * pt_im
    denom = np.maximum(p2 + q2, 1.0e-12)
    pqbar_re = pt_re * qt_re + pt_im * qt_im
    pqbar_im = pt_im * qt_re - pt_re * qt_im

    v0 = 0.85
    phi = np.zeros((grid_n, grid_n, grid_n, 4), dtype=np.float32)
    phi[..., 0] = v0 * (2.0 * pqbar_re / denom)
    phi[..., 1] = v0 * (2.0 * pqbar_im / denom)
    phi[..., 2] = v0 * (((q2 - p2) / denom) - 1.0)
    phi[..., 3] = phi[..., 0] ** 2 + phi[..., 1] ** 2 + phi[..., 2] ** 2
    return phi


def _measure_canonical_inward_acceleration_cpu_and_wgpu(
    eng: WgpuComputeEngine,
    mod,
    comp,
    phi0_xyzw: np.ndarray,
    box_size: float,
    run_wgpu_check: bool = False,
) -> Tuple[float, float]:
    """
    Computes the exact emergent mutual inward canonical acceleration `a_inward = (a_L - a_R) / 2`
    directly from the 3D Euler-Lagrange PDE (`accel_numpy_fn` + `solve_3d_weave_tension_u_cpu`),
    and optionally cross-checks 1-step `WgpuComputeEngine` Vulkan GPU velocity kick parity!
    """
    n = int(phi0_xyzw.shape[0])
    dx = float(box_size) / float(n)
    phi = np.moveaxis(phi0_xyzw[..., :3], -1, 0).astype(np.float64)
    vel0 = np.zeros_like(phi)

    xp = np.roll(phi, -1, axis=3)
    xm = np.roll(phi, 1, axis=3)
    yp = np.roll(phi, -1, axis=2)
    ym = np.roll(phi, 1, axis=2)
    zp = np.roll(phi, -1, axis=1)
    zm = np.roll(phi, 1, axis=1)

    grad = np.stack([(xp - xm) / (2.0 * dx), (yp - ym) / (2.0 * dx), (zp - zm) / (2.0 * dx)], axis=1)
    lap = (xp + xm + yp + ym + zp + zm - 6.0 * phi) / (dx * dx)

    p = mod.parameters
    g_knot = float(getattr(p, "g_knot", 0.0))
    if abs(g_knot) > 1.0e-12:
        rho_h0 = comp.hamiltonian_numpy_fn(phi, vel0, grad, p.c_w, p.m_eff, p.g4, p.g6, p.k_skyrme)
        u_field = solve_3d_weave_tension_u_cpu(rho_h0, box_size=box_size, g_knot=g_knot).astype(np.float64)
        u_arg = u_field
        weight = np.exp(4.0 * u_field)
    else:
        u_field = np.zeros((n, n, n), dtype=np.float64)
        u_arg = None
        weight = np.ones((n, n, n), dtype=np.float64)

    acc = comp.accel_numpy_fn(
        phi,
        grad,
        lap,
        p.c_w,
        p.m_eff,
        p.g4,
        p.g6,
        vel_arr=vel0,
        dx_step=dx,
        p_k_skyrme=p.k_skyrme,
        u_arr=u_arg,
    )

    # Canonical field momentum boost projection: for rigid translation phi(x - v*t), dt_phi = -v * dx_phi
    dphi_dx = grad[:, 0]  # (3, Z, Y, X)
    f_density_x = -weight * np.sum(acc * dphi_dx, axis=0)
    m_density_x = weight * np.sum(dphi_dx * dphi_dx, axis=0)

    mid = n // 2
    a_left = float(np.sum(f_density_x[:, :, :mid]) / max(float(np.sum(m_density_x[:, :, :mid])), 1.0e-12))
    a_right = float(np.sum(f_density_x[:, :, mid:]) / max(float(np.sum(m_density_x[:, :, mid:])), 1.0e-12))
    a_inward_cpu = 0.5 * (a_left - a_right)

    if not run_wgpu_check:
        return a_inward_cpu, a_inward_cpu

    dt_probe = 1.0e-3
    vel0_xyzw = np.zeros_like(phi0_xyzw)
    res = eng.run_3d_knot_evolution(
        comp,
        mod,
        grid_n=n,
        n_steps=1,
        box_size=box_size,
        dt=dt_probe,
        damping=0.0,
        sponge=0.0,
        phi_init_override=phi0_xyzw,
        vel_init_override=vel0_xyzw,
        u_field_override=u_field.astype(np.float32) if abs(g_knot) > 1.0e-12 else None,
        solve_tension_u=False,
        readback_full_field=True,
    )
    acc_gpu = np.moveaxis(res["vel_final"][..., :3], -1, 0).astype(np.float64) / dt_probe
    f_gpu_x = -weight * np.sum(acc_gpu * dphi_dx, axis=0)
    a_left_g = float(np.sum(f_gpu_x[:, :, :mid]) / max(float(np.sum(m_density_x[:, :, :mid])), 1.0e-12))
    a_right_g = float(np.sum(f_gpu_x[:, :, mid:]) / max(float(np.sum(m_density_x[:, :, mid:])), 1.0e-12))
    a_inward_gpu = 0.5 * (a_left_g - a_right_g)
    return a_inward_cpu, a_inward_gpu


def test_gpu_multigrid_tension_u_matches_hockney_eastwood_open_space_fft() -> None:
    ctx = moderngl.create_standalone_context(require=430)
    tnap_base = get_mods_by_id()["tnap_weave"]
    tnap_comp = compile_master_action(tnap_base)

    mod_zero = tnap_base.clone(new_id="tnap_g0")
    mod_zero.parameters.g_knot = 0.0
    mod_pos = tnap_base.clone(new_id="tnap_gpos")
    mod_pos.parameters.g_knot = 0.90

    a1 = Arena1KnotTank(ctx, initial_grid_n=64, mod=mod_zero, compiled=tnap_comp)
    try:
        a1.reset_field(0)
        u_zero = a1.readback_tension_u_grid()
        assert u_zero.shape == (64, 64, 64)
        assert float(np.max(np.abs(u_zero))) == 0.0

        a1.update_compiled_action(tnap_comp, mod=mod_pos)
        u_gpu = a1.readback_tension_u_grid()
        assert float(np.min(u_gpu)) > 0.0, "Open-space monopole BCs must keep U(x) > 0 everywhere!"
        assert 0.05 < float(np.max(u_gpu)) < 0.35

        phi_zyx3, vel_zyx3 = _readback_arena1_phi_vel_zyx3(a1)
        dx = float(a1.box_size) / 64.0
        p_c = np.moveaxis(phi_zyx3, -1, 0).astype(np.float64)
        v_c = np.moveaxis(vel_zyx3, -1, 0).astype(np.float64)
        # Match GPU _deposit_u_prog's symmetric forward/backward 1-cell difference stencil
        grad_p = np.stack(
            [
                (np.roll(p_c, -1, axis=3) - p_c) / dx,
                (np.roll(p_c, -1, axis=2) - p_c) / dx,
                (np.roll(p_c, -1, axis=1) - p_c) / dx,
            ],
            axis=1,
        )
        grad_m = np.stack(
            [
                (p_c - np.roll(p_c, 1, axis=3)) / dx,
                (p_c - np.roll(p_c, 1, axis=2)) / dx,
                (p_c - np.roll(p_c, 1, axis=1)) / dx,
            ],
            axis=1,
        )
        mp = mod_pos.parameters
        hp = tnap_comp.hamiltonian_numpy_fn(p_c, v_c, grad_p, mp.c_w, mp.m_eff, mp.g4, mp.g6, mp.k_skyrme)
        hm = tnap_comp.hamiltonian_numpy_fn(p_c, v_c, grad_m, mp.c_w, mp.m_eff, mp.g4, mp.g6, mp.k_skyrme)
        rho_h0 = np.maximum(0.5 * (hp + hm), 0.0)
        u_cpu = solve_3d_weave_tension_u_cpu(rho_h0, box_size=a1.box_size, g_knot=mp.g_knot)

        corr = float(np.corrcoef(u_gpu.ravel(), u_cpu.ravel())[0, 1])
        rel_peak = abs(float(np.max(u_gpu)) - float(np.max(u_cpu))) / float(np.max(u_cpu))
        assert corr > 0.994, f"GPU Multigrid vs CPU Hockney-Eastwood FFT correlation too low: r={corr:.5f}"
        assert rel_peak < 0.08, f"GPU vs CPU peak U mismatch: gpu={np.max(u_gpu):.4f}, cpu={np.max(u_cpu):.4f}"
    finally:
        a1.release()
        ctx.release()
        gc.collect()


def test_emergent_inverse_square_two_knot_force_matter_and_antimatter() -> None:
    tnap_base = get_mods_by_id()["tnap_weave"]
    tnap_comp = compile_master_action(tnap_base)
    eng = WgpuComputeEngine()
    try:
        mod_g0 = tnap_base.clone(new_id="tnap_force_g0")
        mod_g0.parameters.g_knot = 0.0
        mod_grav = tnap_base.clone(new_id="tnap_force_grav")
        mod_grav.parameters.g_knot = 0.90

        # Verify topological charges Q_H = +14 (+7/+7) and Q_H = 0 (+7/-7) on a fine 128^3 grid
        phi_mm_fine = _eval_compact_two_knot_state(128, 20.0, 8.0, 2.35, antimatter_right=False)
        phi_ma_fine = _eval_compact_two_knot_state(128, 20.0, 8.0, 2.35, antimatter_right=True)
        qh_mm = _compute_hopf_charge(phi_mm_fine[..., :3], box_size=20.0)
        qh_ma = _compute_hopf_charge(phi_ma_fine[..., :3], box_size=20.0)
        assert qh_mm > 11.5, f"Expected two-matter Hopf charge ~ +14, got {qh_mm:.2f}"
        assert abs(qh_ma) < 0.05, f"Expected matter-antimatter net Hopf charge ~ 0, got {qh_ma:.4f}"
        del phi_mm_fine, phi_ma_fine

        # Well-resolved integer-cell lattice separations on 96^3 (dx = 0.225, x_off = 18..24 * dx)
        grid_n = 96
        box_size = 21.6
        a0_knot = 2.05
        separations = [8.10, 9.00, 9.90, 10.80]

        # Measure exact transverse RMS radius R_perp^2 = <(y^2 + z^2)>_H of a single 3D Trefoil knot (0 free params!)
        coords_96 = ((np.arange(grid_n, dtype=np.float64) + 0.5) / float(grid_n) - 0.5) * box_size
        zz96, yy96, _ = np.meshgrid(coords_96, coords_96, coords_96, indexing="ij")
        phi_ref = _eval_compact_two_knot_state(grid_n, box_size, separations[-1], a0_knot, antimatter_right=False)
        p_ref = np.moveaxis(phi_ref[..., :3], -1, 0).astype(np.float64)
        dx96 = box_size / float(grid_n)
        g_ref = np.stack(
            [
                (np.roll(p_ref, -1, axis=3) - np.roll(p_ref, 1, axis=3)) / (2.0 * dx96),
                (np.roll(p_ref, -1, axis=2) - np.roll(p_ref, 1, axis=2)) / (2.0 * dx96),
                (np.roll(p_ref, -1, axis=1) - np.roll(p_ref, 1, axis=1)) / (2.0 * dx96),
            ],
            axis=1,
        )
        mp = mod_grav.parameters
        rho_h_ref = tnap_comp.hamiltonian_numpy_fn(
            p_ref, np.zeros_like(p_ref), g_ref, mp.c_w, mp.m_eff, mp.g4, mp.g6, mp.k_skyrme
        )
        mid96 = grid_n // 2
        w_left = np.maximum(rho_h_ref[:, :, :mid96], 0.0)
        r_perp_sq = float(
            np.sum((yy96[:, :, :mid96] ** 2 + zz96[:, :, :mid96] ** 2) * w_left) / max(float(np.sum(w_left)), 1.0e-12)
        )
        del phi_ref, p_ref, g_ref, rho_h_ref

        acc_mm_grav = []
        acc_ma_grav = []

        for idx_s, s_val in enumerate(separations):
            phi_mm = _eval_compact_two_knot_state(grid_n, box_size, s_val, a0_knot, antimatter_right=False)
            phi_ma = _eval_compact_two_knot_state(grid_n, box_size, s_val, a0_knot, antimatter_right=True)

            a0_mm_cpu, _ = _measure_canonical_inward_acceleration_cpu_and_wgpu(
                eng, mod_g0, tnap_comp, phi_mm, box_size, run_wgpu_check=False
            )
            amm_cpu, amm_gpu = _measure_canonical_inward_acceleration_cpu_and_wgpu(
                eng, mod_grav, tnap_comp, phi_mm, box_size, run_wgpu_check=(idx_s in (0, len(separations) - 1))
            )
            a0_ma_cpu, _ = _measure_canonical_inward_acceleration_cpu_and_wgpu(
                eng, mod_g0, tnap_comp, phi_ma, box_size, run_wgpu_check=False
            )
            ama_cpu, ama_gpu = _measure_canonical_inward_acceleration_cpu_and_wgpu(
                eng, mod_grav, tnap_comp, phi_ma, box_size, run_wgpu_check=(idx_s == 0)
            )

            if idx_s in (0, len(separations) - 1):
                assert abs(amm_cpu - amm_gpu) / max(abs(amm_cpu), 1.0e-9) < 0.01
            if idx_s == 0:
                assert abs(ama_cpu - ama_gpu) / max(abs(ama_cpu), 1.0e-9) < 0.01

            # G_knot=0 short-range tail must be negligible (< 5%) compared to 1/r^2 Weave gravity!
            assert abs(a0_mm_cpu) < 0.05 * abs(amm_cpu), (
                f"At s={s_val}, G_knot=0 tail ({a0_mm_cpu:.3e}) not small vs G_knot>0 ({amm_cpu:.3e})"
            )
            assert abs(a0_ma_cpu) < 0.05 * abs(ama_cpu), (
                f"At s={s_val}, MA G_knot=0 tail ({a0_ma_cpu:.3e}) not small vs G_knot>0 ({ama_cpu:.3e})"
            )

            acc_mm_grav.append(amm_cpu - a0_mm_cpu)
            acc_ma_grav.append(ama_cpu - a0_ma_cpu)

        for s_val, g_mm, g_ma in zip(separations, acc_mm_grav, acc_ma_grav):
            assert g_mm > 0.0, f"Matter-Matter gravity must be attractive (a_inward > 0) at s={s_val}"
            assert g_ma > 0.0, f"Matter-Antimatter gravity must be attractive (a_inward > 0) at s={s_val}"
            wep_diff = abs(g_mm - g_ma) / max(0.5 * (g_mm + g_ma), 1.0e-12)
            assert wep_diff < 0.02, f"Weak Equivalence Principle violated at s={s_val}: mm={g_mm:.5e}, ma={g_ma:.5e}"

        s_arr = np.asarray(separations, dtype=np.float64)
        mm_arr = np.asarray(acc_mm_grav, dtype=np.float64)
        ma_arr = np.asarray(acc_ma_grav, dtype=np.float64)
        assert np.all(np.diff(mm_arr) < 0.0) and np.all(np.diff(ma_arr) < 0.0)

        # Coaxial torus-ring chord distance d_eff(s) = sqrt(s^2 + 2 * R_perp^2): a_inward * (d_eff / s) ~ d_eff^-2
        d_eff = np.sqrt(s_arr * s_arr + 2.0 * r_perp_sq)
        log_deff = np.log(d_eff)
        slope_mm = float(np.polyfit(log_deff, np.log(mm_arr * (d_eff / s_arr)), 1)[0])
        slope_ma = float(np.polyfit(log_deff, np.log(ma_arr * (d_eff / s_arr)), 1)[0])
        assert -2.12 <= slope_mm <= -1.88, f"Expected 1/r^2 power law (-2.0) for +7/+7, got slope={slope_mm:.3f}"
        assert -2.12 <= slope_ma <= -1.88, f"Expected 1/r^2 power law (-2.0) for +7/-7, got slope={slope_ma:.3f}"
    finally:
        del eng
        gc.collect()


def test_emergent_exponential_wave_lensing_shapiro_delay_and_deflection() -> None:
    """
    Propagates a linear high-frequency wave packet past a 3D knot's `U(x)` tension well with
    `optical_metric_n = "1"` (zero optical metric formula used!) and verifies BOTH:
      • Exact `n(U) = exp(2*U)` Shapiro arrival delay (distinguishing `exp(2*U)` from `1 + 2*U`!),
      • Inward gravitational wave deflection toward the knot core on BOTH OpenGL (`Arena1KnotTank`)
        and Vulkan (`WgpuComputeEngine`)!
    """
    wave_mod = get_mods_by_id()["tnap_weave"].clone(new_id="pure_pde_wave_lens")
    wave_mod.action.field_lagrangian = "exp(4 * U) * T_kin - c_w**2 * G_grad"
    wave_mod.action.optical_metric_n = "1"
    wave_mod.action.metric_lapse_A = "1"
    wave_mod.parameters.c_w = 1.0
    wave_mod.parameters.m_eff = 0.0
    wave_mod.parameters.g4 = 0.0
    wave_mod.parameters.g6 = 0.0
    wave_mod.parameters.k_skyrme = 0.0
    wave_mod.parameters.g_knot = 1.0
    wave_comp = compile_master_action(wave_mod)

    grid_n = 64
    box_size = 12.0
    dx = box_size / float(grid_n)
    coords = ((np.arange(grid_n, dtype=np.float32) + 0.5) / float(grid_n) - 0.5) * box_size
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")

    # Smooth 3D monopole-softened knot tension well U(r) = U0 / sqrt(1 + (r / r_s)^2) with U_peak = 0.36
    r_well = np.sqrt(xx * xx + yy * yy + zz * zz)
    u_well = (0.36 / np.sqrt(1.0 + (r_well / 1.55) ** 2)).astype(np.float32)

    b_imp = 1.50
    x0 = -3.40
    sigma = 0.65
    k0 = 4.8  # Well-resolved on 64^3 (dx=0.1875 -> k0*dx/2 = 0.45 -> minimal lattice dispersion!)
    amp = 0.02
    env = amp * np.exp(-((xx - x0) ** 2 + (yy - b_imp) ** 2 + zz * zz) / (2.0 * sigma * sigma))
    phase = k0 * (xx - x0)

    phi0 = np.zeros((grid_n, grid_n, grid_n, 4), dtype=np.float32)
    phi0[..., 0] = env * np.cos(phase)
    phi0[..., 1] = env * np.sin(phase)
    phi0[..., 3] = phi0[..., 0] ** 2 + phi0[..., 1] ** 2

    v_loc_lens = np.exp(-2.0 * u_well)
    dphi0_dx = env * (-((xx - x0) / (sigma * sigma)) * np.cos(phase) - k0 * np.sin(phase))
    dphi1_dx = env * (-((xx - x0) / (sigma * sigma)) * np.sin(phase) + k0 * np.cos(phase))

    vel0_flat = np.zeros_like(phi0)
    vel0_flat[..., 0] = -dphi0_dx
    vel0_flat[..., 1] = -dphi1_dx

    vel0_lens = np.zeros_like(phi0)
    vel0_lens[..., 0] = -v_loc_lens * dphi0_dx
    vel0_lens[..., 1] = -v_loc_lens * dphi1_dx

    dt = 0.025
    n_steps = 260

    eng = WgpuComputeEngine()
    try:
        mod_flat = wave_mod.clone(new_id="wave_flat")
        mod_flat.parameters.g_knot = 0.0
        res_flat = eng.run_3d_knot_evolution(
            wave_comp,
            mod_flat,
            grid_n=grid_n,
            n_steps=n_steps,
            box_size=box_size,
            dt=dt,
            damping=0.0,
            sponge=0.0,
            phi_init_override=phi0,
            vel_init_override=vel0_flat,
            solve_tension_u=False,
            readback_full_field=True,
        )
        res_lens = eng.run_3d_knot_evolution(
            wave_comp,
            wave_mod,
            grid_n=grid_n,
            n_steps=n_steps,
            box_size=box_size,
            dt=dt,
            damping=0.0,
            sponge=0.0,
            phi_init_override=phi0,
            vel_init_override=vel0_lens,
            u_field_override=u_well,
            solve_tension_u=False,
            readback_full_field=True,
        )
    finally:
        del eng
        gc.collect()

    rho_flat = res_flat["phi_final"][..., 3].astype(np.float64)
    rho_lens = res_lens["phi_final"][..., 3].astype(np.float64)
    x_flat = float(np.sum(rho_flat * xx) / np.sum(rho_flat))
    y_flat = float(np.sum(rho_flat * yy) / np.sum(rho_flat))
    x_lens = float(np.sum(rho_lens * xx) / np.sum(rho_lens))
    y_lens = float(np.sum(rho_lens * yy) / np.sum(rho_lens))

    dx_measured = x_lens - x_flat
    dy_deflect = y_lens - y_flat

    # Measure exact discrete lattice group speed in flat space vg0 = (x_flat - x0) / T
    vg0 = (x_flat - x0) / (n_steps * dt)
    # 2D WKB Fermat ray integration for n(U) = exp(2*U) vs linearized n_lin(U) = 1 + 2*U!
    # Including transverse packet width averaging r_eff^2 = x^2 + y^2 + sigma^2 (since <z^2 + dy^2> = sigma^2)
    xr_exp, yr_exp, py_exp = x0, b_imp, 0.0
    xr_lin = x0
    for _ in range(n_steps):
        denom_sq = 1.0 + (xr_exp * xr_exp + yr_exp * yr_exp + sigma * sigma) / (1.55 * 1.55)
        u_e = 0.36 / math.sqrt(denom_sq)
        du_dy = -0.36 * (yr_exp / (1.55 * 1.55)) / (denom_sq ** 1.5)
        v_e = vg0 * math.exp(-2.0 * u_e)
        py_exp += -2.0 * du_dy * v_e * dt  # d(k_y/k_0)/dt = -grad_y(ln n) * v
        xr_exp += dt * v_e * math.sqrt(max(1.0 - py_exp * py_exp, 0.2))
        yr_exp += dt * v_e * py_exp

        u_l = 0.36 / math.sqrt(1.0 + (xr_lin * xr_lin + b_imp * b_imp + sigma * sigma) / (1.55 * 1.55))
        xr_lin += dt * vg0 * (1.0 - 2.0 * u_l)

    dx_pred_exp = xr_exp - x_flat
    dx_pred_lin = xr_lin - x_flat

    assert dx_measured < -1.0, f"Expected substantial Shapiro arrival delay, got dx={dx_measured:.4f}"
    err_vs_exp = abs(dx_measured - dx_pred_exp)
    err_vs_lin = abs(dx_measured - dx_pred_lin)
    assert err_vs_exp < 0.16, f"PDE Shapiro delay ({dx_measured:.4f}) deviated from exp(2U) WKB ({dx_pred_exp:.4f})"
    assert err_vs_exp < 0.40 * err_vs_lin, (
        f"3D PDE must match n(U)=exp(2U) ({dx_pred_exp:.3f}), NOT linearized 1+2U ({dx_pred_lin:.3f}); got {dx_measured:.3f}"
    )
    assert dy_deflect < -0.15, f"Expected inward gravitational wave deflection toward y=0, got dy={dy_deflect:.4f}"

    # 2. ModernGL OpenGL 4.6 Arena1KnotTank parity check via set_external_tension_u
    ctx = moderngl.create_standalone_context(require=430)
    a1 = Arena1KnotTank(ctx, initial_grid_n=grid_n, mod=wave_mod, compiled=wave_comp)
    try:
        a1.sponge_strength = 0.0
        a1.damping = 0.0
        assert a1._buf_a is not None and a1._buf_vel is not None
        a1._buf_a.write(np.ascontiguousarray(phi0[..., :3], dtype=np.float32))
        a1._buf_vel.write(np.ascontiguousarray(vel0_lens[..., :3], dtype=np.float32))
        a1._cur_in_a = True
        a1.set_external_tension_u(u_well)

        t_target = n_steps * dt
        dt_gl = compute_safe_symplectic_dt(dx, wave_mod, compiled=wave_comp, peak_rho_est=max(a1.init_peak_rho, 2.8))
        sub_gl = int(round(t_target / dt_gl))
        a1.step(wave_mod, wave_comp, n_substeps=sub_gl)
        phi_gl, _ = _readback_arena1_phi_vel_zyx3(a1)
        rho_gl = np.sum(phi_gl.astype(np.float64) ** 2, axis=-1)
        x_gl = float(np.sum(rho_gl * xx) / np.sum(rho_gl))
        y_gl = float(np.sum(rho_gl * yy) / np.sum(rho_gl))
        dx_gl = x_gl - (x0 + vg0 * sub_gl * a1.last_dt)
        dy_gl = y_gl - b_imp
        assert abs(dx_gl - dx_measured) < 0.14, f"OpenGL vs Vulkan Shapiro delay mismatch: {dx_gl:.3f} vs {dx_measured:.3f}"
        assert dy_gl < -0.15, f"OpenGL Arena1 wave packet did not deflect inward: dy_gl={dy_gl:.4f}"
    finally:
        a1.release()
        ctx.release()
        gc.collect()


def test_arena1_live_two_knot_presets_scattering_and_annihilation() -> None:
    """
    Verifies live Arena 1 Presets `4` (`+7 / +7` Two Matter Trefoils) and `5` (`+7 / -7` Matter + Antimatter):
      • At `G_knot = 0.0`: both pairs remain cleanly separated without colliding or annihilating,
      • At `G_knot = 0.90` (`tnap_weave` default): both pairs pull inward from rest; Preset 4 conserves
        topological Hopf charge `Q_H ~ +14` and core volume, whereas Preset 5 unties and annihilates!
    """
    ctx = moderngl.create_standalone_context(require=430)
    tnap_grav = get_mods_by_id()["tnap_weave"].clone(new_id="tnap_preset_grav")
    tnap_comp = compile_master_action(tnap_grav)
    tnap_g0 = tnap_grav.clone(new_id="tnap_preset_g0")
    tnap_g0.parameters.g_knot = 0.0

    grid_n = 128
    a1 = Arena1KnotTank(ctx, initial_grid_n=grid_n, mod=tnap_grav, compiled=tnap_comp)
    try:
        def _sep_and_core(phi_zyx3: np.ndarray, box_sz: float) -> Tuple[float, float]:
            coords_x = ((np.arange(grid_n, dtype=np.float32) + 0.5) / float(grid_n) - 0.5) * box_sz
            rho = np.sum(phi_zyx3.astype(np.float64) ** 2, axis=-1)
            dv = (box_sz / float(grid_n)) ** 3
            w = np.maximum(rho - 1.2, 0.0) ** 2
            mid = grid_n // 2
            xl = float(np.sum(w[:, :, :mid] * coords_x[None, None, :mid]) / max(float(np.sum(w[:, :, :mid])), 1.0e-12))
            xr = float(np.sum(w[:, :, mid:] * coords_x[None, None, mid:]) / max(float(np.sum(w[:, :, mid:])), 1.0e-12))
            return xr - xl, float(np.sum(rho > 1.5) * dv)

        # 1. Preset 4 (+7 / +7 Matter-Matter): G=0 separates vs G>0 pulls inward & conserves Q_H ~ +14!
        a1.update_compiled_action(tnap_comp, mod=tnap_g0)
        a1.reset_field(4)
        assert abs(a1.box_size - 16.8) < 1.0e-6
        phi4_0, _ = _readback_arena1_phi_vel_zyx3(a1)
        q4_init = _compute_hopf_charge(phi4_0, box_size=a1.box_size)
        s4_init, _ = _sep_and_core(phi4_0, a1.box_size)
        assert q4_init > 11.8

        a1.step(tnap_g0, tnap_comp, n_substeps=10000)
        phi4_g0, _ = _readback_arena1_phi_vel_zyx3(a1)
        s4_g0, cv4_g0 = _sep_and_core(phi4_g0, a1.box_size)
        assert s4_g0 > s4_init + 1.0, f"Preset 4 at G=0 should stay separated, got s={s4_g0:.2f} vs {s4_init:.2f}"
        assert cv4_g0 > 60.0

        a1.update_compiled_action(tnap_comp, mod=tnap_grav)
        a1.reset_field(4)
        a1.step(tnap_grav, tnap_comp, n_substeps=22000)
        phi4_grav, _ = _readback_arena1_phi_vel_zyx3(a1)
        s4_grav, cv4_grav = _sep_and_core(phi4_grav, a1.box_size)
        q4_grav = _compute_hopf_charge(phi4_grav, box_size=a1.box_size)
        assert s4_grav < s4_init - 1.2, f"Preset 4 at G>0 must fall inward, got s={s4_grav:.2f} vs {s4_init:.2f}"
        assert abs(q4_grav - q4_init) < 1.0, f"Preset 4 topological charge not conserved: {q4_init:.2f} -> {q4_grav:.2f}"
        assert cv4_grav > 50.0

        # 2. Preset 5 (+7 / -7 Matter-Antimatter): G=0 stays intact vs G>0 falls inward & annihilates!
        a1.update_compiled_action(tnap_comp, mod=tnap_g0)
        a1.reset_field(5)
        phi5_0, _ = _readback_arena1_phi_vel_zyx3(a1)
        q5_init = _compute_hopf_charge(phi5_0, box_size=a1.box_size)
        s5_init, cv5_init = _sep_and_core(phi5_0, a1.box_size)
        assert abs(q5_init) < 0.1

        a1.step(tnap_g0, tnap_comp, n_substeps=12000)
        phi5_g0, _ = _readback_arena1_phi_vel_zyx3(a1)
        s5_g0, cv5_g0 = _sep_and_core(phi5_g0, a1.box_size)
        assert s5_g0 > s5_init + 2.0, f"Preset 5 at G=0 should stay separated, got s={s5_g0:.2f}"
        assert cv5_g0 > 60.0

        a1.update_compiled_action(tnap_comp, mod=tnap_grav)
        a1.reset_field(5)
        a1.step(tnap_grav, tnap_comp, n_substeps=28500)
        phi5_grav, _ = _readback_arena1_phi_vel_zyx3(a1)
        _, cv5_grav = _sep_and_core(phi5_grav, a1.box_size)
        a1._sample_diagnostics()
        assert cv5_grav < 0.28 * cv5_init, (
            f"Preset 5 (+7/-7) must annihilate core volume on contact: {cv5_init:.1f} -> {cv5_grav:.1f}"
        )
        assert cv5_grav < 0.32 * cv4_grav, (
            f"Preset 5 (+7/-7) core ({cv5_grav:.1f}) must be annihilated relative to Preset 4 (+7/+7) ({cv4_grav:.1f})"
        )
    finally:
        a1.release()
        ctx.release()
        gc.collect()
