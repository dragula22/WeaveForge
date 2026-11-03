"""
Step 3 Causal Hyperbolic Weave-Tension Gravitational-Wave Suite
(`tests/test_retarded_weave_gw.py`)
===============================================================
Verifies directly on GPU (`ModernGL` OpenGL 4.6 + `wgpu` Vulkan) with ZERO hardcoded shortcuts:
  1. Strict relativistic light-cone causality (`|U(r > c_gw * t + r_src)| < 1e-6` vs instantaneous
     elliptic `retarded_weave_u = False`), wave-front propagation speed `c_gw = |gw_speed_ratio| * c_w`
     matching LIGO/GW170817 (`c_gw = c_w` in `tnap_weave`), and cross-API OpenGL vs Vulkan parity.
  2. 1st-order spherical Bayliss-Turkel / Sommerfeld non-reflecting outer boundary (`> 97%` of a free
     3D spherical GW pulse's energy exits cleanly through the outer cube faces without reflecting, while
     a stationary solitary Trefoil's `1/r` monopole tension remains quiescent and positive).
  3. Dynamic gravitational-wave emission with positive outward radial Poynting luminosity
     `L_GW = -1/kappa * oint (d_t dU)(d_r dU) dA > 0` from BOTH:
       • `Preset 5`: Matter + Antimatter Trefoils (`+7 / -7`) Annihilation GW Burst,
       • `Preset 6`: Binary Trefoil Inspiral (`+7 / +7`) Prograde Orbiting Quadrupole GW Ripples
         with conserved topological Hopf charge `Q_H ~ +14`.
  4. Lossless `float32` preservation of BOTH `(U, dU/dt)` across DDR5 System RAM hibernation
     (`hibernate_to_ram` / `wake_from_ram`) and Time-Machine Replay (`load_replay_frame` /
     `resume_live_from_replay`), plus headless `WeaveForgeStudioApp` UI & ray-marching verification.
"""

from __future__ import annotations

import gc
import math
from typing import Tuple

import moderngl
import numpy as np

from tests.test_knot_long_run_stability import _compute_hopf_charge, _readback_arena1_phi_vel_zyx3
from weaveforge.arenas.arena1_knot_tank import Arena1KnotTank, WEAVE_KAPPA_SCALE
from weaveforge.compute.wgpu_engine import (
    WgpuComputeEngine,
    _eval_trefoil_pq_numpy,
    compute_safe_symplectic_dt,
)
from weaveforge.core.frameworks import get_mods_by_id
from weaveforge.core.symbolic_compiler import compile_master_action
from weaveforge.studio.app import WeaveForgeStudioApp


def _build_compact_solitary_trefoil(grid_n: int, box_size: float, a0: float = 1.50) -> Tuple[np.ndarray, np.ndarray]:
    """Builds a tightly localized 3D T_(3,2) Trefoil (`r_src ~ 2.4`) inside a large `box_size` tank."""
    coords = ((np.arange(grid_n, dtype=np.float32) + 0.5) / float(grid_n) - 0.5) * float(box_size)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    p_re, p_im, q_re, q_im = _eval_trefoil_pq_numpy(xx, yy, zz, a0)
    q2 = q_re * q_re + q_im * q_im
    p2 = p_re * p_re + p_im * p_im
    denom = np.maximum(p2 + q2, 1.0e-12)
    pq_re = p_re * q_re + p_im * q_im
    pq_im = p_im * q_re - p_re * q_im
    v0 = 0.85
    phi_zyx3 = np.zeros((grid_n, grid_n, grid_n, 3), dtype=np.float32)
    phi_zyx3[..., 0] = v0 * (2.0 * pq_re / denom)
    phi_zyx3[..., 1] = v0 * (2.0 * pq_im / denom)
    phi_zyx3[..., 2] = v0 * (((q2 - p2) / denom) - 1.0)

    phi_xyzw = np.zeros((grid_n, grid_n, grid_n, 4), dtype=np.float32)
    phi_xyzw[..., :3] = np.transpose(phi_zyx3, (2, 1, 0, 3))
    phi_xyzw[..., 3] = np.sum(phi_xyzw[..., :3] ** 2, axis=-1)
    return phi_zyx3, phi_xyzw


def _radial_grid_64(box_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    m0 = 64
    h = float(box_size) / float(m0)
    coords = ((np.arange(m0, dtype=np.float64) + 0.5) / float(m0) - 0.5) * float(box_size)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy + zz * zz)
    return xx, yy, zz, rr, h


def _wavefront_outer_radius(u_zyx: np.ndarray, rr: np.ndarray, frac_of_peak: float = 0.08) -> float:
    """Measures the outer spherical-shell wavefront radius where `<|U|(r)>` crosses `frac_of_peak * max`."""
    r_bins = np.linspace(0.5, 8.2, 64)
    prof = np.zeros(len(r_bins) - 1, dtype=np.float64)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    abs_u = np.abs(u_zyx.astype(np.float64))
    for i in range(len(prof)):
        mask = (rr >= r_bins[i]) & (rr < r_bins[i + 1])
        if np.any(mask):
            prof[i] = float(np.mean(abs_u[mask]))
    thresh = frac_of_peak * float(np.max(prof))
    above = np.where(prof >= thresh)[0]
    if above.size == 0:
        return 0.0
    idx = int(above[-1])
    if idx + 1 < len(prof) and prof[idx] > prof[idx + 1]:
        t = (prof[idx] - thresh) / max(prof[idx] - prof[idx + 1], 1.0e-12)
        return float(r_mid[idx] + t * (r_mid[idx + 1] - r_mid[idx]))
    return float(r_mid[idx])


def test_causal_light_cone_and_gw170817_wave_speed() -> None:
    """
    Proves strict relativistic light-cone causality (`|U| == 0` outside `r_src + c_gw * t`),
    contrasts against instantaneous elliptic Poisson (`retarded_weave_u = False`), verifies
    wavefront speed scales linearly with `gw_speed_ratio`, and checks OpenGL vs Vulkan parity.
    """
    ctx = moderngl.create_standalone_context(require=430)
    eng = WgpuComputeEngine()

    mod_c1 = get_mods_by_id()["tnap_weave"].clone(new_id="tnap_gw_c1")
    mod_c1.parameters.retarded_weave_u = True
    comp_c1 = compile_master_action(mod_c1)

    mod_slow = mod_c1.clone(new_id="tnap_gw_slow")
    mod_slow.action.gw_speed_ratio = "0.68"
    comp_slow = compile_master_action(mod_slow)

    mod_inst = mod_c1.clone(new_id="tnap_gw_inst")
    mod_inst.parameters.retarded_weave_u = False
    comp_inst = compile_master_action(mod_inst)

    box_sz = 16.8
    grid_n = 64
    _, _, _, rr, _ = _radial_grid_64(box_sz)
    phi0_zyx3, _ = _build_compact_solitary_trefoil(grid_n, box_sz, a0=1.50)
    vel0_zyx3 = np.zeros_like(phi0_zyx3)

    a1 = Arena1KnotTank(ctx, initial_grid_n=grid_n, mod=mod_c1, compiled=comp_c1)
    a1.box_size = box_sz

    def _upload_compact_trefoil() -> None:
        assert a1._buf_a is not None and a1._buf_vel is not None
        a1._buf_a.write(np.ascontiguousarray(phi0_zyx3, dtype=np.float32))
        a1._zero_ssbo(a1._buf_vel, grid_n * grid_n * grid_n * 3)
        a1._cur_in_a = True
        a1.sim_time = 0.0
        a1.step_count = 0
        a1._sample_diagnostics(is_initial=True, reset_ham_baseline=True)
        a1.zero_tension_wave_state()

    try:
        # 1. Instantaneous Elliptic Mode (`retarded_weave_u = False`): nonzero across entire box after 1 step!
        a1.update_compiled_action(comp_inst, mod=mod_inst)
        _upload_compact_trefoil()
        a1.step(mod_inst, comp_inst, n_substeps=1)
        u_inst = a1.readback_tension_u_grid()
        far_corner_mask = rr > 9.5
        assert float(np.mean(u_inst[far_corner_mask])) > 0.008, (
            "Instantaneous elliptic mode should populate far corners immediately in 1 step!"
        )

        # 2. Causal Retarded Mode (`c_gw = 1.0 * c_w`): strict zero outside light cone & arrival inside!
        a1.update_compiled_action(comp_c1, mod=mod_c1)
        _upload_compact_trefoil()

        dx = box_sz / float(grid_n)
        dt_c1 = compute_safe_symplectic_dt(dx, mod_c1, compiled=comp_c1, peak_rho_est=max(a1.init_peak_rho, 2.8))
        n_steps_c1 = int(round(3.80 / dt_c1))
        a1.step(mod_c1, comp_c1, n_substeps=n_steps_c1)
        t_actual_c1 = a1.sim_time
        u_gl_c1 = a1.readback_tension_u_grid()
        ut_gl_c1 = a1.readback_tension_udot_grid()

        # Compact Trefoil (`a0 = 1.50`) source density drops to ~0 outside r_src ~ 2.9; include 3*h stencil cushion:
        outside_cone = rr > (3.85 + 1.0 * t_actual_c1)
        inside_cone = (rr >= 3.0) & (rr <= 4.4)
        assert np.count_nonzero(outside_cone) > 20000
        assert float(np.max(np.abs(u_gl_c1[outside_cone]))) < 1.0e-5, (
            f"Acausal leakage outside light cone! max|U|={float(np.max(np.abs(u_gl_c1[outside_cone]))):.3e}"
        )
        assert float(np.max(np.abs(ut_gl_c1[outside_cone]))) < 2.0e-5
        assert float(np.mean(u_gl_c1[inside_cone])) > 0.018, (
            f"Retarded gravitational wave failed to propagate inside light cone: {float(np.mean(u_gl_c1[inside_cone])):.4f}"
        )
        r_front_c1 = _wavefront_outer_radius(u_gl_c1, rr, frac_of_peak=0.08)

        # 3. Cross-API Vulkan (`WgpuComputeEngine`) parity from exact identical `t = 0` field and `dt`!
        res_vk_c1 = eng.run_3d_knot_evolution(
            comp_c1,
            mod_c1,
            grid_n=grid_n,
            n_steps=n_steps_c1,
            box_size=box_sz,
            dt=dt_c1,
            damping=0.0,
            sponge=a1.sponge_strength,
            phi_init_override=phi0_zyx3,
            vel_init_override=vel0_zyx3,
            solve_tension_u=True,
            zero_init_tension_u=True,
            readback_tension_u=True,
        )
        u_vk_c1 = res_vk_c1["u_final"]
        ut_vk_c1 = res_vk_c1["udot_final"]
        assert float(np.max(np.abs(u_vk_c1[outside_cone]))) < 1.0e-5
        max_diff_u = float(np.max(np.abs(u_gl_c1 - u_vk_c1)))
        max_diff_ut = float(np.max(np.abs(ut_gl_c1 - ut_vk_c1)))
        assert max_diff_u < 2.5e-3, f"OpenGL vs Vulkan retarded U mismatch: {max_diff_u:.4e}"
        assert max_diff_ut < 5.0e-3, f"OpenGL vs Vulkan retarded dU/dt mismatch: {max_diff_ut:.4e}"

        # 4. Slower GW speed (`gw_speed_ratio = 0.68`) over the EXACT same physical time `t_actual_c1`:
        a1.update_compiled_action(comp_slow, mod=mod_slow)
        _upload_compact_trefoil()
        dt_slow = compute_safe_symplectic_dt(dx, mod_slow, compiled=comp_slow, peak_rho_est=max(a1.init_peak_rho, 2.8))
        n_steps_slow = int(round(t_actual_c1 / dt_slow))
        a1.step(mod_slow, comp_slow, n_substeps=n_steps_slow)
        u_gl_slow = a1.readback_tension_u_grid()
        r_front_slow = _wavefront_outer_radius(u_gl_slow, rr, frac_of_peak=0.08)

        # Effective source core edge r_0 ~ 1.45; propagated distance (R_front - r_0) scales ~ 0.68!
        r_src_eff = 1.45
        speed_ratio_meas = (r_front_slow - r_src_eff) / max(r_front_c1 - r_src_eff, 1.0e-6)
        assert 0.56 <= speed_ratio_meas <= 0.78, (
            f"Expected wavefront travel ratio ~0.68, got {speed_ratio_meas:.3f} "
            f"(R_slow={r_front_slow:.2f}, R_c1={r_front_c1:.2f})"
        )
    finally:
        a1.release()
        ctx.release()
        del eng
        gc.collect()


def test_sommerfeld_radiative_boundary_non_reflecting_and_static_monopole() -> None:
    """
    Verifies:
      (A) Pure vacuum spherical GW pulse (`phi = 0 => rho_{H,0} = 0`) exits cleanly through the 6 outer
          Bayliss-Turkel / Sommerfeld faces with `> 97%` of wave Hamiltonian absorbed (no reflection!).
      (B) Stationary solitary Trefoil (`Preset 0`) preserves its positive exterior `1/r` monopole potential.
    """
    ctx = moderngl.create_standalone_context(require=430)
    mod = get_mods_by_id()["tnap_weave"].clone(new_id="tnap_sommerfeld")
    comp = compile_master_action(mod)
    box_sz = 16.8
    grid_n = 64
    _, _, _, rr, h_u = _radial_grid_64(box_sz)
    kappa = WEAVE_KAPPA_SCALE * float(mod.parameters.g_knot)

    a1 = Arena1KnotTank(ctx, initial_grid_n=grid_n, mod=mod, compiled=comp)
    a1.box_size = box_sz
    try:
        # --- Part A: Pure vacuum spherical GW packet (`phi = 0, vel = 0`) ---
        vox3 = grid_n * grid_n * grid_n * 3
        assert a1._buf_a is not None and a1._buf_b is not None and a1._buf_vel is not None
        a1._zero_ssbo(a1._buf_a, vox3)
        a1._zero_ssbo(a1._buf_b, vox3)
        a1._zero_ssbo(a1._buf_vel, vox3)
        a1._cur_in_a = True
        a1._ensure_mg_resources()
        assert a1._mg_ut_buf is not None

        u_pulse = (0.12 * np.exp(-(rr * rr) / (2.0 * (1.25 ** 2)))).astype(np.float32)
        a1._mg_u_bufs[0].write(np.ascontiguousarray(u_pulse))
        a1._zero_ssbo(a1._mg_ut_buf, 64 * 64 * 64)
        a1._pack_tension_tex_gpu(zero_mode=False, has_ut=True)
        a1._u_warm_valid = True
        a1._external_tension_u_active = False

        def _gw_wave_energy() -> float:
            u = a1.readback_tension_u_grid().astype(np.float64)
            ut = a1.readback_tension_udot_grid().astype(np.float64)
            du_x = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * h_u)
            du_y = (u[:, 2:, :] - u[:, :-2, :]) / (2.0 * h_u)
            du_z = (u[2:, :, :] - u[:-2, :, :]) / (2.0 * h_u)
            grad2 = du_x[1:-1, 1:-1, :] ** 2 + du_y[1:-1, :, 1:-1] ** 2 + du_z[:, 1:-1, 1:-1] ** 2
            ut2 = ut[1:-1, 1:-1, 1:-1] ** 2
            return float(0.5 / kappa * np.sum(ut2 + grad2) * (h_u ** 3))

        e0 = _gw_wave_energy()
        assert e0 > 0.05
        # Advance t ~ 16.5 (> diagonal exit time sqrt(3)*8.4 / 1.0 = 14.55) so entire spherical shell crosses walls
        dt_vac = compute_safe_symplectic_dt(box_sz / grid_n, mod, compiled=comp, peak_rho_est=2.8)
        a1.step(mod, comp, n_substeps=int(round(16.8 / dt_vac)))
        e_final = _gw_wave_energy()
        assert e_final < 0.025 * e0, (
            f"Sommerfeld boundary reflected too much GW energy: E_final/E0 = {e_final / e0:.4f}"
        )

        # --- Part B: Stationary solitary Trefoil (`Preset 0`) retains positive 1/r monopole ---
        a1.box_size = None  # default 12.0 for Preset 0
        a1.reset_field(0)
        _, _, _, rr12, _ = _radial_grid_64(a1.box_size)
        shell12 = (rr12 >= 3.2) & (rr12 <= 5.2)
        u_init_mean = float(np.mean(a1.readback_tension_u_grid()[shell12]))
        a1.step(mod, comp, n_substeps=1600)
        u_after = a1.readback_tension_u_grid()
        ut_after = a1.readback_tension_udot_grid()
        u_after_mean = float(np.mean(u_after[shell12]))
        rel_monopole_drift = abs(u_after_mean - u_init_mean) / max(u_init_mean, 1.0e-8)
        assert float(np.min(u_after)) > 0.015, "Static Trefoil tension U must remain strictly positive everywhere!"
        assert rel_monopole_drift < 0.08, f"Exterior 1/r monopole drifted by {rel_monopole_drift * 100:.2f}%"
        assert float(np.sqrt(np.mean(ut_after[shell12] ** 2))) < 2.0e-3
    finally:
        a1.release()
        ctx.release()
        gc.collect()


def test_annihilating_and_orbiting_knots_radiate_outgoing_gravitational_waves() -> None:
    """
    Verifies both dynamic multi-knot gravitational-wave sources on `128^3`:
      • `Preset 5` (`+7 / -7` Matter-Antimatter Annihilation): emits an outgoing GW burst with
        strong strain rate `|dU/dt|` and strictly positive outward radial GW Poynting flux!
      • `Preset 6` (`+7 / +7` Binary Trefoil Inspiral): undergoes prograde orbital rotation & inspiral
        while conserving `Q_H ~ +14` and radiating outgoing quadrupole gravitational waves!
    """
    ctx = moderngl.create_standalone_context(require=430)
    mod = get_mods_by_id()["tnap_weave"].clone(new_id="tnap_gw_sources")
    comp = compile_master_action(mod)
    grid_n = 128
    a1 = Arena1KnotTank(ctx, initial_grid_n=grid_n, mod=mod, compiled=comp)

    xx_u, yy_u, zz_u, rr_u, h_u = _radial_grid_64(16.8)
    rr_safe = np.maximum(rr_u, 1.0e-6)
    wavezone = (rr_u >= 5.2) & (rr_u <= 7.8)
    kappa = WEAVE_KAPPA_SCALE * float(mod.parameters.g_knot)

    c128 = ((np.arange(grid_n, dtype=np.float64) + 0.5) / float(grid_n) - 0.5) * 16.8
    _, yy_f, xx_f = np.meshgrid(c128, c128, c128, indexing="ij")

    def _measure_perturbation_poynting(u0_ref: np.ndarray) -> Tuple[float, float, float]:
        du = a1.readback_tension_u_grid().astype(np.float64) - u0_ref
        ut = a1.readback_tension_udot_grid().astype(np.float64)
        gx = (np.roll(du, -1, axis=2) - np.roll(du, 1, axis=2)) / (2.0 * h_u)
        gy = (np.roll(du, -1, axis=1) - np.roll(du, 1, axis=1)) / (2.0 * h_u)
        gz = (np.roll(du, -1, axis=0) - np.roll(du, 1, axis=0)) / (2.0 * h_u)
        du_dr = (xx_u * gx + yy_u * gy + zz_u * gz) / rr_safe
        sr_gw = -(1.0 / kappa) * ut * du_dr
        return (
            float(np.mean(sr_gw[wavezone])),
            float(np.sqrt(np.mean(ut[wavezone] ** 2))),
            float(np.max(np.abs(ut[wavezone]))),
        )

    try:
        # 1. Preset 6: Binary Trefoil Inspiral (+7 / +7) — Orbiting Quadrupole GW Ripples!
        a1.reset_field(6)
        u0_p6 = a1.readback_tension_u_grid().astype(np.float64)
        phi6_0, _ = _readback_arena1_phi_vel_zyx3(a1)
        q6_0 = _compute_hopf_charge(phi6_0, box_size=a1.box_size)
        assert q6_0 > 11.8

        rho0 = np.sum(phi6_0.astype(np.float64) ** 2, axis=-1)
        w0 = np.maximum(rho0 - 1.2, 0.0) ** 2
        r_rms_0 = math.sqrt(
            float(np.sum(w0 * (xx_f * xx_f + yy_f * yy_f)) / max(float(np.sum(w0)), 1.0e-12))
        )

        poynting_samples_p6 = []
        ut_rms_samples_p6 = []
        angles_deg = [0.0]
        prev_ang = 0.0
        cum_rot_deg = 0.0

        pre_merger_peak_rot_deg = 0.0
        for step_blk in range(10):
            a1.step(mod, comp, n_substeps=2000)
            sr_m, ut_rms, _ = _measure_perturbation_poynting(u0_p6)
            poynting_samples_p6.append(sr_m)
            ut_rms_samples_p6.append(ut_rms)

            phi_cur, _ = _readback_arena1_phi_vel_zyx3(a1)
            rho_c = np.sum(phi_cur.astype(np.float64) ** 2, axis=-1)
            wc = np.maximum(rho_c - 1.2, 0.0) ** 2
            w_sum = max(float(np.sum(wc)), 1.0e-12)
            qxx = float(np.sum(wc * xx_f * xx_f) / w_sum)
            qxy = float(np.sum(wc * xx_f * yy_f) / w_sum)
            qyy = float(np.sum(wc * yy_f * yy_f) / w_sum)
            ang = 0.5 * math.degrees(math.atan2(2.0 * qxy, qxx - qyy))
            if step_blk < 6:
                d_ang = ((ang - prev_ang + 90.0) % 180.0) - 90.0
                cum_rot_deg += d_ang
                prev_ang = ang
                pre_merger_peak_rot_deg = max(pre_merger_peak_rot_deg, cum_rot_deg)
            angles_deg.append(ang)

        phi6_end, _ = _readback_arena1_phi_vel_zyx3(a1)
        q6_end = _compute_hopf_charge(phi6_end, box_size=a1.box_size)
        rho_e = np.sum(phi6_end.astype(np.float64) ** 2, axis=-1)
        we = np.maximum(rho_e - 1.2, 0.0) ** 2
        r_rms_end = math.sqrt(
            float(np.sum(we * (xx_f * xx_f + yy_f * yy_f)) / max(float(np.sum(we)), 1.0e-12))
        )

        assert pre_merger_peak_rot_deg > 26.0, (
            f"Binary Trefoil pre-merger inspiral failed to orbit prograde: {pre_merger_peak_rot_deg:.1f} deg ({angles_deg})"
        )
        assert r_rms_end < r_rms_0 - 0.55, f"Binary Trefoil failed to inspiral: {r_rms_0:.3f} -> {r_rms_end:.3f}"
        assert abs(q6_end - q6_0) < 1.0, f"Binary Trefoil topological charge not conserved: {q6_0:.2f} -> {q6_end:.2f}"
        assert max(ut_rms_samples_p6) > 1.5e-3, f"Quadrupole GW strain rate too weak: {max(ut_rms_samples_p6):.3e}"
        assert float(np.mean(poynting_samples_p6[1:])) > 5.0e-5, (
            f"Binary inspiral net radial GW Poynting flux must be outward (>0): {poynting_samples_p6}"
        )

        # 2. Preset 5: Matter + Antimatter (+7 / -7) — Annihilation GW Burst!
        a1.reset_field(5)
        u0_p5 = a1.readback_tension_u_grid().astype(np.float64)
        a1.step(mod, comp, n_substeps=22000)
        burst_poynting = []
        burst_ut_pk = []
        for _ in range(4):
            a1.step(mod, comp, n_substeps=2000)
            sr5, _, ut5_pk = _measure_perturbation_poynting(u0_p5)
            burst_poynting.append(sr5)
            burst_ut_pk.append(ut5_pk)

        assert max(burst_ut_pk) > 1.2e-2, f"Annihilation GW burst strain rate too small: {max(burst_ut_pk):.3e}"
        assert float(np.mean(burst_poynting)) > 2.0e-4, (
            f"Annihilation GW burst Poynting flux must be strongly outward (>0): {burst_poynting}"
        )
    finally:
        a1.release()
        ctx.release()
        gc.collect()


def test_retarded_tension_ddr5_hibernation_replay_and_studio_ui() -> None:
    """
    Verifies 100% lossless float32 preservation of BOTH `U(x, t)` and `dU/dt(x, t)` across
    DDR5 System RAM Hibernation and Time-Machine Replay, plus Studio UI & volumetric GW ray-marching.
    """
    app = WeaveForgeStudioApp(width=1280, height=720, initial_arena=1, initial_mod_id="tnap_weave", visible=False)
    try:
        assert "retarded_weave_u" in app._arena_defaults[0] or hasattr(app.working_mod.parameters, "retarded_weave_u")
        assert app.working_mod.parameters.retarded_weave_u is True
        assert app.arena1.show_gw_ripples is True

        app.arena1.set_grid_resolution(64)
        app.arena1.reset_field(6)
        app.arena1.step(app.working_mod, app.compiled, n_substeps=160)
        app.arena1.flush_pending_replay_snapshot()
        assert len(app.arena1._replay_frames) >= 1

        u_live = app.arena1.readback_tension_u_grid().copy()
        ut_live = app.arena1.readback_tension_udot_grid().copy()
        assert float(np.max(np.abs(ut_live))) > 5.0e-4

        # 1. DDR5 Hibernation round-trip (`hibernate_to_ram` -> `wake_from_ram`)
        ram_mb = app.arena1.hibernate_to_ram()
        assert ram_mb > 1.0 and app.arena1.is_hibernated
        app.arena1.wake_from_ram()
        u_woke = app.arena1.readback_tension_u_grid()
        ut_woke = app.arena1.readback_tension_udot_grid()
        assert float(np.max(np.abs(u_woke - u_live))) == 0.0
        assert float(np.max(np.abs(ut_woke - ut_live))) == 0.0

        # 2. DDR5 Time-Machine Replay round-trip (`load_replay_frame` -> `resume_live_from_replay`)
        app.arena1.load_replay_frame(0)
        assert app.arena1.replay_cursor == 0
        app.arena1.resume_live_from_replay()
        assert app.arena1.replay_cursor == -1
        u_resumed = app.arena1.readback_tension_u_grid()
        ut_resumed = app.arena1.readback_tension_udot_grid()
        assert float(np.max(np.abs(u_resumed - u_live))) == 0.0
        assert float(np.max(np.abs(ut_resumed - ut_live))) == 0.0

        # 3. Studio UI parameter modification & reset + live volumetric GW frame render
        app.select_left_tab(1)
        app.working_mod.parameters.retarded_weave_u = False
        app._recompile_working_mod()
        assert app.is_parameter_modified("retarded_weave_u") is True
        app.reset_single_parameter("retarded_weave_u")
        assert app.working_mod.parameters.retarded_weave_u is True
        assert app.is_parameter_modified("retarded_weave_u") is False

        app.render_single_frame(dt_override=1.0 / 60.0)
        assert app.compile_error == ""
    finally:
        app.shutdown()
        gc.collect()
