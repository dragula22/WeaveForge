r"""
Phase 4 (Milestone 4A): Causal Spin-2 Transverse-Traceless Tensor Gravitational Waves
& 3D Finkelstein-Rubinstein Topological Spin-1/2 Fermions Test Suite
(`tests/test_phase4_spin2_gw_and_fermions.py`)
=====================================================================================
Verifies directly on GPU (`ModernGL` OpenGL 4.6 + `wgpu` Vulkan) and exact `SymPy`
algebraic topology with ZERO hardcoded waveforms or shortcuts:
  1. Birkhoff No-Monopole Selection Rule (`max|dU/dt| > 2e-3` while `max|h_+|, max|h_x| < 1e-6`
     and `max|h_+| / max|U| < 1e-5` for a spherically symmetric radial breathing pulse)
     plus cross-API OpenGL 4.6 vs Vulkan `WGSL` Spin-2 TT parity (`< 1e-5`) on an
     anisotropic 3D quadrupole source.
  2. Spin-2 Quadrupole Polarization & GW170817 Inclination Physics:
       • Part A (`Preset 6`: `binary_trefoil_inspiral`): Radiates strong dynamic `(h_+, h_x)`
         waveforms into the virtual LIGO detectors with total face-on (`+z`) strain power
         exceeding edge-on (`+x`) strain power (`rms_face > rms_edge`).
       • Part B ($z \to -z$ Even Equatorial Binary Q-Balls): Satisfies the exact equatorial
         reflection selection rule (`|h_+|_{edge} > 10 * |h_x|_{edge}`) and face-on (`+z`)
         circular polarization with comparable `h_+` / `h_x` amplitudes in $90^\circ$ quadrature.
  3. Finkelstein-Rubinstein Spin-1/2 Sign $(-1)^{Q_H}$ & 3D Two-Hopfion Pauli Core Repulsion:
       • Exact `SymPy` $SU(2)$ $2\pi$-rotation and exchange loop homotopy proof (`sign_q1 == -1`,
         `sign_q2 == +1`, `exchange_sign_q1 == -1`) + 3D discrete $SU(2)$ holonomy on $Q_H = 1$
         vs $Q_H = 2$ Hopfions + static two-Hopfion Pauli barrier curve.
       • Dynamic 3D GPU scattering (`Preset 7`: `fermion_hopfion_pauli` on `96^3`): identical
         in-phase ($\Delta\psi = 0$) fermionic Hopfions bounce outward off the repulsive Pauli
         core barrier while conserving $Q_H \approx +2$, whereas antisymmetric ($\Delta\psi = \pi$)
         Hopfions attract deeply inward.
  4. Lossless `float32` DDR5 Hibernation (`hibernate_to_ddr5` / `wake_from_ddr5`), Time-Machine
     Replay (`load_replay_frame` / `resume_live_from_replay`), and distinct non-saturated
     volumetric ray-marching (`gw_vis_mode = 1`: Spin-2 Tensor `h_+ / h_x` vs `gw_vis_mode = 0`).
"""

from __future__ import annotations

import gc
import math

import moderngl
import numpy as np

from tests.test_knot_long_run_stability import _compute_hopf_charge, _readback_arena1_phi_vel_zyx3
from weaveforge.arenas.arena1_knot_tank import Arena1KnotTank, KNOT_PRESETS
from weaveforge.compute.wgpu_engine import WgpuComputeEngine, compute_safe_symplectic_dt
from weaveforge.core.camera import Camera3D
from weaveforge.core.finkelstein_rubinstein import (
    compute_3d_hopfion_fr_holonomy,
    measure_two_hopfion_pauli_barrier_curve,
    verify_finkelstein_rubinstein_sympy,
)
from weaveforge.core.frameworks import ActionSpec, ModParameters, PhysicsMod, get_mods_by_id
from weaveforge.core.symbolic_compiler import compile_master_action


def test_spin2_birkhoff_no_monopole_selection_rule_and_vulkan_parity() -> None:
    """
    Proves that a spherically symmetric radial breathing pulse excites strong scalar tension
    waves (`max|dU/dt| > 2e-3`) while radiating ZERO Spin-2 TT strain (`max|h_+|, max|h_x| < 1e-6`,
    `max|h_+| / max|U| < 1e-5`), and verifies cross-API OpenGL 4.6 vs Vulkan `WGSL` parity
    on a triaxial quadrupolar source.
    """
    ctx = moderngl.create_standalone_context(require=430)
    eng = WgpuComputeEngine()

    mod = get_mods_by_id()["tnap_weave"].clone(new_id="tnap_p4_birkhoff")
    mod.parameters.retarded_weave_u = True
    comp = compile_master_action(mod)

    grid_n = 64
    box_sz = 16.0
    coords = ((np.arange(grid_n, dtype=np.float32) + 0.5) / float(grid_n) - 0.5) * box_sz
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    rr2 = xx * xx + yy * yy + zz * zz

    a1 = Arena1KnotTank(ctx, initial_grid_n=grid_n, mod=mod, compiled=comp)
    a1.box_size = box_sz

    try:
        # 1. Spherically symmetric radial breathing pulse: Birkhoff no-monopole selection rule
        phi_sph = np.zeros((grid_n, grid_n, grid_n, 3), dtype=np.float32)
        phi_sph[..., 0] = 0.85 * np.exp(-rr2 / (1.85 * 1.85))
        vel_zero = np.zeros_like(phi_sph)

        assert a1._buf_a is not None and a1._buf_vel is not None
        a1._buf_a.write(np.ascontiguousarray(phi_sph, dtype=np.float32))
        a1._zero_ssbo(a1._buf_vel, grid_n * grid_n * grid_n * 3)
        a1._cur_in_a = True
        a1.sim_time = 0.0
        a1.step_count = 0
        a1._sample_diagnostics(is_initial=True, reset_ham_baseline=True)
        a1.zero_tension_wave_state()

        dx = box_sz / float(grid_n)
        rho_guard_sph = a1.current_peak_rho if a1.current_peak_rho > 4.5 else max(a1.init_peak_rho, 2.8)
        dt_sph = compute_safe_symplectic_dt(dx, mod, compiled=comp, peak_rho_est=rho_guard_sph)

        a1.step(mod, comp, n_substeps=45)
        u_sph = a1.readback_tension_u_grid()
        ut_sph = a1.readback_tension_udot_grid()
        hp_sph, hx_sph = a1.readback_spin2_tt_grids()

        max_u = float(np.max(np.abs(u_sph)))
        max_ut = float(np.max(np.abs(ut_sph)))
        max_hp_sph = float(np.max(np.abs(hp_sph)))
        max_hx_sph = float(np.max(np.abs(hx_sph)))

        assert max_u > 1.0e-4 and max_ut > 2.0e-3, (
            f"Spherical breathing pulse failed to excite scalar tension channel: max|U|={max_u:.3e}, max|dU/dt|={max_ut:.3e}"
        )
        assert max_hp_sph < 1.0e-6 and max_hx_sph < 1.0e-6, (
            f"Birkhoff violation in OpenGL Spin-2 solver: max|h+|={max_hp_sph:.3e}, max|hx|={max_hx_sph:.3e}"
        )
        assert max(max_hp_sph, max_hx_sph) / max(max_u, 1.0e-12) < 1.0e-5

        res_vk_sph = eng.run_3d_knot_evolution(
            comp,
            mod,
            grid_n=grid_n,
            n_steps=45,
            box_size=box_sz,
            dt=dt_sph,
            damping=0.0,
            sponge=a1.sponge_strength,
            phi_init_override=phi_sph,
            vel_init_override=vel_zero,
            solve_tension_u=True,
            zero_init_tension_u=True,
            enable_spin2_gw=True,
            readback_spin2_gw=True,
        )
        assert float(np.max(np.abs(res_vk_sph["h_plus_final"]))) < 1.0e-6
        assert float(np.max(np.abs(res_vk_sph["h_cross_final"]))) < 1.0e-6

        # 2. Anisotropic quadrupolar source: OpenGL 4.6 vs Vulkan WGSL Spin-2 TT parity
        phi_ani = np.zeros((grid_n, grid_n, grid_n, 3), dtype=np.float32)
        phi_ani[..., 0] = 1.15 * np.exp(-(xx * xx / 1.40 + yy * yy / 3.40 + zz * zz / 2.10))
        phi_ani[..., 1] = 0.75 * (xx * yy / 2.20) * np.exp(-rr2 / 2.80)

        a1._buf_a.write(np.ascontiguousarray(phi_ani, dtype=np.float32))
        a1._zero_ssbo(a1._buf_vel, grid_n * grid_n * grid_n * 3)
        a1._cur_in_a = True
        a1.sim_time = 0.0
        a1.step_count = 0
        a1._sample_diagnostics(is_initial=True, reset_ham_baseline=True)
        a1.zero_tension_wave_state()

        rho_guard_ani = a1.current_peak_rho if a1.current_peak_rho > 4.5 else max(a1.init_peak_rho, 2.8)
        dt_ani = compute_safe_symplectic_dt(dx, mod, compiled=comp, peak_rho_est=rho_guard_ani)
        n_steps_ani = 320

        a1.step(mod, comp, n_substeps=n_steps_ani)
        q_gl, _ = a1.readback_spin2_q_grids()
        hp_gl, hx_gl = a1.readback_spin2_tt_grids()

        res_vk_ani = eng.run_3d_knot_evolution(
            comp,
            mod,
            grid_n=grid_n,
            n_steps=n_steps_ani,
            box_size=box_sz,
            dt=dt_ani,
            damping=0.0,
            sponge=a1.sponge_strength,
            phi_init_override=phi_ani,
            vel_init_override=vel_zero,
            solve_tension_u=True,
            zero_init_tension_u=True,
            enable_spin2_gw=True,
            readback_spin2_gw=True,
        )
        q_vk = res_vk_ani["gw_q_final"]
        hp_vk = res_vk_ani["h_plus_final"]
        hx_vk = res_vk_ani["h_cross_final"]

        assert float(np.max(np.abs(hp_gl))) > 1.0e-5 and float(np.max(np.abs(hx_gl))) > 1.5e-6, (
            f"Anisotropic quadrupole source failed to radiate Spin-2 TT strain: "
            f"max|h+|={np.max(np.abs(hp_gl)):.3e}, max|hx|={np.max(np.abs(hx_gl)):.3e}"
        )
        err_q = float(np.max(np.abs(q_gl - q_vk)))
        err_hp = float(np.max(np.abs(hp_gl - hp_vk)))
        err_hx = float(np.max(np.abs(hx_gl - hx_vk)))
        assert err_q < 1.0e-5 and err_hp < 1.0e-5 and err_hx < 1.0e-5, (
            f"OpenGL vs Vulkan Spin-2 parity mismatch: err_q={err_q:.3e}, err_hp={err_hp:.3e}, err_hx={err_hx:.3e}"
        )
    finally:
        a1.release()
        ctx.release()
        gc.collect()


def test_binary_trefoil_inspiral_radiates_quadrupole_hplus_hcross_polarizations() -> None:
    """
    Verifies Spin-2 tensor GW radiation and LIGO inclination physics in two parts:
      • Part A (`Preset 6` `binary_trefoil_inspiral` — untouched chiral `(2,3)` Trefoil geometry):
        Radiates strong dynamic `(h_+, h_x)` waveforms into `gw_hplus_faceon` / `gw_hcross_faceon`
        with total face-on (`+z`) strain power exceeding edge-on (`+x`) strain power.
      • Part B (Equatorial $z \\to -z$ Reflection-Symmetric Binary Q-Balls):
        Demonstrates the exact GW170817 equatorial inclination selection rule
        (`|h_+|_{edge} > 10 * |h_x|_{edge}`) and face-on (`+z`) circular polarization with
        comparable `h_+` and `h_x` amplitudes oscillating in $90^\\circ$ phase quadrature.
    """
    ctx = moderngl.create_standalone_context(require=430)

    # Part A: Preset 6 (`binary_trefoil_inspiral`) on 64^3
    mod_p6 = get_mods_by_id()["tnap_weave"].clone(new_id="tnap_p4_inspiral")
    mod_p6.parameters.retarded_weave_u = True
    comp_p6 = compile_master_action(mod_p6)

    a1 = Arena1KnotTank(ctx, initial_grid_n=64, mod=mod_p6, compiled=comp_p6)
    try:
        a1.reset_field(preset_idx=6)
        for _ in range(10):
            a1.step(mod_p6, comp_p6, n_substeps=900)

        t_p6 = np.asarray(a1.gw_strain_t, dtype=np.float64)
        hpf_p6 = np.asarray(a1.gw_hplus_faceon, dtype=np.float64)
        hxf_p6 = np.asarray(a1.gw_hcross_faceon, dtype=np.float64)
        hpe_p6 = np.asarray(a1.gw_hplus_edgeon, dtype=np.float64)
        hxe_p6 = np.asarray(a1.gw_hcross_edgeon, dtype=np.float64)

        assert t_p6.size >= 25 and float(t_p6[-1]) >= 12.0
        win_p6 = t_p6 >= 6.5
        assert int(np.sum(win_p6)) >= 12

        rms_hpf_p6 = float(np.sqrt(np.mean(hpf_p6[win_p6] ** 2)))
        rms_hxf_p6 = float(np.sqrt(np.mean(hxf_p6[win_p6] ** 2)))
        rms_hpe_p6 = float(np.sqrt(np.mean(hpe_p6[win_p6] ** 2)))
        rms_hxe_p6 = float(np.sqrt(np.mean(hxe_p6[win_p6] ** 2)))

        total_rms_face_p6 = math.hypot(rms_hpf_p6, rms_hxf_p6)
        total_rms_edge_p6 = math.hypot(rms_hpe_p6, rms_hxe_p6)

        assert rms_hpf_p6 > 3.0e-4 and rms_hxf_p6 > 3.0e-4, (
            f"Preset 6 face-on Spin-2 strain too weak: rms(h+)={rms_hpf_p6:.3e}, rms(hx)={rms_hxf_p6:.3e}"
        )
        assert float(np.std(hpf_p6[win_p6])) > 1.5e-4 and float(np.std(hxf_p6[win_p6])) > 1.5e-4, (
            "Preset 6 face-on (h+, hx) time series must oscillate dynamically!"
        )
        assert total_rms_face_p6 > 1.35 * total_rms_edge_p6, (
            f"Preset 6 face-on total strain power ({total_rms_face_p6:.3e}) must exceed "
            f"edge-on total strain power ({total_rms_edge_p6:.3e})!"
        )

        # Part B: Equatorial z -> -z reflection-symmetric Binary Q-Balls in (x, y) orbit at z = 0
        qball_mod = PhysicsMod(
            id="equatorial_qball_gw",
            name="Equatorial Binary Q-Balls (Self-Gravitating Weave)",
            author="WeaveForge Phase 4 Suite",
            summary="SO(3)-symmetric self-gravitating Q-balls testing z -> -z equatorial GW polarization",
            action=ActionSpec(
                field_lagrangian=(
                    "exp(4 * U) * T_kin - c_w**2 * G_grad "
                    "- exp(2 * U) * (0.5 * m_eff**2 * rho - 0.25 * g4 * rho**2 + (1.0/6.0) * g6 * rho**3)"
                ),
                optical_metric_n="exp(2 * U)",
                metric_lapse_A="exp(-2 * U)",
                gw_speed_ratio="1",
            ),
            parameters=ModParameters(
                c_w=1.0,
                m_eff=2.50,
                g4=9.80,
                g6=3.10,
                k_skyrme=0.0,
                g_knot=1.60,
                retarded_weave_u=True,
            ),
        )
        qball_comp = compile_master_action(qball_mod)
        a1.update_compiled_action(qball_comp, mod=qball_mod)

        grid_n = 64
        box_sz = 16.0
        a1.box_size = box_sz
        coords = ((np.arange(grid_n, dtype=np.float32) + 0.5) / float(grid_n) - 0.5) * box_sz
        # GPU 3D buffer layout is (z * N + y) * N + x -> (Z, Y, X)
        zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")

        d0 = 2.15
        w = 1.30
        amp = 1.35
        omega_q = 1.65
        v_orb = 0.36

        env_l = amp * np.exp(-((xx + d0) ** 2 + yy * yy + zz * zz) / (w * w))
        env_r = amp * np.exp(-((xx - d0) ** 2 + yy * yy + zz * zz) / (w * w))

        phi_eq = np.zeros((grid_n, grid_n, grid_n, 3), dtype=np.float32)
        vel_eq = np.zeros((grid_n, grid_n, grid_n, 3), dtype=np.float32)
        phi_eq[..., 0] = env_l + env_r
        dy_env_l = (-2.0 * yy / (w * w)) * env_l
        dy_env_r = (-2.0 * yy / (w * w)) * env_r
        vel_eq[..., 0] = (+v_orb) * dy_env_l + (-v_orb) * dy_env_r
        vel_eq[..., 1] = omega_q * (env_l + env_r)

        assert a1._buf_a is not None and a1._buf_vel is not None
        a1._buf_a.write(np.ascontiguousarray(phi_eq, dtype=np.float32))
        a1._buf_vel.write(np.ascontiguousarray(vel_eq, dtype=np.float32))
        a1._cur_in_a = True
        a1.sim_time = 0.0
        a1.step_count = 0
        a1._sample_diagnostics(is_initial=True, reset_ham_baseline=True)
        a1._u_warm_valid = False
        a1.zero_spin2_gw_state()
        a1.clear_gw_strain_history()

        for _ in range(5):
            a1.step(qball_mod, qball_comp, n_substeps=320)

        t_eq = np.asarray(a1.gw_strain_t, dtype=np.float64)
        hpf_eq = np.asarray(a1.gw_hplus_faceon, dtype=np.float64)
        hxf_eq = np.asarray(a1.gw_hcross_faceon, dtype=np.float64)
        hpe_eq = np.asarray(a1.gw_hplus_edgeon, dtype=np.float64)
        hxe_eq = np.asarray(a1.gw_hcross_edgeon, dtype=np.float64)

        win_eq = t_eq >= 5.5
        assert int(np.sum(win_eq)) >= 20

        rms_hpf_eq = float(np.sqrt(np.mean(hpf_eq[win_eq] ** 2)))
        rms_hxf_eq = float(np.sqrt(np.mean(hxf_eq[win_eq] ** 2)))
        rms_hpe_eq = float(np.sqrt(np.mean(hpe_eq[win_eq] ** 2)))
        rms_hxe_eq = float(np.sqrt(np.mean(hxe_eq[win_eq] ** 2)))

        # Edge-on (+x, iota = pi/2): exact z -> -z reflection symmetry zeroes h_x = -h_yz while preserving h_+
        assert rms_hpe_eq > 5.0e-5, f"Equatorial edge-on h_+ too weak: {rms_hpe_eq:.3e}"
        assert rms_hpe_eq > 10.0 * max(rms_hxe_eq, 1.0e-15), (
            f"Equatorial inclination law violated: edge-on rms(h+)={rms_hpe_eq:.3e} vs rms(hx)={rms_hxe_eq:.3e}"
        )

        # Face-on (+z, iota = 0): circular polarization with comparable h_+ / h_x in 90-degree quadrature
        assert rms_hpf_eq > 8.0e-5 and rms_hxf_eq > 8.0e-5, (
            f"Equatorial face-on circular strain too weak: rms(h+)={rms_hpf_eq:.3e}, rms(hx)={rms_hxf_eq:.3e}"
        )
        amp_ratio_face = rms_hpf_eq / max(rms_hxf_eq, 1.0e-15)
        assert 0.30 < amp_ratio_face < 3.30, (
            f"Face-on h_+ / h_x amplitudes not comparable: ratio={amp_ratio_face:.3f}"
        )

        hpf_c = hpf_eq[win_eq] - np.mean(hpf_eq[win_eq])
        hxf_c = hxf_eq[win_eq] - np.mean(hxf_eq[win_eq])
        dhpf_dt = np.gradient(hpf_c, t_eq[win_eq])
        corr_quad = float(
            np.dot(dhpf_dt, hxf_c) / max(float(np.linalg.norm(dhpf_dt) * np.linalg.norm(hxf_c)), 1.0e-15)
        )
        assert abs(corr_quad) > 0.75, (
            f"Face-on h_+(t) and h_x(t) failed 90-degree circular quadrature check: corr(dh+/dt, hx)={corr_quad:.4f}"
        )
    finally:
        a1.release()
        ctx.release()
        gc.collect()


def test_finkelstein_rubinstein_spin_half_sign_and_hopfion_pauli_repulsion() -> None:
    """
    Verifies the Finkelstein-Rubinstein spin-statistics sign $(-1)^{Q_H}$ symbolically via `SymPy`
    and numerically via 3D $SU(2)$ holonomy + static Pauli core barrier curves, then evolves
    `Preset 7` (`fermion_hopfion_pauli`) dynamically in 3D on GPU (`96^3`) to prove identical
    in-phase ($\\Delta\\psi = 0$) Hopfions bounce off the Pauli core barrier while conserving $Q_H \\approx +2$.
    """
    # 1. Exact SymPy homotopy proof on SU(2)
    sym_res = verify_finkelstein_rubinstein_sympy()
    assert sym_res["sign_q1"] == -1
    assert sym_res["sign_q2"] == 1
    assert sym_res["exchange_sign_q1"] == -1
    assert sym_res["general_2pi_proved"] is True
    assert sym_res["exchange_equals_2pi_rotation"] is True

    # 2. 3D lattice SU(2) 2pi-rotation holonomy on Q_H = 1 (fermion) vs Q_H = 2 (boson)
    hol_q1 = compute_3d_hopfion_fr_holonomy(winding_m=1, grid_n=40, n_angle_steps=48)
    hol_q2 = compute_3d_hopfion_fr_holonomy(winding_m=2, grid_n=40, n_angle_steps=48)

    assert 0.90 < hol_q1["hopf_charge"] < 1.10
    assert abs(hol_q1["holonomy_sign"] - (-1.0)) < 1.0e-6
    assert abs(hol_q1["phase_rad"] - math.pi) < 1.0e-4
    assert hol_q1["is_fermionic"] is True

    assert 1.85 < hol_q2["hopf_charge"] < 2.15
    assert abs(hol_q2["holonomy_sign"] - (+1.0)) < 1.0e-6
    assert abs(hol_q2["phase_rad"] - 0.0) < 1.0e-4
    assert hol_q2["is_fermionic"] is False

    # 3. Static 3D Skyrme-Faddeev Pauli core barrier curve (Delta psi = 0 vs Delta psi = pi)
    barrier = measure_two_hopfion_pauli_barrier_curve(grid_n=48)
    assert barrier["pauli_repulsive_core_verified"] is True
    assert barrier["singlet_lower_than_symmetric"] is True
    assert barrier["delta_e_sym"][0] > 80.0
    assert (barrier["delta_e_sym"][0] - barrier["delta_e_anti"][0]) > 60.0

    # 4. Dynamic 3D GPU two-Hopfion scattering on Preset 7 (`96^3`): Pauli bounce vs singlet attraction
    assert len(KNOT_PRESETS) >= 8 and KNOT_PRESETS[7][0] == "fermion_hopfion_pauli"

    ctx = moderngl.create_standalone_context(require=430)
    eng = WgpuComputeEngine()
    mod = get_mods_by_id()["tnap_weave"].clone(new_id="tnap_p4_pauli")
    mod.parameters.retarded_weave_u = True
    comp = compile_master_action(mod)

    a1 = Arena1KnotTank(ctx, initial_grid_n=96, mod=mod, compiled=comp)
    try:
        def _evolve_preset7(pauli_pi: bool) -> tuple[list[float], float, float]:
            a1.pauli_twist_pi = pauli_pi
            a1.reset_field(preset_idx=7)
            box_l = a1.box_size
            c1d = ((np.arange(96, dtype=np.float64) + 0.5) / 96.0 - 0.5) * box_l
            _, _, xx96 = np.meshgrid(c1d, c1d, c1d, indexing="ij")

            phi0, _ = _readback_arena1_phi_vel_zyx3(a1)
            q_init = _compute_hopf_charge(phi0, box_size=box_l)

            seps: list[float] = []
            for blk in range(8):
                phi_k, _ = _readback_arena1_phi_vel_zyx3(a1)
                rho_k = np.sum(phi_k.astype(np.float64) ** 2, axis=-1)
                w = np.maximum(rho_k - 0.25, 0.0) ** 2
                mr = xx96 > 0.0
                ml = xx96 < 0.0
                xr = float(np.sum(xx96[mr] * w[mr]) / max(float(np.sum(w[mr])), 1.0e-12))
                xl = float(np.sum(xx96[ml] * w[ml]) / max(float(np.sum(w[ml])), 1.0e-12))
                seps.append(xr - xl)
                if blk < 7:
                    a1.step(mod, comp, n_substeps=220)

            phi_end, _ = _readback_arena1_phi_vel_zyx3(a1)
            q_final = _compute_hopf_charge(phi_end, box_size=box_l)
            return seps, q_init, q_final

        seps_sym, q0_sym, qend_sym = _evolve_preset7(pauli_pi=False)
        seps_anti, q0_anti, qend_anti = _evolve_preset7(pauli_pi=True)

        min_sym = min(seps_sym)
        min_anti = min(seps_anti)

        # Identical in-phase fermions (Delta psi = 0) decelerate, reach a turning point, and bounce outward!
        assert min_sym < seps_sym[0] - 0.10, f"Symmetric Hopfions never approached core barrier: {seps_sym}"
        assert seps_sym[-1] > min_sym + 0.10, (
            f"Symmetric (Delta psi = 0) fermionic Hopfions failed to bounce off Pauli core barrier: {seps_sym}"
        )
        # Antisymmetric singlet channel (Delta psi = pi) plunges deeply inward instead of bouncing at ~4.7!
        assert min_anti < min_sym - 1.20 and seps_anti[-1] < seps_sym[-1] - 1.50, (
            f"Expected strong Pauli repulsion gap between sym={seps_sym} and anti={seps_anti}"
        )
        # Topological Hopf charge Q_H ~ +2 is conserved across the collision in both channels
        assert 1.70 < q0_sym < 2.15 and 1.45 < qend_sym < 2.25, (
            f"Symmetric two-Hopfion Q_H not conserved: {q0_sym:.3f} -> {qend_sym:.3f}"
        )
        assert 1.70 < q0_anti < 2.15 and 1.45 < qend_anti < 2.25, (
            f"Antisymmetric two-Hopfion Q_H not conserved: {q0_anti:.3f} -> {qend_anti:.3f}"
        )

        # Verify Vulkan WGSL Preset 7 (`fermion_hopfion_pauli`) vs Preset 8 (`fermion_hopfion_singlet`) parity
        vk_p7 = eng.run_3d_knot_evolution(
            comp, mod, grid_n=48, n_steps=1, box_size=15.6, dt=1.0e-4, preset="fermion_hopfion_pauli"
        )
        vk_p8 = eng.run_3d_knot_evolution(
            comp, mod, grid_n=48, n_steps=1, box_size=15.6, dt=1.0e-4, preset="fermion_hopfion_singlet"
        )
        assert float(np.max(np.abs(vk_p7["phi_final"] - vk_p8["phi_final"]))) > 0.50
    finally:
        a1.release()
        ctx.release()
        gc.collect()


def test_spin2_ddr5_hibernation_replay_and_volumetric_render() -> None:
    """
    Verifies exact `0.0` `float32` round-trip preservation of Spin-2 `(gw_q, gw_qt)` and `(h_+, h_x)`
    across DDR5 System RAM Hibernation (`hibernate_to_ddr5` / `wake_from_ddr5`) and Time-Machine
    Replay (`load_replay_frame` / `resume_live_from_replay`), plus non-empty distinct volumetric
    ray-marched rendering in `gw_vis_mode = 1` (Spin-2 Tensor `h_+ / h_x`).
    """
    ctx = moderngl.create_standalone_context(require=430)
    mod = get_mods_by_id()["tnap_weave"].clone(new_id="tnap_p4_hib")
    mod.parameters.retarded_weave_u = True
    comp = compile_master_action(mod)

    a1 = Arena1KnotTank(ctx, initial_grid_n=64, mod=mod, compiled=comp)
    tex_fbo = ctx.texture((256, 256), 4)
    fbo = ctx.framebuffer(color_attachments=[tex_fbo])
    cam = Camera3D()

    try:
        a1.reset_field(preset_idx=6)
        a1.step(mod, comp, n_substeps=2400)
        a1.flush_pending_replay_snapshot()
        a1._capture_replay_snapshot()
        assert len(a1._replay_frames) >= 2

        q_live, qt_live = a1.readback_spin2_q_grids()
        hp_live, hx_live = a1.readback_spin2_tt_grids()
        assert float(np.max(np.abs(q_live))) > 5.0e-4
        assert float(np.max(np.abs(qt_live))) > 5.0e-4
        assert float(np.max(np.abs(hp_live))) > 2.0e-4

        # 1. DDR5 Hibernation round-trip (`hibernate_to_ddr5` -> `wake_from_ddr5`)
        ram_mb = a1.hibernate_to_ddr5()
        assert ram_mb > 1.0 and a1.is_hibernated and a1.allocated_vram_mb == 0.0
        a1.wake_from_ddr5()
        assert not a1.is_hibernated

        q_wake, qt_wake = a1.readback_spin2_q_grids()
        hp_wake, hx_wake = a1.readback_spin2_tt_grids()
        assert float(np.max(np.abs(q_wake - q_live))) == 0.0
        assert float(np.max(np.abs(qt_wake - qt_live))) == 0.0
        assert float(np.max(np.abs(hp_wake - hp_live))) == 0.0
        assert float(np.max(np.abs(hx_wake - hx_live))) == 0.0

        # 2. Time-Machine Replay round-trip (`load_replay_frame` zeroes scrubbed wave state ->
        #    `resume_live_from_replay` restores live `(gw_q, gw_qt)` & `(h_+, h_x)` with 0.0 error!)
        a1.load_replay_frame(0)
        assert a1.replay_cursor == 0
        q_scrub_0, _ = a1.readback_spin2_q_grids()
        assert float(np.max(np.abs(q_scrub_0))) == 0.0

        a1.resume_live_from_replay()
        assert a1.replay_cursor == -1
        q_resumed, qt_resumed = a1.readback_spin2_q_grids()
        hp_resumed, hx_resumed = a1.readback_spin2_tt_grids()
        assert float(np.max(np.abs(q_resumed - q_live))) == 0.0
        assert float(np.max(np.abs(qt_resumed - qt_live))) == 0.0
        assert float(np.max(np.abs(hp_resumed - hp_live))) == 0.0
        assert float(np.max(np.abs(hx_resumed - hx_live))) == 0.0

        # 3. Offscreen volumetric ray-march FBO render: Scalar dU/dt (mode 0) vs Spin-2 h_+/h_x (mode 1)
        a1.show_gw_ripples = True
        a1.gw_ripple_gain = 1.5

        a1.gw_vis_mode = 0
        a1.render(fbo, cam, 0.5)
        img_mode0 = np.frombuffer(fbo.read(components=4), dtype=np.uint8).reshape((256, 256, 4)).copy()

        a1.gw_vis_mode = 1
        a1.render(fbo, cam, 0.5)
        img_mode1 = np.frombuffer(fbo.read(components=4), dtype=np.uint8).reshape((256, 256, 4)).copy()

        mean_rgb_1 = float(np.mean(img_mode1[..., :3]))
        max_px_diff = int(np.max(np.abs(img_mode1[..., :3].astype(int) - img_mode0[..., :3].astype(int))))
        mean_px_diff = float(np.mean(np.abs(img_mode1[..., :3].astype(np.float64) - img_mode0[..., :3].astype(np.float64))))

        assert 5.0 < mean_rgb_1 < 235.0, f"Spin-2 volumetric render blank or saturated: mean_rgb={mean_rgb_1:.2f}"
        assert max_px_diff >= 25 and mean_px_diff > 1.0, (
            f"Spin-2 volumetric render (gw_vis_mode=1) indistinguishable from scalar mode 0: "
            f"max_diff={max_px_diff}, mean_diff={mean_px_diff:.3f}"
        )
    finally:
        fbo.release()
        tex_fbo.release()
        a1.release()
        ctx.release()
        gc.collect()


def test_studio_phase4_controls_cmb_pol_and_gaia_rehearsal() -> None:
    """
    Verifies headless 4K `WeaveForgeStudioApp` integration of all Phase 4 UI hooks:
      • `Preset 7` (`Fermionic Hopfions`) + `pauli_twist_pi` + `gw_vis_mode = 1` + live `LIGO Strain` plot
      • `CMB_PLOT_MODES` (`0 = TT 1k`, `1 = TT 2.5k`, `2 = TE Pol`, `3 = EE Pol`) up to `l = 2500`
      • `run_gaia_day1_rehearsal` blind sky-level astrometric + RV vetting (`FAVORS_TNAP_WEAVE` vs `FAVORS_LAMBDA_CDM`)
      • `reset_all_sidebar_defaults()` restoring `gw_vis_mode = 0` and `pauli_twist_pi = False`
    """
    from weaveforge.studio.app import WeaveForgeStudioApp

    app = WeaveForgeStudioApp(
        width=1680,
        height=960,
        initial_arena=1,
        initial_mod_id="tnap_weave",
        vsync=False,
        visible=False,
    )
    try:
        app.select_left_tab(1)
        app.arena1.set_grid_resolution(64)
        app.arena1.reset_field(6)
        app.reset_active_camera()
        app.arena1.gw_vis_mode = 1
        app.arena1.step(app.working_mod, app.compiled, n_substeps=160)
        assert len(app.arena1.gw_strain_t) >= 4

        # Cycle through all 4 CMB channels (TT 1k, TT 2.5k, TE Pol, EE Pol) while drawing LIGO strain plot
        for cm in range(4):
            app.cmb_plot_mode = cm
            app.render_single_frame(dt_override=0.016)

        # Switch to Preset 7 (Fermionic Hopfions) and toggle Antisymmetric Singlet Twist
        app.arena1.pauli_twist_pi = True
        app.arena1.reset_field(7)
        app.reset_active_camera()
        app.render_single_frame(dt_override=0.016)

        # Run blind Gaia DR4 Day-1 rehearsals under TNaP Weave and Lambda-CDM
        reh_tnap = app.run_gaia_day1_rehearsal(n_binaries=3600, truth_mode="tnap_weave")
        assert reh_tnap["verdict"] == "FAVORS_TNAP_WEAVE" and reh_tnap["delta_chi2"] > 16.0
        app.render_single_frame(dt_override=0.016)

        reh_cdm = app.run_gaia_day1_rehearsal(n_binaries=3600, truth_mode="lambda_cdm")
        assert reh_cdm["verdict"] == "FAVORS_LAMBDA_CDM" and reh_cdm["delta_chi2"] < -16.0
        app.render_single_frame(dt_override=0.016)

        # Reset all sidebar settings and verify Phase 4 Arena 1 defaults restore cleanly
        app.reset_all_sidebar_defaults()
        assert app.arena1.gw_vis_mode == 0
        assert app.arena1.pauli_twist_pi is False
    finally:
        app.shutdown()
        gc.collect()

