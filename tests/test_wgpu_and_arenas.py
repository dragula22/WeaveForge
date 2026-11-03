"""
GPU Numerical Parity & 4-Arena Real Hardware Tests (`tests/test_wgpu_and_arenas.py`)
====================================================================================
Tests:
  • Exact 1-step 3D PDE numerical parity between CPU NumPy (`accel_numpy_fn`) and Vulkan `wgpu` (`WGSL`)!
  • Cross-vendor Vulkan `wgpu` parity between NVIDIA DiscreteGPU and AMD Radeon Vulkan adapter (when both present)!
  • Coexistence of `ModernGL` OpenGL 4.6 and `wgpu` Vulkan in the same process without WGL context corruption!
  • Real offscreen GPU execution & finite-pixel rendering across all 4 Arenas (`Arena 1`–`Arena 4`) +
    full `WeaveForgeStudioApp` frame loop!
"""

from __future__ import annotations

from pathlib import Path
import moderngl
import numpy as np

from weaveforge.arenas.arena1_knot_tank import Arena1KnotTank
from weaveforge.arenas.arena2_black_hole_forge import Arena2BlackHoleForge
from weaveforge.arenas.arena3_galaxy_collider import Arena3GalaxyCollider
from weaveforge.arenas.arena4_cosmic_crucible import Arena4CosmicCrucible
from weaveforge.compute.wgpu_engine import WgpuComputeEngine, build_3d_topological_initial_state
from weaveforge.core.camera import Camera3D
from weaveforge.core.frameworks import get_mods_by_id
from weaveforge.core.symbolic_compiler import compile_master_action


def _cpu_single_step_reference(
    phi0_xyzw: np.ndarray,
    vel0_xyzw: np.ndarray,
    mod,
    comp,
    box_size: float,
    dt: float,
) -> np.ndarray:
    """Computes 1 exact periodic finite-difference Störmer-Verlet step on CPU using SymPy's accel_numpy_fn."""
    n = phi0_xyzw.shape[0]
    dx = box_size / float(n)
    # phi_arr has shape (3, N, N, N) matching (z, y, x) layout in phi0_xyzw
    phi = np.transpose(phi0_xyzw[..., :3], (3, 0, 1, 2)).astype(np.float64)
    vel = np.transpose(vel0_xyzw[..., :3], (3, 0, 1, 2)).astype(np.float64)

    # Note: X is axis 3, Y is axis 2, Z is axis 1 in (comp, Z, Y, X)!
    xp = np.roll(phi, -1, axis=3)
    xm = np.roll(phi, 1, axis=3)
    yp = np.roll(phi, -1, axis=2)
    ym = np.roll(phi, 1, axis=2)
    zp = np.roll(phi, -1, axis=1)
    zm = np.roll(phi, 1, axis=1)

    grad = np.zeros((3, 3, n, n, n), dtype=np.float64)
    grad[:, 0] = (xp - xm) / (2.0 * dx)
    grad[:, 1] = (yp - ym) / (2.0 * dx)
    grad[:, 2] = (zp - zm) / (2.0 * dx)
    lap = (xp + xm + yp + ym + zp + zm - 6.0 * phi) / (dx * dx)

    p = mod.parameters
    acc = comp.accel_numpy_fn(phi, grad, lap, p.c_w, p.m_eff, p.g4, p.g6)
    v_new = np.clip(vel + dt * acc, -50.0, 50.0)
    p_new = np.clip(phi + dt * v_new, -25.0, 25.0)
    return np.transpose(p_new, (1, 2, 3, 0)).astype(np.float32)


def test_cpu_numpy_vs_wgpu_vulkan_3d_voxel_parity() -> None:
    mod = get_mods_by_id()["tnap_weave"]
    comp = compile_master_action(mod)

    grid_n = 32
    box_size = 12.0
    dt = 0.02
    phi0, vel0 = build_3d_topological_initial_state(grid_n=grid_n, preset="trefoil_knot", box_size=box_size)

    cpu_next = _cpu_single_step_reference(phi0, vel0, mod, comp, box_size=box_size, dt=dt)

    eng = WgpuComputeEngine()
    gpu_res = eng.run_3d_knot_evolution(
        comp,
        mod,
        grid_n=grid_n,
        n_steps=1,
        box_size=box_size,
        dt=dt,
        phi_init_override=phi0,
        vel_init_override=vel0,
    )
    gpu_next = gpu_res["phi_final"][..., :3]
    max_abs_err = float(np.max(np.abs(cpu_next - gpu_next)))
    assert max_abs_err < 1.0e-4, f"CPU vs WGSL 3D PDE mismatch: max_abs_err={max_abs_err}"


def test_cross_vendor_nvidia_and_amd_vulkan_parity_if_available() -> None:
    adapters = WgpuComputeEngine.list_available_adapters()
    has_nv = any("nvidia" in a.lower() for a in adapters)
    has_amd = any("amd" in a.lower() or "radeon" in a.lower() for a in adapters)
    if not (has_nv and has_amd):
        return

    mod = get_mods_by_id()["tnap_weave"]
    comp = compile_master_action(mod)
    eng_nv = WgpuComputeEngine(prefer_vendor="nvidia")
    eng_amd = WgpuComputeEngine(prefer_vendor="amd")

    r_nv = eng_nv.run_3d_knot_evolution(comp, mod, grid_n=48, n_steps=30, preset="z3_braid")
    r_amd = eng_amd.run_3d_knot_evolution(comp, mod, grid_n=48, n_steps=30, preset="z3_braid")
    assert abs(float(r_nv["retention_ratio"]) - float(r_amd["retention_ratio"])) < 1.0e-3


def test_all_four_arenas_and_moderngl_wgpu_coexistence() -> None:
    ctx = moderngl.create_standalone_context(require=430)
    # Instantiate WgpuComputeEngine AFTER ModernGL context to verify WGL context is never unbound!
    wgpu_eng = WgpuComputeEngine()
    assert "via OpenGL" not in wgpu_eng.adapter_summary

    mods = get_mods_by_id()
    tnap_mod = mods["tnap_weave"]
    tnap_comp = compile_master_action(tnap_mod)
    cdm_mod = mods["lambda_cdm"]
    cdm_comp = compile_master_action(cdm_mod)

    tex = ctx.texture((320, 200), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    cam = Camera3D(distance=14.0, yaw=0.5, pitch=0.3)

    # Arena 1: test all 4 presets + resolution switch
    a1 = Arena1KnotTank(ctx, initial_grid_n=64)
    for p_idx in range(4):
        a1.reset_field(p_idx)
        a1.step(tnap_mod, tnap_comp, n_substeps=2)
        a1.render(fbo, cam, wall_time=0.5)
        arr = np.frombuffer(fbo.read(components=4), dtype=np.uint8)
        assert arr.max() > 20
    a1.set_grid_resolution(128)
    a1.step(tnap_mod, tnap_comp, n_substeps=1)
    a1.release()

    # Arena 2: test all 3 submodes + exact 3,189.0685 km zero-crossing
    a2 = Arena2BlackHoleForge(ctx)
    for sm in (0, 1, 2):
        a2.submode = sm
        a2.render(fbo, cam, tnap_comp, wall_time=1.0)
        arr = np.frombuffer(fbo.read(components=4), dtype=np.uint8)
        assert arr.max() > 20
    a2.probe_alt_km = 0.5 * 6378.137
    tel = a2.compute_orbital_clock_telemetry(tnap_comp)
    assert abs(tel["net_us_day"]) < 1.0e-9
    a2.release()

    # Arena 3: test 14 km/s slow dwarf splash difference (TNaP > Lambda-CDM)
    a3 = Arena3GalaxyCollider(ctx, particle_count=100_000)
    a3.reset_galaxy(3)
    for _ in range(60):
        a3.step(cdm_mod, cdm_comp, dt=0.04)
    a3._sample_telemetry()
    splash_cdm = a3.transverse_splash_ratio

    a3.reset_galaxy(3)
    for _ in range(60):
        a3.step(tnap_mod, tnap_comp, dt=0.04)
    a3._sample_telemetry()
    splash_tnap = a3.transverse_splash_ratio
    assert splash_tnap > splash_cdm * 1.35
    a3.render(fbo, cam, wall_time=1.0)
    a3.release()

    # Arena 4: test CPT Hourglass render & finite waist curvature
    a4 = Arena4CosmicCrucible(ctx)
    assert np.isfinite(a4.waist_kretschmann())
    a4.render(fbo, cam, tnap_mod, wall_time=1.0)
    arr4 = np.frombuffer(fbo.read(components=4), dtype=np.uint8)
    assert arr4.max() > 20
    a4.release()

    fbo.release()
    tex.release()
    ctx.release()


def test_studio_hidden_smoke_launch(tmp_path: Path) -> None:
    from weaveforge.studio.app import WeaveForgeStudioApp

    out_png = tmp_path / "smoke_studio.png"
    app = WeaveForgeStudioApp(width=960, height=600, initial_arena=2, vsync=False, visible=False)
    app.run(max_frames=4, screenshot_out=out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 15_000


def test_wgpu_knot_boundary_zero_clipping_and_long_horizon_energy_stability() -> None:
    phi0, _ = build_3d_topological_initial_state(grid_n=64, preset="trefoil_knot", box_size=12.0)
    rho0 = np.sum(phi0[..., :3] ** 2, axis=-1)
    boundary_max = max(
        float(np.max(rho0[0, :, :])),
        float(np.max(rho0[-1, :, :])),
        float(np.max(rho0[:, 0, :])),
        float(np.max(rho0[:, -1, :])),
        float(np.max(rho0[:, :, 0])),
        float(np.max(rho0[:, :, -1])),
    )
    assert boundary_max < 1.0e-3

    ctx = moderngl.create_standalone_context(require=430)
    tnap_mod = get_mods_by_id()["tnap_weave"]
    tnap_comp = compile_master_action(tnap_mod)
    a1 = Arena1KnotTank(ctx, initial_grid_n=64)
    a1.step(tnap_mod, tnap_comp, n_substeps=2)
    l2_init = a1.l2_norm
    # Evolve 600 steps (well past the ~500-step false-vacuum explosion point of prior parameters!)
    for _ in range(60):
        a1.step(tnap_mod, tnap_comp, n_substeps=10)
    assert 0.65 * l2_init < a1.l2_norm < 1.45 * l2_init
    assert a1.current_peak_rho < 6.0
    assert abs(a1.hamiltonian_drift_pct) < 2.5
    a1.release()
    ctx.release()


def test_arena2_fermat_raytracer_shadow_scaling_across_metrics() -> None:
    ctx = moderngl.create_standalone_context(require=430)
    mods = get_mods_by_id()
    tex = ctx.texture((512, 512), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    # Polar view along +Y axis with spin=0 and overlays disabled so swallowed shadow core is pure black
    cam = Camera3D(distance=28.0, yaw=0.0, pitch=1.568, fov_deg=30.0)
    a2 = Arena2BlackHoleForge(ctx)
    a2.spin_a = 0.0
    a2.show_bhex_gauge = False
    a2.show_phase_slips = False

    dark_counts = {}
    for mid in ("ghost_redteam_demo", "lambda_cdm", "tnap_weave"):
        comp = compile_master_action(mods[mid])
        a2.render(fbo, cam, comp, wall_time=0.0)
        img = np.frombuffer(fbo.read(components=4), dtype=np.uint8).reshape(512, 512, 4)
        luma = img[..., :3].max(axis=-1)
        # Swallowed pixels have near-zero HDR emission (mapped only to base tone floor <= 4)
        dark_counts[mid] = int(np.sum(luma <= 5))

    # Shadow area scales as b_crit^2: TNaP (5.437^2) > GR (5.196^2) >> 1911 half-bending (2.718^2)
    assert dark_counts["tnap_weave"] > dark_counts["lambda_cdm"] * 1.06
    assert dark_counts["lambda_cdm"] > dark_counts["ghost_redteam_demo"] * 3.10

    a2.release()
    fbo.release()
    tex.release()
    ctx.release()


def test_arena3_multigrid_bullet_cluster_offset_and_sparc_spawner() -> None:
    from weaveforge.arenas.arena4_cosmic_crucible import get_sparc_175_cached

    ctx = moderngl.create_standalone_context(require=430)
    mods = get_mods_by_id()
    a3 = Arena3GalaxyCollider(ctx, particle_count=120_000)

    # 1. Bullet Cluster (preset 2): CDM & TNaP moving halos/pleats separate from stalled ICM gas;
    #    pure baryon-only Classic MOND keeps lensing pinned near the 66%-mass central gas cloud!
    offsets = {}
    for mid in ("lambda_cdm", "tnap_weave", "classic_mond"):
        m = mods[mid]
        c = compile_master_action(m)
        a3.reset_galaxy(2)
        for _ in range(72):
            a3.step(m, c, dt=0.04)
        a3._sample_telemetry()
        offsets[mid] = a3.lensing_gas_offset_kpc

    assert offsets["lambda_cdm"] > 80.0
    assert offsets["tnap_weave"] > 80.0
    assert offsets["classic_mond"] < 45.0

    # 2. Live 175-galaxy SPARC 3D spawner
    gals = get_sparc_175_cached()
    a3.load_sparc_galaxy(gals[0])
    a3.step(mods["tnap_weave"], compile_master_action(mods["tnap_weave"]), dt=0.03)
    a3._sample_telemetry()
    assert a3.v_outer_kms > 5.0

    a3.release()
    ctx.release()


def test_arena1_256_nonblocking_replay_zero_stall_and_lossless_resume() -> None:
    import gc
    import time

    gc.collect()
    ctx = moderngl.create_standalone_context(require=430)
    mod = get_mods_by_id()["tnap_weave"]
    cam = Camera3D()
    tex = ctx.texture((3840, 2160), 4)
    fbo = ctx.framebuffer(color_attachments=[tex])

    # Cold instantiation matching exact task signature (no pre-compiled shader arg, no untimed warmup frame)
    a1 = Arena1KnotTank(ctx, grid_n=256)
    a1.replay_enabled = True
    try:
        frame_ms: list[float] = []
        for i in range(40):
            t0 = time.perf_counter()
            a1.step(mod)
            a1.render(fbo, cam, (i + 1) * 0.016)
            ctx.finish()
            frame_ms.append((time.perf_counter() - t0) * 1000.0)

        max_ms = max(frame_ms)
        max_steady_ms = max(frame_ms[2:])
        assert max_ms < 25.0, f"Expected all 40 cold 4K 256^3 frames < 25 ms, got max={max_ms:.2f} ms ({frame_ms})"
        assert max_steady_ms < 18.0, (
            f"Expected steady-state 4K 256^3 frames[2:] < 18 ms, got max={max_steady_ms:.2f} ms ({frame_ms[2:]})"
        )
        assert a1.replay_frames_count >= 2
        # Verify 256^3 keyframes are 2x2x2 downsampled to 128^3 float16 (24.0 MiB instead of 192.0 MiB)
        snap0 = a1._replay_frames[0]
        assert snap0.dtype == np.float16
        assert snap0.shape == (2, 128 * 128 * 128 * 3)
        assert abs(snap0.nbytes / (1024.0 * 1024.0) - 24.0) < 1e-6

        h_live_before = float(a1.hamiltonian_energy)
        step_live_before = int(a1.step_count)
        time_live_before = float(a1.sim_time)
        assert h_live_before > 10.0

        # Multi-keyframe scrub (trilinearly upsampled 128^3 fp16 -> 256^3 fp32 on GPU) + index clamping
        a1.load_replay_frame(0)
        assert a1.replay_cursor == 0
        assert np.isfinite(a1.hamiltonian_energy) and a1.hamiltonian_energy > 10.0
        a1.load_replay_frame(1)
        assert a1.replay_cursor == 1
        assert np.isfinite(a1.hamiltonian_energy) and a1.hamiltonian_energy > 10.0
        a1.load_replay_frame(-1)
        assert a1.replay_cursor == a1.replay_frames_count - 1
        a1.load_replay_frame(999)
        assert a1.replay_cursor == a1.replay_frames_count - 1
        a1.load_replay_frame(-999)
        assert a1.replay_cursor == 0

        # Resume live from lossless 256^3 float32 backup and verify Hamiltonian energy within < 1e-2
        a1.resume_live_from_replay()
        assert a1.replay_cursor == -1
        assert a1.step_count == step_live_before
        assert abs(a1.sim_time - time_live_before) < 1e-6
        assert abs(a1.hamiltonian_energy - h_live_before) < 1e-2

        # Mid-flight pending GL_FenceSync snapshot + hibernate_to_ram() must flush cleanly to 0.0 MiB VRAM
        a1._enqueue_replay_snapshot()
        assert a1._pending_replay_fence is not None
        prev_snaps = a1.replay_frames_count
        ddr5_256 = a1.hibernate_to_ram()
        assert a1.is_hibernated is True
        assert a1.allocated_vram_mb == 0.0
        assert a1._pending_replay_fence is None
        assert a1.replay_frames_count == prev_snaps + 1
        assert ddr5_256 > 380.0
        a1.wake_from_ram()
        assert a1.is_hibernated is False
        assert a1.allocated_vram_mb > 0.0

        # Empty-buffer edge case + native <= 128^3 and odd (128, 256] resolution replay check
        a1.clear_replay_buffer()
        assert a1.replay_frames_count == 0
        a1.load_replay_frame(0)
        a1.resume_live_from_replay()
        a1.set_resolution(64)
        for _ in range(4):
            a1.step(mod)
            time.sleep(0.002)
        assert a1.replay_frames_count >= 1
        assert a1._replay_frames[0].shape == (2, 64 * 64 * 64 * 3)
        a1.set_resolution(131)
        a1.flush_pending_replay_snapshot()
        assert a1.replay_frames_count == 1
        assert a1._replay_frames[0].shape == (2, 65 * 65 * 65 * 3)
        a1.load_replay_frame(0)
        assert np.isfinite(a1.hamiltonian_energy)
    finally:
        a1.release()
        fbo.release()
        tex.release()
        ctx.release()


def test_arena1_and_arena3_hibernate_wake_and_arena3_async_replay() -> None:
    import gc
    import time

    gc.collect()
    ctx = moderngl.create_standalone_context(require=430)
    mod = get_mods_by_id()["tnap_weave"]
    comp = compile_master_action(mod)
    cam = Camera3D()
    tex = ctx.texture((1920, 1080), 4)
    fbo = ctx.framebuffer(color_attachments=[tex])

    a1 = Arena1KnotTank(ctx, grid_n=128)
    # Cold 1,000,000-star Arena 3 (no pre-compiled shader arg, no untimed warmup frame)
    a3 = Arena3GalaxyCollider(ctx, particle_count=1_000_000)
    try:
        # 1. Arena 1 hibernate_to_ram() + wake_from_ram()
        for _ in range(12):
            a1.step(mod)
        a1._poll_async_telemetry(wait=True)
        h_a1_pre = float(a1.hamiltonian_energy)
        steps_a1_pre = int(a1.step_count)
        assert a1.allocated_vram_mb > 0.0

        ddr5_a1 = a1.hibernate_to_ram()
        assert a1.is_hibernated is True
        assert a1.allocated_vram_mb == 0.0
        assert ddr5_a1 > 0.0
        # Idempotent second call when already hibernated
        assert a1.hibernate_to_ram() == ddr5_a1
        assert a1.allocated_vram_mb == 0.0

        a1.wake_from_ram()
        assert a1.is_hibernated is False
        assert a1.allocated_vram_mb > 0.0
        assert a1.step_count == steps_a1_pre
        assert abs(a1.hamiltonian_energy - h_a1_pre) < 1e-4

        # 2. Cold Frame-0 check + 36 Bullet Cluster frames at 1M stars (16 MiB GL_FenceSync replay + async telemetry)
        t_f0 = time.perf_counter()
        a3.step(mod, comp, dt=0.03)
        a3.render(fbo, cam, 0.0)
        ctx.finish()
        cold_f0_ms = (time.perf_counter() - t_f0) * 1000.0
        assert cold_f0_ms < 25.0, f"Expected cold 1M-star Arena 3 Frame 0 < 25 ms, got {cold_f0_ms:.2f} ms"

        a3.reset_galaxy(2)
        a3_frame_ms: list[float] = []
        offsets_seen: list[float] = []
        for i in range(36):
            t0 = time.perf_counter()
            a3.step(mod, comp, dt=0.03)
            a3.render(fbo, cam, (i + 1) * 0.016)
            offsets_seen.append(float(a3.lensing_gas_offset_kpc))
            ctx.finish()
            a3_frame_ms.append((time.perf_counter() - t0) * 1000.0)

        assert max(a3_frame_ms) < 25.0, (
            f"Expected all 36 1M-star Bullet Cluster frames < 25 ms, got max={max(a3_frame_ms):.2f} ms ({a3_frame_ms})"
        )
        assert a3.replay_frames_count >= 2
        assert np.isfinite(offsets_seen[-1]) and offsets_seen[-1] >= 0.0
        steps_a3_pre = int(a3.step_count)
        assert a3._pos_buf is not None
        pos_pre = a3._copy_ssbo_to_ram_f32_vec4(a3._pos_buf, a3.particle_count)

        a3.load_replay_frame(0)
        assert a3.replay_cursor == 0
        a3.load_replay_frame(1)
        assert a3.replay_cursor == 1
        a3.resume_live_from_replay()
        assert a3.replay_cursor == -1
        assert a3.step_count == steps_a3_pre

        # Enqueue mid-flight replay fence immediately before hibernate_to_ram()
        a3._enqueue_replay_snapshot()
        assert a3._pending_replay_fence is not None
        ddr5_a3 = a3.hibernate_to_ram()
        assert a3.is_hibernated is True
        assert a3.allocated_vram_mb == 0.0
        assert a3._pending_replay_fence is None
        assert ddr5_a3 > 0.0

        a3.wake_from_ram()
        assert a3.is_hibernated is False
        assert a3.allocated_vram_mb > 0.0
        assert a3.step_count == steps_a3_pre
        assert a3._pos_buf is not None
        pos_post = a3._copy_ssbo_to_ram_f32_vec4(a3._pos_buf, a3.particle_count)
        assert np.allclose(pos_pre, pos_post, atol=0.0)
    finally:
        a1.release()
        a3.release()
        fbo.release()
        tex.release()
        ctx.release()


def test_arena4_camb_npz_disk_cache_fast_load_and_slider_evaluate_mod(tmp_path: Path) -> None:
    import copy
    import gc
    import pytest
    import time
    from weaveforge.arenas import arena4_cosmic_crucible as a4m

    gc.collect()
    npz_path = a4m._CAMB_LUT_DISK_CACHE_PATH
    assert npz_path.is_file(), f"Missing pre-generated CAMB LUT cache: {npz_path}"
    assert npz_path.stat().st_size < 250 * 1024

    # Verify corrupt / mismatched signature .npz fails gracefully without corrupting state
    bad_npz = tmp_path / "bad_sig.npz"
    np.savez_compressed(bad_npz, signature=np.array("v0|mismatch"))
    assert a4m.load_camb_lut_disk_cache(cache_path=bad_npz, force_reload=True) is False

    # Cold reload of real camb_tt_lut_cache.npz into DDR5 RAM must complete in < 50 ms with zero background thread
    a4m._CAMB_WARM_THREAD = None
    t0 = time.perf_counter()
    loaded = a4m.load_camb_lut_disk_cache(force_reload=True)
    load_ms = (time.perf_counter() - t0) * 1000.0
    assert loaded is True
    assert load_ms < 50.0, f"Expected CAMB .npz DDR5 load < 50 ms, got {load_ms:.2f} ms"
    assert a4m.is_camb_ddr5_cache_ready() is True
    a4m.warm_camb_ddr5_cache_async()
    assert a4m._CAMB_WARM_THREAD is None

    ctx = moderngl.create_standalone_context(require=430)
    a4 = Arena4CosmicCrucible(ctx)
    try:
        # Missing mod error path
        with pytest.raises(ValueError):
            a4.evaluate_mod(None)

        base_mod = get_mods_by_id()["tnap_weave"]
        comp = compile_master_action(base_mod)
        # Warm one-time SPARC-175 catalog fit so slider drags isolate cosmological evaluation
        a4.evaluate_mod(base_mod, comp)

        couplings = (0.0, 0.25, 0.50, 0.75, 1.00, 1.35)
        variants = []
        for ppc in couplings:
            m_var = copy.deepcopy(base_mod)
            m_var.parameters.pinned_pleat_coupling = ppc
            variants.append(m_var)
        gc.collect()

        eval_times_ms: list[float] = []
        omega_vals: list[float] = []
        for m_var in variants:
            t_ev = time.perf_counter()
            rep = a4.evaluate_mod(m_var, comp)
            eval_times_ms.append((time.perf_counter() - t_ev) * 1000.0)
            omega_vals.append(rep.omega_m_effective)
            assert np.all(np.isfinite(rep.cmb_dl_mod))

        assert max(eval_times_ms) < 15.0, (
            f"Expected evaluate_mod across pinned_pleat_coupling < 15 ms, got {eval_times_ms}"
        )
        assert all(omega_vals[i] < omega_vals[i + 1] for i in range(len(omega_vals) - 1))

        # High-Omega_m boundary check (cdm_halo_scale + pinned_pleat_coupling -> Omega_m_eff ~ 0.88 in [0.60, 1.00])
        lcdm = copy.deepcopy(get_mods_by_id()["lambda_cdm"])
        lcdm.parameters.cdm_halo_scale = 1.75
        lcdm_comp = compile_master_action(lcdm)
        a4.evaluate_mod(lcdm, lcdm_comp)
        lcdm.parameters.pinned_pleat_coupling = 1.40
        t_hi = time.perf_counter()
        rep_hi = a4.evaluate_mod(lcdm, lcdm_comp)
        hi_ms = (time.perf_counter() - t_hi) * 1000.0
        assert 0.80 < rep_hi.omega_m_effective <= 1.00
        assert hi_ms < 15.0

        # Invalidated/missing .npz fallback check: async background rebuild must NEVER stall UI slider drags
        orig_cache_path = a4m._CAMB_LUT_DISK_CACHE_PATH
        try:
            a4m._stop_camb_warmer()
            a4m._CAMB_LUT_DISK_CACHE_PATH = tmp_path / "missing_rebuild_lut.npz"
            with a4m._CAMB_LUT_LOCK:
                a4m._CAMB_LUT_READY = False
                a4m._CAMB_DDR5_SPLINES.clear()
                a4m._EXACT_CAMB_CURVES.clear()
                a4m._CAMB_FAILED_NPZ_STAT = None
            a4m._PLANCK_REF_CACHE = None

            a4m.warm_camb_ddr5_cache_async(wait=False)
            time.sleep(0.05)
            fallback_ms: list[float] = []
            for m_var in variants[:4]:
                t_fb = time.perf_counter()
                rep_fb = a4.evaluate_mod(m_var, comp)
                fallback_ms.append((time.perf_counter() - t_fb) * 1000.0)
                assert np.all(np.isfinite(rep_fb.cmb_dl_mod))
            assert max(fallback_ms) < 15.0, (
                f"Expected non-blocking fallback slider drags < 15 ms during async CAMB rebuild, got {fallback_ms}"
            )
        finally:
            a4m._stop_camb_warmer()
            if a4m._CAMB_WARM_THREAD is not None:
                a4m._CAMB_WARM_THREAD.join(timeout=2.0)
                a4m._CAMB_WARM_THREAD = None
            a4m._CAMB_LUT_DISK_CACHE_PATH = orig_cache_path
            assert a4m.load_camb_lut_disk_cache(force_reload=True) is True
    finally:
        a4.release()
        ctx.release()


def test_lan_compute_dispatcher_fast_offline_failover_and_local_igpu_subprocess() -> None:
    import os
    import pytest
    import time
    from pydantic import ValidationError
    from weaveforge.compute import LanComputeDispatcher

    tnap = get_mods_by_id()["tnap_weave"]

    # 1. Unreachable RFC 5737 TEST-NET-1 blackhole IP must fail over / report offline in < 500 ms
    disp_lan = LanComputeDispatcher(worker_url="http://192.0.2.1:8765", execution_mode="lan")
    t0 = time.perf_counter()
    reachable = disp_lan.probe_lan_reachable()
    probe_ms = (time.perf_counter() - t0) * 1000.0
    assert reachable is False
    assert probe_ms < 500.0, f"Expected unreachable LAN probe < 500 ms, got {probe_ms:.2f} ms"

    t_disp = time.perf_counter()
    disp_lan.dispatch_async(tnap, execution_mode="lan")
    deadline = time.perf_counter() + 2.0
    while disp_lan.is_running and time.perf_counter() < deadline:
        time.sleep(0.01)
    lan_fail_ms = (time.perf_counter() - t_disp) * 1000.0
    assert disp_lan.is_running is False
    assert lan_fail_ms < 500.0, f"Expected LAN offline failover < 500 ms, got {lan_fail_ms:.2f} ms"
    assert "Offline" in disp_lan.last_status
    assert disp_lan.last_result is None

    # Empty endpoint, out-of-range port, and invalid request_overrides edge cases
    disp_empty = LanComputeDispatcher(worker_url="", execution_mode="lan")
    disp_empty.dispatch_async(tnap, execution_mode="lan")
    time.sleep(0.05)
    assert disp_empty.is_running is False
    assert "Offline" in disp_empty.last_status

    disp_bad_port = LanComputeDispatcher(worker_url="http://192.0.2.1:99999", execution_mode="lan")
    assert disp_bad_port.probe_lan_reachable() is False
    with pytest.raises(ValidationError):
        disp_bad_port.dispatch_async(tnap, request_overrides={"knot_grid_n": "invalid_int"})
    assert disp_bad_port.is_running is False

    # 2. Explicit local_igpu subprocess execution mode (isolated spawn child process)
    disp_igpu = LanComputeDispatcher(worker_url="http://192.0.2.1:8765", execution_mode="local_igpu")
    assert disp_igpu.prefer_igpu_on_local is True
    fast_overrides = {
        "g4_grid": [0.0, 9.8],
        "g6_grid": [2.5],
        "knot_grid_n": 24,
        "knot_steps": 6,
        "optimize_sparc_ml": False,
        "gaia_mock_pairs": 200,
    }
    disp_igpu.dispatch_async(tnap, execution_mode="local_igpu", request_overrides=fast_overrides)
    deadline_igpu = time.perf_counter() + 30.0
    while disp_igpu.is_running and time.perf_counter() < deadline_igpu:
        time.sleep(0.05)

    assert disp_igpu.is_running is False
    assert disp_igpu.last_result is not None
    assert disp_igpu.last_result["status"] == "ok"
    assert disp_igpu.last_result["prefer_igpu"] is True
    assert disp_igpu.last_result["execution_backend"] == "local_subprocess"
    assert disp_igpu.last_result["worker_pid"] != os.getpid()
    assert disp_igpu.last_result["sparc_175"]["n_galaxies"] == 175

    # 3. Auto mode fast failover (< 500 ms LAN probe) from unreachable blackhole IP to local subprocess
    disp_auto = LanComputeDispatcher(worker_url="http://192.0.2.1:8765", execution_mode="auto")
    disp_auto.dispatch_async(tnap, execution_mode="auto", request_overrides=fast_overrides)
    deadline_auto = time.perf_counter() + 30.0
    while disp_auto.is_running and time.perf_counter() < deadline_auto:
        time.sleep(0.05)

    assert disp_auto.is_running is False
    assert disp_auto.last_result is not None
    assert disp_auto.last_result["lan_probe_ms"] < 500.0
    assert disp_auto.last_result["execution_backend"] == "local_subprocess"


