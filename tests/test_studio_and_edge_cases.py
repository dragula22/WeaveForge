"""
End-to-End Hardware & UI Regression Tests (`tests/test_studio_and_edge_cases.py`)
=================================================================================
Exercises and proves fixes for:
  1. Studio UI Tab 2 (`Arena & Couplings`) `hamiltonian_drift_pct` telemetry in Arena 1,
     full traversal across all 3 left-panel tabs, all 4 arenas, all submodes, and Shader Inspector modal.
  2. All 3 Arena 1 volumetric color modes (`Sapphire-Gold Phase Weave`, `Topological Rainbow Phase`,
     `Thermal Energy Density`) producing distinct GPU ray-marched images.
  3. Real `512³` (`134,217,728` voxels) packed 3-float (`12` B/voxel = `1.50 GiB` < `2^31`) storage buffers
     on BOTH OpenGL 4.6 (`Arena1KnotTank`) and Vulkan `wgpu` (`WgpuComputeEngine` with `required_limits`).
  4. Non-colliding custom `.toml` saving (`save_custom_mod`) + immediate preset dropdown reload.
  5. Honest 3-state 5-Gate Health Audit header (`PASS`, `CAUTION` on false-vacuum `WARN`, `RED-TEAM ALERT` on `FAIL`).
  6. Arena 3 `t=0` & paused Multigrid potential solve respecting active `has_cdm_halo` / `pinned_pleat_coupling`,
     plus Tab 3 (`Rig 2 LAN Worker`) 2D `(g4, g6)` `knot_phase_sweep` heatmap & matrix rendering.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import moderngl
import numpy as np
import pytest

from weaveforge.arenas.arena1_knot_tank import Arena1KnotTank, KNOT_COLOR_MODES
from weaveforge.arenas.arena3_galaxy_collider import Arena3GalaxyCollider
from weaveforge.compute.wgpu_engine import WgpuComputeEngine
from weaveforge.core.camera import Camera3D
from weaveforge.core.frameworks import DEFAULT_MODS_DIR, get_mods_by_id
from weaveforge.core.symbolic_compiler import compile_master_action
from weaveforge.data.gaia_dr4_pipeline import load_gaia_dr4_csv
from weaveforge.studio.app import WeaveForgeStudioApp, health_audit_header
from weaveforge.studio.theme import COL_FAIL, COL_PASS, COL_WARN
from weaveforge.studio.video_recorder import (
    AsyncVideoRecorder,
    probe_nvenc_support,
    probe_video_stream_info,
    resolve_ffmpeg_exe,
)


def test_arena1_all_three_color_modes_produce_distinct_renders() -> None:
    assert len(KNOT_COLOR_MODES) == 3
    ctx = moderngl.create_standalone_context(require=430)
    tex = ctx.texture((256, 192), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    a1 = Arena1KnotTank(ctx, initial_grid_n=64)
    try:
        mod = get_mods_by_id()["tnap_weave"]
        comp = compile_master_action(mod)
        a1.step(mod, comp, n_substeps=2)
        assert a1.energy_drift_pct == pytest.approx(a1.hamiltonian_drift_pct)

        cam = Camera3D(yaw=0.62, pitch=0.38, distance=13.5)
        renders = []
        for mode_idx in range(len(KNOT_COLOR_MODES)):
            a1.color_mode = mode_idx
            a1.render(fbo, cam, wall_time=0.5)
            img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape((192, 256, 3)).astype(np.float32)
            renders.append(img)

        diff_01 = float(np.mean(np.abs(renders[0] - renders[1])))
        diff_02 = float(np.mean(np.abs(renders[0] - renders[2])))
        diff_12 = float(np.mean(np.abs(renders[1] - renders[2])))
        assert diff_01 > 0.8, f"Palette 0 vs 1 too similar: {diff_01}"
        assert diff_02 > 0.8, f"Palette 0 vs 2 too similar: {diff_02}"
        assert diff_12 > 0.8, f"Palette 1 vs 2 too similar: {diff_12}"
    finally:
        a1.release()
        fbo.release()
        tex.release()
        ctx.release()


def test_512_cubed_storage_buffer_limit_wgpu_and_opengl() -> None:
    expected_bytes = (512 ** 3) * 12
    assert expected_bytes == 1_610_612_736
    assert expected_bytes < (2 ** 31) - 4

    mod = get_mods_by_id()["tnap_weave"]
    comp = compile_master_action(mod)

    # Part A: Vulkan/DX12 wgpu 512^3 allocation, init, 2-step PDE evolution, & exact 134M-voxel GPU reduction
    eng = WgpuComputeEngine()
    max_bind = int(eng.device.limits.get("max-storage-buffer-binding-size", 0))
    max_buf = int(eng.device.limits.get("max-buffer-size", 0))
    assert max_bind >= expected_bytes, f"Device storage binding limit too small: {max_bind}"
    assert max_buf >= expected_bytes, f"Device max buffer size too small: {max_buf}"

    res_512 = eng.run_3d_knot_evolution(
        comp,
        mod,
        grid_n=512,
        n_steps=2,
        preset="trefoil_knot",
    )
    assert res_512["storage_bytes_per_field"] == expected_bytes
    assert res_512["phi_final"] is None  # Avoids 2 GiB host copy by default at 512^3
    assert float(res_512["init_peak_rho"]) > 1.0
    assert 0.85 <= float(res_512["retention_ratio"]) <= 1.20
    mid_xy = res_512["mid_slice_xy"]
    assert isinstance(mid_xy, np.ndarray) and mid_xy.shape == (512, 512)
    assert np.all(np.isfinite(mid_xy))

    # Part B: OpenGL 4.6 Arena1KnotTank 512^3 allocation, init, step, 3D Hamiltonian reduction, & ray-march
    ctx = moderngl.create_standalone_context(require=430)
    tex = ctx.texture((256, 192), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    a1 = Arena1KnotTank(ctx, initial_grid_n=512)
    try:
        assert a1._buf_a is not None and a1._buf_b is not None and a1._buf_vel is not None
        assert a1._buf_a.size == expected_bytes
        assert a1._buf_b.size == expected_bytes
        assert a1._buf_vel.size == expected_bytes
        assert a1.allocated_vram_mb == pytest.approx(5632.0)

        a1.step(mod, comp, n_substeps=2)
        a1._sample_diagnostics(is_initial=False)
        assert np.isfinite(a1.hamiltonian_energy)
        assert np.isfinite(a1.hamiltonian_drift_pct)
        assert 0.85 <= a1.retention_ratio <= 1.20

        cam = Camera3D(yaw=0.5, pitch=0.35, distance=14.0)
        a1.render(fbo, cam, wall_time=0.2)
        rgb = np.frombuffer(fbo.read(components=3), dtype=np.uint8)
        assert float(np.std(rgb)) > 4.0
    finally:
        a1.release()
        fbo.release()
        tex.release()
        ctx.release()


def test_health_audit_header_pass_warn_and_fail_states() -> None:
    mods = get_mods_by_id()

    # 1. All 5 gates PASS -> green COL_PASS
    tnap_comp = compile_master_action(mods["tnap_weave"])
    assert tnap_comp.all_critical_passed is True
    assert tnap_comp.all_gates_passed is True
    assert tnap_comp.has_warnings is False
    col_p, txt_p = health_audit_header(tnap_comp)
    assert col_p == COL_PASS
    assert "ALL 5 GATES PASSED" in txt_p

    # 2. Coleman false-vacuum detonation (Gate 5 WARN: g4^2 > 16/3 m_eff^2 g6, g6 > 0) -> amber COL_WARN
    warn_mod = mods["tnap_weave"].clone()
    warn_mod.parameters.m_eff = 2.0
    warn_mod.parameters.g4 = 14.0
    warn_mod.parameters.g6 = 2.5
    warn_comp = compile_master_action(warn_mod)
    assert warn_comp.all_critical_passed is True
    assert warn_comp.all_gates_passed is False
    assert warn_comp.has_warnings is True
    col_w, txt_w = health_audit_header(warn_comp)
    assert col_w == COL_WARN
    assert "CAUTION" in txt_w
    assert "ALL CRITICAL GATES PASSED" not in txt_w

    # 3. Critical FAIL -> red COL_FAIL
    fail_comp = compile_master_action(mods["ghost_redteam_demo"])
    assert fail_comp.all_critical_passed is False
    assert fail_comp.all_gates_passed is False
    col_f, txt_f = health_audit_header(fail_comp)
    assert col_f == COL_FAIL
    assert "RED-TEAM ALERT" in txt_f

    # 4. Non-positive asymptotic optical metric n(0) <= 0 rejected by Gate 4
    neg_n_mod = mods["tnap_weave"].clone()
    neg_n_mod.action.optical_metric_n = "-1 - 2*U"
    neg_n_comp = compile_master_action(neg_n_mod)
    ppn_gate = next(g for g in neg_n_comp.health_gates if g.gate_id == "ppn_solar")
    assert ppn_gate.passed is False
    assert neg_n_comp.all_critical_passed is False


def test_arena3_t0_multigrid_potential_respects_cdm_halo_and_paused_sync() -> None:
    ctx = moderngl.create_standalone_context(require=430)
    a3 = Arena3GalaxyCollider(ctx, particle_count=100_000)
    try:
        mods = get_mods_by_id()
        tnap_mod = mods["tnap_weave"]
        tnap_comp = compile_master_action(tnap_mod)
        cdm_mod = mods["lambda_cdm"]
        cdm_comp = compile_master_action(cdm_mod)

        # Reset Preset 0 (NGC 3198) at t=0 without calling step()
        a3.reset_galaxy(0, mod=tnap_mod, compiled=tnap_comp)
        assert a3.step_count == 0
        phi_bary = np.frombuffer(a3._mg_ua[0].read(), dtype=np.float32).copy()

        a3.reset_galaxy(0, mod=cdm_mod, compiled=cdm_comp)
        assert a3.step_count == 0
        phi_cdm = np.frombuffer(a3._mg_ua[0].read(), dtype=np.float32).copy()

        depth_bary = float(np.max(np.abs(phi_bary)))
        depth_cdm = float(np.max(np.abs(phi_cdm)))
        mean_bary = float(np.mean(np.abs(phi_bary)))
        mean_cdm = float(np.mean(np.abs(phi_cdm)))
        # NGC 3198 has extended NFW CDM halo (M_halo/M_bary ~ 4.6 -> total mass ~5.6x):
        # central peak |Phi_N| is >2.0x deeper and box-averaged |Phi_N| across the 38 kpc domain is >4.0x deeper at t=0!
        assert depth_cdm > 2.0 * depth_bary, f"Expected deeper central CDM potential at t=0: {depth_cdm} vs {depth_bary}"
        assert mean_cdm > 4.0 * mean_bary, f"Expected deeper extended NFW halo potential at t=0: {mean_cdm} vs {mean_bary}"

        # Paused mod switching via sync_mod_state(..., re_solve_field=True) updates potential immediately
        a3.sync_mod_state(tnap_mod, tnap_comp, re_solve_field=True)
        phi_synced_bary = np.frombuffer(a3._mg_ua[0].read(), dtype=np.float32).copy()
        assert float(np.max(np.abs(phi_synced_bary))) == pytest.approx(depth_bary, rel=0.05)

        # Bullet Cluster (Preset 2) at t=0: pre-collision subclusters co-locate gas & halos (offset ~ 0),
        # while Multigrid |Phi_N| is >3.2x deeper for CDM and ballistic-pleat TNAP than Classic MOND!
        mond_mod = mods["classic_mond"]
        mond_comp = compile_master_action(mond_mod)
        a3.reset_galaxy(2, mod=mond_mod, compiled=mond_comp)
        phi_bullet_mond = float(np.max(np.abs(np.frombuffer(a3._mg_ua[0].read(), dtype=np.float32))))

        a3.reset_galaxy(2, mod=tnap_mod, compiled=tnap_comp)
        phi_bullet_tnap = float(np.max(np.abs(np.frombuffer(a3._mg_ua[0].read(), dtype=np.float32))))

        a3.reset_galaxy(2, mod=cdm_mod, compiled=cdm_comp)
        phi_bullet_cdm = float(np.max(np.abs(np.frombuffer(a3._mg_ua[0].read(), dtype=np.float32))))
        assert phi_bullet_tnap > 3.2 * phi_bullet_mond
        assert phi_bullet_cdm > 3.2 * phi_bullet_mond

        # Evolve past pericenter so X-ray gas shocks at x~0 while halos/galaxies pass through, then test paused sync!
        for _ in range(48):
            a3.step(cdm_mod, cdm_comp, dt=0.04)
        offset_cdm = a3.lensing_gas_offset_kpc
        assert offset_cdm > 80.0

        a3.sync_mod_state(mond_mod, mond_comp, re_solve_field=True)
        offset_mond_paused = a3.lensing_gas_offset_kpc
        assert offset_cdm > offset_mond_paused + 40.0
    finally:
        a3.release()
        ctx.release()


def test_studio_full_ui_tabs_arenas_custom_save_and_phase_sweep(tmp_path: Path) -> None:
    tmp_mods = tmp_path / "mods"
    tmp_mods.mkdir(parents=True, exist_ok=True)
    for src_toml in sorted(DEFAULT_MODS_DIR.glob("*.toml")):
        shutil.copy2(src_toml, tmp_mods / src_toml.name)

    app = WeaveForgeStudioApp(
        width=1280,
        height=760,
        initial_arena=1,
        initial_mod_id="tnap_weave",
        vsync=False,
        visible=False,
        mods_dir=tmp_mods,
    )
    try:
        # Render cold-boot frame at default 256^3 + 1M stars, then downscale for rapid full-matrix sweep
        app.render_single_frame()
        app.arena1.set_grid_resolution(64)
        app.arena3.set_particle_count(100_000, mod=app.working_mod, compiled=app.compiled)

        # 1. Exercise Tab 1 ("Arena & Couplings") across ALL 4 arenas, ALL 3 Arena 1 palettes, & ALL 3 Arena 2 submodes
        for arena_idx in range(4):
            app.active_arena = arena_idx
            app.select_left_tab(1)
            if arena_idx == 0:
                for pal_idx in range(len(KNOT_COLOR_MODES)):
                    app.arena1.color_mode = pal_idx
                    app.render_single_frame()
            elif arena_idx == 1:
                for sub_idx in range(3):
                    app.arena2.submode = sub_idx
                    app.render_single_frame()
            elif arena_idx == 2:
                for gal_preset in (0, 2):
                    app.arena3.reset_galaxy(gal_preset, mod=app.working_mod, compiled=app.compiled)
                    app.render_single_frame()
            else:
                app.render_single_frame()

        # 2. Exercise Custom .toml Saving (non-colliding ID + immediate reload + idempotent re-save + #2 disambiguation)
        n_initial = len(app.mods)
        app.working_mod.parameters.g4 = 10.25
        saved_1 = app.save_custom_mod()
        assert saved_1.exists()
        assert app.working_mod.id == "user_tnap_weave"
        assert app.working_mod.name.startswith("[Custom]")
        assert len(app.mods) == n_initial + 1
        assert app.mods[app.active_mod_idx].id == "user_tnap_weave"

        by_id = get_mods_by_id(tmp_mods)
        assert "tnap_weave" in by_id and "user_tnap_weave" in by_id
        assert by_id["tnap_weave"].parameters.g4 != pytest.approx(10.25)
        assert by_id["user_tnap_weave"].parameters.g4 == pytest.approx(10.25)

        # Re-saving while editing the custom mod updates in place without creating duplicates or double-prefixing
        app.working_mod.parameters.g4 = 10.75
        saved_1b = app.save_custom_mod()
        assert saved_1b == saved_1
        assert len(app.mods) == n_initial + 1
        assert not app.working_mod.name.startswith("[Custom] [Custom]")
        assert get_mods_by_id(tmp_mods)["user_tnap_weave"].parameters.g4 == pytest.approx(10.75)

        # Saving again starting from built-in tnap_weave creates user_tnap_weave_2 without overwriting user_tnap_weave
        tnap_idx = next(i for i, m in enumerate(app.mods) if m.id == "tnap_weave")
        app.select_preset_mod(tnap_idx)
        app.working_mod.parameters.g4 = 11.50
        saved_2 = app.save_custom_mod()
        assert saved_2.exists() and saved_2 != saved_1
        assert app.working_mod.id == "user_tnap_weave_2"
        assert len(app.mods) == n_initial + 2
        by_id_2 = get_mods_by_id(tmp_mods)
        assert by_id_2["user_tnap_weave"].parameters.g4 == pytest.approx(10.75)
        assert by_id_2["user_tnap_weave_2"].parameters.g4 == pytest.approx(11.50)

        # 3. Exercise Tab 0 ("Master Action") in WARN state, FAIL state, syntax error state, & Shader Inspector modal
        app.select_left_tab(0)
        app.show_shader_modal = True
        app.working_mod.parameters.g4 = 15.0
        app.working_mod.parameters.g6 = 2.0
        app._recompile_working_mod()
        assert app.compiled is not None and app.compiled.has_warnings is True
        app.render_single_frame()

        ghost_idx = next(i for i, m in enumerate(app.mods) if m.id == "ghost_redteam_demo")
        app.select_preset_mod(ghost_idx)
        app.select_left_tab(0)
        app.render_single_frame()

        app.working_mod.action.field_lagrangian = "T_kin + ("
        app._recompile_working_mod()
        assert app.compile_error != ""
        app.select_left_tab(0)
        app.render_single_frame()
        app.show_shader_modal = False

        # Restore clean preset
        app.select_preset_mod(tnap_idx)

        # 4. Exercise Tab 2 ("Rig 2 LAN Worker") both empty and with a populated 2D (g4, g6) knot_phase_sweep
        app.select_left_tab(2)
        app.render_single_frame()

        app.lan_dispatcher.last_status = "Completed test sweep on Vulkan"
        app.lan_dispatcher.last_result = {
            "gpu_adapter": "Vulkan Test Adapter",
            "sparc_175": {"median_chi2_red": 1.18, "rar_rms_dex": 0.128},
            "knot_3d": {"retention_ratio": 0.96, "elapsed_ms": 14.2},
            "gaia_dr4_mock": {"halo_minus_disk_pct": 14.8},
            "knot_phase_sweep": {
                "g4_values": [0.0, 8.0, 14.0],
                "g6_values": [1.0, 3.0, 6.0],
                "retention_grid": [
                    [0.22, 1.35, 4.10],
                    [0.24, 0.98, 1.85],
                    [0.25, 0.72, 1.12],
                ],
                "elapsed_ms": 95.0,
            },
        }
        app.select_left_tab(2)
        snap_path = tmp_path / "studio_tab3_sweep.png"
        app.render_single_frame(capture_out=snap_path)
        assert snap_path.exists() and snap_path.stat().st_size > 20_000
    finally:
        app.shutdown()


def test_gaia_csv_malformed_rows_and_empty_phase_sweep_edges(tmp_path: Path) -> None:
    csv_file = tmp_path / "gaia_edge.csv"
    csv_file.write_text(
        "pair_id,sep_au,z_gal_kpc,m_tot_msun,v_tilde_obs,ruwe_max\n"
        "VALID-1,12000.0,5.2,1.5,1.02,1.08\n"
        "VALID-BLANK-OPT,950.0,0.3,,,\n"
        "NAN-SEP,nan,1.0,1.6,0.8,1.05\n"
        "INF-VEL,8500.0,2.0,1.6,inf,1.05\n"
        "NEG-MASS,8500.0,2.0,-1.2,0.85,1.05\n"
        "HIGH-RUWE,8500.0,2.0,1.6,0.85,1.45\n"
        "BAD-FLOAT,not_a_number,1.0,1.6,0.8,1.05\n",
        encoding="utf-8",
    )
    # Row 2 has blank v_tilde_obs -> rejected; add one row with blank ruwe/m_tot but valid v_tilde_obs:
    csv_file_2 = tmp_path / "gaia_edge2.csv"
    csv_file_2.write_text(
        csv_file.read_text(encoding="utf-8") + "VALID-DEFAULTS,1400.0,0.4,,0.79,\n",
        encoding="utf-8",
    )
    pairs = load_gaia_dr4_csv(csv_file_2, max_ruwe=1.20)
    ids = [p.pair_id for p in pairs]
    assert ids == ["VALID-1", "VALID-DEFAULTS"]
    assert pairs[1].m_tot_msun == pytest.approx(1.6)
    assert pairs[1].ruwe_max == pytest.approx(1.0)

    eng = WgpuComputeEngine()
    mod = get_mods_by_id()["tnap_weave"]
    empty_sweep = eng.run_knot_stability_phase_sweep(mod, g4_values=[], g6_values=[2.0, 4.0], grid_n=16, n_steps=4)
    assert empty_sweep["retention_grid"] == []


def test_multiline_lagrangian_toml_roundtrip_and_studio_save_recompile(tmp_path: Path) -> None:
    from weaveforge.core.frameworks import PhysicsMod

    mods = get_mods_by_id()
    multi = mods["tnap_weave"].clone(new_id="user_multi", new_name="Multiline\nHeader\tMod")
    multiline_lag = (
        "T_kin\n"
        "  - c_w**2 * G_grad\r\n"
        "\t- 0.5 * m_eff**2 * rho + 0.25 * g4 * rho**2 - (g6 / 6.0) * rho**3"
    )
    multi.action.field_lagrangian = multiline_lag
    raw_toml = multi.to_toml_string().encode("utf-8")
    parsed = PhysicsMod.from_toml_bytes(raw_toml)
    assert parsed.action.field_lagrangian == multiline_lag
    assert parsed.name == "Multiline\nHeader\tMod"

    # Verify save_custom_mod() inside Studio handles multiline edits and recompiles immediately
    tmp_mods = tmp_path / "mods_multi"
    tmp_mods.mkdir(parents=True, exist_ok=True)
    for src_toml in sorted(DEFAULT_MODS_DIR.glob("*.toml")):
        shutil.copy2(src_toml, tmp_mods / src_toml.name)

    app = WeaveForgeStudioApp(
        width=1024,
        height=680,
        initial_arena=1,
        initial_mod_id="tnap_weave",
        vsync=False,
        visible=False,
        mods_dir=tmp_mods,
    )
    try:
        app.working_mod.action.field_lagrangian = (
            "T_kin\n - c_w**2 * G_grad\n - 0.5 * m_eff**2 * rho + 0.25 * g4 * rho**2 - (g6 / 6.0) * rho**3"
        )
        app.working_mod.action.optical_metric_n = "1 + 2 * U"
        # Save WITHOUT clicking Compile first: save_custom_mod() must recompile & keep multiline action intact!
        out_file = app.save_custom_mod()
        assert out_file.exists()
        assert "\n" in app.working_mod.action.field_lagrangian
        assert app.compiled is not None
        assert app.compiled.mod_id == "user_tnap_weave"
        assert app.compile_error == ""
    finally:
        app.shutdown()


def test_lan_worker_gaia_dr4_evaluates_exact_compiled_interpolation_nu() -> None:
    from weaveforge.compute.lan_worker import WorkerSweepRequest, execute_crunch_job_sync
    from weaveforge.data.gaia_dr4_pipeline import generate_mock_gaia_dr4_catalog

    mods = get_mods_by_id()
    eng = WgpuComputeEngine()

    # Bit-for-bit parity between truth_mode='tnap_weave' and compiled nu_numpy_fn
    tnap = mods["tnap_weave"]
    tnap_comp = compile_master_action(tnap)
    cat_legacy = generate_mock_gaia_dr4_catalog(240, truth_mode="tnap_weave", seed=42)
    cat_nufn = generate_mock_gaia_dr4_catalog(240, truth_mode="custom", seed=42, nu_fn=tnap_comp.nu_numpy_fn)
    assert [p.v_tilde_obs for p in cat_legacy] == pytest.approx([p.v_tilde_obs for p in cat_nufn], rel=1e-12)

    # Pure Newtonian nu(y) == "1.0" WITHOUT CDM halo (e.g. ghost_redteam_demo) must NOT fake a +15% TNaP boost!
    res_ghost = execute_crunch_job_sync(
        WorkerSweepRequest(
            toml_source=mods["ghost_redteam_demo"].to_toml_string(),
            knot_grid_n=16,
            knot_steps=2,
            g4_grid=[0.0],
            g6_grid=[1.0],
            gaia_mock_count=2000,
            optimize_sparc_ml=False,
        ),
        engine=eng,
    )
    assert res_ghost["sparc_175"]["optimized_ml"] is False
    assert abs(float(res_ghost["gaia_dr4_mock"]["disk_boost_pct"])) < 4.5
    assert abs(float(res_ghost["gaia_dr4_mock"]["halo_minus_disk_pct"])) < 4.5

    # Lambda-CDM with float literal "1.0" instead of "1" must also yield ~0% boost
    cdm_float = mods["lambda_cdm"].clone()
    cdm_float.action.interpolation_nu = "1.0"
    res_cdm = execute_crunch_job_sync(
        WorkerSweepRequest(
            toml_source=cdm_float.to_toml_string(),
            knot_grid_n=16,
            knot_steps=2,
            g4_grid=[0.0],
            g6_grid=[1.0],
            gaia_mock_count=2000,
            optimize_sparc_ml=False,
        ),
        engine=eng,
    )
    assert abs(float(res_cdm["gaia_dr4_mock"]["halo_minus_disk_pct"])) < 4.5

    # TNaP weave must exhibit strong positive Halo - Disk boost (> +8%)
    res_tnap = execute_crunch_job_sync(
        WorkerSweepRequest(
            toml_source=tnap.to_toml_string(),
            knot_grid_n=16,
            knot_steps=2,
            g4_grid=[0.0],
            g6_grid=[1.0],
            gaia_mock_count=2400,
            optimize_sparc_ml=False,
        ),
        engine=eng,
    )
    assert float(res_tnap["gaia_dr4_mock"]["halo_minus_disk_pct"]) > 8.0


def test_compiler_fractional_px_superfluid_and_metric_asymptotic_validation() -> None:
    import math
    from weaveforge.core.symbolic_compiler import SymbolicCompilationError

    mods = get_mods_by_id()

    # 1. Berezhiani-Khoury fractional P(X) ~ X^(3/2) superfluid compiles cleanly & matches physics at v=0
    sf_frac = mods["superfluid_dm"].clone(new_id="user_bk_superfluid")
    sf_frac.action.field_lagrangian = "T_kin + 0.1 * T_kin**1.5 - c_w**2 * G_grad - 0.5 * m_eff**2 * rho"
    comp_sf = compile_master_action(sf_frac)
    assert comp_sf.all_gates_passed is True
    assert comp_sf.kinetic_eigenvalues == pytest.approx((1.0, 1.0, 1.0), abs=1e-5)

    z = np.zeros((3, 8, 8, 8), dtype=float)
    gz = np.zeros((3, 3, 8, 8, 8), dtype=float)
    phi_static = np.ones((3, 8, 8, 8), dtype=float)
    acc_static = comp_sf.accel_numpy_fn(phi_static, gz, z, 1.0, 2.0, 0.0, 0.0, vel_arr=z, dx_step=0.5)
    assert np.all(np.isfinite(acc_static))
    assert acc_static == pytest.approx(-4.0, abs=1e-4)

    eng = WgpuComputeEngine()
    res_sf = eng.run_3d_knot_evolution(comp_sf, sf_frac, grid_n=16, n_steps=6, readback_full_field=True)
    assert res_sf["phi_final"] is not None and np.all(np.isfinite(res_sf["phi_final"]))

    # 2. Cuspy non-C^2 kinetic terms and non-analytic metrics raise SymbolicCompilationError cleanly
    cusp = mods["superfluid_dm"].clone()
    cusp.action.field_lagrangian = "sqrt(T_kin) - c_w**2 * G_grad - 0.5 * m_eff**2 * rho"
    with pytest.raises(SymbolicCompilationError):
        compile_master_action(cusp)

    for bad_metric in ("1 + sqrt(U)", "1 / U"):
        m_bad = mods["lambda_cdm"].clone()
        m_bad.action.optical_metric_n = bad_metric
        with pytest.raises(SymbolicCompilationError):
            compile_master_action(m_bad)

    # 3. Unnormalized asymptotic metric n(U) = 10 * exp(2*U) must FAIL Gate 4 (non-Minkowski infinity!)
    unnorm = mods["tnap_weave"].clone()
    unnorm.action.optical_metric_n = "10 * exp(2 * U)"
    comp_unnorm = compile_master_action(unnorm)
    ppn_gate = next(g for g in comp_unnorm.health_gates if g.gate_id == "ppn_solar")
    assert ppn_gate.passed is False
    assert comp_unnorm.all_critical_passed is False
    assert comp_unnorm.shadow_radius_rg == pytest.approx(2.0 * math.e, rel=2e-3)


def test_arena1_non_multiple_of_64_reduction_and_vram_oom_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = moderngl.create_standalone_context(require=430)
    tex = ctx.texture((160, 120), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    # grid_n=96 previously overflowed _buf_stats (3456 > 2048 workgroups) in _sample_diagnostics!
    a1 = Arena1KnotTank(ctx, initial_grid_n=96)
    try:
        mod = get_mods_by_id()["tnap_weave"]
        comp = compile_master_action(mod)
        a1.step(mod, comp, n_substeps=24)
        assert a1.step_count == 24
        assert np.isfinite(a1.retention_ratio) and a1.retention_ratio > 0.4
        assert np.isfinite(a1.hamiltonian_energy)

        # Simulate GPU VRAM exhaustion when attempting to upscale from 96^3 to 512^3
        orig_buffer = ctx.buffer

        def _oom_on_huge_buffer(*args: object, **kwargs: object) -> moderngl.Buffer:
            if int(kwargs.get("reserve", 0)) > 200_000_000:
                raise MemoryError("Simulated 8GB GPU VRAM exhaustion at 512^3")
            return orig_buffer(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(ctx, "buffer", _oom_on_huge_buffer)
        with pytest.raises(RuntimeError, match="rolled back"):
            a1.set_grid_resolution(512)

        # Arena 1 must still be 100% live and renderable at 96^3!
        assert a1.grid_n == 96
        a1.step(mod, comp, n_substeps=2)
        a1.render(fbo, Camera3D(), wall_time=0.1)
    finally:
        a1.release()
        fbo.release()
        tex.release()
        ctx.release()


def test_studio_interactive_clicks_cdm_toggle_sweep_cell_and_wasd_guard() -> None:
    import glfw
    from imgui_bundle import imgui

    app = WeaveForgeStudioApp(
        width=1280,
        height=760,
        initial_arena=4,
        initial_mod_id="tnap_weave",
        vsync=False,
        visible=False,
    )
    try:
        app.arena1.set_grid_resolution(64)
        app.arena3.set_particle_count(100_000, mod=app.working_mod, compiled=app.compiled)
        io = imgui.get_io()

        # 1. Interactive click on Tab 2 'Include NFW Cold Dark Matter Halo' checkbox must auto-set scale=1.0
        assert app.working_mod.parameters.has_cdm_halo is False
        assert app.working_mod.parameters.cdm_halo_scale == 0.0
        app.select_left_tab(1)
        app.render_single_frame()
        app.render_single_frame()
        for y_try in range(440, 600, 12):
            glfw.set_cursor_pos(app.window, 24.0, float(y_try))
            io.add_mouse_pos_event(24.0, float(y_try))
            app.render_single_frame()
            io.add_mouse_button_event(0, True)
            app.render_single_frame()
            io.add_mouse_button_event(0, False)
            app.render_single_frame()
            if app.working_mod.parameters.has_cdm_halo:
                break
        assert app.working_mod.parameters.has_cdm_halo is True
        assert app.working_mod.parameters.cdm_halo_scale == pytest.approx(1.0)

        # 2. Interactive click on Tab 3 #KnotSweepMatrix cell button applies (g4, g6) and recompiles
        app.lan_dispatcher.last_result = {
            "gpu_adapter": "Vulkan",
            "sparc_175": {"median_chi2_red": 1.18, "rar_rms_dex": 0.128, "optimized_ml": True},
            "knot_3d": {"retention_ratio": 0.96, "elapsed_ms": 14.2},
            "gaia_dr4_mock": {"n_clean": 2400, "halo_minus_disk_pct": 14.8},
            "knot_phase_sweep": {
                "g4_values": [4.5, 11.2],
                "g6_values": [2.2, 5.4],
                "retention_grid": [[0.62, 1.35], [0.71, 1.18]],
                "elapsed_ms": 40.0,
            },
        }
        app.select_left_tab(2)
        app.render_single_frame()
        app.render_single_frame()
        for y_try in range(640, 745, 12):
            glfw.set_cursor_pos(app.window, 275.0, float(y_try))
            io.add_mouse_pos_event(275.0, float(y_try))
            app.render_single_frame()
            io.add_mouse_button_event(0, True)
            app.render_single_frame()
            io.add_mouse_button_event(0, False)
            app.render_single_frame()
            if abs(app.working_mod.parameters.g4 - 11.2) < 1e-4:
                break
        assert app.working_mod.parameters.g4 == pytest.approx(11.2)
        assert app.working_mod.parameters.g6 == pytest.approx(5.4)
        assert "g4=11.20, g6=5.40" in app.status_banner

        # 3. Focusing Tab 1 multiline Lagrangian editor blocks WASD camera motion while hovering 3D viewport
        app.select_left_tab(0)
        app.render_single_frame()
        app.render_single_frame()
        glfw.set_cursor_pos(app.window, 80.0, 195.0)
        io.add_mouse_pos_event(80.0, 195.0)
        app.render_single_frame()
        io.add_mouse_button_event(0, True)
        app.render_single_frame()
        io.add_mouse_button_event(0, False)
        app.render_single_frame()
        assert imgui.get_io().want_text_input is True

        cam = app.cameras[app.active_arena]
        eye_before = cam.basis_vectors()[0].copy()
        glfw.set_cursor_pos(app.window, 640.0, 400.0)
        io.add_mouse_pos_event(640.0, 400.0)
        io.add_key_event(imgui.Key.w, True)
        app.render_single_frame()
        io.add_key_event(imgui.Key.w, False)
        eye_after = cam.basis_vectors()[0].copy()
        assert np.allclose(eye_after, eye_before, atol=1e-9)
    finally:
        app.shutdown()


def test_studio_4k_high_dpi_scaling_render_scale_and_screenshot_fps_guard(tmp_path: Path) -> None:
    import time
    import glfw
    from PIL import Image
    from imgui_bundle import imgui

    # 1. Verify Auto High-DPI scale selection across 4K, 1080p, and non-finite inputs
    assert WeaveForgeStudioApp._compute_auto_ui_scale(1.0, 3840) == pytest.approx(1.75)
    assert WeaveForgeStudioApp._compute_auto_ui_scale(2.25, 3840) == pytest.approx(2.25)
    assert WeaveForgeStudioApp._compute_auto_ui_scale(1.0, 1920) == pytest.approx(1.0)
    assert WeaveForgeStudioApp._compute_auto_ui_scale(float("nan"), 3840) == pytest.approx(1.75)

    ini_cwd = Path("imgui.ini")
    had_ini_before = ini_cwd.exists()

    app = WeaveForgeStudioApp(
        width=3840,
        height=2088,
        initial_arena=1,
        initial_mod_id="tnap_weave",
        vsync=False,
        visible=False,
        ui_scale=1.75,
        render_scale=0.75,
    )
    try:
        assert app.ui_scale == pytest.approx(1.75)
        assert app.render_scale == pytest.approx(0.75)

        # Verify FPS cap clamping & switching (60 -> 120 -> 240 -> Uncapped)
        for target_cap, expected_cap in ((60, 60), (120, 120), (240, 240), (-15, 0), (0, 0)):
            app.set_fps_cap(target_cap)
            assert app.fps_cap == expected_cap

        app.arena1.set_grid_resolution(64)
        app.render_single_frame()
        assert app._view_tex is not None
        size_rs_075 = app._view_tex.size

        # 2. Live UI scale changes (1.0 -> 2.0 -> 1.75 + boundary/NaN guards + deferred mid-frame apply)
        app.set_ui_scale(float("nan"))
        assert app.ui_scale == pytest.approx(1.0)
        app.set_ui_scale(0.1)
        assert app.ui_scale == pytest.approx(0.75)
        app.set_ui_scale(9.0)
        assert app.ui_scale == pytest.approx(3.0)

        for target_ui in (1.0, 2.0, 1.75):
            app.set_ui_scale(target_ui)
            app.render_single_frame()
            assert app.ui_scale == pytest.approx(target_ui)
            assert imgui.get_style().font_scale_main == pytest.approx(target_ui)

        pad_x_first = float(imgui.get_style().window_padding.x)
        # Simulate deferred mid-ImGui-frame scale change -> must apply cleanly on next frame without compounding
        app._in_imgui_frame = True
        app.set_ui_scale(2.0)
        app._in_imgui_frame = False
        assert app._pending_theme_apply is True
        app.render_single_frame()
        assert app._pending_theme_apply is False
        app.set_ui_scale(1.75)
        app.render_single_frame()
        assert float(imgui.get_style().window_padding.x) == pytest.approx(pad_x_first)

        # 3. Live decoupled 3D render_scale changes (0.5 -> 1.0 -> 0.75 + boundary clamping)
        app.set_render_scale(0.1)
        assert app.render_scale == pytest.approx(0.35)
        app.set_render_scale(float("inf"))
        assert app.render_scale == pytest.approx(1.0)

        app.set_render_scale(0.5)
        app.render_single_frame()
        size_rs_050 = app._view_tex.size

        app.set_render_scale(1.0)
        app.render_single_frame()
        size_rs_100 = app._view_tex.size

        app.set_render_scale(0.75)
        app.render_single_frame()
        assert app._view_tex.size == size_rs_075
        assert size_rs_050[0] < size_rs_075[0] < size_rs_100[0]
        assert size_rs_050[1] < size_rs_075[1] < size_rs_100[1]

        # 4. Verify save_window_screenshot() & preset-mod switches never poison steady-state frame_ms_smooth / fps_smooth
        for _ in range(12):
            app.render_single_frame()
        ms_before = app.frame_ms_smooth
        assert ms_before < 25.0

        fb_size = glfw.get_framebuffer_size(app.window)
        snap_direct = tmp_path / "4k_direct_snap.png"
        t0 = time.perf_counter()
        app.save_window_screenshot(snap_direct)
        save_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert snap_direct.exists() and snap_direct.stat().st_size > 20_000
        assert save_elapsed_ms > 15.0  # Real PNG compression stall occurred
        with Image.open(snap_direct) as im:
            assert im.size == fb_size  # Full backbuffer resolution preserved even when 3D render_scale == 0.75

        app.render_single_frame()
        app.render_single_frame()
        assert app.frame_ms_smooth < max(ms_before + 4.0, 22.0)

        snap_inline = tmp_path / "4k_inline_snap.png"
        app.render_single_frame(capture_out=snap_inline)
        assert snap_inline.exists() and snap_inline.stat().st_size > 20_000
        app.render_single_frame()
        assert app.frame_ms_smooth < max(ms_before + 4.0, 22.0)

        # Regression check: switching to Arena 4 + selecting ghost_redteam_demo (~400ms SymPy/SPARC/CAMB compile)
        # + capturing a 4K screenshot must NOT drag fps_smooth into amber (< 50 FPS)!
        ghost_idx = next(i for i, m in enumerate(app.mods) if m.id == "ghost_redteam_demo")
        tnap_idx = next(i for i, m in enumerate(app.mods) if m.id == "tnap_weave")
        app._switch_active_arena(3)
        app.select_preset_mod(ghost_idx)
        for _ in range(5):
            app.render_single_frame()
        assert app.fps_smooth >= 50.0
        assert app.frame_ms_smooth < 20.0

        snap_a4 = tmp_path / "4k_arena4_redteam.png"
        app.save_window_screenshot(snap_a4)
        assert snap_a4.exists() and snap_a4.stat().st_size > 20_000
        app.render_single_frame()
        app.render_single_frame()
        assert app.fps_smooth >= 50.0
        assert app.frame_ms_smooth < 20.0

        app.select_preset_mod(tnap_idx)
        app.render_single_frame()
        assert app.fps_smooth >= 50.0
    finally:
        app.shutdown()

    if not had_ini_before:
        assert not ini_cwd.exists(), "Studio must never create imgui.ini in working directory"


def test_studio_ddr5_outgoing_first_hibernation_and_replay_backup_dedup() -> None:
    import time

    app = WeaveForgeStudioApp(
        width=1280,
        height=760,
        initial_arena=1,
        initial_mod_id="tnap_weave",
        vsync=False,
        visible=False,
    )
    try:
        # Cold-boot on Arena 1 (idx 0): inactive arenas start lazy-hibernated with 0.0 MB VRAM
        assert app.active_arena == 0
        assert app.arena1.is_hibernated is False and app.arena1.allocated_vram_mb > 0.0
        assert app.arena2.is_hibernated is True and app.arena2.allocated_vram_mb == 0.0
        assert app.arena3.is_hibernated is True and app.arena3.allocated_vram_mb == 0.0
        assert app.arena4.is_hibernated is True and app.arena4.allocated_vram_mb == 0.0

        app.arena1.set_grid_resolution(64)
        for _ in range(8):
            app.render_single_frame()
        if app.arena1.replay_frames_count == 0:
            app.arena1._capture_replay_snapshot()
        assert app.arena1.replay_frames_count >= 1
        live_steps_a1 = app.arena1.step_count
        assert live_steps_a1 > 0

        # 1. Verify _stash_live_before_replay does NOT duplicate Arena 1's _live_backup in _replay_live_backup
        app._stash_live_before_replay(app.arena1, 0)
        assert 0 not in app._replay_live_backup
        app.arena1.load_replay_frame(0)
        assert app.arena1.replay_cursor == 0
        assert app.arena1._live_backup is not None
        assert app._is_arena_in_replay(app.arena1, 0) is True
        assert 0 not in app._replay_live_backup

        app._poll_memory_telemetry(force=True)
        expected_tracked = sum(
            float(getattr(ar, "ddr5_ram_mb", 0.0))
            for ar in (app.arena1, app.arena2, app.arena3, app.arena4)
        )
        assert app.tracked_ddr5_mb == pytest.approx(expected_tracked, rel=1e-5)
        assert app.ddr5_used_mb >= app.tracked_ddr5_mb > 0.0

        # Verify _poll_memory_telemetry handles np.float32 scalars and ignores bools on legacy objects
        setattr(app.arena2, "_test_dummy_flag", True)
        orig_a2_ddr5 = getattr(app.arena2, "ddr5_ram_mb", None)
        try:
            setattr(app.arena2, "ddr5_ram_mb", np.float32(3.5))
            app._poll_memory_telemetry(force=True)
            assert app.tracked_ddr5_mb == pytest.approx(expected_tracked + 3.5, rel=1e-5)
        finally:
            if orig_a2_ddr5 is None:
                delattr(app.arena2, "ddr5_ram_mb")
            else:
                setattr(app.arena2, "ddr5_ram_mb", orig_a2_ddr5)
            delattr(app.arena2, "_test_dummy_flag")
            app._poll_memory_telemetry(force=True)

        # Simulate user pausing/inspecting replay for 45ms, then clicking Resume -> must not poison frame_ms_smooth
        ms_pre_resume = app.frame_ms_smooth
        time.sleep(0.045)
        app._exit_replay_to_live(0, unpause=True)
        assert app.arena1.replay_cursor == -1
        assert app.arena1._live_backup is None
        assert app._is_arena_in_replay(app.arena1, 0) is False
        assert app.arena1.step_count == live_steps_a1
        app.render_single_frame()
        assert app.frame_ms_smooth < max(ms_pre_resume + 5.0, 25.0)

        # Fallback path check: mock object WITHOUT resume_live_from_replay still populates _replay_live_backup
        class _LegacyMockArena:
            is_hibernated = False
            replay_cursor = -1
            grid_n = 4
            _cur_in_a = True
            _buf_a = object()
            _buf_b = object()
            _buf_vel = object()

            @staticmethod
            def _copy_ssbo_to_ram_f32(_buf: object, n: int) -> np.ndarray:
                return np.ones(n, dtype=np.float32)

        app._stash_live_before_replay(_LegacyMockArena(), 0)
        assert 0 in app._replay_live_backup
        app._clear_replay_backup(0)
        assert 0 not in app._replay_live_backup

        # 2. Verify outgoing-first DDR5 hibernation order & full traversal 0 -> 1 -> 2 -> 3 -> 0
        call_order: list[str] = []
        orig_hib_a1 = app.arena1.hibernate_to_ram
        orig_wake_a2 = app.arena2.wake_from_ram

        def _spy_hib_a1() -> float:
            call_order.append("hibernate_a1")
            return orig_hib_a1()

        def _spy_wake_a2() -> None:
            call_order.append("wake_a2")
            orig_wake_a2()

        app.arena1.hibernate_to_ram = _spy_hib_a1  # type: ignore[method-assign]
        app.arena2.wake_from_ram = _spy_wake_a2  # type: ignore[method-assign]

        # 0 -> 1
        app._switch_active_arena(1)
        assert call_order == ["hibernate_a1", "wake_a2"]
        assert app.arena1.is_hibernated is True
        assert app.arena1.allocated_vram_mb == 0.0
        assert app.arena1.ddr5_ram_mb > 0.0
        assert app.arena2.is_hibernated is False
        assert app.arena3.is_hibernated is True and app.arena3.allocated_vram_mb == 0.0
        assert app.ddr5_used_mb >= app.arena1.ddr5_ram_mb > 0.0
        app.render_single_frame()

        # Restore unwrapped methods
        app.arena1.hibernate_to_ram = orig_hib_a1  # type: ignore[method-assign]
        app.arena2.wake_from_ram = orig_wake_a2  # type: ignore[method-assign]

        # 1 -> 2 (wake Arena 3, downscale to 100k stars for fast test, verify Arena 3 replay dedup too)
        app._switch_active_arena(2)
        assert app.arena2.is_hibernated is True and app.arena2.allocated_vram_mb == 0.0
        assert app.arena3.is_hibernated is False and app.arena3.allocated_vram_mb > 0.0
        assert app.arena1.is_hibernated is True and app.arena1.allocated_vram_mb == 0.0
        app.arena3.set_particle_count(100_000, mod=app.working_mod, compiled=app.compiled)
        app.render_single_frame()
        if app.arena3.replay_frames_count == 0:
            app.arena3._capture_replay_snapshot()
        app._stash_live_before_replay(app.arena3, 2)
        assert 2 not in app._replay_live_backup

        # 2 -> 3 (both heavy arenas 1 & 3 hibernated in DDR5 with 0.0 MB VRAM!)
        app._switch_active_arena(3)
        assert app.arena1.is_hibernated is True and app.arena1.allocated_vram_mb == 0.0
        assert app.arena2.is_hibernated is True and app.arena2.allocated_vram_mb == 0.0
        assert app.arena3.is_hibernated is True and app.arena3.allocated_vram_mb == 0.0
        assert app.arena4.is_hibernated is False and app.arena4.allocated_vram_mb > 0.0
        assert app.arena1.ddr5_ram_mb > 0.0 and app.arena3.ddr5_ram_mb > 0.0
        assert app.ddr5_used_mb >= (app.arena1.ddr5_ram_mb + app.arena3.ddr5_ram_mb)
        app.render_single_frame()

        # 3 -> 0 (Arena 1 wakes back from DDR5 with evolved state intact; frame_ms_smooth stays unpoisoned!)
        ms_pre_wake = app.frame_ms_smooth
        app._switch_active_arena(0)
        assert app.arena1.is_hibernated is False and app.arena1.allocated_vram_mb > 0.0
        assert app.arena1.step_count >= live_steps_a1
        assert app.arena3.is_hibernated is True and app.arena3.allocated_vram_mb == 0.0
        assert app.arena4.is_hibernated is True and app.arena4.allocated_vram_mb == 0.0
        app.render_single_frame()
        app.render_single_frame()
        assert app.frame_ms_smooth < max(ms_pre_wake + 5.0, 25.0)

        # 3. Rapid mid-scrubbing tab switches (scrub A1 -> switch A3 -> scrub A3 -> switch A1)
        # Must restore true live state BEFORE DDR5 hibernation with zero leaked _live_backup buffers!
        app.arena1.load_replay_frame(0)
        assert app.arena1.replay_cursor == 0 and app.arena1._live_backup is not None
        app._switch_active_arena(2)
        assert app.arena1.is_hibernated is True and app.arena1.allocated_vram_mb == 0.0
        assert app.arena1.replay_cursor == -1 and app.arena1._live_backup is None
        assert 0 not in app._replay_live_backup
        assert app.arena3.is_hibernated is False and app.arena3.allocated_vram_mb > 0.0

        if app.arena3.replay_frames_count == 0:
            app.arena3._capture_replay_snapshot()
        live_steps_a3 = app.arena3.step_count
        app.arena3.load_replay_frame(0)
        assert app.arena3.replay_cursor == 0 and app.arena3._live_backup is not None
        app._switch_active_arena(0)
        assert app.arena3.is_hibernated is True and app.arena3.allocated_vram_mb == 0.0
        assert app.arena3.replay_cursor == -1 and app.arena3._live_backup is None
        assert 2 not in app._replay_live_backup
        assert app.arena1.is_hibernated is False and app.arena1.allocated_vram_mb > 0.0
        assert app.arena1.step_count >= live_steps_a1

        app._poll_memory_telemetry(force=True)
        expected_after_rapid = sum(
            float(getattr(ar, "ddr5_ram_mb", 0.0))
            for ar in (app.arena1, app.arena2, app.arena3, app.arena4)
        )
        assert app.tracked_ddr5_mb == pytest.approx(expected_after_rapid, rel=1e-5)

        # Wake Arena 3 once more to verify its live step_count survived mid-scrubbing hibernation intact
        app._switch_active_arena(2)
        assert app.arena3.is_hibernated is False
        assert app.arena3.step_count == live_steps_a3
    finally:
        app.shutdown()


def test_studio_left_sidebar_reset_defaults_per_slider_and_mod_equations() -> None:
    import glfw
    from imgui_bundle import imgui

    app = WeaveForgeStudioApp(
        width=1280,
        height=760,
        initial_arena=1,
        initial_mod_id="tnap_weave",
        vsync=False,
        visible=False,
    )
    try:
        app.arena1.set_grid_resolution(64)
        app.arena3.set_particle_count(100_000, mod=app.working_mod, compiled=app.compiled)
        io = imgui.get_io()

        def _click_rect(rect: tuple[float, float, float, float]) -> None:
            cx = 0.5 * (rect[0] + rect[2])
            cy = 0.5 * (rect[1] + rect[3])
            glfw.set_cursor_pos(app.window, cx, cy)
            io.add_mouse_pos_event(cx, cy)
            app.render_single_frame()
            io.add_mouse_button_event(0, True)
            app.render_single_frame()
            io.add_mouse_button_event(0, False)
            app.render_single_frame()

        # 1. Clean startup & rendering Tab 1 + Tab 2 must show 0 modified parameters (no float32 drift!)
        app.select_left_tab(0)
        app.render_single_frame()
        app.select_left_tab(1)
        app.render_single_frame()
        app.render_single_frame()
        assert app.count_modified_couplings() == 0
        assert app.count_modified_equations() == 0
        assert app.count_modified_parameters() == 0
        assert len(app._last_slider_reset_rects) == 0

        pristine_g4 = float(app._pristine_mod.parameters.g4)
        pristine_cw = float(app._pristine_mod.parameters.c_w)
        pristine_w0 = float(app._pristine_mod.parameters.w0_de)

        # 2. Modify 5 coupling parameters -> verify dirty counts & per-slider [R] micro-buttons appear
        app.working_mod.parameters.g4 = 9.25
        app.working_mod.parameters.c_w = 0.88
        app.working_mod.parameters.has_cdm_halo = True
        app.working_mod.parameters.cdm_halo_scale = 1.35
        app.working_mod.parameters.w0_de = -1.22
        app._recompile_working_mod()
        app.render_single_frame()
        app.render_single_frame()

        assert app.count_modified_couplings() == 5
        assert app.count_modified_parameters() == 5
        for expected_dirty in ("g4", "c_w", "has_cdm_halo", "cdm_halo_scale", "w0_de"):
            assert app.is_parameter_modified(expected_dirty) is True
            assert expected_dirty in app._last_slider_reset_rects

        # 3. Click individual [R] micro-button on 'g4' -> ONLY 'g4' resets to pristine .toml default!
        _click_rect(app._last_slider_reset_rects["g4"])
        assert app.is_parameter_modified("g4") is False
        assert app.working_mod.parameters.g4 == pytest.approx(pristine_g4)
        assert app.count_modified_couplings() == 4
        assert "Reset g4 ->" in app.status_banner

        # 4. Click individual [R] micro-button on 'has_cdm_halo' -> restores both has_cdm_halo & cdm_halo_scale
        _click_rect(app._last_slider_reset_rects["has_cdm_halo"])
        assert app.working_mod.parameters.has_cdm_halo is False
        assert app.working_mod.parameters.cdm_halo_scale == pytest.approx(0.0)
        assert app.count_modified_couplings() == 2

        # 5. Click prominent full-width 'Reset Modified Sliders to Defaults' button at top of Tab 2
        assert app._last_top_reset_btn_rect is not None
        _click_rect(app._last_top_reset_btn_rect)
        assert app.count_modified_couplings() == 0
        assert app.count_modified_parameters() == 0
        assert app.working_mod.parameters.c_w == pytest.approx(pristine_cw)
        assert app.working_mod.parameters.w0_de == pytest.approx(pristine_w0)
        assert len(app._last_slider_reset_rects) == 0

        # 6. Switching preset mod updates _pristine_mod cleanly
        cdm_idx = next(i for i, m in enumerate(app.mods) if m.id == "lambda_cdm")
        app.select_preset_mod(cdm_idx)
        assert app.count_modified_parameters() == 0
        app.working_mod.parameters.cdm_halo_scale = 0.42
        app.working_mod.parameters.a0_si = 1.95e-10
        assert app.count_modified_couplings() == 2
        app.reset_sliders_to_mod_defaults()
        assert app.count_modified_couplings() == 0
        assert app.working_mod.parameters.cdm_halo_scale == pytest.approx(
            app._pristine_mod.parameters.cdm_halo_scale
        )
        assert app.working_mod.parameters.a0_si == pytest.approx(app._pristine_mod.parameters.a0_si)

        # 7. Tab 1 symbolic equation edits & restore_default_mod_equations()
        tnap_idx = next(i for i, m in enumerate(app.mods) if m.id == "tnap_weave")
        app.select_preset_mod(tnap_idx)
        orig_lag = app._pristine_mod.action.field_lagrangian
        orig_opt = app._pristine_mod.action.optical_metric_n
        app.working_mod.action.field_lagrangian = "0.5 * T_kin - 0.5 * G_grad"
        app.working_mod.action.optical_metric_n = "1 + 2*U"
        assert app.count_modified_equations() == 2
        assert app.count_modified_parameters() == 2
        app.select_left_tab(0)
        app.render_single_frame()
        app.restore_default_mod_equations()
        assert app.count_modified_equations() == 0
        assert app.working_mod.action.field_lagrangian == orig_lag
        assert app.working_mod.action.optical_metric_n == orig_opt
        assert app.compile_error == ""

        # 8. Full sidebar reset restores equations + couplings + all 4 arenas' view settings
        def_gain = app._arena_defaults[0]["density_gain"]
        def_spin = app._arena_defaults[1]["spin_a"]
        def_lens = app._arena_defaults[2]["show_lensing"]
        def_waist = app._arena_defaults[3]["waist_r0"]

        app.arena1.density_gain = 5.2
        app.arena1.color_mode = 2
        app.arena2.spin_a = 0.22
        app.arena2.show_bhex_gauge = False
        app.arena3.show_lensing = False
        app.arena4.waist_r0 = 1.33
        app.working_mod.parameters.g6 = 5.5
        app.working_mod.action.gw_speed_ratio = "0.98"
        assert app.count_modified_parameters() == 2

        app.reset_all_sidebar_defaults()
        assert app.count_modified_parameters() == 0
        assert app.arena1.density_gain == pytest.approx(def_gain)
        assert app.arena1.color_mode == app._arena_defaults[0]["color_mode"]
        assert app.arena2.spin_a == pytest.approx(def_spin)
        assert app.arena2.show_bhex_gauge is True
        assert app.arena3.show_lensing is bool(def_lens)
        assert app.arena4.waist_r0 == pytest.approx(def_waist)
    finally:
        app.shutdown()


def test_studio_right_click_context_menu_vs_rmb_pan_and_hotkeys(tmp_path: Path) -> None:
    import glfw
    from PIL import Image
    from imgui_bundle import imgui

    app = WeaveForgeStudioApp(
        width=1280,
        height=760,
        initial_arena=1,
        initial_mod_id="tnap_weave",
        vsync=False,
        visible=False,
    )
    app.captures_dir = tmp_path / "captures"
    try:
        app.arena1.set_grid_resolution(64)
        app.arena3.set_particle_count(100_000, mod=app.working_mod, compiled=app.compiled)
        io = imgui.get_io()
        app.render_single_frame()
        app.render_single_frame()

        def _clean_right_click(x: float, y: float) -> None:
            glfw.set_cursor_pos(app.window, x, y)
            io.add_mouse_pos_event(x, y)
            app.render_single_frame()
            io.add_mouse_button_event(1, True)
            app.render_single_frame()
            io.add_mouse_button_event(1, False)
            app.render_single_frame()
            # Render one settle frame after popup Appearing so Dear ImGui finalizes clamped item rects
            app.render_single_frame()

        def _left_click_rect(rect: tuple[float, float, float, float]) -> None:
            cx = 0.5 * (rect[0] + rect[2])
            cy = 0.5 * (rect[1] + rect[3])
            glfw.set_cursor_pos(app.window, cx, cy)
            io.add_mouse_pos_event(cx, cy)
            app.render_single_frame()
            io.add_mouse_button_event(0, True)
            app.render_single_frame()
            io.add_mouse_button_event(0, False)
            app.render_single_frame()

        def _tap_key(key: imgui.Key) -> None:
            io.add_key_event(key, True)
            app.render_single_frame()
            io.add_key_event(key, False)
            app.render_single_frame()

        # 1. Right-mouse DRAG (> 5 px) over 3D viewport MUST pan camera AND NOT open context menu!
        cam = app.cameras[app.active_arena]
        target_before = cam.target.copy()
        glfw.set_cursor_pos(app.window, 620.0, 380.0)
        io.add_mouse_pos_event(620.0, 380.0)
        app.render_single_frame()
        io.add_mouse_button_event(1, True)
        app.render_single_frame()
        for dx, dy in ((20.0, 15.0), (48.0, 32.0), (80.0, 52.0)):
            glfw.set_cursor_pos(app.window, 620.0 + dx, 380.0 + dy)
            io.add_mouse_pos_event(620.0 + dx, 380.0 + dy)
            app.render_single_frame()
        io.add_mouse_button_event(1, False)
        app.render_single_frame()
        assert not np.allclose(cam.target, target_before, atol=1e-5)
        assert app.is_context_menu_open is False

        # 2. Guard: Clean RMB click while an InputText has focus (want_text_input=True) does NOT open popup
        app.select_left_tab(0)
        app.render_single_frame()
        app.render_single_frame()
        glfw.set_cursor_pos(app.window, 80.0, 195.0)
        io.add_mouse_pos_event(80.0, 195.0)
        app.render_single_frame()
        io.add_mouse_button_event(0, True)
        app.render_single_frame()
        io.add_mouse_button_event(0, False)
        app.render_single_frame()
        assert io.want_text_input is True
        _clean_right_click(640.0, 360.0)
        assert app.is_context_menu_open is False

        # Unfocus InputText by switching to Tab 2 and clicking neutral area
        _tap_key(imgui.Key.escape)
        app.select_left_tab(1)
        app.render_single_frame()
        glfw.set_cursor_pos(app.window, 640.0, 360.0)
        io.add_mouse_pos_event(640.0, 360.0)
        io.add_mouse_button_event(0, True)
        app.render_single_frame()
        io.add_mouse_button_event(0, False)
        app.render_single_frame()
        assert io.want_text_input is False

        # 3. Clean stationary RMB click opens ##StudioContextMenu with all core action items
        assert app.paused is False
        _clean_right_click(600.0, 260.0)
        assert app.is_context_menu_open is True
        assert app._last_context_menu_rect is not None
        for required_key in (
            "pause_resume",
            "step_1",
            "reset_arena",
            "save_png",
            "toggle_rec",
            "open_captures",
            "reset_sliders",
            "reset_all_sidebar",
            "recompile_action",
            "toggle_shader_inspector",
            "reset_camera",
            "toggle_hibernate",
        ):
            assert required_key in app._context_menu_item_rects

        # Click 'Pause Simulation' in context menu -> pauses & closes menu
        _left_click_rect(app._context_menu_item_rects["pause_resume"])
        assert app.paused is True
        assert app.is_context_menu_open is False

        # Open menu again while paused -> click 'Step 1 Frame Forward' -> advances Arena 1 by 1 frame (2 ping-pong substeps) and stays paused
        step_before = app.arena1.step_count
        _clean_right_click(600.0, 260.0)
        assert app.is_context_menu_open is True
        _left_click_rect(app._context_menu_item_rects["step_1"])
        assert app.arena1.step_count == step_before + 2
        assert app.paused is True
        app.render_single_frame()
        assert app.arena1.step_count == step_before + 2

        # Dirty coupling slider -> open menu -> click 'Reset Coupling Sliders to Mod Defaults'
        app.working_mod.parameters.g4 = 8.50
        assert app.count_modified_couplings() == 1
        _clean_right_click(600.0, 260.0)
        _left_click_rect(app._context_menu_item_rects["reset_sliders"])
        assert app.count_modified_couplings() == 0
        assert app.working_mod.parameters.g4 == pytest.approx(app._pristine_mod.parameters.g4)

        # Offset camera -> open menu -> click 'Reset 3D Camera View'
        cam.yaw += 1.1
        cam.target[0] += 4.0
        _clean_right_click(600.0, 260.0)
        _left_click_rect(app._context_menu_item_rects["reset_camera"])
        assert cam.yaw == pytest.approx(0.62)
        assert np.allclose(cam.target, 0.0, atol=1e-6)

        # 4. Verify non-conflicting keyboard hotkeys: Space, Home, R, F12, F9
        assert app.paused is True
        _tap_key(imgui.Key.space)
        assert app.paused is False
        _tap_key(imgui.Key.space)
        assert app.paused is True

        cam.pitch = -0.75
        _tap_key(imgui.Key.home)
        assert cam.pitch == pytest.approx(0.38)

        assert app.arena1.step_count > 0
        _tap_key(imgui.Key.r)
        assert app.arena1.step_count == 0

        _tap_key(imgui.Key.f12)
        snaps = sorted(app.captures_dir.glob("snap_*.png"))
        assert len(snaps) == 1
        with Image.open(snaps[0]) as im:
            assert im.size == glfw.get_framebuffer_size(app.window)

        assert app.is_recording_video is False
        _tap_key(imgui.Key.f9)
        assert app.is_recording_video is True
        for _ in range(12):
            app.render_single_frame()
        _tap_key(imgui.Key.f9)
        assert app.is_recording_video is False
        mp4s = sorted(app.captures_dir.glob("rec_*.mp4"))
        assert len(mp4s) == 1
        assert mp4s[0].stat().st_size > 2_000
    finally:
        app.shutdown()


def test_studio_60fps_video_recorder_nvenc_and_libx264_plus_4k_topbar_layout(tmp_path: Path) -> None:
    import glfw
    from PIL import Image

    exe = resolve_ffmpeg_exe()
    assert Path(exe).exists()
    nvenc_hw_ok = probe_nvenc_support(exe)

    # 1. Verify 4K (3840x2088, ui_scale=1.75, render_scale=0.75) Top Bar non-overlap & 4K NVENC capture
    app_4k = WeaveForgeStudioApp(
        width=3840,
        height=2088,
        initial_arena=2,
        initial_mod_id="tnap_weave",
        vsync=False,
        visible=False,
        ui_scale=1.75,
        render_scale=0.75,
    )
    app_4k.captures_dir = tmp_path / "captures_4k"
    try:
        app_4k.arena1.set_grid_resolution(64)
        app_4k.arena3.set_particle_count(100_000, mod=app_4k.working_mod, compiled=app_4k.compiled)
        app_4k.render_single_frame()
        app_4k.render_single_frame()
        win_w_4k = float(max(glfw.get_window_size(app_4k.window)[0], 640))
        if app_4k._last_topbar_telemetry_drawn:
            assert app_4k._last_topbar_widgets_end_x <= app_4k._last_topbar_telemetry_x < win_w_4k

        fb_4k_w, fb_4k_h = glfw.get_framebuffer_size(app_4k.window)
        even_4k_w = fb_4k_w - (fb_4k_w % 2)
        even_4k_h = fb_4k_h - (fb_4k_h % 2)

        mp4_4k = app_4k.captures_dir / "test_4k_60fps.mp4"
        app_4k.start_video_recording(
            output_path=mp4_4k,
            fps=60,
            resolution_mode="native",
            record_every_frame=True,
            prefer_nvenc=True,
        )
        assert app_4k.is_recording_video is True
        for _ in range(12):
            app_4k.render_single_frame()
        # Verify active crimson 'Stop ● MM:SS' button in 4K Top Bar NEVER overlaps right-aligned telemetry!
        if app_4k._last_topbar_telemetry_drawn:
            assert app_4k._last_topbar_widgets_end_x <= app_4k._last_topbar_telemetry_x < win_w_4k

        res_4k = app_4k.stop_video_recording()
        assert res_4k is not None
        assert mp4_4k.exists() and mp4_4k.stat().st_size > 8_000
        assert res_4k.frames_written == 12
        assert res_4k.width == even_4k_w and res_4k.height == even_4k_h
        if nvenc_hw_ok and (160 <= even_4k_w <= 4096) and (128 <= even_4k_h <= 4096):
            assert res_4k.encoder_name == "h264_nvenc"
        info_4k = probe_video_stream_info(mp4_4k)
        assert info_4k["returncode"] == 0
        assert info_4k["codec"] == "h264"
        assert info_4k["width"] == even_4k_w and info_4k["height"] == even_4k_h
        assert info_4k["fps"] == pytest.approx(60.0, abs=0.5)
        assert info_4k["decoded_frames"] == 12
    finally:
        app_4k.shutdown()

    # 2. Record >= 60 real GPU-rendered frames @ 60 FPS + mid-recording PNG screenshot + 120 FPS 1080p & libx264
    app = WeaveForgeStudioApp(
        width=961,  # Deliberately odd width/height to verify automatic even-dimension alignment!
        height=541,
        initial_arena=1,
        initial_mod_id="tnap_weave",
        vsync=False,
        visible=False,
    )
    app.captures_dir = tmp_path / "captures"
    try:
        app.arena1.set_grid_resolution(64)
        app.arena3.set_particle_count(100_000, mod=app.working_mod, compiled=app.compiled)

        fb_w, fb_h = glfw.get_framebuffer_size(app.window)
        exp_w = fb_w - (fb_w % 2)
        exp_h = fb_h - (fb_h % 2)

        mp4_60 = app.captures_dir / "knot_65frames_60fps.mp4"
        mid_snap = app.captures_dir / "mid_recording_snap.png"
        app.start_video_recording(
            output_path=mp4_60,
            fps=60,
            resolution_mode="native",
            record_every_frame=True,
            prefer_nvenc=True,
        )
        for i in range(65):
            cap = mid_snap if i == 30 else None
            app.render_single_frame(capture_out=cap)

        res_60 = app.stop_video_recording()
        assert res_60 is not None
        assert mp4_60.exists() and mp4_60.stat().st_size > 10_000
        assert res_60.frames_written == 65
        assert res_60.fps == 60
        assert res_60.width == exp_w and res_60.height == exp_h
        assert mid_snap.exists()
        with Image.open(mid_snap) as im_mid:
            assert im_mid.size == (fb_w, fb_h)
            arr_mid = np.asarray(im_mid)
            assert float(arr_mid[..., :3].mean()) > 5.0  # Non-black valid screenshot during PBO capture!

        info_60 = probe_video_stream_info(mp4_60)
        assert info_60["returncode"] == 0
        assert info_60["codec"] == "h264"
        assert info_60["width"] == exp_w and info_60["height"] == exp_h
        assert info_60["fps"] == pytest.approx(60.0, abs=0.5)
        assert info_60["decoded_frames"] == 65

        # 3. Verify 120 FPS + 1080p hardware glBlitFramebuffer scaling + CPU libx264 fallback path
        mp4_1080p_cpu = app.captures_dir / "crucible_120fps_1080p_libx264.mp4"
        app._switch_active_arena(3)
        app.start_video_recording(
            output_path=mp4_1080p_cpu,
            fps=120,
            resolution_mode="1080p",
            record_every_frame=True,
            prefer_nvenc=False,
        )
        for _ in range(20):
            app.render_single_frame()
        res_cpu = app.stop_video_recording()
        assert res_cpu is not None
        assert res_cpu.encoder_name == "libx264"
        assert res_cpu.fps == 120
        assert res_cpu.width == 1920 and res_cpu.height == 1080
        assert res_cpu.frames_written == 20

        info_cpu = probe_video_stream_info(mp4_1080p_cpu)
        assert info_cpu["returncode"] == 0
        assert info_cpu["codec"] == "h264"
        assert info_cpu["width"] == 1920 and info_cpu["height"] == 1080
        assert info_cpu["fps"] == pytest.approx(120.0, abs=0.5)
        assert info_cpu["decoded_frames"] == 20
    finally:
        app.shutdown()
