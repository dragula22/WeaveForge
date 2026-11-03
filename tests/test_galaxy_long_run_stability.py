"""
Long-Run Galactic Dynamical Stability & Zero-Cheat Calibration Tests (`tests/test_galaxy_long_run_stability.py`)
=================================================================================================================
Verifies:
  1. Exact `(1 kpc, 100 km/s, 10^9 M_sun)` unit calibration (`A0_REF_CODE == 0.371238`) and zero-cheat
     bit-for-bit identical `t = 0` initial positions and velocities across all `PhysicsMod` frameworks.
  2. Dynamic atomic fixed-point grid mass conservation on ultra-low-mass dwarfs (`DDO 154`, zero `m_fix >= 1`
     35x mass inflation) and live `SPARC-175` catalog spawning.
  3. 600-step (`dt = 0.030`) bound galactic rotation across `NGC 3198`, `DDO 154`, and `Milky Way` for
     `tnap_weave`, `classic_mond`, `aest`, `superfluid_dm`, and `lambda_cdm` (`R_half` within `+-18%`,
     `> 97%` bound mass, live CDM halo `R_half` within `+-15%`) vs genuine outward dispersion
     (`R_half(600) > 1.65 * R_half(0)`) for the baryon-only `ghost_redteam_demo` control.
  4. 4K (`3840x2160`) 1,000,000-particle GPU `step + render` median frame time `< 5.0 ms`, DDR5 RAM
     hibernation, and `DDO 154` adaptive `fp16` DDR5 Time-Machine mass preservation.
  5. Conservative curl-free 3D QUMOND potential (`rel_curl < 1e-6`), logarithmic outer wall potential
     rise, and deterministic GPU `t = 0` `R_CoM` / `V_CoM` zeroing (`< 1e-5`).
  6. Full 60-second (`3,600`-step, `dt = 0.030`, `sim_time = 108.0` ~ `1.06 Gyr`) multi-rotation
     disk stability (`R_half` within `+-16%`, `bound >= 96%`, `|R_com| < 1.5 kpc`) vs Red-Team dispersion.
"""

from __future__ import annotations

import time
import moderngl
import numpy as np
import pytest

from weaveforge.arenas.arena3_galaxy_collider import A0_REF_CODE, Arena3GalaxyCollider
from weaveforge.arenas.arena4_cosmic_crucible import get_sparc_175_cached
from weaveforge.core.camera import Camera3D
from weaveforge.core.frameworks import get_mods_by_id
from weaveforge.core.symbolic_compiler import compile_master_action


def _half_mass_and_bound_fraction(
    pos: np.ndarray,
    mask: np.ndarray,
    half_box: float,
) -> tuple[float, float]:
    sub = pos[mask]
    masses = np.maximum(sub[:, 3], 0.0).astype(np.float64)
    tot_m = float(np.sum(masses))
    if tot_m <= 0.0:
        return 0.0, 0.0
    r3d = np.linalg.norm(sub[:, :3], axis=1)
    order = np.argsort(r3d)
    cum = np.cumsum(masses[order])
    idx = min(int(np.searchsorted(cum, 0.5 * tot_m)), len(r3d) - 1)
    r_half = float(r3d[order[idx]])
    bound_frac = float(np.sum(masses[r3d < half_box])) / tot_m
    return r_half, bound_frac


def test_arena3_unit_calibration_and_mod_independent_t0_state() -> None:
    ctx = moderngl.create_standalone_context(require=430)
    try:
        mods = get_mods_by_id()
        comps = {mid: compile_master_action(m) for mid, m in mods.items()}
        a3 = Arena3GalaxyCollider(ctx, particle_count=120_000)
        try:
            # 1. Exact (1 kpc, 100 km/s) unit calibration for a0
            assert A0_REF_CODE == pytest.approx(0.371238, rel=1e-3)
            a3.sync_mod_state(mods["tnap_weave"], comps["tnap_weave"])
            a3.reset_galaxy(0, mods["tnap_weave"], comps["tnap_weave"])
            assert a3.last_a0_code == pytest.approx(0.371238, rel=1e-3)
            assert a3.half_box == pytest.approx(38.0)

            # 2. Zero-cheat check: t=0 positions & velocities must be bit-for-bit identical across all mods
            check_mods = (
                "tnap_weave",
                "classic_mond",
                "aest",
                "superfluid_dm",
                "lambda_cdm",
                "ghost_redteam_demo",
            )
            for preset in (0, 1, 4):
                ref_pos = None
                ref_vel = None
                for mid in check_mods:
                    a3.sync_mod_state(mods[mid], comps[mid])
                    a3.reset_galaxy(preset, mods[mid], comps[mid])
                    pos = np.frombuffer(a3._pos_buf.read(), dtype=np.float32).reshape(-1, 4).copy()
                    vel = np.frombuffer(a3._vel_buf.read(), dtype=np.float32).reshape(-1, 4).copy()
                    if ref_pos is None:
                        ref_pos, ref_vel = pos, vel
                    else:
                        assert np.array_equal(pos, ref_pos), f"t=0 pos differed on preset {preset} for {mid}"
                        assert np.array_equal(vel, ref_vel), f"t=0 vel differed on preset {preset} for {mid}"

            # 3. Dynamic fixed-point atomic grid mass accuracy on low-mass dwarf DDO 154 (zero 35x m_fix>=1 inflation)
            a3.reset_galaxy(1, mods["tnap_weave"], comps["tnap_weave"])
            pos_ddo = np.frombuffer(a3._pos_buf.read(), dtype=np.float32).reshape(-1, 4)
            vel_ddo = np.frombuffer(a3._vel_buf.read(), dtype=np.float32).reshape(-1, 4)
            bary_mass_particles = float(np.sum(pos_ddo[vel_ddo[:, 3] < 1.5, 3], dtype=np.float64))
            rho_grid = np.frombuffer(a3._rho_flt_buf.read(), dtype=np.float32)
            dx = (2.0 * a3.half_box) / float(a3.grid_m)
            # _rho_flt_buf stores raw mass density rho (M_sun_1e9 / kpc^3) before 4*pi*G Poisson RHS scaling
            grid_mass = float(np.sum(rho_grid, dtype=np.float64)) * (dx ** 3)
            assert bary_mass_particles == pytest.approx(0.38, rel=0.02)
            assert grid_mass == pytest.approx(bary_mass_particles, rel=0.02)

            # 4. SPARC-175 live catalog spawner velocity calibration check
            catalog = {g.name: g for g in get_sparc_175_cached()}
            for gname, v_lo, v_hi in (("NGC3198", 125.0, 178.0), ("DDO154", 38.0, 62.0)):
                if gname in catalog:
                    a3.load_sparc_galaxy(catalog[gname], mods["tnap_weave"], comps["tnap_weave"])
                    v_kms = a3.telemetry_dict()["v_outer_kms"]
                    assert v_lo <= v_kms <= v_hi, f"SPARC {gname} v_outer_kms={v_kms:.1f} outside [{v_lo}, {v_hi}]"
        finally:
            a3.release()
    finally:
        ctx.release()


def test_galaxy_600_step_bound_rotation_vs_redteam_dispersion() -> None:
    ctx = moderngl.create_standalone_context(require=430)
    try:
        mods = get_mods_by_id()
        comps = {mid: compile_master_action(m) for mid, m in mods.items()}
        a3 = Arena3GalaxyCollider(ctx, particle_count=160_000)
        try:
            bound_mods = ("tnap_weave", "classic_mond", "aest", "superfluid_dm", "lambda_cdm")
            presets = (
                (0, "NGC 3198", 120.0, 185.0),
                (1, "DDO 154", 40.0, 62.0),
                (4, "Milky Way", 175.0, 245.0),
            )

            for preset_idx, pname, v_min, v_max in presets:
                for mid in (*bound_mods, "ghost_redteam_demo"):
                    m = mods[mid]
                    c = comps[mid]
                    a3.sync_mod_state(m, c)
                    a3.reset_galaxy(preset_idx, m, c)

                    pos0 = np.frombuffer(a3._pos_buf.read(), dtype=np.float32).reshape(-1, 4)
                    vel0 = np.frombuffer(a3._vel_buf.read(), dtype=np.float32).reshape(-1, 4)
                    bary0 = vel0[:, 3] < 1.5
                    cdm0 = vel0[:, 3] > 1.5
                    rh0, _ = _half_mass_and_bound_fraction(pos0, bary0, a3.half_box)
                    cdm_rh0, _ = _half_mass_and_bound_fraction(pos0, cdm0, a3.half_box)

                    for _ in range(600):
                        a3.step(m, c, dt=0.030)

                    pos1 = np.frombuffer(a3._pos_buf.read(), dtype=np.float32).reshape(-1, 4)
                    vel1 = np.frombuffer(a3._vel_buf.read(), dtype=np.float32).reshape(-1, 4)
                    assert np.all(np.isfinite(pos1)) and np.all(np.isfinite(vel1))

                    bary1 = vel1[:, 3] < 1.5
                    cdm1 = vel1[:, 3] > 1.5
                    rh600, bound600 = _half_mass_and_bound_fraction(pos1, bary1, a3.half_box)
                    cdm_rh600, _ = _half_mass_and_bound_fraction(pos1, cdm1, a3.half_box)
                    v_out = a3.telemetry_dict()["v_outer_kms"]

                    if mid == "ghost_redteam_demo":
                        assert rh600 > 1.65 * rh0, (
                            f"{pname} ghost_redteam_demo failed to disperse: "
                            f"R_half(0)={rh0:.2f}, R_half(600)={rh600:.2f}"
                        )
                    else:
                        rel_drift = abs(rh600 / rh0 - 1.0)
                        assert rel_drift <= 0.18, (
                            f"{pname} [{mid}] R_half drifted {rel_drift*100:.1f}% "
                            f"(R_half(0)={rh0:.2f} -> R_half(600)={rh600:.2f})"
                        )
                        assert bound600 > 0.97, f"{pname} [{mid}] bound fraction={bound600:.4f} <= 0.97"
                        assert v_min <= v_out <= v_max, (
                            f"{pname} [{mid}] v_outer_kms={v_out:.1f} outside [{v_min}, {v_max}]"
                        )
                        if mid == "lambda_cdm":
                            cdm_drift = abs(cdm_rh600 / cdm_rh0 - 1.0)
                            assert cdm_drift <= 0.15, (
                                f"{pname} [lambda_cdm] CDM halo R_half drifted {cdm_drift*100:.1f}%"
                            )
        finally:
            a3.release()
    finally:
        ctx.release()


def test_arena3_4k_1m_particle_frametime_hibernation_and_ddo154_fp16_replay() -> None:
    ctx = moderngl.create_standalone_context(require=430)
    try:
        mods = get_mods_by_id()
        m = mods["tnap_weave"]
        c = compile_master_action(m)

        a3 = Arena3GalaxyCollider(ctx, particle_count=1_000_000)
        tex_4k = ctx.texture((3840, 2160), components=4, dtype="f1")
        fbo_4k = ctx.framebuffer(color_attachments=[tex_4k])
        cam = Camera3D(yaw=0.68, pitch=0.58, distance=31.0)
        try:
            a3.sync_mod_state(m, c)
            a3.reset_galaxy(0, m, c)

            # Warmup 10 4K 1M-particle frames
            for i in range(10):
                a3.step(m, c, dt=0.030)
                a3.render(fbo_4k, cam, wall_time=i * 0.016)
            ctx.finish()

            # Measure per-frame GPU latency across 36 frames at 3840x2160, 1,000,000 particles
            frame_ms = []
            for i in range(36):
                t0 = time.perf_counter()
                a3.step(m, c, dt=0.030)
                a3.render(fbo_4k, cam, wall_time=(10 + i) * 0.016)
                ctx.finish()
                frame_ms.append((time.perf_counter() - t0) * 1000.0)

            median_4k_ms = float(np.median(frame_ms))
            assert median_4k_ms < 5.0, f"4K 1M-particle median frame time {median_4k_ms:.2f} ms >= 5.0 ms"

            # Verify DDO 154 ultra-low particle mass (~5.4e-7) survives fp16 Time-Machine replay & DDR5 hibernation
            a3.reset_galaxy(1, m, c)
            pos_before = np.frombuffer(a3._pos_buf.read(), dtype=np.float32).reshape(-1, 4).copy()
            vel_before = np.frombuffer(a3._vel_buf.read(), dtype=np.float32).reshape(-1, 4).copy()
            bary_before = vel_before[:, 3] < 1.5
            m_sum_before = float(np.sum(pos_before[bary_before, 3], dtype=np.float64))
            assert m_sum_before == pytest.approx(0.38, rel=0.02)

            a3._capture_replay_snapshot()
            for _ in range(5):
                a3.step(m, c, dt=0.030)
            a3.load_replay_frame(0)
            pos_replayed = np.frombuffer(a3._pos_buf.read(), dtype=np.float32).reshape(-1, 4)
            m_sum_replayed = float(np.sum(pos_replayed[bary_before, 3], dtype=np.float64))
            assert m_sum_replayed == pytest.approx(m_sum_before, rel=0.01)

            a3.hibernate_to_ram()
            assert a3.is_hibernated
            assert a3.allocated_vram_mb == 0.0
            a3.wake_from_ram()
            assert not a3.is_hibernated
            pos_woken = np.frombuffer(a3._pos_buf.read(), dtype=np.float32).reshape(-1, 4)
            m_sum_woken = float(np.sum(pos_woken[bary_before, 3], dtype=np.float64))
            assert m_sum_woken == pytest.approx(m_sum_before, rel=0.01)
        finally:
            fbo_4k.release()
            tex_4k.release()
            a3.release()
    finally:
        ctx.release()


def _com_and_z_rms(pos: np.ndarray, vel: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    sub_p = pos[mask]
    sub_v = vel[mask]
    masses = np.maximum(sub_p[:, 3], 0.0).astype(np.float64)
    tot_m = float(np.sum(masses))
    if tot_m <= 0.0:
        return 0.0, 0.0, 0.0
    r_com = float(np.linalg.norm(np.sum(sub_p[:, :3].astype(np.float64) * masses[:, None], axis=0) / tot_m))
    v_com = float(np.linalg.norm(np.sum(sub_v[:, :3].astype(np.float64) * masses[:, None], axis=0) / tot_m))
    z_rms = float(np.sqrt(np.sum((sub_p[:, 1].astype(np.float64) ** 2) * masses) / tot_m))
    return r_com, v_com, z_rms


def test_qumond_conservative_curl_free_field_and_exact_com_zeroing() -> None:
    ctx = moderngl.create_standalone_context(require=430)
    try:
        mods = get_mods_by_id()
        comps = {mid: compile_master_action(m) for mid, m in mods.items()}
        a3 = Arena3GalaxyCollider(ctx, particle_count=160_000)
        try:
            # 1. Verify Stage-2 QUMOND activation vs unity-nu bypass
            for mid in ("tnap_weave", "classic_mond", "aest", "superfluid_dm"):
                a3.sync_mod_state(mods[mid], comps[mid])
                assert a3._nu_is_unity is False, f"{mid} should enable Stage-2 QUMOND solve"
                assert a3._qumond_rhs_prog is not None
            for mid in ("lambda_cdm", "ghost_redteam_demo"):
                a3.sync_mod_state(mods[mid], comps[mid])
                assert a3._nu_is_unity is True, f"{mid} has nu==1 and should bypass Stage-2 QUMOND solve"
                assert a3._qumond_rhs_prog is None

            # 2. Verify exact t=0 R_CoM and V_CoM GPU tree-reduction zeroing across all isolated galaxy presets
            m_tnap, c_tnap = mods["tnap_weave"], comps["tnap_weave"]
            a3.sync_mod_state(m_tnap, c_tnap)
            for preset_idx in (0, 1, 4):
                a3.reset_galaxy(preset_idx, m_tnap, c_tnap)
                pos0 = np.frombuffer(a3._pos_buf.read(), dtype=np.float32).reshape(-1, 4)
                vel0 = np.frombuffer(a3._vel_buf.read(), dtype=np.float32).reshape(-1, 4)
                rcom_b, vcom_b, _ = _com_and_z_rms(pos0, vel0, vel0[:, 3] < 1.5)
                rcom_c, vcom_c, _ = _com_and_z_rms(pos0, vel0, vel0[:, 3] > 1.5)
                assert rcom_b < 1e-5 and vcom_b < 1e-5, f"Preset {preset_idx} baryon CoM not zeroed: R={rcom_b}, V={vcom_b}"
                assert rcom_c < 1e-5 and vcom_c < 1e-5, f"Preset {preset_idx} CDM CoM not zeroed: R={rcom_c}, V={vcom_c}"

            # 3. Verify 3D QUMOND grid acceleration is strictly curl-free (-grad Phi_eff) with logarithmic outer potential
            a3.reset_galaxy(0, m_tnap, c_tnap)
            ftex = np.frombuffer(a3._force_tex.read(), dtype=np.float32).reshape((64, 64, 64, 4))
            phi_eff = np.frombuffer(a3._mg_eff_u0.read(), dtype=np.float32).reshape((64, 64, 64))
            phi_n = np.frombuffer(a3._mg_ua[0].read(), dtype=np.float32).reshape((64, 64, 64))

            h = (2.0 * a3.half_box) / 64.0
            inv_2h = 0.5 / h
            gx, gy, gz = ftex[..., 0], ftex[..., 1], ftex[..., 2]
            dgx_dy = (gx[2:-2, 3:-1, 2:-2] - gx[2:-2, 1:-3, 2:-2]) * inv_2h
            dgy_dx = (gy[2:-2, 2:-2, 3:-1] - gy[2:-2, 2:-2, 1:-3]) * inv_2h
            dgx_dz = (gx[3:-1, 2:-2, 2:-2] - gx[1:-3, 2:-2, 2:-2]) * inv_2h
            dgz_dx = (gz[2:-2, 2:-2, 3:-1] - gz[2:-2, 2:-2, 1:-3]) * inv_2h
            dgy_dz = (gy[3:-1, 2:-2, 2:-2] - gy[1:-3, 2:-2, 2:-2]) * inv_2h
            dgz_dy = (gz[2:-2, 3:-1, 2:-2] - gz[2:-2, 1:-3, 2:-2]) * inv_2h
            curl_mag = np.sqrt((dgz_dy - dgy_dz) ** 2 + (dgx_dz - dgz_dx) ** 2 + (dgy_dx - dgx_dy) ** 2)
            g_max = float(np.max(np.linalg.norm(ftex[2:-2, 2:-2, 2:-2, :3], axis=-1)))
            rel_curl = float(np.max(curl_mag)) / max(g_max, 1e-8)
            assert rel_curl < 1e-6, f"QUMOND 3D force field is not curl-free: rel_curl={rel_curl:.3e}"

            g_eff_outer = float(np.linalg.norm(ftex[32, 32, 48, :3]))
            gn_outer = float(abs(phi_n[32, 32, 49] - phi_n[32, 32, 47]) * inv_2h)
            assert (g_eff_outer / max(gn_outer, 1e-8)) > 3.0, "Expected strong QUMOND radial boost at R ~ 19.6 kpc"

            dphi_eff_outer = float(phi_eff[32, 32, 62] - phi_eff[32, 32, 54])
            assert dphi_eff_outer > 0.45, f"Expected confining logarithmic QUMOND potential rise near wall, got {dphi_eff_outer}"
        finally:
            a3.release()
    finally:
        ctx.release()


def test_galaxy_3600_step_60s_multi_rotation_stability() -> None:
    ctx = moderngl.create_standalone_context(require=430)
    try:
        mods = get_mods_by_id()
        comps = {mid: compile_master_action(m) for mid, m in mods.items()}
        a3 = Arena3GalaxyCollider(ctx, particle_count=160_000)
        try:
            presets = (
                (0, "NGC 3198", 120.0, 185.0, ("tnap_weave", "classic_mond", "lambda_cdm", "ghost_redteam_demo")),
                (1, "DDO 154", 38.0, 65.0, ("tnap_weave", "lambda_cdm", "ghost_redteam_demo")),
                (4, "Milky Way", 170.0, 245.0, ("tnap_weave", "lambda_cdm", "ghost_redteam_demo")),
            )

            for preset_idx, pname, v_min, v_max, check_mods in presets:
                for mid in check_mods:
                    m = mods[mid]
                    c = comps[mid]
                    a3.sync_mod_state(m, c)
                    a3.reset_galaxy(preset_idx, m, c)

                    pos0 = np.frombuffer(a3._pos_buf.read(), dtype=np.float32).reshape(-1, 4)
                    vel0 = np.frombuffer(a3._vel_buf.read(), dtype=np.float32).reshape(-1, 4)
                    bary0 = vel0[:, 3] < 1.5
                    cdm0 = vel0[:, 3] > 1.5
                    rh0, _ = _half_mass_and_bound_fraction(pos0, bary0, a3.half_box)
                    cdm_rh0, _ = _half_mass_and_bound_fraction(pos0, cdm0, a3.half_box)

                    for _ in range(3600):
                        a3.step(m, c, dt=0.030)

                    assert a3.sim_time == pytest.approx(108.0, rel=1e-4)
                    pos1 = np.frombuffer(a3._pos_buf.read(), dtype=np.float32).reshape(-1, 4)
                    vel1 = np.frombuffer(a3._vel_buf.read(), dtype=np.float32).reshape(-1, 4)
                    assert np.all(np.isfinite(pos1)) and np.all(np.isfinite(vel1))

                    bary1 = vel1[:, 3] < 1.5
                    cdm1 = vel1[:, 3] > 1.5
                    rh3600, bound3600 = _half_mass_and_bound_fraction(pos1, bary1, a3.half_box)
                    cdm_rh3600, _ = _half_mass_and_bound_fraction(pos1, cdm1, a3.half_box)
                    r_com, _, z_rms = _com_and_z_rms(pos1, vel1, bary1)
                    v_out = a3.telemetry_dict()["v_outer_kms"]

                    if mid == "ghost_redteam_demo":
                        assert rh3600 > 1.65 * rh0, (
                            f"{pname} ghost_redteam_demo failed to disperse at 3600 steps: "
                            f"R_half(0)={rh0:.2f}, R_half(3600)={rh3600:.2f}"
                        )
                        assert bound3600 < 0.80, (
                            f"{pname} ghost_redteam_demo remained artificially bound at 3600 steps: bound={bound3600:.3f}"
                        )
                    else:
                        rel_drift = abs(rh3600 / rh0 - 1.0)
                        assert rel_drift <= 0.16, (
                            f"{pname} [{mid}] 60s R_half drifted {rel_drift*100:.1f}% "
                            f"(R_half(0)={rh0:.2f} -> R_half(3600)={rh3600:.2f})"
                        )
                        assert bound3600 >= 0.96, f"{pname} [{mid}] 60s bound fraction={bound3600:.4f} < 0.96"
                        assert r_com < 1.5, f"{pname} [{mid}] 60s CoM drifted {r_com:.3f} kpc >= 1.5 kpc"
                        assert z_rms < 1.5, f"{pname} [{mid}] 60s disk puffed vertically z_rms={z_rms:.3f} kpc >= 1.5 kpc"
                        assert v_min <= v_out <= v_max, (
                            f"{pname} [{mid}] 60s v_outer_kms={v_out:.1f} outside [{v_min}, {v_max}]"
                        )
                        if mid == "lambda_cdm":
                            cdm_drift = abs(cdm_rh3600 / cdm_rh0 - 1.0)
                            assert cdm_drift <= 0.18, (
                                f"{pname} [lambda_cdm] 60s CDM halo R_half drifted {cdm_drift*100:.1f}%"
                            )
        finally:
            a3.release()
    finally:
        ctx.release()

