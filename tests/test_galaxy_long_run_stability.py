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
