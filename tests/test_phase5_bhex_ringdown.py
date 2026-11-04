"""
Phase 5 (Milestone 5A-2) Automated Verification Suite — Strong-Field BHEX QNM Ringdown & Arena 2 Submode 3
===========================================================================================================
Verifies:
  1. Symbolic (`SymPy`) derivation of exact covariant axial spin-2 Regge–Wheeler potential `V_ell^RW(r)`
     purely from `(metric_lapse_A(U), optical_metric_n(U))`, exact simplification to `(1 - 2M/R)[l(l+1)/R^2 - 6M/R^3]`
     on isotropic Schwarzschild GR, exact photon-sphere BHEX shadows & Lyapunov exponents, and `tnap_weave`'s
     smooth horizonless catenoid/wormhole throat `R_min = e * M` at `r = M`.
  2. 4th-order FD + RK4 1+1D wave PDE + SVD Matrix-Pencil QNM calibration against Leaver's exact Schwarzschild
     `M * omega_220 = 0.37367168 - 0.08896232i` (`< 0.4%` omega_R, `< 1.5%` |omega_I|), genuine `tnap_weave`
     strong-field ringdown shift, and candid Red-Team LVK O3b/O4 `TGR` spectroscopy classification (`PASS`/`WARN`/`FAIL`).
  3. Gibbons Planck-spool partial reflectivity (`R_refl = 0.65`) post-ringdown gravitational-wave echo generation
     at `Delta t_echo ~ 2|r_{*,peak} - r_{*,wall}|` and `< 0.5 ms` warm-cached `get_bhex_ringdown_summary` latency.
  4. Headless `moderngl` OpenGL 4.30 rendering of `Arena2BlackHoleForge` Submode 3 (`bssn_ringdown`) at
     `spool_reflectivity = 0.0` vs `0.75` without breaking Submodes `0, 1, 2`.
"""

from __future__ import annotations

import math
import time

import moderngl
import numpy as np
import pytest
import sympy as sp

from weaveforge.arenas.arena2_black_hole_forge import Arena2BlackHoleForge, FORGE_SUBMODES
from weaveforge.core.bhex_ringdown import (
    LEAVER_SCHWARZSCHILD_L2_OMEGA_I,
    LEAVER_SCHWARZSCHILD_L2_OMEGA_R,
    clear_bhex_ringdown_cache,
    derive_mod_strong_field_qnm_potential_sympy,
    get_bhex_ringdown_summary,
    simulate_bhex_qnm_ringdown,
)
from weaveforge.core.camera import Camera3D
from weaveforge.core.frameworks import get_mods_by_id
from weaveforge.core.symbolic_compiler import compile_master_action


def test_sympy_covariant_regge_wheeler_exact_gr_and_weave_throat() -> None:
    """
    Proves exact SymPy equivalence of `V_ell^RW(r)` on isotropic Schwarzschild GR to Regge & Wheeler (1957),
    exact photon-sphere BHEX shadow & Lyapunov instability formulas, and `tnap_weave`'s horizonless throat.
    """
    clear_bhex_ringdown_cache()
    mods = get_mods_by_id()

    # 1. Isotropic Schwarzschild GR (`lambda_cdm`)
    gr_sym = derive_mod_strong_field_qnm_potential_sympy(mods["lambda_cdm"], ell=2)
    r_sym = gr_sym["symbols"]["r"]
    M_sym = gr_sym["symbols"]["M"]
    ell_sym = gr_sym["symbols"]["ell"]

    R_gr = gr_sym["areal_radius_R_expr"]
    expected_R_gr = r_sym * (1 + M_sym / (2 * r_sym)) ** 2
    assert sp.simplify(R_gr - expected_R_gr) == 0

    expected_rw_general = (1 - 2 * M_sym / R_gr) * (
        ell_sym * (ell_sym + 1) / (R_gr**2) - 6 * M_sym / (R_gr**3)
    )
    assert sp.simplify(gr_sym["v_rw_general_ell_expr"] - expected_rw_general) == 0
    assert sp.simplify(gr_sym["gr_rw_symbolic_residual"]) == 0
    assert gr_sym["is_exact_gr_regge_wheeler"] is True

    assert gr_sym["has_event_horizon"] is True
    assert gr_sym["has_wormhole_throat"] is False
    assert gr_sym["r_horizon_over_m"] == pytest.approx(0.5, rel=1e-12)
    assert gr_sym["areal_horizon_over_m"] == pytest.approx(2.0, rel=1e-12)
    assert gr_sym["r_photon_over_m"] == pytest.approx(1.0 + 0.5 * math.sqrt(3.0), rel=1e-12)
    assert gr_sym["areal_photon_over_m"] == pytest.approx(3.0, rel=1e-12)
    assert gr_sym["b_crit_over_m"] == pytest.approx(3.0 * math.sqrt(3.0), rel=1e-12)
    assert gr_sym["omega_ph_m"] == pytest.approx(1.0 / (3.0 * math.sqrt(3.0)), rel=1e-12)
    assert gr_sym["lyapunov_m"] == pytest.approx(1.0 / (3.0 * math.sqrt(3.0)), rel=1e-12)
    assert gr_sym["eikonal_omega_r_m"] == pytest.approx(2.5 / (3.0 * math.sqrt(3.0)), rel=1e-12)
    assert gr_sym["eikonal_omega_i_m"] == pytest.approx(-0.5 / (3.0 * math.sqrt(3.0)), rel=1e-12)

    # 2. TNaP Exponential Weave (`tnap_weave`: A = exp(-2U), n = exp(2U))
    wv_sym = derive_mod_strong_field_qnm_potential_sympy(mods["tnap_weave"], ell=2)
    r_w = wv_sym["symbols"]["r"]
    M_w = wv_sym["symbols"]["M"]
    assert sp.simplify(wv_sym["areal_radius_R_expr"] - r_w * sp.exp(M_w / r_w)) == 0
    assert wv_sym["is_exact_gr_regge_wheeler"] is False

    assert wv_sym["has_event_horizon"] is False
    assert wv_sym["has_wormhole_throat"] is True
    assert sp.simplify(wv_sym["r_throat_expr"] - M_w) == 0
    assert sp.simplify(wv_sym["areal_throat_expr"] - sp.E * M_w) == 0
    assert wv_sym["r_throat_over_m"] == pytest.approx(1.0, rel=1e-12)
    assert wv_sym["areal_throat_over_m"] == pytest.approx(math.e, rel=1e-12)

    assert sp.simplify(wv_sym["r_photon_expr"] - 2 * M_w) == 0
    assert sp.simplify(wv_sym["b_crit_expr"] - 2 * sp.E * M_w) == 0
    assert wv_sym["r_photon_over_m"] == pytest.approx(2.0, rel=1e-12)
    assert wv_sym["areal_photon_over_m"] == pytest.approx(2.0 * math.sqrt(math.e), rel=1e-12)
    assert wv_sym["b_crit_over_m"] == pytest.approx(2.0 * math.e, rel=1e-12)
    assert wv_sym["omega_ph_m"] == pytest.approx(1.0 / (2.0 * math.e), rel=1e-12)
    assert wv_sym["lyapunov_m"] == pytest.approx(1.0 / (2.0 * math.e), rel=1e-12)
    expected_eik_shift = ((3.0 * math.sqrt(3.0)) / (2.0 * math.e) - 1.0) * 100.0
    assert wv_sym["omega_ph_vs_gr_pct"] == pytest.approx(expected_eik_shift, rel=1e-10)

    # 3. Seamless compilation across every registered PhysicsMod
    for mod_id, mod_obj in mods.items():
        derived = derive_mod_strong_field_qnm_potential_sympy(mod_obj, ell=2)
        assert derived["mod_id"] == mod_id
        assert derived["b_crit_over_m"] > 2.0
        assert derived["v_peak_m2"] > 0.0


def test_leaver_gr_qnm_calibration_and_tnap_ringdown_shift() -> None:
    """
    Verifies zero-fudge PDE + Matrix-Pencil recovery of Leaver's exact Schwarzschild l=2 QNM,
    measures `tnap_weave`'s genuine strong-field QNM shift, and checks candid LVK O3b/O4 TGR verdicts.
    """
    mods = get_mods_by_id()

    # 1. Isotropic Schwarzschild GR calibration against Leaver (1985)
    gr_res = simulate_bhex_qnm_ringdown(mods["lambda_cdm"], ell=2, spool_reflectivity=0.0)
    assert np.all(np.isfinite(gr_res["v_rw_grid"]))
    assert 0.0 < float(gr_res["v_rw_grid"][0]) < 1.0e-14  # Exact near-horizon tail at r_* = -90M

    rel_err_r = abs(gr_res["omega_r"] - LEAVER_SCHWARZSCHILD_L2_OMEGA_R) / LEAVER_SCHWARZSCHILD_L2_OMEGA_R * 100.0
    rel_err_i = (
        abs(abs(gr_res["omega_i"]) - abs(LEAVER_SCHWARZSCHILD_L2_OMEGA_I))
        / abs(LEAVER_SCHWARZSCHILD_L2_OMEGA_I)
        * 100.0
    )
    assert rel_err_r < 0.4, f"GR omega_R calibration error {rel_err_r:.4f}% exceeded 0.4%"
    assert rel_err_i < 1.5, f"GR |omega_I| calibration error {rel_err_i:.4f}% exceeded 1.5%"
    assert gr_res["omega_i"] < 0.0
    assert gr_res["fit_rel_rms"] < 1.0e-3
    assert abs(gr_res["delta_f220_vs_gr_pct"]) < 1.0e-9
    assert gr_res["ligo_status"] == "PASS"
    assert gr_res["ligo_o4_audit"]["status"] == "PASS"

    # 2. TNaP Exponential Weave (`tnap_weave`) genuine non-perturbative l=2 QNM shift & candid WARN audit
    wv_res = simulate_bhex_qnm_ringdown(mods["tnap_weave"], ell=2, spool_reflectivity=0.0)
    df_weave = float(wv_res["delta_f220_vs_gr_pct"])
    dtau_weave = float(wv_res["delta_tau220_vs_gr_pct"])
    assert -9.0 <= df_weave <= -2.5
    assert dtau_weave > 1.0  # Longer damping time tau = 1 / |omega_I| on wider 2*e*M barrier
    assert wv_res["ligo_status"] == "WARN"
    audit_wv = wv_res["ligo_o4_audit"]
    assert audit_wv["status"] == "WARN"
    assert audit_wv["exceeds_hierarchical_o3b"] is True
    verdict_txt = str(wv_res["ligo_honest_verdict"]).lower()
    assert "degeneracy" in verdict_txt
    assert "gw250114" in verdict_txt or "o4/o5" in verdict_txt

    # 3. 1911 Half-Bending / Adversarial Ghost Mod (`ghost_redteam_demo`: n(U) = exp(U)) -> candid FAIL
    bad_res = simulate_bhex_qnm_ringdown(mods["ghost_redteam_demo"], ell=2, spool_reflectivity=0.0)
    assert abs(float(bad_res["delta_f220_vs_gr_pct"])) > 8.0
    assert bad_res["ligo_status"] == "FAIL"
    assert bad_res["ligo_o4_audit"]["status"] == "FAIL"


def test_planck_spool_echo_generation_and_warm_cache_latency() -> None:
    """
    Verifies post-ringdown Gibbons Planck-spool gravitational-wave echoes at `spool_reflectivity = 0.65`
    and `< 0.5 ms` warm-cache latency on `get_bhex_ringdown_summary`.
    """
    mods = get_mods_by_id()
    weave_mod = mods["tnap_weave"]

    s_zero = get_bhex_ringdown_summary(weave_mod, spool_reflectivity=0.0)
    s_echo = get_bhex_ringdown_summary(weave_mod, spool_reflectivity=0.65)

    echo_zero = float(s_zero["echo_ratio"])
    echo_refl = float(s_echo["echo_ratio"])
    assert echo_refl > 5.0 * echo_zero
    assert echo_refl > 0.15

    # Verify echo delay matches round-trip cavity travel time 2 * |r_{*,peak} - r_{*,wall}|
    th_delay = float(s_echo["echo_delay_m"])
    meas_delay = float(s_echo["measured_echo_delay_m"])
    assert 65.0 <= th_delay <= 100.0
    assert abs(meas_delay - th_delay) < 18.0

    # Primary ringdown QNM extracted before first echo must remain consistent with R=0
    assert abs(float(s_echo["omega_r"]) - float(s_zero["omega_r"])) / float(s_zero["omega_r"]) < 0.01

    # Verify ImPlot float32 downsampled waveform arrays & scalar keys
    required_keys = {
        "omega_r",
        "omega_i",
        "gr_omega_r",
        "gr_omega_i",
        "delta_f220_vs_gr_pct",
        "delta_tau220_vs_gr_pct",
        "echo_delay_m",
        "echo_ratio",
        "ligo_status",
        "ligo_honest_verdict",
        "t_over_m",
        "h_mod",
        "h_gr",
    }
    assert required_keys.issubset(s_echo.keys())
    for arr_key in ("t_over_m", "h_mod", "h_gr"):
        arr = s_echo[arr_key]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.flags["C_CONTIGUOUS"]
        assert arr.shape == (320,)
        assert np.all(np.isfinite(arr))

    # Warm-cache latency benchmark (< 0.5 ms required)
    n_iters = 100
    t0 = time.perf_counter()
    for _ in range(n_iters):
        res_warm = get_bhex_ringdown_summary(weave_mod, spool_reflectivity=0.65)
    elapsed_ms_per_call = ((time.perf_counter() - t0) / n_iters) * 1000.0
    assert res_warm is s_echo
    assert elapsed_ms_per_call < 0.5, f"Warm summary took {elapsed_ms_per_call:.4f} ms (>= 0.5 ms)"


def test_arena2_submode3_bssn_ringdown_glsl_render() -> None:
    """
    Creates a headless `moderngl` standalone context, renders `Arena2BlackHoleForge` in `submode = 3`
    at `spool_reflectivity = 0.0` and `0.75`, and asserts finite non-blank RGB pixels, distinct echo
    visuals, `qnm_freq_ratio` spiral response, and intact Submodes `0, 1, 2`.
    """
    assert len(FORGE_SUBMODES) >= 4
    assert FORGE_SUBMODES[3] == ("bssn_ringdown", "Strong-Field BSSN Funnel & BHEX QNM Ringdown")

    ctx = moderngl.create_standalone_context(require=430)
    tex_f32 = ctx.texture((160, 120), 4, dtype="f4")
    fbo_f32 = ctx.framebuffer(color_attachments=[tex_f32])
    arena = None
    try:
        mods = get_mods_by_id()
        comp_gr = compile_master_action(mods["lambda_cdm"])
        comp_wv = compile_master_action(mods["tnap_weave"])

        arena = Arena2BlackHoleForge(ctx, compiled=comp_wv, mod=mods["tnap_weave"])
        assert arena.spool_reflectivity == 0.0
        assert arena.qnm_freq_ratio == 1.0

        cam = Camera3D(distance=11.5, pitch=0.45, yaw=0.65)

        def _render_rgb(submode: int, comp, wall_t: float = 1.25) -> np.ndarray:
            arena.submode = submode
            arena.render(fbo_f32, cam, comp, wall_time=wall_t)
            raw = np.frombuffer(tex_f32.read(), dtype=np.float32).reshape(120, 160, 4)
            assert np.all(np.isfinite(raw)), f"Non-finite pixel in Arena 2 submode {submode}"
            rgb = raw[..., :3].copy()
            assert float(np.mean(rgb)) > 0.03, f"Blank frame in Arena 2 submode {submode}"
            assert float(np.std(rgb)) > 0.02, f"Flat zero-contrast frame in Arena 2 submode {submode}"
            return rgb

        # 1. Verify Submodes 0, 1, 2 remain 100% intact and distinct
        rgb_0 = _render_rgb(0, comp_wv)
        rgb_1 = _render_rgb(1, comp_wv)
        rgb_2 = _render_rgb(2, comp_wv)
        assert float(np.mean(np.abs(rgb_0 - rgb_1))) > 0.02
        assert float(np.mean(np.abs(rgb_1 - rgb_2))) > 0.02

        # 2. Render Submode 3 (`bssn_ringdown`) at spool_reflectivity = 0.0 vs 0.75
        arena.qnm_freq_ratio = 0.9228
        arena.spool_reflectivity = 0.0
        rgb_3_no_echo = _render_rgb(3, comp_wv)

        arena.spool_reflectivity = 0.75
        rgb_3_echo = _render_rgb(3, comp_wv)

        echo_diff = np.abs(rgb_3_echo - rgb_3_no_echo)
        assert float(np.max(echo_diff)) > 0.15
        assert float(np.mean(echo_diff)) > 0.01
        assert float(np.mean(np.abs(rgb_3_no_echo - rgb_2))) > 0.02

        # 3. Verify QNM frequency ratio & compiled optical metric response in Submode 3
        arena.qnm_freq_ratio = 1.0
        rgb_3_gr_freq = _render_rgb(3, comp_gr)
        assert float(np.max(np.abs(rgb_3_gr_freq - rgb_3_echo))) > 0.05
    finally:
        if arena is not None:
            arena.release()
        fbo_f32.release()
        tex_f32.release()
        ctx.release()


def test_studio_phase5_bssn_ringdown_and_kz_telemetry(tmp_path) -> None:
    """
    Verifies full headless `WeaveForgeStudioApp` rendering of Arena 2 Submode 3 (`bssn_ringdown`
    live `ImPlot` QNM oscilloscope + `Spool Echo R` controls + 3+1D BSSNok telemetry) and Arena 4
    3D Kibble–Zurek & Chiral Baryogenesis telemetry, plus sidebar reset restoration.
    """
    from weaveforge.studio.app import WeaveForgeStudioApp

    app = WeaveForgeStudioApp(
        initial_mod_id="tnap_weave",
        initial_arena=2,
        width=1600,
        height=960,
        vsync=False,
        visible=False,
    )
    try:
        app.active_arena = 1
        app.select_left_tab(1)
        app.arena2.submode = 3
        app.reset_active_camera()
        app.arena2.spin_a = 0.0
        app.arena2.spool_reflectivity = 0.0
        snap_r0 = tmp_path / "studio_submode3_r0.png"
        app.render_single_frame(dt_override=0.016)
        app.render_single_frame(capture_out=snap_r0, dt_override=0.016)
        assert snap_r0.exists() and snap_r0.stat().st_size > 25_000
        assert 0.88 <= app.arena2.qnm_freq_ratio <= 0.98

        # Switch to reflective Gibbons Planck-spool (R = 0.65)
        app.arena2.spool_reflectivity = 0.65
        snap_r65 = tmp_path / "studio_submode3_r65.png"
        app.render_single_frame(capture_out=snap_r65, dt_override=0.016)
        assert snap_r65.exists() and snap_r65.stat().st_size > 25_000

        # Verify Arena 4 3D Kibble-Zurek & Red-Team Crucible UI renders cleanly
        app.active_arena = 3
        app.render_single_frame(dt_override=0.016)
        assert app.crucible is not None
        assert app.crucible.cmb_xi_kz_max > app.crucible.cmb_xi_kz_min > 0.0

        # Verify reset_all_sidebar_defaults restores spool_reflectivity -> 0.0
        app.reset_all_sidebar_defaults()
        assert app.arena2.spool_reflectivity == pytest.approx(0.0, abs=1e-9)
    finally:
        app.shutdown()
