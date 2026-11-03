"""
WeaveForge Interactive 60–120+ FPS 3D Studio (`weaveforge/studio/app.py`)
=========================================================================
Unified Desktop GPU Sandbox (`GLFW` + `ModernGL` OpenGL 4.6 + `Dear ImGui` + `ImPlot`):
  • Top Command Bar: Live Physics Framework Switcher (6 built-in mods + custom `.toml` save),
    4 Multi-Scale 3D Arena Tabs, Pause/Reset/Snapshot, and live FPS + RTX 5080 telemetry.
  • Left Studio Panel: "One Master Cost Sheet" Live Symbolic Lagrangian Editor (`SymPy` ->
    `GLSL 430` + `WGSL` in `< 50 ms`), 5-Gate Red-Team Health Audit, Coupling Sliders,
    Context-Sensitive Arena Controls, and Rig 2 (`AMD RX 5700 XT`) LAN Worker Dispatcher.
  • Center 3D Viewport: Zero-copy OpenGL FBO texture display (`ImTextureRef`) with full
    60+ FPS `WASD + Q/E + Shift` Free-Fly & Turntable Orbit/Pan/Zoom camera.
  • Right Studio Panel: Ruthlessly Honest 5-Dataset Observational Pass/Fail Crucible +
    3 Real-Time `ImPlot` Scientific Charts (SPARC Rotation Curves, Planck CMB Peaks,
    and Dec 2, 2026 `Gaia DR4` Wide-Binary `|z_gal|` Velocity Boost).
"""

from __future__ import annotations

import copy
import ctypes
import ctypes.wintypes as _wintypes
import inspect
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import glfw
from imgui_bundle import imgui, implot
import moderngl
import numpy as np
from OpenGL import GL
from PIL import Image

from weaveforge.arenas import arena4_cosmic_crucible as arena4_mod
from weaveforge.arenas.arena1_knot_tank import Arena1KnotTank, KNOT_COLOR_MODES, KNOT_PRESETS
from weaveforge.arenas.arena2_black_hole_forge import Arena2BlackHoleForge, FORGE_SUBMODES
from weaveforge.arenas.arena3_galaxy_collider import Arena3GalaxyCollider, GALAXY_PRESETS
from weaveforge.arenas.arena4_cosmic_crucible import (
    Arena4CosmicCrucible,
    CrucibleReport,
    evaluate_observational_crucible,
    get_sparc_175_cached,
)
from weaveforge.compute.lan_worker import AsyncLANWorkerDispatcher
from weaveforge.core.camera import Camera3D
from weaveforge.core.frameworks import DEFAULT_MODS_DIR, PhysicsMod, load_all_mods
from weaveforge.core.symbolic_compiler import (
    CompiledMasterAction,
    SymbolicCompilationError,
    compile_master_action,
)
from weaveforge.studio.theme import (
    COL_ACCENT,
    COL_FAIL,
    COL_FORECAST,
    COL_MUTED,
    COL_PASS,
    COL_WARN,
    apply_luminous_sapphire_theme,
    badge_color,
    load_studio_fonts,
)
from weaveforge.studio.video_recorder import (
    AsyncVideoRecorder,
    VIDEO_FPS_OPTIONS,
    VIDEO_RESOLUTION_OPTIONS,
    VideoRecordingResult,
)


ARENA_TITLES = [
    "[1] Knot Tank",
    "[2] Black-Hole Forge",
    "[3] 1M-Star Collider",
    "[4] Hourglass & Crucible",
]
_ARENA_TITLES_MED = [
    "[1] Knots",
    "[2] BH Forge",
    "[3] 1M-Galaxy",
    "[4] Crucible",
]
_ARENA_TITLES_COMPACT = [
    "[1] Knots",
    "[2] BH",
    "[3] Galaxy",
    "[4] Cosmic",
]
_ARENA_TITLES_TINY = [
    "[1]",
    "[2]",
    "[3]",
    "[4]",
]

UI_SCALE_STOPS: Tuple[float, ...] = (1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50)
RENDER_SCALE_OPTIONS: Tuple[Tuple[float, str], ...] = (
    (0.50, "50% Perf"),
    (0.67, "67% Crisp"),
    (0.75, "75% Balanced"),
    (1.00, "100% Native"),
)
FPS_CAP_OPTIONS: Tuple[Tuple[int, str], ...] = (
    (60, "60 FPS"),
    (120, "120 FPS"),
    (240, "240 FPS"),
    (0, "Uncapped"),
)
COMPUTE_MODE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("auto", "Auto (LAN Rig 2 -> Local Zen 5 + Radeon iGPU Fallback)"),
    ("local", "Single-PC Only (Local Zen 5 + Radeon iGPU + DDR5, Skip LAN)"),
    ("lan", "LAN Rig 2 Only"),
)
_DEFAULT_CAMERA_POSES: Tuple[Tuple[float, float, float], ...] = (
    (0.62, 0.38, 15.0),
    (0.45, 0.24, 18.5),
    (0.68, 0.52, 28.0),
    (0.55, 0.20, 12.6),
)
_COUPLING_PARAM_FIELDS: Tuple[str, ...] = (
    "c_w",
    "m_eff",
    "g4",
    "g6",
    "a0_si",
    "has_cdm_halo",
    "cdm_halo_scale",
    "pinned_pleat_coupling",
    "waist_clamping_ell",
    "w0_de",
    "wa_de",
)
_ACTION_EXPR_FIELDS: Tuple[str, ...] = (
    "field_lagrangian",
    "optical_metric_n",
    "metric_lapse_A",
    "interpolation_nu",
    "gw_speed_ratio",
)

_GL_GPU_MEM_INFO_TOTAL_NVX = 0x9048
_GL_GPU_MEM_INFO_CURRENT_AVAIL_NVX = 0x9049


class _Win32ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", _wintypes.DWORD),
        ("PageFaultCount", _wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _query_process_working_set_mb() -> float:
    """Queries live OS DDR5 Working Set RAM (MB) of the current process via Win32 kernel32."""
    if sys.platform != "win32":
        return 0.0
    try:
        pmc = _Win32ProcessMemoryCountersEx()
        pmc.cb = ctypes.sizeof(pmc)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Win32ProcessMemoryCountersEx),
            _wintypes.DWORD,
        ]
        if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def _construct_arena_compat(cls: type, *pos_args: Any, **kwargs: Any) -> Tuple[Any, bool]:
    """
    Constructs an Arena class cleanly whether or not sibling backend kwargs
    (`compiled`, `mod`, `lazy_gpu`) are supported on `cls.__init__`.
    Returns `(instance, accepted_compiled_kwarg)`.
    """
    accepted_compiled = False
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_varkw:
            filtered = dict(kwargs)
        else:
            filtered = {k: v for k, v in kwargs.items() if k in params}
        accepted_compiled = "compiled" in filtered
        return cls(*pos_args, **filtered), accepted_compiled
    except TypeError:
        base_only = {k: v for k, v in kwargs.items() if k in ("initial_grid_n", "particle_count")}
        return cls(*pos_args, **base_only), False


def health_audit_header(compiled: CompiledMasterAction) -> Tuple[imgui.ImVec4, str]:
    """
    Computes the honest 5-Gate Health Audit banner `(color, text)`:
      • Any `FAIL` gate      -> Red `RED-TEAM ALERT`
      • Any `WARN` gate      -> Amber `CAUTION — STABILITY / FALSE-VACUUM WARNING!`
      • All 5 gates `PASS`   -> Green `5-GATE HEALTH AUDIT: ALL 5 GATES PASSED`
    """
    if not compiled.all_critical_passed:
        return COL_FAIL, "RED-TEAM ALERT: BROKEN / UNSTABLE PHYSICS DETECTED!"
    if compiled.has_warnings or not compiled.all_gates_passed:
        return COL_WARN, "5-GATE HEALTH AUDIT: CAUTION — STABILITY / FALSE-VACUUM WARNING!"
    return COL_PASS, "5-GATE HEALTH AUDIT: ALL 5 GATES PASSED"


class WeaveForgeStudioApp:
    """Main 60–120+ FPS Interactive 3D Physics Studio & Observational Crucible."""

    def __init__(
        self,
        width: int = 1680,
        height: int = 960,
        initial_arena: int = 1,
        initial_mod_id: str = "tnap_weave",
        vsync: bool = True,
        visible: bool = True,
        mods_dir: Optional[Path] = DEFAULT_MODS_DIR,
        ui_scale: Optional[float] = None,
        render_scale: Optional[float] = None,
        fps_cap: Optional[int] = None,
    ) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW for WeaveForge Studio.")

        self._visible: bool = bool(visible)
        self.vsync: bool = bool(vsync)
        self._in_imgui_frame: bool = False
        self._in_render_frame: bool = False
        self._one_shot_stall_in_frame: bool = False
        self._pending_theme_apply: bool = False
        self._shut_down: bool = False
        self._timer_period_active: bool = False

        # Detect primary monitor resolution, refresh rate, DPI scale, & desktop workarea
        self._mon_width: int = 1920
        self._mon_height: int = 1080
        self._mon_refresh: int = 60
        self._mon_scale: float = 1.0
        work_x, work_y, work_w, work_h = 0, 0, 1920, 1040

        if self._visible:
            mon = glfw.get_primary_monitor()
            if mon:
                vm = glfw.get_video_mode(mon)
                if vm is not None:
                    self._mon_width = int(vm.size.width)
                    self._mon_height = int(vm.size.height)
                    self._mon_refresh = int(vm.refresh_rate)
                sx, sy = glfw.get_monitor_content_scale(mon)
                self._mon_scale = float(max(sx, sy, 1.0))
                wx, wy, ww, wh = glfw.get_monitor_workarea(mon)
                if ww > 0 and wh > 0:
                    work_x, work_y, work_w, work_h = int(wx), int(wy), int(ww), int(wh)

        # Headless test guard: when visible=False and ui_scale is None, preserve exact 1.0 scale & window size!
        if not self._visible:
            self._ui_scale_auto: bool = False
            self.ui_scale: float = self._sanitize_ui_scale(ui_scale) if ui_scale is not None else 1.0
            self.render_scale: float = (
                self._sanitize_render_scale(render_scale) if render_scale is not None else 1.0
            )
            self.fps_cap: int = self._sanitize_fps_cap(fps_cap) if fps_cap is not None else 0
            init_w, init_h = int(width), int(height)
            center_on_workarea = False
        else:
            if ui_scale is None:
                self._ui_scale_auto = True
                self.ui_scale = self._compute_auto_ui_scale(self._mon_scale, self._mon_width)
            else:
                self._ui_scale_auto = False
                self.ui_scale = self._sanitize_ui_scale(ui_scale)

            if render_scale is None:
                self.render_scale = 0.75 if self._mon_width >= 3200 else 1.0
            else:
                self.render_scale = self._sanitize_render_scale(render_scale)

            if fps_cap is None:
                self.fps_cap = 120 if self._mon_refresh >= 165 else 0
            else:
                self.fps_cap = self._sanitize_fps_cap(fps_cap)

            init_w, init_h = int(width), int(height)
            center_on_workarea = False
            if int(width) == 1680 and int(height) == 960 and self.ui_scale > 1.05:
                init_w = min(round(width * self.ui_scale), int(work_w * 0.92))
                init_h = min(round(height * self.ui_scale), int(work_h * 0.90))
                center_on_workarea = True

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if self._visible else glfw.FALSE)

        self.window = glfw.create_window(
            init_w,
            init_h,
            "WeaveForge Studio — Multi-Scale 3D Physics Sandbox & Observational Crucible",
            None,
            None,
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create OpenGL 4.6 GLFW window (fallback 4.3 check needed).")

        self._hires_timer: Optional[int] = None
        if self._visible and sys.platform == "win32":
            try:
                if ctypes.windll.winmm.timeBeginPeriod(1) == 0:
                    self._timer_period_active = True
            except Exception:
                pass
            try:
                k32 = ctypes.windll.kernel32
                k32.CreateWaitableTimerExW.restype = ctypes.c_void_p
                k32.CreateWaitableTimerExW.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_wchar_p,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                ]
                k32.SetWaitableTimer.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int32,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int32,
                ]
                k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                k32.CloseHandle.argtypes = [ctypes.c_void_p]
                th = k32.CreateWaitableTimerExW(None, None, 0x00000002, 0x001F0003)
                if th:
                    self._hires_timer = int(th)
            except Exception:
                self._hires_timer = None

        if self._visible:
            if center_on_workarea:
                pos_x = work_x + max((work_w - init_w) // 2, 0)
                pos_y = work_y + max((work_h - init_h) // 2, 0)
                glfw.set_window_pos(self.window, pos_x, pos_y)
            wsx, wsy = glfw.get_window_content_scale(self.window)
            win_scale = float(max(wsx, wsy, 1.0))
            if win_scale > 0.5:
                self._mon_scale = max(self._mon_scale, win_scale)
                if self._ui_scale_auto:
                    self.ui_scale = self._compute_auto_ui_scale(self._mon_scale, self._mon_width)

        glfw.make_context_current(self.window)
        self._apply_swap_interval(explicit_uncapped=False)

        self.ctx = moderngl.create_context()
        self.gpu_renderer = str(self.ctx.info.get("GL_RENDERER", "OpenGL 4.6 GPU"))

        # Register GLFW content-scale callback BEFORE glfw_init_for_opengl so ImGui chains onto it!
        self._content_scale_cb = self._on_window_content_scale
        glfw.set_window_content_scale_callback(self.window, self._content_scale_cb)

        # Initialize Dear ImGui + ImPlot + GLFW/OpenGL3 backends
        imgui.create_context()
        implot.create_context()
        io = imgui.get_io()
        if hasattr(io, "set_ini_filename"):
            io.set_ini_filename(None)
        else:
            try:
                io.ini_filename = None  # type: ignore[attr-defined]
            except Exception:
                pass
        io.ini_saving_rate = 0.0

        self.ui_font, self.mono_font = load_studio_fonts()
        apply_luminous_sapphire_theme(self.ui_scale)

        win_addr = ctypes.cast(self.window, ctypes.c_void_p).value
        assert win_addr is not None
        imgui.backends.glfw_init_for_opengl(int(win_addr), True)
        imgui.backends.opengl3_init("#version 430")
        apply_luminous_sapphire_theme(self.ui_scale)

        # Load all PhysicsMods & initialize active working copy (exact ID match BEFORE substring fallback!)
        self.mods_dir: Path = Path(mods_dir) if mods_dir is not None else DEFAULT_MODS_DIR
        self.mods: List[PhysicsMod] = load_all_mods(self.mods_dir)
        if not self.mods:
            raise RuntimeError(f"No PhysicsMod .toml files found in {self.mods_dir}")
        self.active_mod_idx = self._resolve_initial_mod_index(initial_mod_id)

        self.working_mod: PhysicsMod = self.mods[self.active_mod_idx].clone()
        self._pristine_mod: PhysicsMod = self.mods[self.active_mod_idx].clone()
        self.compile_error: str = ""
        self.selected_sparc_idx: int = 0
        self.sparc_names: List[str] = [g.name for g in get_sparc_175_cached()]
        self.show_shader_modal: bool = False
        self.status_banner: str = ""
        self._requested_left_tab: Optional[int] = None

        # 1. Compile working_mod FIRST & asynchronously warm CAMB DDR5 cache if available
        t_comp0 = time.perf_counter()
        self.compiled: Optional[CompiledMasterAction] = compile_master_action(self.working_mod)
        self.last_compile_ms: float = (time.perf_counter() - t_comp0) * 1000.0
        if hasattr(arena4_mod, "warm_camb_ddr5_cache_async"):
            try:
                arena4_mod.warm_camb_ddr5_cache_async()
            except Exception:
                pass

        self._active_arena: int = int(np.clip(initial_arena - 1, 0, 3))
        self.ddr5_hibernate_enabled: bool = True
        self.paused: bool = False
        self._arena1_needs_kernel_sync: bool = False
        self._arena3_needs_field_sync: bool = False
        self._replay_idx: Dict[int, int] = {}
        self._replay_live_backup: Dict[int, Dict[str, Any]] = {}

        # 2. Construct 4 Connected 3D GPU Arenas with lazy_gpu=(arena_idx != active_arena)
        self.arena1, a1_got_comp = _construct_arena_compat(
            Arena1KnotTank,
            self.ctx,
            initial_grid_n=256,
            compiled=self.compiled,
            mod=self.working_mod,
            lazy_gpu=(self._active_arena != 0),
        )
        self.arena2, _ = _construct_arena_compat(
            Arena2BlackHoleForge,
            self.ctx,
            compiled=self.compiled,
            mod=self.working_mod,
            lazy_gpu=(self._active_arena != 1),
        )
        self.arena3, a3_got_comp = _construct_arena_compat(
            Arena3GalaxyCollider,
            self.ctx,
            particle_count=1_000_000,
            compiled=self.compiled,
            mod=self.working_mod,
            lazy_gpu=(self._active_arena != 2),
        )
        self.arena4, _ = _construct_arena_compat(
            Arena4CosmicCrucible,
            self.ctx,
            compiled=self.compiled,
            mod=self.working_mod,
            lazy_gpu=(self._active_arena != 3),
        )

        if not a1_got_comp and self.compiled is not None:
            self.arena1.update_compiled_action(self.compiled, mod=self.working_mod)
        if not a3_got_comp and self.compiled is not None:
            solve_a3_init = self._active_arena == 2
            self.arena3.sync_mod_state(self.working_mod, self.compiled, re_solve_field=solve_a3_init)
            self._arena3_needs_field_sync = not solve_a3_init

        self.crucible: Optional[CrucibleReport] = evaluate_observational_crucible(
            self.working_mod,
            self.compiled,
            selected_galaxy_idx=self.selected_sparc_idx,
        )

        self.cameras: List[Camera3D] = [
            Camera3D(yaw=0.62, pitch=0.38, distance=15.0),  # Arena 1: 256^3 Knot Tank
            Camera3D(yaw=0.45, pitch=0.24, distance=18.5),  # Arena 2: Kerr/Weave BH Forge
            Camera3D(yaw=0.68, pitch=0.52, distance=28.0),  # Arena 3: 1M-Star Galaxy Collider
            Camera3D(yaw=0.55, pitch=0.20, distance=12.6),  # Arena 4: Two-Wing CPT Hourglass
        ]

        # Exact snapshot of per-arena visual/slider defaults from constructed Arena __init__ state
        self._arena_defaults: Dict[int, Dict[str, Any]] = {
            0: {
                "density_gain": float(self.arena1.density_gain),
                "slice_x": float(self.arena1.slice_x),
                "damping": float(self.arena1.damping),
                "color_mode": int(self.arena1.color_mode),
            },
            1: {
                "submode": int(self.arena2.submode),
                "spin_a": float(self.arena2.spin_a),
                "show_bhex_gauge": bool(self.arena2.show_bhex_gauge),
                "show_phase_slips": bool(self.arena2.show_phase_slips),
                "mass_solar": float(self.arena2.mass_solar),
                "probe_alt_km": float(self.arena2.probe_alt_km),
            },
            2: {
                "show_lensing": bool(self.arena3.show_lensing),
            },
            3: {
                "waist_r0": float(self.arena4.waist_r0),
                "tau_now": float(self.arena4.tau_now),
                "auto_scrub": bool(self.arena4.auto_scrub),
                "show_creases": bool(self.arena4.show_creases),
            },
        }

        # Captures directory, async 60/120 FPS video recorder, & Right-Click Context Menu state
        self.captures_dir: Path = Path("captures")
        self._video_recorder: AsyncVideoRecorder = AsyncVideoRecorder()
        self.video_target_fps: int = 60
        self.video_resolution_mode: str = "native"
        self._last_rec_tick_wall: float = time.perf_counter()

        self._rmb_down: bool = False
        self._rmb_down_pos: Tuple[float, float] = (0.0, 0.0)
        self._rmb_drag_dist: float = 0.0
        self._pending_open_context_menu: bool = False
        self.is_context_menu_open: bool = False
        self._last_context_menu_rect: Optional[Tuple[float, float, float, float]] = None

        self._last_topbar_widgets_end_x: float = 0.0
        self._last_topbar_telemetry_x: float = 0.0
        self._last_topbar_telemetry_drawn: str = ""
        self._pending_screenshot_out: Optional[Path] = None
        self._last_slider_reset_rects: Dict[str, Tuple[float, float, float, float]] = {}
        self._last_top_reset_btn_rect: Optional[Tuple[float, float, float, float]] = None

        # Viewport offscreen FBO (zero-copy texture bound to ImGui viewport)
        self._view_tex: Optional[moderngl.Texture] = None
        self._view_fbo: Optional[moderngl.Framebuffer] = None
        init_fbo_w = max(int(round(init_w * 0.45 * self.render_scale)), 64)
        init_fbo_h = max(int(round(max(init_h - round(52.0 * self.ui_scale), 200) * self.render_scale)), 64)
        self._ensure_viewport_fbo(init_fbo_w, init_fbo_h)

        # Async Rig 2 LAN Worker Dispatcher + Single-PC execution mode
        self.lan_dispatcher = AsyncLANWorkerDispatcher(worker_url="http://192.168.1.50:8765")
        self.compute_execution_mode: str = str(getattr(self.lan_dispatcher, "execution_mode", "auto"))
        setattr(self.lan_dispatcher, "execution_mode", self.compute_execution_mode)

        # Honest post-swap wall-clock frame-time & VRAM/DDR5 telemetry state
        self._start_wall = time.perf_counter()
        self._prev_wall = self._start_wall
        self._prev_swap_wall = self._start_wall
        self._rendered_frames: int = 0
        init_fps = float(self.fps_cap) if self.fps_cap > 0 else 60.0
        self._dt_smooth: float = 1.0 / init_fps
        self.fps_smooth: float = init_fps
        self.frame_ms_smooth: float = self._dt_smooth * 1000.0
        self.vram_used_gb: float = 0.0
        self.arena_vram_mb: float = 0.0
        self.tracked_ddr5_mb: float = 0.0
        self.ddr5_used_mb: float = 0.0
        self._last_mem_poll_wall: float = -10.0
        self._poll_memory_telemetry(force=True)

    # =========================================================================
    # SCALE, DPI, HIBERNATION & TELEMETRY HELPERS
    # =========================================================================
    @staticmethod
    def _sanitize_ui_scale(val: float) -> float:
        f = float(val)
        if not math.isfinite(f):
            return 1.0
        return float(np.clip(f, 0.75, 3.0))

    @staticmethod
    def _sanitize_render_scale(val: float) -> float:
        f = float(val)
        if not math.isfinite(f):
            return 1.0
        return float(np.clip(f, 0.35, 1.0))

    @staticmethod
    def _sanitize_fps_cap(val: int) -> int:
        try:
            v = int(val)
        except Exception:
            return 0
        return max(0, min(v, 1000))

    def _resolve_initial_mod_index(self, requested_id: str) -> int:
        req = str(requested_id or "").strip()
        if not req:
            return 0
        # Pass 1: Exact mod ID match (prevents `tnap_weave` from shadowing `user_tnap_weave`!)
        for i, m in enumerate(self.mods):
            if m.id == req:
                return i
        # Pass 2: Exact file stem match (with or without leading numeric prefix like `05_` / `99_`)
        for i, m in enumerate(self.mods):
            stem = m.source_path.stem if m.source_path is not None else ""
            if stem and (stem == req or re.sub(r"^\d+_", "", stem) == req):
                return i
        # Pass 3: Substring fallback for shorthand CLI arguments (e.g. `--mod khronon`)
        for i, m in enumerate(self.mods):
            stem = m.source_path.stem if m.source_path is not None else ""
            if req in m.id or m.id in req or (stem and req in stem):
                return i
        return 0

    @staticmethod
    def _compute_auto_ui_scale(mon_scale: float, mon_width: int) -> float:
        s = float(mon_scale) if math.isfinite(float(mon_scale)) else 1.0
        if int(mon_width) >= 3200:
            return max(s, 1.75)
        return max(s, 1.0)

    def _get_window_monitor_metrics(self) -> Tuple[int, int, int, float]:
        """Finds the monitor containing `self.window`'s center (with primary-monitor fallback)."""
        best_mon = None
        if self._visible and getattr(self, "window", None) is not None:
            try:
                wx, wy = glfw.get_window_pos(self.window)
                ww, wh = glfw.get_window_size(self.window)
                cx = int(wx) + max(int(ww), 1) // 2
                cy = int(wy) + max(int(wh), 1) // 2
                monitors = glfw.get_monitors() or []
                for m in monitors:
                    mx, my, mw, mh = glfw.get_monitor_workarea(m)
                    vm_m = glfw.get_video_mode(m)
                    if vm_m is not None:
                        px, py = glfw.get_monitor_pos(m)
                        mx, my, mw, mh = int(px), int(py), int(vm_m.size.width), int(vm_m.size.height)
                    if mx <= cx < (mx + mw) and my <= cy < (my + mh):
                        best_mon = m
                        break
            except Exception:
                best_mon = None

        if best_mon is None:
            try:
                best_mon = glfw.get_primary_monitor()
            except Exception:
                best_mon = None

        mon_w, mon_h, mon_ref, mon_s = self._mon_width, self._mon_height, self._mon_refresh, self._mon_scale
        if best_mon is not None:
            try:
                vm = glfw.get_video_mode(best_mon)
                if vm is not None:
                    mon_w = int(vm.size.width)
                    mon_h = int(vm.size.height)
                    mon_ref = int(vm.refresh_rate)
                sx, sy = glfw.get_monitor_content_scale(best_mon)
                mon_s = float(max(sx, sy, 1.0))
            except Exception:
                pass
        return mon_w, mon_h, mon_ref, mon_s

    def _apply_swap_interval(self, explicit_uncapped: bool = False) -> None:
        """
        Configures GLFW swap interval:
          • `0` when Uncapped, VSync-off, or when a sub-refresh software `fps_cap` (e.g. `120 FPS` on `240 Hz`)
            paces frames via Win32 high-res waitable timer (prevents DWM VBlank-1 double-blocking!).
          • `1` otherwise for native-refresh VSync.
        """
        try:
            sub_refresh_cap = self._visible and (0 < self.fps_cap < self._mon_refresh)
            if explicit_uncapped or (not self.vsync) or sub_refresh_cap:
                glfw.swap_interval(0)
            else:
                glfw.swap_interval(1)
        except Exception:
            pass

    def _sleep_precise(self, duration_s: float) -> None:
        """Sleeps `duration_s` with `~0.5 ms` kernel precision via Win32 `CreateWaitableTimerExW`."""
        if duration_s <= 0.0:
            return
        th = getattr(self, "_hires_timer", None)
        if th is not None and sys.platform == "win32":
            try:
                due = ctypes.c_int64(-max(int(duration_s * 10_000_000.0), 1000))
                k32 = ctypes.windll.kernel32
                if k32.SetWaitableTimer(ctypes.c_void_p(th), ctypes.byref(due), 0, None, None, 0):
                    timeout_ms = max(int(duration_s * 1000.0) + 15, 2)
                    k32.WaitForSingleObject(ctypes.c_void_p(th), timeout_ms)
                    return
            except Exception:
                pass
        time.sleep(duration_s)

    def _on_window_content_scale(self, _window: Any, xscale: float, yscale: float) -> None:
        new_mon_scale = float(max(xscale, yscale, 1.0))
        self._mon_scale = new_mon_scale
        if self._visible:
            mon_w, mon_h, mon_ref, _ = self._get_window_monitor_metrics()
            self._mon_width = mon_w
            self._mon_height = mon_h
            self._mon_refresh = mon_ref
            self._apply_swap_interval(explicit_uncapped=False)
            if self._ui_scale_auto:
                target = self._compute_auto_ui_scale(new_mon_scale, mon_w)
                self.set_ui_scale(target, keep_auto=True)

    def set_ui_scale(self, new_scale: float, keep_auto: bool = False) -> None:
        """Updates UI DPI scale live without compounding styles or corrupting active ImGui stacks."""
        self._ui_scale_auto = bool(keep_auto)
        sanitized = self._sanitize_ui_scale(new_scale)
        if abs(sanitized - self.ui_scale) < 1e-4 and not self._pending_theme_apply:
            return
        self.ui_scale = sanitized
        if self._in_imgui_frame:
            self._pending_theme_apply = True
        else:
            apply_luminous_sapphire_theme(self.ui_scale)
            self._pending_theme_apply = False

    def reset_ui_scale_auto(self) -> None:
        """Resets UI scale to active-monitor Auto (4K -> >=175%, 1080p -> OS scale, headless -> 100%)."""
        if not self._visible:
            self.set_ui_scale(1.0, keep_auto=False)
            return
        mon_w, mon_h, mon_ref, mon_s = self._get_window_monitor_metrics()
        wsx, wsy = glfw.get_window_content_scale(self.window)
        self._mon_width = mon_w
        self._mon_height = mon_h
        self._mon_refresh = mon_ref
        self._mon_scale = float(max(wsx, wsy, mon_s, 1.0))
        target = self._compute_auto_ui_scale(self._mon_scale, self._mon_width)
        self.set_ui_scale(target, keep_auto=True)

    def set_render_scale(self, new_render_scale: float) -> None:
        self.render_scale = self._sanitize_render_scale(new_render_scale)

    def set_fps_cap(self, new_cap: int, explicit_user_choice: bool = False) -> None:
        self.fps_cap = self._sanitize_fps_cap(new_cap)
        self._apply_swap_interval(explicit_uncapped=(explicit_user_choice and self.fps_cap == 0))

    def _reset_frame_pacing_clock(self) -> None:
        """
        Resets `_prev_wall` and `_prev_swap_wall` after one-shot non-rendering stalls
        (offline PNG screenshot encoding or DDR5 arena hibernation/wake transfers) so
        steady-state `fps_smooth` and `frame_ms_smooth` never absorb offline I/O spikes.
        """
        now_t = time.perf_counter()
        self._prev_wall = now_t
        self._prev_swap_wall = now_t
        if getattr(self, "_in_render_frame", False):
            self._one_shot_stall_in_frame = True

    def _is_arena_in_replay(self, arena: Any, arena_idx: int) -> bool:
        return (
            int(getattr(arena, "replay_cursor", -1)) >= 0
            or (arena_idx in self._replay_live_backup)
            or (getattr(arena, "_live_backup", None) is not None)
            or bool(getattr(arena, "in_replay_mode", False))
        )

    @property
    def active_arena(self) -> int:
        return self._active_arena

    @active_arena.setter
    def active_arena(self, new_idx: int) -> None:
        self._switch_active_arena(new_idx)

    def _switch_active_arena(self, new_idx: int) -> None:
        """Switches active 3D Arena: hibernates outgoing VRAM to DDR5 FIRST, then wakes incoming arena."""
        clamped = int(np.clip(new_idx, 0, 3))
        old_idx = self._active_arena
        self._active_arena = clamped
        if not hasattr(self, "arena1") or not hasattr(self, "arena4"):
            return

        arenas = (self.arena1, self.arena2, self.arena3, self.arena4)

        # 1. Offload outgoing arena to DDR5 System RAM FIRST to minimize peak GDDR7 VRAM footprint!
        if clamped != old_idx:
            out_arena = arenas[old_idx]
            if self._is_arena_in_replay(out_arena, old_idx):
                self._exit_replay_to_live(old_idx, unpause=False)
            if self.ddr5_hibernate_enabled:
                hib_fn = getattr(out_arena, "hibernate_to_ram", None)
                if callable(hib_fn):
                    try:
                        hib_fn()
                    except Exception as exc:
                        self.status_banner = f"Arena {old_idx + 1} DDR5 hibernate warning: {exc}"

        in_arena = arenas[clamped]
        was_hibernated = bool(getattr(in_arena, "is_hibernated", False))

        # 2. If Arena 3 is hibernated and needs a field sync, mark it BEFORE wake_from_ram so it solves once!
        if clamped == 2 and self._arena3_needs_field_sync and self.compiled is not None and was_hibernated:
            if self.arena3.step_count == 0 or self.paused:
                self.arena3.sync_mod_state(self.working_mod, self.compiled, re_solve_field=True)
            self._arena3_needs_field_sync = False

        # 3. Wake incoming arena into GDDR7 VRAM
        wake_fn = getattr(in_arena, "wake_from_ram", None)
        if callable(wake_fn):
            try:
                wake_fn()
            except Exception as exc:
                self.status_banner = f"Arena {clamped + 1} wake warning: {exc}"

        if clamped == 0 and self._arena1_needs_kernel_sync and self.compiled is not None:
            self.arena1.update_compiled_action(self.compiled, mod=self.working_mod)
            self._arena1_needs_kernel_sync = False

        if clamped == 2 and self._arena3_needs_field_sync and self.compiled is not None:
            if self.arena3.step_count == 0 or self.paused:
                self.arena3.sync_mod_state(self.working_mod, self.compiled, re_solve_field=True)
            self._arena3_needs_field_sync = False

        self._poll_memory_telemetry(force=True)
        if clamped != old_idx or was_hibernated:
            self._reset_frame_pacing_clock()

    def _set_ddr5_hibernate_enabled(self, enabled: bool) -> None:
        self.ddr5_hibernate_enabled = bool(enabled)
        arenas = (self.arena1, self.arena2, self.arena3, self.arena4)
        if self.ddr5_hibernate_enabled:
            for idx, ar in enumerate(arenas):
                if idx != self._active_arena:
                    if self._is_arena_in_replay(ar, idx):
                        self._exit_replay_to_live(idx, unpause=False)
                    hib_fn = getattr(ar, "hibernate_to_ram", None)
                    if callable(hib_fn):
                        try:
                            hib_fn()
                        except Exception as exc:
                            self.status_banner = f"Arena {idx + 1} DDR5 hibernate warning: {exc}"
        self._poll_memory_telemetry(force=True)
        if self.ddr5_hibernate_enabled:
            self._reset_frame_pacing_clock()

    def _poll_memory_telemetry(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_mem_poll_wall) < 0.25:
            return
        self._last_mem_poll_wall = now

        arenas = (self.arena1, self.arena2, self.arena3, self.arena4)
        fbo_mb = 0.0
        if self._view_tex is not None:
            tw, th = self._view_tex.size
            fbo_mb = (tw * th * 4.0) / (1024.0 * 1024.0)

        self.arena_vram_mb = fbo_mb + sum(
            float(getattr(ar, "allocated_vram_mb", 0.0) or 0.0) for ar in arenas
        )
        vram_gb = self.arena_vram_mb / 1024.0
        try:
            for _ in range(8):
                if GL.glGetError() == 0:
                    break
            tot_kb = int(GL.glGetIntegerv(_GL_GPU_MEM_INFO_TOTAL_NVX))
            avail_kb = int(GL.glGetIntegerv(_GL_GPU_MEM_INFO_CURRENT_AVAIL_NVX))
            if GL.glGetError() == 0 and tot_kb > 0 and 0 <= avail_kb <= tot_kb:
                nvx_gb = float(tot_kb - avail_kb) / (1024.0 * 1024.0)
                vram_gb = max(nvx_gb, vram_gb)
        except Exception:
            pass
        self.vram_used_gb = vram_gb

        tracked_ddr5_mb = 0.0
        arena4_reported_ddr5 = False
        for ar_i, ar in enumerate(arenas):
            for attr in ("ddr5_ram_mb", "allocated_ddr5_mb", "ddr5_mb", "system_ram_mb", "ram_cache_mb"):
                val = getattr(ar, attr, None)
                if (not isinstance(val, bool)) and isinstance(val, (int, float, np.number)) and math.isfinite(float(val)):
                    tracked_ddr5_mb += float(val)
                    if ar_i == 3:
                        arena4_reported_ddr5 = True
                    break
        for ar_idx, bk in self._replay_live_backup.items():
            ar_obj = arenas[ar_idx] if 0 <= ar_idx < len(arenas) else None
            if ar_obj is not None and (
                callable(getattr(ar_obj, "resume_live_from_replay", None))
                or getattr(ar_obj, "_live_backup", None) is not None
            ):
                continue
            for v in bk.values():
                if isinstance(v, np.ndarray):
                    tracked_ddr5_mb += float(v.nbytes) / (1024.0 * 1024.0)
        if not arena4_reported_ddr5:
            camb_mb_fn = getattr(arena4_mod, "get_camb_ddr5_cache_mb", None)
            camb_attr = getattr(arena4_mod, "camb_ddr5_cache_mb", None)
            if callable(camb_mb_fn):
                try:
                    tracked_ddr5_mb += float(camb_mb_fn() or 0.0)
                except Exception:
                    pass
            elif (not isinstance(camb_attr, bool)) and isinstance(camb_attr, (int, float, np.number)):
                tracked_ddr5_mb += float(camb_attr)

        self.tracked_ddr5_mb = tracked_ddr5_mb
        proc_ws_mb = _query_process_working_set_mb()
        self.ddr5_used_mb = max(proc_ws_mb, tracked_ddr5_mb)

    def _ensure_viewport_fbo(self, w: int, h: int) -> None:
        w_clamped = max(int(w), 64)
        h_clamped = max(int(h), 64)
        if self._view_tex is not None and self._view_tex.size == (w_clamped, h_clamped):
            return
        if self._view_fbo is not None:
            self._view_fbo.release()
        if self._view_tex is not None:
            self._view_tex.release()
        self._view_tex = self.ctx.texture((w_clamped, h_clamped), components=4, dtype="f1")
        self._view_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._view_fbo = self.ctx.framebuffer(color_attachments=[self._view_tex])

    def select_left_tab(self, tab_idx: int) -> None:
        """Selects Left Studio Panel tab (`0`: Master Action, `1`: Arena & Couplings, `2`: Rig 2 Worker)."""
        self._requested_left_tab = int(np.clip(tab_idx, 0, 2))

    def select_preset_mod(self, idx: int) -> None:
        self.active_mod_idx = idx % len(self.mods)
        self.working_mod = self.mods[self.active_mod_idx].clone()
        self._pristine_mod = self.mods[self.active_mod_idx].clone()
        self._recompile_working_mod()
        self._reset_frame_pacing_clock()

    def save_custom_mod(self) -> Path:
        """
        Saves `self.working_mod` to `self.mods_dir` under a non-colliding `user_*` id,
        reloads `self.mods` immediately, and selects & compiles the newly saved custom mod!
        """
        self.mods_dir.mkdir(parents=True, exist_ok=True)
        existing_ids = {m.id for m in load_all_mods(self.mods_dir)}

        raw_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", self.working_mod.id.strip()).strip("_")[:48].strip("_") or "mod"
        clean_base_name = (
            re.sub(r"^\[Custom(?:\s*#\d+)?\]\s*", "", " ".join(self.working_mod.name.splitlines()).strip())
            or raw_slug
        )

        active_cur = self.mods[self.active_mod_idx] if (0 <= self.active_mod_idx < len(self.mods)) else None
        is_editing_custom = (
            raw_slug.startswith("user_")
            and active_cur is not None
            and (active_cur.id == raw_slug)
        )

        if is_editing_custom:
            custom_id = raw_slug
            self.working_mod.id = custom_id
            if not self.working_mod.name.strip().startswith("[Custom"):
                self.working_mod.name = f"[Custom] {clean_base_name}"
            else:
                self.working_mod.name = " ".join(self.working_mod.name.splitlines()).strip()
            if (
                active_cur is not None
                and active_cur.source_path is not None
                and active_cur.source_path.parent.resolve() == self.mods_dir.resolve()
            ):
                out_p = active_cur.source_path
            else:
                out_p = self.mods_dir / f"99_{custom_id}.toml"
        else:
            stem = (raw_slug[5:] if raw_slug.startswith("user_") else raw_slug)[:40].strip("_") or "mod"
            custom_id = f"user_{stem}"
            suffix = 1
            while (custom_id in existing_ids) or (self.mods_dir / f"99_{custom_id}.toml").exists():
                suffix += 1
                custom_id = f"user_{stem}_{suffix}"
            self.working_mod.id = custom_id
            self.working_mod.name = (
                f"[Custom] {clean_base_name}" if suffix == 1 else f"[Custom #{suffix}] {clean_base_name}"
            )
            out_p = self.mods_dir / f"99_{custom_id}.toml"

        self.working_mod.save_toml(out_p)

        self.mods = load_all_mods(self.mods_dir)
        for i, m in enumerate(self.mods):
            if m.id == custom_id:
                self.active_mod_idx = i
                break
        self.working_mod = self.mods[self.active_mod_idx].clone()
        self._pristine_mod = self.mods[self.active_mod_idx].clone()
        self._recompile_working_mod()
        self.status_banner = f"Saved mod [{custom_id}] -> {out_p.name}"
        self._reset_frame_pacing_clock()
        return out_p

    # =========================================================================
    # LEFT-SIDEBAR RESET DEFAULTS, CAMERA, STEPPING & VIDEO CAPTURE HELPERS
    # =========================================================================
    def is_parameter_modified(self, param_name: str) -> bool:
        """Returns `True` if `working_mod.parameters.<param_name>` differs from `_pristine_mod`."""
        cur = getattr(self.working_mod.parameters, param_name)
        base = getattr(self._pristine_mod.parameters, param_name)
        if isinstance(base, bool) or param_name == "has_cdm_halo":
            return bool(cur) != bool(base)
        if param_name == "a0_si":
            return not math.isclose(float(cur) * 1.0e10, float(base) * 1.0e10, rel_tol=1e-4, abs_tol=1e-4)
        return not math.isclose(float(cur), float(base), rel_tol=1e-4, abs_tol=1e-5)

    def count_modified_couplings(self) -> int:
        """Counts how many `ModParameters` sliders/checkboxes differ from `_pristine_mod`."""
        return sum(1 for name in _COUPLING_PARAM_FIELDS if self.is_parameter_modified(name))

    def count_modified_equations(self) -> int:
        """Counts how many symbolic `ActionSpec` equation strings differ from `_pristine_mod`."""
        return sum(
            1
            for name in _ACTION_EXPR_FIELDS
            if getattr(self.working_mod.action, name) != getattr(self._pristine_mod.action, name)
        )

    def count_modified_parameters(self) -> int:
        """Counts total modified `ModParameters` fields and `ActionSpec` expressions vs `_pristine_mod`."""
        return self.count_modified_couplings() + self.count_modified_equations()

    def reset_single_parameter(self, param_name: str) -> None:
        """Resets a single coupling slider/checkbox to its pristine `.toml` mod default."""
        if not hasattr(self._pristine_mod.parameters, param_name):
            return
        default_val = copy.deepcopy(getattr(self._pristine_mod.parameters, param_name))
        setattr(self.working_mod.parameters, param_name, default_val)
        if param_name == "has_cdm_halo":
            self.working_mod.parameters.cdm_halo_scale = float(self._pristine_mod.parameters.cdm_halo_scale)
        if param_name in ("c_w", "m_eff", "g4", "g6", "a0_si"):
            self._recompile_working_mod()
        else:
            self._refresh_crucible_fast()
        disp_str = f"{default_val * 1.0e10:.4g}e-10" if param_name == "a0_si" else f"{default_val}"
        self.status_banner = f"Reset {param_name} -> {disp_str} ({self._pristine_mod.name} default)"
        self._reset_frame_pacing_clock()

    def restore_default_mod_equations(self) -> None:
        """Restores all 5 symbolic `ActionSpec` equations to `self._pristine_mod.action`."""
        self.working_mod.action = copy.deepcopy(self._pristine_mod.action)
        self._recompile_working_mod()
        self.status_banner = f"Restored default Master Action equations for {self._pristine_mod.name}"
        self._reset_frame_pacing_clock()

    def reset_sliders_to_mod_defaults(self) -> None:
        """Restores `self.working_mod.parameters` to `self._pristine_mod.parameters` and syncs arenas."""
        self.working_mod.parameters = copy.deepcopy(self._pristine_mod.parameters)
        self._recompile_working_mod()
        self.status_banner = f"Reset all coupling sliders to {self._pristine_mod.name} defaults"
        self._reset_frame_pacing_clock()

    def reset_arena_view_settings(self, arena_idx: Optional[int] = None) -> None:
        """Restores arena visual/simulation slider settings to their clean `__init__` defaults."""
        arenas = (self.arena1, self.arena2, self.arena3, self.arena4)
        targets = range(4) if arena_idx is None else (int(np.clip(arena_idx, 0, 3)),)
        for idx in targets:
            defs = self._arena_defaults.get(idx, {})
            ar = arenas[idx]
            for attr, val in defs.items():
                if hasattr(ar, attr):
                    setattr(ar, attr, copy.deepcopy(val))

    def reset_all_sidebar_defaults(self) -> None:
        """Restores `self.working_mod` (equations + coupling sliders) AND arena sidebar settings to defaults."""
        self.working_mod = self._pristine_mod.clone()
        self.reset_arena_view_settings(arena_idx=None)
        self._recompile_working_mod()
        self.status_banner = (
            f"Reset all sidebar sliders, equations & arena settings to {self._pristine_mod.name} defaults"
        )
        self._reset_frame_pacing_clock()

    def reset_active_camera(self) -> None:
        """Resets the active 3D arena camera to its canonical orbit pose."""
        idx = int(np.clip(self._active_arena, 0, 3))
        if idx == 1:
            sub = int(getattr(self.arena2, "submode", 0))
            if sub == 1:
                self.cameras[1].reset(yaw=0.52, pitch=0.38, distance=8.2)
            elif sub == 2:
                self.cameras[1].reset(yaw=0.66, pitch=0.45, distance=11.5)
            else:
                self.cameras[1].reset(*_DEFAULT_CAMERA_POSES[1])
        else:
            self.cameras[idx].reset(*_DEFAULT_CAMERA_POSES[idx])
        self.status_banner = f"Reset 3D camera view for {ARENA_TITLES[idx]}"

    def toggle_pause(self) -> None:
        """Toggles simulation pause/resume cleanly (exiting DDR5 replay if resuming from replay)."""
        if self.paused:
            cur_ar = (self.arena1, self.arena2, self.arena3, self.arena4)[self._active_arena]
            if self._is_arena_in_replay(cur_ar, self._active_arena):
                self._exit_replay_to_live(self._active_arena, unpause=True)
            else:
                self.paused = False
        else:
            self.paused = True

    def step_single_simulation_frame(self) -> None:
        """Advances the active 3D arena by exactly 1 simulation step (even while paused) and keeps it paused."""
        if self.compiled is None:
            return
        self.paused = True
        if self._active_arena == 0:
            if self._is_arena_in_replay(self.arena1, 0):
                self._exit_replay_to_live(0, unpause=False)
            self.arena1.step(self.working_mod, self.compiled)
        elif self._active_arena == 2:
            if self._is_arena_in_replay(self.arena3, 2):
                self._exit_replay_to_live(2, unpause=False)
            self.arena3.step(self.working_mod, self.compiled, dt=0.018)
        elif self._active_arena == 3:
            self.arena4.tau_now = float((self.arena4.tau_now + 0.04) % 3.65)
        self.status_banner = f"Stepped {ARENA_TITLES[self._active_arena]} +1 frame"
        self._reset_frame_pacing_clock()

    def _recompile_working_mod(self) -> None:
        t0 = time.perf_counter()
        try:
            comp = compile_master_action(self.working_mod)
            self.compiled = comp
            self.crucible = evaluate_observational_crucible(
                self.working_mod,
                comp,
                selected_galaxy_idx=self.selected_sparc_idx,
            )
            self.arena1.update_compiled_action(comp, mod=self.working_mod)
            self._arena1_needs_kernel_sync = False

            # Only solve Arena 3 multigrid on GPU when Arena 3 is the active arena!
            solve_a3 = (self._active_arena == 2) and (self.arena3.step_count == 0 or self.paused)
            self.arena3.sync_mod_state(
                self.working_mod,
                comp,
                re_solve_field=solve_a3,
            )
            self._arena3_needs_field_sync = self._active_arena != 2
            self.compile_error = ""
            self.last_compile_ms = (time.perf_counter() - t0) * 1000.0
        except SymbolicCompilationError as exc:
            self.compile_error = str(exc)
        except Exception as exc:
            self.compile_error = f"Compilation error: {exc}"

    def _refresh_crucible_fast(self) -> None:
        if self.compiled is not None:
            self.crucible = evaluate_observational_crucible(
                self.working_mod,
                self.compiled,
                selected_galaxy_idx=self.selected_sparc_idx,
            )
            # Never trigger GPU multigrid readbacks on inactive Arena 3 while dragging sliders elsewhere!
            solve_a3 = (self._active_arena == 2) and (self.arena3.step_count == 0 or self.paused)
            self.arena3.sync_mod_state(
                self.working_mod,
                self.compiled,
                re_solve_field=solve_a3,
            )
            if self._active_arena != 2:
                self._arena3_needs_field_sync = True

    def _clear_replay_backup(self, arena_idx: int) -> None:
        self._replay_live_backup.pop(arena_idx, None)
        self._replay_idx.pop(arena_idx, None)

    def _stash_live_before_replay(self, arena: Any, arena_idx: int) -> None:
        """Snapshots live conjugate momenta/fields in DDR5 RAM before entering keyframe replay mode."""
        if callable(getattr(arena, "resume_live_from_replay", None)):
            return
        if arena_idx in self._replay_live_backup or int(getattr(arena, "replay_cursor", -1)) >= 0:
            return
        try:
            if arena_idx == 0:
                if bool(getattr(arena, "is_hibernated", False)) or getattr(arena, "_buf_a", None) is None:
                    arena.wake_from_ram()
                poll_fn = getattr(arena, "_poll_async_telemetry", None)
                if callable(poll_fn):
                    poll_fn(wait=True)
                vox = int(arena.grid_n) ** 3
                num_floats = vox * 3
                latest_buf = arena._buf_a if arena._cur_in_a else arena._buf_b
                copy_fn = getattr(arena, "_copy_ssbo_to_ram_f32", None)
                if callable(copy_fn) and latest_buf is not None and arena._buf_vel is not None:
                    self._replay_live_backup[0] = {
                        "kind": "arena1",
                        "grid_n": int(arena.grid_n),
                        "phi": copy_fn(latest_buf, num_floats),
                        "vel": copy_fn(arena._buf_vel, num_floats),
                        "init_hamiltonian": float(getattr(arena, "init_hamiltonian", 0.0)),
                        "init_peak_rho": float(getattr(arena, "init_peak_rho", 1.0)),
                        "step_count": int(getattr(arena, "step_count", 0)),
                        "sim_time": float(getattr(arena, "sim_time", 0.0)),
                    }
            elif arena_idx == 2:
                if bool(getattr(arena, "is_hibernated", False)) or getattr(arena, "_pos_buf", None) is None:
                    arena.wake_from_ram()
                cnt = int(arena.particle_count)
                copy_v4 = getattr(arena, "_copy_ssbo_to_ram_f32_vec4", None)
                if callable(copy_v4) and arena._pos_buf is not None and arena._vel_buf is not None:
                    self._replay_live_backup[2] = {
                        "kind": "arena3",
                        "particle_count": cnt,
                        "pos": copy_v4(arena._pos_buf, cnt),
                        "vel": copy_v4(arena._vel_buf, cnt),
                        "step_count": int(getattr(arena, "step_count", 0)),
                        "sim_time": float(getattr(arena, "sim_time", 0.0)),
                    }
        except Exception:
            pass

    def _exit_replay_to_live(self, arena_idx: int, unpause: bool = True) -> None:
        """Exits DDR5 keyframe replay mode and restores full-precision live fields & velocities."""
        arenas = (self.arena1, self.arena2, self.arena3, self.arena4)
        idx_clamped = int(np.clip(arena_idx, 0, 3))
        arena = arenas[idx_clamped]
        n_replay = int(getattr(arena, "replay_frames_count", 0) or 0)
        max_idx = max(n_replay - 1, 0)
        self._replay_idx[idx_clamped] = max_idx

        resume_fn = getattr(arena, "resume_live_from_replay", None)
        bk = self._replay_live_backup.pop(idx_clamped, None)
        if callable(resume_fn):
            resume_fn()
        elif bk is not None:
            try:
                if bk.get("kind") == "arena1" and int(getattr(arena, "grid_n", -1)) == int(bk["grid_n"]):
                    if bool(getattr(arena, "is_hibernated", False)) or getattr(arena, "_buf_a", None) is None:
                        arena.wake_from_ram()
                    arena._buf_a.write(np.ascontiguousarray(bk["phi"], dtype=np.float32))
                    arena._buf_vel.write(np.ascontiguousarray(bk["vel"], dtype=np.float32))
                    arena._cur_in_a = True
                    arena.init_hamiltonian = float(bk["init_hamiltonian"])
                    arena.init_peak_rho = float(bk["init_peak_rho"])
                    arena.step_count = int(bk["step_count"])
                    arena.sim_time = float(bk["sim_time"])
                    n = int(arena.grid_n)
                    arena._buf_a.bind_to_storage_buffer(0)
                    arena._vol_tex.bind_to_image(0, read=False, write=True)
                    arena._pack_prog["u_N"].value = n
                    arena._pack_prog.run((n + 7) // 8, (n + 7) // 8, (n + 3) // 4)
                    self.ctx.memory_barrier()
                    arena._sample_diagnostics(is_initial=False, reset_ham_baseline=False)
                elif bk.get("kind") == "arena3" and int(getattr(arena, "particle_count", -1)) == int(
                    bk["particle_count"]
                ):
                    if bool(getattr(arena, "is_hibernated", False)) or getattr(arena, "_pos_buf", None) is None:
                        arena.wake_from_ram()
                    arena._pos_buf.write(np.ascontiguousarray(bk["pos"], dtype=np.float32))
                    arena._vel_buf.write(np.ascontiguousarray(bk["vel"], dtype=np.float32))
                    arena.step_count = int(bk["step_count"])
                    arena.sim_time = float(bk["sim_time"])
                    eff_has, eff_scale, _ = arena._effective_halo_params(self.working_mod)
                    arena._solve_field_gpu(has_cdm=eff_has, cdm_scale=eff_scale, n_vcycles=2)
                    arena._sample_telemetry()
            except Exception:
                pass
        elif int(getattr(arena, "replay_cursor", -1)) >= 0 and n_replay > 0:
            if int(getattr(arena, "replay_cursor", -1)) != max_idx:
                load_fn = getattr(arena, "load_replay_frame", None)
                if callable(load_fn):
                    load_fn(max_idx)

        if hasattr(arena, "replay_cursor"):
            try:
                arena.replay_cursor = -1
            except Exception:
                pass
        if unpause:
            self.paused = False
        self._poll_memory_telemetry(force=True)
        self._reset_frame_pacing_clock()

    def _reset_active_arena(self) -> None:
        if self._active_arena == 0:
            self._clear_replay_backup(0)
            self.arena1.reset_field()
            self._poll_memory_telemetry(force=True)
            self._reset_frame_pacing_clock()
        elif self._active_arena == 1:
            self.cameras[1].reset(yaw=0.45, pitch=0.24, distance=18.5)
        elif self._active_arena == 2:
            self._clear_replay_backup(2)
            self.arena3.reset_galaxy(mod=self.working_mod, compiled=self.compiled)
            self._arena3_needs_field_sync = False
            self._poll_memory_telemetry(force=True)
            self._reset_frame_pacing_clock()
        elif self._active_arena == 3:
            self.arena4.tau_now = 2.65
            self.cameras[3].reset(yaw=0.55, pitch=0.20, distance=12.6)

    def save_window_screenshot(self, out_path: Path) -> Path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        GL.glReadBuffer(GL.GL_BACK)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        GL.glPixelStorei(GL.GL_PACK_ROW_LENGTH, 0)
        raw = GL.glReadPixels(0, 0, fb_w, fb_h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        img = Image.frombytes("RGBA", (fb_w, fb_h), raw).transpose(Image.FLIP_TOP_BOTTOM)
        img.save(out_p)
        self.status_banner = f"Saved screenshot -> {out_p.name}"
        self._reset_frame_pacing_clock()
        return out_p

    def save_default_screenshot(self) -> Path:
        """Saves a timestamped PNG screenshot into `self.captures_dir` (`captures/snap_...png`)."""
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        now_f = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_f))
        ms = int((now_f % 1.0) * 1000.0)
        out_p = self.captures_dir / f"snap_{stamp}_{ms:03d}.png"
        if self._in_imgui_frame:
            self._pending_screenshot_out: Optional[Path] = out_p
            self.status_banner = f"Saved screenshot -> {out_p.name}"
            return out_p
        return self.save_window_screenshot(out_p)

    @property
    def is_recording_video(self) -> bool:
        return bool(self._video_recorder.is_recording)

    def start_video_recording(
        self,
        output_path: Optional[Path] = None,
        fps: Optional[int] = None,
        resolution_mode: Optional[str] = None,
        record_every_frame: bool = False,
        prefer_nvenc: bool = True,
        out_path: Optional[Path] = None,
    ) -> Path:
        """Starts zero-stall triple-PBO 60/120 FPS `.mp4` video capture into `captures/`."""
        if self._video_recorder.is_recording:
            self.stop_video_recording()

        chosen_out = output_path if output_path is not None else out_path
        if chosen_out is None:
            self.captures_dir.mkdir(parents=True, exist_ok=True)
            now_f = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_f))
            ms = int((now_f % 1.0) * 1000.0)
            out_p = self.captures_dir / f"rec_{stamp}_{ms:03d}.mp4"
        else:
            out_p = Path(chosen_out)

        target_fps = int(fps if fps is not None else self.video_target_fps)
        target_fps = 120 if target_fps >= 100 else max(target_fps, 1)
        res_mode = str(resolution_mode if resolution_mode is not None else self.video_resolution_mode)

        # Ensure software FPS cap never throttles below requested recording frame rate
        if 0 < self.fps_cap < target_fps:
            self.set_fps_cap(target_fps)

        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        started_path = self._video_recorder.start(
            out_path=out_p,
            fb_width=max(fb_w, 2),
            fb_height=max(fb_h, 2),
            fps=target_fps,
            resolution_mode=res_mode,
            record_every_frame=record_every_frame,
            prefer_nvenc=prefer_nvenc,
        )
        self._last_rec_tick_wall = time.perf_counter()
        self.status_banner = (
            f"● REC {target_fps} FPS ({self._video_recorder.encoder_short_label}) -> {started_path.name}"
        )
        self._reset_frame_pacing_clock()
        return started_path

    def stop_video_recording(self) -> Optional[VideoRecordingResult]:
        """Stops active `.mp4` capture, drains in-flight GPU PBOs, and finalizes the MP4 container."""
        if not self._video_recorder.is_recording:
            return None
        res = self._video_recorder.stop()
        self.status_banner = (
            f"Saved {res.fps} FPS video ({res.frames_written} frames, {res.duration_s:.1f}s, "
            f"{res.encoder_name}) -> {res.output_path.name}"
        )
        self._reset_frame_pacing_clock()
        return res

    def toggle_video_recording(self) -> Optional[VideoRecordingResult]:
        """Toggles live `.mp4` capture on/off (Top Bar `Rec` button / `F9` / Context Menu)."""
        if self._video_recorder.is_recording:
            return self.stop_video_recording()
        self.start_video_recording()
        return None

    def open_captures_folder(self) -> Path:
        """Creates `self.captures_dir` and opens it in OS Explorer when running interactively."""
        target = self.captures_dir.resolve()
        target.mkdir(parents=True, exist_ok=True)
        if self._visible and sys.platform == "win32" and hasattr(os, "startfile"):
            try:
                os.startfile(str(target))  # type: ignore[attr-defined]
            except Exception as exc:
                self.status_banner = f"Captures folder ({target}): {exc}"
                return target
        self.status_banner = f"Captures folder ready: {target}"
        return target

    # =========================================================================
    # SIMULATION & ZERO-COPY 3D VIEWPORT RENDER
    # =========================================================================
    def _step_and_render_arena(self, dt: float, wall_t: float) -> None:
        if self._view_fbo is None or self.compiled is None:
            return
        # Ensure ImGui's scissor test never clips ModernGL offscreen FBO passes!
        GL.glDisable(GL.GL_SCISSOR_TEST)

        cam = self.cameras[self._active_arena]
        if self._active_arena == 0:
            if not self.paused:
                if self._is_arena_in_replay(self.arena1, 0):
                    self._exit_replay_to_live(0, unpause=False)
                self.arena1.step(self.working_mod, self.compiled)
            self.arena1.render(self._view_fbo, cam, wall_t)
        elif self._active_arena == 1:
            self.arena2.render(self._view_fbo, cam, self.compiled, wall_t)
        elif self._active_arena == 2:
            if not self.paused:
                if self._is_arena_in_replay(self.arena3, 2):
                    self._exit_replay_to_live(2, unpause=False)
                self.arena3.step(self.working_mod, self.compiled, dt=min(dt * 1.8, 0.036))
            self.arena3.render(self._view_fbo, cam, wall_t)
        else:
            self.arena4.render(self._view_fbo, cam, self.working_mod, wall_t)

    # =========================================================================
    # STUDIO UI WIDGETS & PANELS
    # =========================================================================
    def _handle_ui_scale_hotkeys(self) -> None:
        io = imgui.get_io()
        ctrl = bool(
            io.key_ctrl
            or imgui.is_key_down(imgui.Key.left_ctrl)
            or imgui.is_key_down(imgui.Key.right_ctrl)
        )
        if ctrl:
            zoom_in = imgui.is_key_pressed(imgui.Key.equal, False) or imgui.is_key_pressed(
                imgui.Key.keypad_add, False
            )
            zoom_out = imgui.is_key_pressed(imgui.Key.minus, False) or imgui.is_key_pressed(
                imgui.Key.keypad_subtract, False
            )
            zoom_reset = imgui.is_key_pressed(imgui.Key._0, False) or imgui.is_key_pressed(
                imgui.Key.keypad0, False
            )

            if zoom_reset:
                self.reset_ui_scale_auto()
            elif zoom_in:
                higher = [st for st in UI_SCALE_STOPS if st > self.ui_scale + 0.03]
                target = higher[0] if higher else UI_SCALE_STOPS[-1]
                self.set_ui_scale(target, keep_auto=False)
            elif zoom_out:
                lower = [st for st in UI_SCALE_STOPS if st < self.ui_scale - 0.03]
                target = lower[-1] if lower else UI_SCALE_STOPS[0]
                self.set_ui_scale(target, keep_auto=False)
            return

        # Studio simulation, camera & capture hotkeys when no InputText box has keyboard focus
        if not io.want_text_input:
            if imgui.is_key_pressed(imgui.Key.space, False):
                self.toggle_pause()
            elif imgui.is_key_pressed(imgui.Key.r, False):
                self._reset_active_arena()
            elif imgui.is_key_pressed(imgui.Key.home, False):
                self.reset_active_camera()
            elif imgui.is_key_pressed(imgui.Key.f12, False):
                self.save_default_screenshot()
            elif imgui.is_key_pressed(imgui.Key.f9, False):
                self.toggle_video_recording()

        # Track Right-Mouse-Button press/drag vs clean right-click across the entire Studio window
        mx, my = float(io.mouse_pos.x), float(io.mouse_pos.y)
        valid_pos = math.isfinite(mx) and math.isfinite(my) and (mx > -1e5) and (my > -1e5)
        if imgui.is_mouse_clicked(imgui.MouseButton_.right):
            self._rmb_down = True
            self._rmb_down_pos = (mx, my) if valid_pos else (0.0, 0.0)
            self._rmb_drag_dist = 0.0

        if self._rmb_down and valid_pos:
            d_origin = math.hypot(mx - self._rmb_down_pos[0], my - self._rmb_down_pos[1])
            d_delta = math.hypot(float(io.mouse_delta.x), float(io.mouse_delta.y))
            self._rmb_drag_dist = max(self._rmb_drag_dist, d_origin, d_delta)

        if imgui.is_mouse_released(imgui.MouseButton_.right):
            was_down = self._rmb_down
            drag_px = self._rmb_drag_dist
            self._rmb_down = False
            self._rmb_drag_dist = 0.0
            if was_down and (drag_px <= 5.0 * self.ui_scale) and (not io.want_text_input):
                self._pending_open_context_menu = True

    def _draw_ui_scale_combo(
        self, combo_id: str, compact_preview: bool = False, tiny_preview: bool = False
    ) -> None:
        s = self.ui_scale
        pct = int(round(s * 100.0))
        if tiny_preview:
            preview = f"{pct}%"
        elif compact_preview:
            preview = f"UI: {pct}%"
        elif self._ui_scale_auto:
            preview = f"Auto ({pct}%)"
        else:
            preview = f"{pct}%"

        auto_pct = int(
            round(
                (self._compute_auto_ui_scale(self._mon_scale, self._mon_width) if self._visible else 1.0)
                * 100.0
            )
        )
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(155.0 * s, 0.0), imgui.ImVec2(360.0 * s, 520.0 * s)
        )
        if imgui.begin_combo(combo_id, preview):
            is_auto = self._ui_scale_auto
            clicked_auto, _ = imgui.selectable(f"Auto ({auto_pct}%)", is_auto)
            if clicked_auto:
                self.reset_ui_scale_auto()
            if is_auto:
                imgui.set_item_default_focus()

            for stop in UI_SCALE_STOPS:
                stop_pct = int(round(stop * 100.0))
                is_sel = (not self._ui_scale_auto) and (abs(self.ui_scale - stop) < 0.03)
                clicked, _ = imgui.selectable(f"{stop_pct}%", is_sel)
                if clicked:
                    self.set_ui_scale(stop, keep_auto=False)
                if is_sel:
                    imgui.set_item_default_focus()
            imgui.end_combo()

    def _draw_render_scale_combo(
        self, combo_id: str, compact_preview: bool = False, tiny_preview: bool = False
    ) -> None:
        s = self.ui_scale
        cur_pct = int(round(self.render_scale * 100.0))
        if tiny_preview:
            preview = f"3D {cur_pct}%"
        elif compact_preview:
            preview = f"3D: {cur_pct}%"
        else:
            best_lbl = f"{cur_pct}%"
            for val, lbl in RENDER_SCALE_OPTIONS:
                if abs(self.render_scale - val) < 0.04:
                    best_lbl = lbl
                    break
            preview = best_lbl

        imgui.set_next_window_size_constraints(
            imgui.ImVec2(165.0 * s, 0.0), imgui.ImVec2(360.0 * s, 400.0 * s)
        )
        if imgui.begin_combo(combo_id, preview):
            for val, lbl in RENDER_SCALE_OPTIONS:
                is_sel = abs(self.render_scale - val) < 0.04
                clicked, _ = imgui.selectable(lbl, is_sel)
                if clicked:
                    self.set_render_scale(val)
                if is_sel:
                    imgui.set_item_default_focus()
            imgui.end_combo()

    def _draw_ddr5_replay_scrubber(self, arena: Any, arena_idx: int) -> None:
        n_replay = int(getattr(arena, "replay_frames_count", 0) or 0)
        if n_replay <= 0:
            return

        s = self.ui_scale
        max_idx = max(n_replay - 1, 0)
        arena_cursor = int(getattr(arena, "replay_cursor", -1))
        if arena_cursor >= 0:
            cur_idx = arena_cursor
        elif not self.paused and not bool(getattr(arena, "in_replay_mode", False)):
            cur_idx = max_idx
        else:
            cur_idx = self._replay_idx.get(arena_idx, max_idx)
        cur_idx = int(np.clip(cur_idx, 0, max_idx))
        self._replay_idx[arena_idx] = cur_idx

        imgui.text_colored(COL_FORECAST, f"DDR5 Time-Machine Replay ({n_replay} frames in RAM):")
        st = imgui.get_style()
        resume_lbl = "Resume Live"
        resume_w = imgui.calc_text_size(resume_lbl).x + st.frame_padding.x * 2.0
        slider_w = max(imgui.get_content_region_avail().x - resume_w - st.item_spacing.x, 80.0 * s)
        imgui.set_next_item_width(slider_w)
        ch_rep, new_idx = imgui.slider_int(
            f"##DDR5Replay_{arena_idx}",
            cur_idx,
            0,
            max_idx,
            f"Frame %d / {max_idx}",
        )
        if ch_rep:
            was_live = not self._is_arena_in_replay(arena, arena_idx)
            self._stash_live_before_replay(arena, arena_idx)
            self.paused = True
            self._replay_idx[arena_idx] = int(new_idx)
            load_fn = getattr(arena, "load_replay_frame", None)
            if callable(load_fn):
                load_fn(int(new_idx))
            self._poll_memory_telemetry(force=True)
            if was_live:
                self._reset_frame_pacing_clock()
        imgui.same_line()
        if imgui.button(f"{resume_lbl}##rep_{arena_idx}"):
            self._exit_replay_to_live(arena_idx, unpause=True)

    def _draw_top_bar(self, win_w: float, top_bar_h: float) -> None:
        s = self.ui_scale
        logical_w = win_w / max(s, 0.5)
        tight_bar = logical_w < 920.0
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse
        )
        imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(win_w, top_bar_h), imgui.Cond_.always)
        imgui.begin("##TopCommandRibbon", None, flags)

        if tight_bar:
            imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(4.0 * s, 5.0 * s))
            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(5.0 * s, 5.0 * s))

        frame_h = imgui.get_frame_height()
        row_y = max((top_bar_h - frame_h) * 0.5, 3.0 * s)
        imgui.set_cursor_pos_y(row_y)
        imgui.align_text_to_frame_padding()

        brand_lbl = "WEAVEFORGE" if logical_w >= 1360.0 else "WF"
        imgui.text_colored(COL_ACCENT, brand_lbl)
        imgui.same_line()

        if logical_w >= 1620.0:
            mod_combo_w = min(245.0 * s, max(win_w * 0.16, 140.0 * s))
        elif logical_w >= 1160.0:
            mod_combo_w = min(195.0 * s, max(win_w * 0.125, 115.0 * s))
        else:
            mod_combo_w = min(128.0 * s, max(win_w * 0.12, 78.0 * s))
        imgui.set_next_item_width(float(round(mod_combo_w)))
        mod_labels = [m.name for m in self.mods]
        changed, new_idx = imgui.combo("##ModCombo", self.active_mod_idx, mod_labels)
        if changed:
            self.select_preset_mod(new_idx)

        if logical_w >= 1980.0:
            arena_labels = ARENA_TITLES
        elif logical_w >= 1520.0:
            arena_labels = _ARENA_TITLES_MED
        elif logical_w >= 1160.0:
            arena_labels = _ARENA_TITLES_COMPACT
        else:
            arena_labels = _ARENA_TITLES_TINY

        imgui.same_line()
        for idx, title in enumerate(arena_labels):
            is_sel = idx == self._active_arena
            if is_sel:
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.22, 0.52, 0.82, 1.0))
            if imgui.button(f"{title}##top_arena_{idx}"):
                self._switch_active_arena(idx)
            if is_sel:
                imgui.pop_style_color()
            imgui.same_line()

        if imgui.button("Resume" if self.paused else "Pause"):
            self.toggle_pause()
        imgui.same_line()
        rst_lbl = "Rst##top_reset" if tight_bar else "Reset##top_reset"
        if imgui.button(rst_lbl):
            self._reset_active_arena()
        imgui.same_line()
        snap_lbl = (
            "Snap PNG##top_snap"
            if logical_w >= 1840.0
            else ("Png##top_snap" if tight_bar else "Snap##top_snap")
        )
        if imgui.button(snap_lbl):
            self.save_default_screenshot()

        # Built-in 60/120 FPS Zero-Stall MP4 Video Capture Button (glowing crimson while recording)
        imgui.same_line()
        is_rec = self.is_recording_video
        if is_rec:
            dur_sec = max(int(self._video_recorder.duration_s), 0)
            mm, ss = divmod(dur_sec, 60)
            if logical_w >= 1480.0:
                rec_lbl = f"Stop ● {mm:02d}:{ss:02d}##top_rec"
            elif tight_bar:
                rec_lbl = f"●{mm:02d}:{ss:02d}##top_rec"
            else:
                rec_lbl = f"Stop {mm:02d}:{ss:02d}##top_rec"
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.78, 0.14, 0.20, 0.98))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.92, 0.20, 0.28, 1.00))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.98, 0.28, 0.36, 1.00))
        else:
            tfps = int(self.video_target_fps)
            if logical_w >= 1800.0:
                rec_lbl = f"Rec {tfps}fps##top_rec"
            elif tight_bar:
                rec_lbl = "Rec##top_rec"
            else:
                rec_lbl = f"Rec {tfps}##top_rec"
        if imgui.button(rec_lbl):
            self.toggle_video_recording()
        if is_rec:
            imgui.pop_style_color(3)
        if imgui.is_item_hovered():
            if is_rec:
                imgui.set_tooltip(
                    f"Stop recording ({self._video_recorder.frames_captured} frames @ "
                    f"{self._video_recorder.fps} FPS, {self._video_recorder.encoder_name}) [F9]"
                )
            else:
                imgui.set_tooltip(
                    f"Start {self.video_target_fps} FPS .mp4 video capture to captures/ [F9] (Right-click for settings)"
                )

        st_top = imgui.get_style()
        tiny_combos = logical_w < 1020.0
        arrow_pad = imgui.get_frame_height() + st_top.frame_padding.x * 2.0 + 4.0 * s
        imgui.same_line()
        ui_ref = "250%" if tiny_combos else "UI: 250%"
        ui_w = float(round(imgui.calc_text_size(ui_ref).x + arrow_pad))
        imgui.set_next_item_width(ui_w)
        self._draw_ui_scale_combo("##TopUiScale", compact_preview=True, tiny_preview=tiny_combos)

        imgui.same_line()
        r3d_ref = "3D 100%" if tiny_combos else "3D: 100%"
        r3d_w = float(round(imgui.calc_text_size(r3d_ref).x + arrow_pad))
        imgui.set_next_item_width(r3d_w)
        self._draw_render_scale_combo("##TopRenderScale", compact_preview=True, tiny_preview=tiny_combos)

        # Measure actual screen-space right edge of the last Top Bar widget (never fooled by cursor line-wrap!)
        widgets_end_x = (
            (imgui.get_item_rect_max().x - imgui.get_window_pos().x) + st_top.item_spacing.x + 6.0 * s
        )
        self._last_topbar_widgets_end_x = float(widgets_end_x)

        # Right-aligned honest post-swap FPS / ms / VRAM / DDR5 telemetry pill with 4-tier non-overlapping fallback
        full_tel = (
            f"{self.fps_smooth:5.1f} FPS ({self.frame_ms_smooth:4.1f} ms) | "
            f"VRAM {self.vram_used_gb:.2f} GB | DDR5 {self.ddr5_used_mb:.0f} MB"
        )
        compact_tel = (
            f"{self.fps_smooth:4.0f} FPS ({self.frame_ms_smooth:.1f}ms) | "
            f"VRAM {self.vram_used_gb:.2f}G | DDR5 {self.ddr5_used_mb:.0f}M"
        )
        mini_tel = (
            f"{self.fps_smooth:.0f} FPS ({self.frame_ms_smooth:.1f}ms) | "
            f"{self.vram_used_gb:.1f}G/{self.ddr5_used_mb / 1024.0:.1f}G"
        )
        micro_tel = f"{self.fps_smooth:.0f} FPS"

        pad_r = 12.0 * s if tight_bar else 14.0 * s
        perf_col = COL_PASS if self.fps_smooth >= 50.0 else (COL_WARN if self.fps_smooth >= 22.0 else COL_FAIL)
        self._last_topbar_telemetry_x = float(win_w)
        self._last_topbar_telemetry_drawn = ""
        for cand_tel in (full_tel, compact_tel, mini_tel, micro_tel):
            tw = imgui.calc_text_size(cand_tel).x
            x_pos = win_w - tw - pad_r
            if x_pos >= widgets_end_x:
                self._last_topbar_telemetry_x = float(x_pos)
                self._last_topbar_telemetry_drawn = cand_tel
                imgui.same_line(x_pos)
                imgui.text_colored(perf_col, cand_tel)
                break

        if tight_bar:
            imgui.pop_style_var(2)

        imgui.end()

    def _draw_coupling_slider_float(
        self,
        label: str,
        param_name: str,
        vmin: float,
        vmax: float,
        scale: float = 1.0,
    ) -> Tuple[bool, bool]:
        """
        Draws a live coupling slider with an unmistakable inline `R` quick-reset micro-button
        (plus hover tooltip) whenever the slider differs from `self._pristine_mod.parameters`.
        Returns `(slider_changed, reset_clicked)`.
        """
        s = self.ui_scale
        is_dirty = self.is_parameter_modified(param_name)
        def_val = float(getattr(self._pristine_mod.parameters, param_name)) * scale
        reset_clicked = False

        cur_val = float(getattr(self.working_mod.parameters, param_name)) * scale
        imgui.set_next_item_width(-185.0 * s)
        ch, new_val = imgui.slider_float(label, cur_val, vmin, vmax)
        if is_dirty and imgui.is_item_hovered():
            imgui.set_tooltip(f"Modified (Mod default: {def_val:g} — click [R] to reset)")

        if is_dirty:
            imgui.same_line()
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.68, 0.42, 0.08, 0.95))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.84, 0.56, 0.14, 1.00))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.94, 0.66, 0.20, 1.00))
            if imgui.small_button(f"R##rst_{param_name}"):
                self.reset_single_parameter(param_name)
                reset_clicked = True
            rmin = imgui.get_item_rect_min()
            rmax = imgui.get_item_rect_max()
            self._last_slider_reset_rects[param_name] = (
                float(rmin.x),
                float(rmin.y),
                float(rmax.x),
                float(rmax.y),
            )
            imgui.pop_style_color(3)
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Reset to default ({def_val:g})")

        if ch and not reset_clicked:
            setattr(self.working_mod.parameters, param_name, float(new_val) / scale)
        return bool(ch and not reset_clicked), reset_clicked

    def _draw_left_panel(self, left_w: float, panel_h: float, top_bar_h: float) -> None:
        s = self.ui_scale
        flags = imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse
        imgui.set_next_window_pos(imgui.ImVec2(0.0, top_bar_h), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(left_w, panel_h), imgui.Cond_.always)
        imgui.begin("One Master Cost Sheet & Lab Controls", None, flags)

        req_tab = self._requested_left_tab
        self._requested_left_tab = None
        t0_flags = int(imgui.TabItemFlags_.set_selected) if req_tab == 0 else 0
        t1_flags = int(imgui.TabItemFlags_.set_selected) if req_tab == 1 else 0
        t2_flags = int(imgui.TabItemFlags_.set_selected) if req_tab == 2 else 0

        if imgui.begin_tab_bar("##LeftTabs", imgui.TabBarFlags_.fitting_policy_scroll):
            # -----------------------------------------------------------------
            # TAB 1: LIVE SYMBOLIC COMPILER & 5-GATE HEALTH AUDIT
            # -----------------------------------------------------------------
            if imgui.begin_tab_item("Master Action (SymPy->GPU)", None, t0_flags)[0]:
                imgui.push_text_wrap_pos(0.0)
                imgui.text_colored(COL_MUTED, "Edit equations live — SymPy derives PDEs & compiles GPU kernels:")
                imgui.pop_text_wrap_pos()

                if self.mono_font is not None:
                    imgui.push_font(self.mono_font, float(self.mono_font.legacy_size))

                imgui.text("3D Field Lagrangian L(T_kin, G_grad, rho):")
                _, self.working_mod.action.field_lagrangian = imgui.input_text_multiline(
                    "##LagrangianEdit",
                    self.working_mod.action.field_lagrangian,
                    imgui.ImVec2(-1.0, float(round(58.0 * s))),
                )

                imgui.text("Optical Metric n(U) & Lapse A(U) [U = GM/rc^2]:")
                sp_x = imgui.get_style().item_spacing.x
                half_w = max((imgui.get_content_region_avail().x - sp_x) * 0.5, 90.0 * s)
                imgui.set_next_item_width(half_w)
                _, self.working_mod.action.optical_metric_n = imgui.input_text(
                    "##OpticalEdit", self.working_mod.action.optical_metric_n
                )
                imgui.same_line()
                imgui.set_next_item_width(-1.0)
                _, self.working_mod.action.metric_lapse_A = imgui.input_text(
                    "##LapseEdit", self.working_mod.action.metric_lapse_A
                )

                imgui.text("Low-Accel Interpolation nu(y) [y = |g_N|/a0]:")
                imgui.set_next_item_width(-1.0)
                _, self.working_mod.action.interpolation_nu = imgui.input_text(
                    "##NuEdit", self.working_mod.action.interpolation_nu
                )

                imgui.text("Tensor GW Speed Ratio c_T / c:")
                imgui.set_next_item_width(-1.0)
                _, self.working_mod.action.gw_speed_ratio = imgui.input_text(
                    "##GwEdit", self.working_mod.action.gw_speed_ratio
                )

                if self.mono_font is not None:
                    imgui.pop_font()

                if imgui.button(
                    "Compile Master Action -> GLSL 430 + WGSL",
                    imgui.ImVec2(-1.0, float(round(32.0 * s))),
                ):
                    self._recompile_working_mod()
                    self._reset_frame_pacing_clock()

                n_eq_mod = self.count_modified_equations()
                if n_eq_mod > 0:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.68, 0.42, 0.08, 0.96))
                    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.84, 0.56, 0.14, 1.00))
                    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.94, 0.66, 0.20, 1.00))
                    if imgui.button(
                        f"Restore Default Mod Equations ({n_eq_mod} changed)##restore_mod_eqs",
                        imgui.ImVec2(-1.0, float(round(28.0 * s))),
                    ):
                        self.restore_default_mod_equations()
                    imgui.pop_style_color(3)

                avail_btn_w = max(imgui.get_content_region_avail().x, 120.0 * s)
                st_t1 = imgui.get_style()
                spacing_x = st_t1.item_spacing.x
                usable_btn_w = max(avail_btn_w - spacing_x, 110.0 * s)
                ins_lbl = "Inspect Generated GLSL / WGSL Kernels"
                sav_lbl = "Save Custom .toml"
                w_ins_need = imgui.calc_text_size(ins_lbl).x + st_t1.frame_padding.x * 2.0 + 8.0 * s
                w_sav_need = imgui.calc_text_size(sav_lbl).x + st_t1.frame_padding.x * 2.0 + 8.0 * s
                if (w_ins_need + w_sav_need) > usable_btn_w:
                    ins_lbl = "Inspect GLSL / WGSL"
                    w_ins_need = imgui.calc_text_size(ins_lbl).x + st_t1.frame_padding.x * 2.0 + 8.0 * s
                if (w_ins_need + w_sav_need) > usable_btn_w:
                    sav_lbl = "Save .toml"
                    w_sav_need = imgui.calc_text_size(sav_lbl).x + st_t1.frame_padding.x * 2.0 + 8.0 * s
                btn1_w = float(
                    round(
                        np.clip(
                            max(usable_btn_w * 0.62, w_ins_need),
                            60.0 * s,
                            max(usable_btn_w - w_sav_need, 60.0 * s),
                        )
                    )
                )
                btn2_w = max(avail_btn_w - btn1_w - spacing_x, 55.0 * s)
                if imgui.button(f"{ins_lbl}##inspect_kernels", imgui.ImVec2(btn1_w, 0.0)):
                    self.show_shader_modal = not self.show_shader_modal
                imgui.same_line()
                if imgui.button(f"{sav_lbl}##save_toml", imgui.ImVec2(btn2_w, 0.0)):
                    self.save_custom_mod()

                if self.compile_error:
                    imgui.separator()
                    imgui.push_text_wrap_pos(0.0)
                    imgui.text_colored(COL_FAIL, f"COMPILER REJECTED: {self.compile_error}")
                    imgui.pop_text_wrap_pos()

                if self.compiled is not None:
                    imgui.separator()
                    hdr_col, hdr_txt = health_audit_header(self.compiled)
                    imgui.push_text_wrap_pos(0.0)
                    imgui.text_colored(hdr_col, hdr_txt)
                    imgui.pop_text_wrap_pos()

                    for gate in self.compiled.health_gates:
                        gc = badge_color(gate.severity if not gate.passed else "PASS")
                        imgui.begin_child(
                            f"##gate_{gate.gate_id}",
                            imgui.ImVec2(-1.0, 0.0),
                            imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
                        )
                        imgui.push_text_wrap_pos(0.0)
                        imgui.text_colored(gc, f"[{'PASS' if gate.passed else gate.severity}] {gate.title}")
                        imgui.text_colored(COL_ACCENT, f"Measured: {gate.measured_str}  (Bound: {gate.bound_str})")
                        imgui.text_colored(COL_MUTED, gate.plain_english)
                        imgui.pop_text_wrap_pos()
                        imgui.end_child()

                imgui.end_tab_item()

            # -----------------------------------------------------------------
            # TAB 2: COUPLING SLIDERS, ACTIVE ARENA & 4K / DDR5 CONTROLS
            # -----------------------------------------------------------------
            if imgui.begin_tab_item("Arena & Couplings", None, t1_flags)[0]:
                self._last_slider_reset_rects.clear()
                n_mod_total = self.count_modified_parameters()
                n_coup_mod = self.count_modified_couplings()
                if n_mod_total > 0:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.68, 0.42, 0.08, 0.96))
                    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.84, 0.56, 0.14, 1.00))
                    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.94, 0.66, 0.20, 1.00))
                    clicked_top_rst = imgui.button(
                        f"Reset Modified Sliders to Defaults ({n_mod_total} changed)##tab2_top_reset",
                        imgui.ImVec2(-1.0, 0.0),
                    )
                    rmin_t = imgui.get_item_rect_min()
                    rmax_t = imgui.get_item_rect_max()
                    self._last_top_reset_btn_rect = (
                        float(rmin_t.x),
                        float(rmin_t.y),
                        float(rmax_t.x),
                        float(rmax_t.y),
                    )
                    imgui.pop_style_color(3)
                    if clicked_top_rst:
                        if self.count_modified_equations() > 0:
                            self.reset_all_sidebar_defaults()
                        else:
                            self.reset_sliders_to_mod_defaults()
                else:
                    clicked_top_rst = imgui.button(
                        "Reset All Sliders & Arena Settings to Default##tab2_top_reset",
                        imgui.ImVec2(-1.0, 0.0),
                    )
                    rmin_t = imgui.get_item_rect_min()
                    rmax_t = imgui.get_item_rect_max()
                    self._last_top_reset_btn_rect = (
                        float(rmin_t.x),
                        float(rmin_t.y),
                        float(rmax_t.x),
                        float(rmax_t.y),
                    )
                    if clicked_top_rst:
                        self.reset_all_sidebar_defaults()

                imgui.separator()
                p = self.working_mod.parameters
                imgui.text_colored(COL_ACCENT, "Active Arena Controls:")
                imgui.push_item_width(-165.0 * s)

                if self._active_arena == 0:
                    p_names = [lbl for _, lbl in KNOT_PRESETS]
                    ch, new_p = imgui.combo("Topology", self.arena1.preset_idx, p_names)
                    if ch:
                        self._clear_replay_backup(0)
                        self.arena1.reset_field(new_p)
                        self._poll_memory_telemetry(force=True)
                        self._reset_frame_pacing_clock()
                    res_vals = [128, 256, 512]
                    res_opts = ["128^3 (Fast)", "256^3 (60+ FPS HD)", "512^3 (134M Max Crunch)"]
                    if self.arena1.grid_n not in res_vals:
                        res_vals.append(int(self.arena1.grid_n))
                        res_opts.append(f"{self.arena1.grid_n}^3 (Custom)")
                    cur_r = res_vals.index(int(self.arena1.grid_n))
                    ch_r, nr = imgui.combo("Grid N^3", cur_r, res_opts)
                    if ch_r and 0 <= nr < len(res_vals):
                        try:
                            self._clear_replay_backup(0)
                            self.arena1.set_grid_resolution(res_vals[nr])
                        except Exception as exc:
                            self.status_banner = str(exc)
                        finally:
                            self._poll_memory_telemetry(force=True)
                            self._reset_frame_pacing_clock()
                    _, self.arena1.color_mode = imgui.combo(
                        "Palette",
                        int(np.clip(self.arena1.color_mode, 0, len(KNOT_COLOR_MODES) - 1)),
                        KNOT_COLOR_MODES,
                    )
                    _, self.arena1.density_gain = imgui.slider_float("Glow Gain", self.arena1.density_gain, 0.6, 6.0)
                    _, self.arena1.slice_x = imgui.slider_float("X Cutaway", self.arena1.slice_x, -0.8, 1.0)
                    _, self.arena1.damping = imgui.slider_float("Cooling", self.arena1.damping, 0.0, 0.8)
                    imgui.push_text_wrap_pos(0.0)
                    imgui.text_colored(
                        COL_PASS if self.arena1.retention_ratio > 0.70 else COL_WARN,
                        f"Soliton Retention = {self.arena1.retention_ratio:.3f}x | L2 = {self.arena1.l2_norm:.1f} | Step {self.arena1.step_count}",
                    )
                    imgui.text_colored(
                        COL_PASS if abs(self.arena1.hamiltonian_drift_pct) < 2.5 else COL_WARN,
                        f"3D Hamiltonian H = {self.arena1.hamiltonian_energy:.2f} ({self.arena1.hamiltonian_drift_pct:+.2f}%) | {self.arena1.allocated_vram_mb:.0f} MB VRAM",
                    )
                    imgui.pop_text_wrap_pos()
                    self._draw_ddr5_replay_scrubber(self.arena1, 0)

                elif self._active_arena == 1:
                    s_names = [lbl for _, lbl in FORGE_SUBMODES]
                    ch_s, ns = imgui.combo("Lab Mode", self.arena2.submode, s_names)
                    if ch_s:
                        self.arena2.submode = ns
                        if ns == 0:
                            self.cameras[1].reset(yaw=0.45, pitch=0.24, distance=18.5)
                        elif ns == 1:
                            self.cameras[1].reset(yaw=0.52, pitch=0.38, distance=8.2)
                        else:
                            self.cameras[1].reset(yaw=0.66, pitch=0.45, distance=11.5)
                    if self.arena2.submode == 0:
                        _, self.arena2.spin_a = imgui.slider_float("Kerr Spin a/M", self.arena2.spin_a, 0.0, 0.998)
                        _, self.arena2.show_bhex_gauge = imgui.checkbox("BHEX Shadow Gauge", self.arena2.show_bhex_gauge)
                        _, self.arena2.show_phase_slips = imgui.checkbox(
                            "3D Superfluid Phase-Slip Threads", self.arena2.show_phase_slips
                        )
                        if self.compiled is not None:
                            imgui.push_text_wrap_pos(0.0)
                            imgui.text_colored(
                                COL_FORECAST,
                                f"Fermat 2PN Shadow = {self.compiled.shadow_radius_rg:.3f} r_g ({self.compiled.shadow_vs_gr_pct:+.2f}% vs GR)",
                            )
                            imgui.pop_text_wrap_pos()
                    elif self.arena2.submode == 1:
                        _, self.arena2.mass_solar = imgui.slider_float("Mass (M_sun)", self.arena2.mass_solar, 0.5, 2.0)
                        imgui.push_text_wrap_pos(0.0)
                        imgui.text_colored(COL_PASS, "Wound Horizon Ratio R_spool / (2GM/c^2) = 1.000000")
                        imgui.pop_text_wrap_pos()
                    else:
                        _, self.arena2.probe_alt_km = imgui.slider_float(
                            "Orbit Alt (km)", self.arena2.probe_alt_km, 200.0, 36000.0
                        )
                        st_orb = imgui.get_style()
                        avail_orb = imgui.get_content_region_avail().x
                        b1_lbl, b2_lbl, b3_lbl = "ISS 408 km", "Zero-Cross 3,189.1 km", "GPS 20,184 km"
                        w1 = imgui.calc_text_size(b1_lbl).x + st_orb.frame_padding.x * 2.0
                        w2 = imgui.calc_text_size(b2_lbl).x + st_orb.frame_padding.x * 2.0
                        w3 = imgui.calc_text_size(b3_lbl).x + st_orb.frame_padding.x * 2.0
                        if imgui.button(b1_lbl):
                            self.arena2.probe_alt_km = 408.0
                        line1_has_b2 = (w1 + st_orb.item_spacing.x + w2) <= avail_orb
                        if line1_has_b2:
                            imgui.same_line()
                        if imgui.button(b2_lbl):
                            self.arena2.probe_alt_km = 3189.07
                        if line1_has_b2:
                            if (w1 + w2 + w3 + st_orb.item_spacing.x * 2.0) <= avail_orb:
                                imgui.same_line()
                        elif (w2 + st_orb.item_spacing.x + w3) <= avail_orb:
                            imgui.same_line()
                        if imgui.button(b3_lbl):
                            self.arena2.probe_alt_km = 20184.0

                        if self.compiled is not None:
                            tel = self.arena2.compute_orbital_clock_telemetry(self.compiled)
                            imgui.push_text_wrap_pos(0.0)
                            imgui.text_colored(
                                COL_PASS,
                                f"SR {tel['sr_us_day']:+.2f} + GR {tel['gr_us_day']:+.2f} = Net {tel['net_us_day']:+.2f} us/day",
                            )
                            imgui.text_colored(
                                COL_MUTED,
                                f"Nav Drift: {tel['drift_km_day']:.2f} km/day | Mercury: {tel['mercury_arcsec_cy']:.2f}\"/cy",
                            )
                            imgui.pop_text_wrap_pos()

                elif self._active_arena == 2:
                    g_names = [lbl for _, lbl in GALAXY_PRESETS]
                    ch_g, ng = imgui.combo("Scenario", self.arena3.preset_idx, g_names)
                    if ch_g:
                        self._clear_replay_backup(2)
                        if ng == 5:
                            gals = get_sparc_175_cached()
                            self.arena3.load_sparc_galaxy(
                                gals[self.selected_sparc_idx % len(gals)],
                                mod=self.working_mod,
                                compiled=self.compiled,
                            )
                        else:
                            self.arena3.reset_galaxy(ng, mod=self.working_mod, compiled=self.compiled)
                        self._arena3_needs_field_sync = False
                        self._poll_memory_telemetry(force=True)
                        self._reset_frame_pacing_clock()
                    if imgui.button(
                        f"Spawn Selected SPARC Galaxy ({self.sparc_names[self.selected_sparc_idx]}) in 3D",
                        imgui.ImVec2(-1.0, 0.0),
                    ):
                        self._clear_replay_backup(2)
                        gals = get_sparc_175_cached()
                        self.arena3.load_sparc_galaxy(
                            gals[self.selected_sparc_idx % len(gals)],
                            mod=self.working_mod,
                            compiled=self.compiled,
                        )
                        self._arena3_needs_field_sync = False
                        self._poll_memory_telemetry(force=True)
                        self._reset_frame_pacing_clock()
                    cnt_opts = ["250,000 Particles", "1,000,000 Particles (Default)", "2,000,000 Particles (RTX 5080)"]
                    cur_c = {250_000: 0, 1_000_000: 1, 2_000_000: 2}.get(self.arena3.particle_count, 1)
                    ch_c, nc = imgui.combo("N-Body Scale", cur_c, cnt_opts)
                    if ch_c:
                        self._clear_replay_backup(2)
                        self.arena3.set_particle_count(
                            [250_000, 1_000_000, 2_000_000][nc],
                            mod=self.working_mod,
                            compiled=self.compiled,
                        )
                        self._arena3_needs_field_sync = False
                        self._poll_memory_telemetry(force=True)
                        self._reset_frame_pacing_clock()
                    _, self.arena3.show_lensing = imgui.checkbox(
                        "3D Volumetric Lensing Contours", self.arena3.show_lensing
                    )
                    imgui.push_text_wrap_pos(0.0)
                    imgui.text_colored(
                        COL_PASS,
                        f"Outer V_rot = {self.arena3.v_outer_kms:.1f} km/s | Splash Ratio = {self.arena3.transverse_splash_ratio:.2f}",
                    )
                    imgui.text_colored(
                        COL_FORECAST,
                        f"Weak-Lensing vs Gas Offset = {self.arena3.lensing_gas_offset_kpc:.1f} kpc",
                    )
                    imgui.pop_text_wrap_pos()
                    self._draw_ddr5_replay_scrubber(self.arena3, 2)

                else:
                    _, self.arena4.waist_r0 = imgui.slider_float("Waist Radius r0", self.arena4.waist_r0, 0.35, 1.50)
                    _, self.arena4.tau_now = imgui.slider_float("Observer Slice |tau|", self.arena4.tau_now, 0.0, 3.65)
                    _, self.arena4.auto_scrub = imgui.checkbox("Auto-Scrub Cosmic Time", self.arena4.auto_scrub)
                    _, self.arena4.show_creases = imgui.checkbox(
                        "Landauer Memory Creases (dQ >= k_B T ln 2)", self.arena4.show_creases
                    )
                    imgui.push_text_wrap_pos(0.0)
                    imgui.text_colored(
                        COL_PASS,
                        f"Waist K_max = {self.arena4.waist_kretschmann():.1f} < inf | ||J|Psi>|| = 0.000000",
                    )
                    imgui.text_colored(
                        COL_MUTED, "Symmetric Slices: Observer Channel A (+tau) & Observer Channel B (-tau)"
                    )
                    imgui.pop_text_wrap_pos()

                imgui.pop_item_width()

                imgui.separator()
                n_coup_hdr = self.count_modified_couplings()
                imgui.text_colored(COL_ACCENT, "Theory Coupling Parameters (Live):")
                st_hdr = imgui.get_style()
                hdr_rst_txt = f"Reset ({n_coup_hdr})" if n_coup_hdr > 0 else "Reset"
                hdr_btn_w = imgui.calc_text_size(hdr_rst_txt).x + st_hdr.frame_padding.x * 2.0 + 2.0 * s
                hdr_x = imgui.get_window_width() - hdr_btn_w - st_hdr.window_padding.x
                cur_label_end_x = (imgui.get_item_rect_max().x - imgui.get_window_pos().x) + 6.0 * s
                if hdr_x >= cur_label_end_x:
                    imgui.same_line(hdr_x)
                if n_coup_hdr > 0:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.68, 0.42, 0.08, 0.96))
                    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.84, 0.56, 0.14, 1.00))
                    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.94, 0.66, 0.20, 1.00))
                if imgui.small_button(f"{hdr_rst_txt}##hdr_coup_reset"):
                    self.reset_sliders_to_mod_defaults()
                if n_coup_hdr > 0:
                    imgui.pop_style_color(3)

                p = self.working_mod.parameters
                ch1, _ = self._draw_coupling_slider_float("Wave Speed c_w", "c_w", 0.2, 1.4)
                ch_m, _ = self._draw_coupling_slider_float("Mass Gap m_eff", "m_eff", 0.0, 5.0)
                ch2, _ = self._draw_coupling_slider_float("Quartic Binding g4", "g4", -2.0, 18.0)
                ch3, _ = self._draw_coupling_slider_float("Planck Core g6", "g6", -1.0, 12.0)
                ch4, _ = self._draw_coupling_slider_float("a0 (x10^-10 m/s^2)", "a0_si", 0.40, 2.50, scale=1.0e10)

                ch5, new_has_cdm = imgui.checkbox("Include NFW Cold Dark Matter Halo", bool(p.has_cdm_halo))
                if ch5:
                    p.has_cdm_halo = bool(new_has_cdm)
                    if p.has_cdm_halo and p.cdm_halo_scale <= 1.0e-6:
                        p.cdm_halo_scale = 1.0
                if self.is_parameter_modified("has_cdm_halo"):
                    imgui.same_line()
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.68, 0.42, 0.08, 0.95))
                    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.84, 0.56, 0.14, 1.00))
                    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.94, 0.66, 0.20, 1.00))
                    if imgui.small_button("R##rst_has_cdm_halo"):
                        self.reset_single_parameter("has_cdm_halo")
                        ch5 = False
                    rmin_c = imgui.get_item_rect_min()
                    rmax_c = imgui.get_item_rect_max()
                    self._last_slider_reset_rects["has_cdm_halo"] = (
                        float(rmin_c.x),
                        float(rmin_c.y),
                        float(rmax_c.x),
                        float(rmax_c.y),
                    )
                    imgui.pop_style_color(3)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(
                            f"Reset to default ({self._pristine_mod.parameters.has_cdm_halo})"
                        )

                if p.has_cdm_halo:
                    ch6, _ = self._draw_coupling_slider_float("CDM Halo Scale", "cdm_halo_scale", 0.0, 1.8)
                else:
                    ch6 = False
                ch7, _ = self._draw_coupling_slider_float(
                    "Pinned Pleat Coupling", "pinned_pleat_coupling", 0.0, 1.5
                )
                ch8, _ = self._draw_coupling_slider_float(
                    "Waist Clamping l_cut", "waist_clamping_ell", 0.0, 8.0
                )
                ch9, _ = self._draw_coupling_slider_float("Dark Energy w0", "w0_de", -1.6, -0.4)
                ch10, _ = self._draw_coupling_slider_float("Dark Energy wa", "wa_de", -1.5, 1.0)

                changed_any = ch1 or ch_m or ch2 or ch3 or ch4 or ch5 or ch6 or ch7 or ch8 or ch9 or ch10
                if changed_any:
                    if ch1 or ch_m or ch2 or ch3 or ch4:
                        self._recompile_working_mod()
                    else:
                        self._refresh_crucible_fast()

                # 4K Scale, Decoupled 3D Render Scale, FPS Pacing & DDR5 RAM Hibernation Controls
                imgui.separator()
                imgui.text_colored(COL_ACCENT, "Display, 4K Scale & DDR5 RAM:")
                imgui.push_item_width(-165.0 * s)
                self._draw_ui_scale_combo("UI Scale", compact_preview=False)
                self._draw_render_scale_combo("3D Scale", compact_preview=False)

                fps_vals = [val for val, _ in FPS_CAP_OPTIONS]
                fps_lbls = [lbl for _, lbl in FPS_CAP_OPTIONS]
                cur_fps_idx = fps_vals.index(self.fps_cap) if self.fps_cap in fps_vals else len(fps_vals) - 1
                ch_fps, new_fps_idx = imgui.combo("FPS Cap", cur_fps_idx, fps_lbls)
                if ch_fps and 0 <= new_fps_idx < len(fps_vals):
                    self.set_fps_cap(fps_vals[new_fps_idx], explicit_user_choice=True)
                imgui.pop_item_width()

                ch_hib, new_hib = imgui.checkbox(
                    "Hibernate Inactive Arenas in 32GB DDR5 RAM", self.ddr5_hibernate_enabled
                )
                if ch_hib:
                    self._set_ddr5_hibernate_enabled(new_hib)

                imgui.push_text_wrap_pos(0.0)
                imgui.text_colored(
                    COL_MUTED,
                    f"Real Post-Swap: {self.fps_smooth:.1f} FPS ({self.frame_ms_smooth:.1f} ms) | "
                    f"VRAM: {self.vram_used_gb:.2f} GB | DDR5: {self.ddr5_used_mb:.0f} MB (Ctrl +/-/0 Zoom)",
                )
                imgui.pop_text_wrap_pos()

                imgui.end_tab_item()

            # -----------------------------------------------------------------
            # TAB 3: RIG 2 CROSS-PLATFORM WGPU LAN / SINGLE-PC WORKER
            # -----------------------------------------------------------------
            if imgui.begin_tab_item("Rig 2 LAN Worker", None, t2_flags)[0]:
                imgui.text_colored(COL_ACCENT, "Single-PC Zen 5 / Dual-PC Vulkan Compute")
                imgui.push_text_wrap_pos(0.0)
                imgui.text_colored(
                    COL_MUTED,
                    "Single-PC Ready: Rig 2 being off causes zero slowdown — runs locally on Zen 5 + DDR5/wgpu or over LAN.",
                )
                imgui.pop_text_wrap_pos()

                mode_keys = [k for k, _ in COMPUTE_MODE_OPTIONS]
                mode_lbls = [lbl for _, lbl in COMPUTE_MODE_OPTIONS]
                cur_m = mode_keys.index(self.compute_execution_mode) if self.compute_execution_mode in mode_keys else 0
                imgui.set_next_item_width(-1.0)
                ch_m_mode, new_m = imgui.combo("##ComputeModeCombo", cur_m, mode_lbls)
                if ch_m_mode and 0 <= new_m < len(mode_keys):
                    self.compute_execution_mode = mode_keys[new_m]
                    setattr(self.lan_dispatcher, "execution_mode", self.compute_execution_mode)

                imgui.set_next_item_width(-1.0)
                _, self.lan_dispatcher.worker_url = imgui.input_text(
                    "##WorkerUrl", self.lan_dispatcher.worker_url
                )

                btn_lbl = (
                    "Crunching Async Job..."
                    if self.lan_dispatcher.is_running
                    else "Dispatch Full SPARC-175 + 3D Knot + Gaia DR4 Sweep"
                )
                if (
                    imgui.button(btn_lbl, imgui.ImVec2(-1.0, float(round(36.0 * s))))
                    and not self.lan_dispatcher.is_running
                ):
                    self._dispatch_compute_sweep()

                imgui.separator()
                imgui.push_text_wrap_pos(0.0)
                imgui.text_colored(COL_FORECAST, f"Status: {self.lan_dispatcher.last_status}")
                imgui.pop_text_wrap_pos()

                res = self.lan_dispatcher.last_result
                if res is not None:
                    imgui.separator()
                    imgui.text_colored(COL_PASS, f"GPU Adapter: {res.get('gpu_adapter', 'Vulkan')}")
                    sp_r = res.get("sparc_175", {})
                    kn_r = res.get("knot_3d", {})
                    ga_r = res.get("gaia_dr4_mock", {})
                    sp_mode_lbl = "Optimized Fit" if sp_r.get("optimized_ml", True) else "Fiducial M/L"
                    n_gaia_clean = int(ga_r.get("n_clean", 2400))
                    imgui.bullet_text(
                        f"SPARC-175 {sp_mode_lbl}: median chi²={sp_r.get('median_chi2_red', 0.0):.2f}, "
                        f"RAR={sp_r.get('rar_rms_dex', 0.0):.3f} dex"
                    )
                    imgui.bullet_text(
                        f"3D Vulkan Knot Run: retention={kn_r.get('retention_ratio', 0.0):.3f}x "
                        f"({kn_r.get('elapsed_ms', 0.0):.1f} ms)"
                    )
                    imgui.bullet_text(
                        f"Gaia DR4 Mock ({n_gaia_clean:,} pairs): Halo-Disk = {ga_r.get('halo_minus_disk_pct', 0.0):+.1f}%"
                    )

                    sweep = res.get("knot_phase_sweep")
                    if isinstance(sweep, dict):
                        g4_vals = sweep.get("g4_values") or []
                        g6_vals = sweep.get("g6_values") or []
                        grid_raw = sweep.get("retention_grid") or []
                        grid_arr = np.asarray(grid_raw, dtype=np.float64)
                        if (
                            len(g4_vals) > 0
                            and len(g6_vals) > 0
                            and grid_arr.ndim == 2
                            and grid_arr.shape == (len(g6_vals), len(g4_vals))
                        ):
                            imgui.separator()
                            imgui.text_colored(
                                COL_ACCENT,
                                f"2D 3D-Knot Stability Phase Map ({len(g6_vals)}x{len(g4_vals)} GPU Sweep, "
                                f"{float(sweep.get('elapsed_ms', 0.0)):.0f} ms):",
                            )
                            imgui.push_text_wrap_pos(0.0)
                            imgui.text_colored(
                                COL_MUTED,
                                "Retention (0.55x-3.50x = Stable Q-Ball; click cell to apply):",
                            )
                            imgui.pop_text_wrap_pos()

                            n_cols = len(g4_vals)
                            n_rows = len(g6_vals)
                            x_ticks = [float(c) + 0.5 for c in range(n_cols)]
                            y_ticks = [float(r) + 0.5 for r in range(n_rows)]
                            x_labels = [f"{float(v):g}" for v in g4_vals]
                            y_labels = [f"{float(v):g}" for v in g6_vals]

                            # Full 200*s height on 4K/tall windows, gracefully compact on short windows
                            heat_h = (
                                float(round(200.0 * s))
                                if panel_h >= 820.0 * s
                                else float(round(min(200.0 * s, max(150.0 * s, panel_h * 0.22))))
                            )
                            heat_vals = np.ascontiguousarray(grid_arr[::-1, :], dtype=np.float64)
                            implot.push_colormap(implot.Colormap_.viridis)
                            if implot.begin_plot("Knot Retention (g4 vs g6)", imgui.ImVec2(-1.0, heat_h)):
                                implot.setup_axes("Quartic Binding g4", "Planck Core g6")
                                implot.setup_axes_limits(0.0, float(n_cols), 0.0, float(n_rows), implot.Cond_.always)
                                implot.setup_axis_ticks(implot.ImAxis_.x1, x_ticks, x_labels)
                                implot.setup_axis_ticks(implot.ImAxis_.y1, y_ticks, y_labels)
                                implot.plot_heatmap(
                                    "##KnotRetentionHeat",
                                    heat_vals,
                                    0.0,
                                    3.5,
                                    "%.2fx",
                                    implot.Point(0.0, 0.0),
                                    implot.Point(float(n_cols), float(n_rows)),
                                )
                                implot.end_plot()
                            implot.pop_colormap()

                            tbl_flags = int(
                                imgui.TableFlags_.borders
                                | imgui.TableFlags_.row_bg
                                | imgui.TableFlags_.sizing_stretch_same
                            )
                            if imgui.begin_table("##KnotSweepMatrix", len(g4_vals) + 1, tbl_flags):
                                imgui.table_next_row()
                                imgui.table_set_column_index(0)
                                imgui.text_colored(COL_MUTED, "g6 \\ g4")
                                for c_i, g4_v in enumerate(g4_vals):
                                    imgui.table_set_column_index(c_i + 1)
                                    imgui.text_colored(COL_ACCENT, f"{float(g4_v):.1f}")

                                clicked_pair: Optional[Tuple[float, float]] = None
                                for r_i in range(len(g6_vals) - 1, -1, -1):
                                    g6_v = float(g6_vals[r_i])
                                    imgui.table_next_row()
                                    imgui.table_set_column_index(0)
                                    imgui.text_colored(COL_ACCENT, f"{g6_v:.1f}")
                                    for c_i, g4_raw in enumerate(g4_vals):
                                        g4_v = float(g4_raw)
                                        ret_v = float(grid_arr[r_i, c_i])
                                        imgui.table_set_column_index(c_i + 1)
                                        if 0.55 <= ret_v <= 3.50:
                                            cell_col = COL_PASS
                                        elif ret_v > 3.50:
                                            cell_col = COL_FAIL
                                        else:
                                            cell_col = COL_WARN
                                        imgui.push_style_color(imgui.Col_.text, cell_col)
                                        if imgui.small_button(f"{ret_v:.2f}x##cell_{r_i}_{c_i}"):
                                            clicked_pair = (g4_v, g6_v)
                                        imgui.pop_style_color()
                                imgui.end_table()

                                if clicked_pair is not None:
                                    self.working_mod.parameters.g4 = clicked_pair[0]
                                    self.working_mod.parameters.g6 = clicked_pair[1]
                                    self._recompile_working_mod()
                                    self.status_banner = (
                                        f"Applied sweep point g4={clicked_pair[0]:.2f}, g6={clicked_pair[1]:.2f}"
                                    )
                                    self._reset_frame_pacing_clock()

                imgui.end_tab_item()

            imgui.end_tab_bar()

        if self.status_banner:
            imgui.separator()
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(COL_PASS, self.status_banner)
            imgui.pop_text_wrap_pos()

        imgui.end()

    def _dispatch_compute_sweep(self) -> None:
        """Dispatches async SPARC-175 + 3D Knot + Gaia DR4 sweep while strictly honoring Compute Mode."""
        if self.lan_dispatcher.is_running:
            return
        mode = self.compute_execution_mode if self.compute_execution_mode in ("auto", "local", "lan") else "auto"
        raw_url = str(self.lan_dispatcher.worker_url or "").strip().rstrip("/")
        if raw_url and not raw_url.startswith(("http://", "https://")):
            raw_url = f"http://{raw_url}"
        self.lan_dispatcher.worker_url = raw_url
        setattr(self.lan_dispatcher, "execution_mode", mode)

        if mode == "lan":
            if not raw_url:
                self.lan_dispatcher.last_status = (
                    "LAN Rig 2 Offline: enter a valid http://host:port URL (or switch to Auto / Local)"
                )
                return
            # If the backend dispatcher natively honors execution_mode == 'lan', delegate directly
            init_params = inspect.signature(self.lan_dispatcher.__init__).parameters
            if "execution_mode" in init_params:
                self.lan_dispatcher.dispatch_async(self.working_mod, use_lan_if_online=True)
                return

            # Fallback strict LAN-only probe/post so 'LAN Rig 2 Only' NEVER runs locally on Rig 1
            import httpx
            from weaveforge.compute.lan_worker import WorkerSweepRequest

            self.lan_dispatcher.is_running = True
            self.lan_dispatcher.last_status = f"Probing LAN Rig 2 ({raw_url})..."
            req = WorkerSweepRequest(toml_source=self.working_mod.to_toml_string())

            def _lan_only_worker() -> None:
                try:
                    with httpx.Client(timeout=1.5) as probe:
                        h = probe.get(f"{raw_url}/health")
                        h.raise_for_status()
                        gpu_lbl = h.json().get("gpu_adapter", "Remote LAN GPU")
                    self.lan_dispatcher.last_status = f"Crunching on Remote LAN Node ({gpu_lbl})..."
                    with httpx.Client(timeout=120.0) as client:
                        r = client.post(f"{raw_url}/crunch", json=req.model_dump())
                        r.raise_for_status()
                        self.lan_dispatcher.last_result = r.json()
                        self.lan_dispatcher.last_status = (
                            f"LAN Complete ({self.lan_dispatcher.last_result['total_elapsed_ms']:.0f} ms on {gpu_lbl})"
                        )
                except Exception as exc:
                    self.lan_dispatcher.last_status = (
                        f"LAN Node Offline ({raw_url}): {exc} -- switch mode to Auto or Single-PC Local"
                    )
                finally:
                    self.lan_dispatcher.is_running = False

            threading.Thread(target=_lan_only_worker, daemon=True).start()
            return

        use_lan = mode != "local"
        try:
            sig_d = inspect.signature(self.lan_dispatcher.dispatch_async)
            if "execution_mode" in sig_d.parameters:
                self.lan_dispatcher.dispatch_async(
                    self.working_mod,
                    use_lan_if_online=use_lan,
                    execution_mode=mode,  # type: ignore[call-arg]
                )
            else:
                self.lan_dispatcher.dispatch_async(self.working_mod, use_lan_if_online=use_lan)
        except TypeError:
            self.lan_dispatcher.dispatch_async(self.working_mod, use_lan_if_online=use_lan)

    def _draw_studio_context_menu(self) -> None:
        """
        Comprehensive High-DPI Right-Click Context Menu (`##StudioContextMenu`).
        Opens on any clean right-click (`drag_dist <= 5 * ui_scale`, when no `InputText` is active)
        without interfering with Right-Mouse-Drag 3D camera panning.
        """
        s = self.ui_scale
        if self._pending_open_context_menu:
            imgui.open_popup("##StudioContextMenu")
            self._pending_open_context_menu = False

        self.is_context_menu_open = bool(imgui.is_popup_open("##StudioContextMenu"))
        if not self.is_context_menu_open:
            self._last_context_menu_rect = None

        imgui.set_next_window_size_constraints(
            imgui.ImVec2(310.0 * s, 0.0),
            imgui.ImVec2(560.0 * s, 860.0 * s),
        )
        if not imgui.begin_popup("##StudioContextMenu"):
            return

        self.is_context_menu_open = True
        wpos = imgui.get_window_pos()
        wsz = imgui.get_window_size()
        self._last_context_menu_rect = (
            float(wpos.x),
            float(wpos.y),
            float(wpos.x + wsz.x),
            float(wpos.y + wsz.y),
        )
        if not hasattr(self, "_context_menu_item_rects"):
            self._context_menu_item_rects: Dict[str, Tuple[float, float, float, float]] = {}
        self._context_menu_item_rects.clear()

        def _rec_rect(key: str) -> None:
            mn = imgui.get_item_rect_min()
            mx = imgui.get_item_rect_max()
            self._context_menu_item_rects[key] = (float(mn.x), float(mn.y), float(mx.x), float(mx.y))

        imgui.text_colored(COL_ACCENT, "WeaveForge Studio — Quick Actions")
        imgui.separator()

        # 1. Simulation Controls
        pause_lbl = "Resume Simulation" if self.paused else "Pause Simulation"
        if imgui.menu_item_simple(pause_lbl, "Space"):
            self.toggle_pause()
        _rec_rect("pause_resume")

        if imgui.menu_item_simple("Step 1 Frame Forward"):
            self.step_single_simulation_frame()
        _rec_rect("step_1")

        if imgui.menu_item_simple("Reset Active Arena Simulation", "R"):
            self._reset_active_arena()
        _rec_rect("reset_arena")

        cur_arena_obj = (self.arena1, self.arena2, self.arena3, self.arena4)[self._active_arena]
        if self._is_arena_in_replay(cur_arena_obj, self._active_arena):
            if imgui.menu_item_simple("Exit DDR5 Replay -> Resume Live"):
                self._exit_replay_to_live(self._active_arena, unpause=True)
            _rec_rect("exit_replay")

        imgui.separator()

        # 2. Capture & Video Recording
        if imgui.menu_item_simple("Save PNG Screenshot", "F12"):
            self.save_default_screenshot()
        _rec_rect("save_png")

        if self.is_recording_video:
            dur_s = max(int(self._video_recorder.duration_s), 0)
            mm, ss = divmod(dur_s, 60)
            rec_menu_lbl = (
                f"Stop Video Capture (● REC {mm:02d}:{ss:02d} {self._video_recorder.encoder_short_label})"
            )
        else:
            rec_menu_lbl = f"Start {self.video_target_fps} FPS Video Capture (.mp4)"
        if imgui.menu_item_simple(rec_menu_lbl, "F9"):
            self.toggle_video_recording()
        _rec_rect("toggle_rec")

        if imgui.begin_menu("Video Recorder Settings"):
            imgui.text_colored(COL_MUTED, "Target Frame Rate:")
            for fps_opt, fps_lbl in VIDEO_FPS_OPTIONS:
                if imgui.menu_item_simple(fps_lbl, None, self.video_target_fps == fps_opt):
                    self.video_target_fps = int(fps_opt)
            imgui.separator()
            imgui.text_colored(COL_MUTED, "Capture Resolution:")
            for res_key, res_lbl in VIDEO_RESOLUTION_OPTIONS:
                if imgui.menu_item_simple(res_lbl, None, self.video_resolution_mode == res_key):
                    self.video_resolution_mode = str(res_key)
            imgui.end_menu()

        if imgui.menu_item_simple("Open Captures Folder"):
            self.open_captures_folder()
        _rec_rect("open_captures")

        imgui.separator()

        # 3. Physics Mod & Left-Sidebar Defaults
        n_coup = self.count_modified_couplings()
        n_eq = self.count_modified_equations()
        rst_coup_lbl = (
            f"Reset Coupling Sliders to Mod Defaults ({n_coup} changed)"
            if n_coup > 0
            else "Reset Coupling Sliders to Mod Defaults"
        )
        if imgui.menu_item_simple(rst_coup_lbl):
            self.reset_sliders_to_mod_defaults()
        _rec_rect("reset_sliders")

        if imgui.menu_item_simple("Reset All Sidebar & Arena Settings"):
            self.reset_all_sidebar_defaults()
        _rec_rect("reset_all_sidebar")

        if n_eq > 0:
            if imgui.menu_item_simple(f"Restore Default Mod Equations ({n_eq} changed)"):
                self.restore_default_mod_equations()
            _rec_rect("restore_equations")

        if imgui.begin_menu("Switch Physics Mod"):
            for m_i, mod_item in enumerate(self.mods):
                if imgui.menu_item_simple(mod_item.name, None, m_i == self.active_mod_idx):
                    self.select_preset_mod(m_i)
            imgui.end_menu()

        if imgui.menu_item_simple("Recompile Master Action (SymPy -> GPU)"):
            self._recompile_working_mod()
            self._reset_frame_pacing_clock()
        _rec_rect("recompile_action")

        if imgui.menu_item_simple(
            "Toggle Generated GLSL / WGSL Shader Inspector", None, self.show_shader_modal
        ):
            self.show_shader_modal = not self.show_shader_modal
        _rec_rect("toggle_shader_inspector")

        imgui.separator()

        # 4. Arena, Presets & Camera
        if imgui.begin_menu("Switch Arena"):
            ctx_arena_names = (
                "[1] 3D Knot Tank",
                "[2] Black Hole & 2PN",
                "[3] 1M-Star Collider",
                "[4] Cosmic Crucible",
            )
            for ar_i, ar_lbl in enumerate(ctx_arena_names):
                if imgui.menu_item_simple(ar_lbl, None, ar_i == self._active_arena):
                    self._switch_active_arena(ar_i)
            imgui.end_menu()

        if imgui.begin_menu("Active Arena Presets"):
            if self._active_arena == 0:
                imgui.text_colored(COL_MUTED, "Knot Topologies:")
                for kp_i, (_, kp_lbl) in enumerate(KNOT_PRESETS):
                    if imgui.menu_item_simple(kp_lbl, None, self.arena1.preset_idx == kp_i):
                        self._clear_replay_backup(0)
                        self.arena1.reset_field(kp_i)
                        self._poll_memory_telemetry(force=True)
                        self._reset_frame_pacing_clock()
                imgui.separator()
                imgui.text_colored(COL_MUTED, "Grid Resolution:")
                for gn, glbl in (
                    (128, "128^3 (Fast)"),
                    (256, "256^3 (60+ FPS HD)"),
                    (512, "512^3 (134M Max Crunch)"),
                ):
                    if imgui.menu_item_simple(glbl, None, int(self.arena1.grid_n) == gn):
                        try:
                            self._clear_replay_backup(0)
                            self.arena1.set_grid_resolution(gn)
                        except Exception as exc:
                            self.status_banner = str(exc)
                        finally:
                            self._poll_memory_telemetry(force=True)
                            self._reset_frame_pacing_clock()
                imgui.separator()
                imgui.text_colored(COL_MUTED, "Color Palette:")
                for pal_i, pal_lbl in enumerate(KNOT_COLOR_MODES):
                    if imgui.menu_item_simple(pal_lbl, None, int(self.arena1.color_mode) == pal_i):
                        self.arena1.color_mode = pal_i
            elif self._active_arena == 1:
                imgui.text_colored(COL_MUTED, "Strong-Gravity Lab Mode:")
                for sm_i, (_, sm_lbl) in enumerate(FORGE_SUBMODES):
                    if imgui.menu_item_simple(sm_lbl, None, int(self.arena2.submode) == sm_i):
                        self.arena2.submode = sm_i
                        self.reset_active_camera()
                imgui.separator()
                imgui.text_colored(COL_MUTED, "Orbital Clock Altitude Presets:")
                for alt_km, alt_lbl in (
                    (408.0, "ISS Orbit (408 km)"),
                    (3189.07, "Zero-Crossing Ring (3,189.1 km)"),
                    (20184.0, "GPS Constellation (20,184 km)"),
                ):
                    is_alt = abs(float(self.arena2.probe_alt_km) - alt_km) < 2.0
                    if imgui.menu_item_simple(alt_lbl, None, is_alt):
                        self.arena2.submode = 2
                        self.arena2.probe_alt_km = alt_km
                        self.reset_active_camera()
                imgui.separator()
                if imgui.menu_item_simple("BHEX Shadow Gauge", None, bool(self.arena2.show_bhex_gauge)):
                    self.arena2.show_bhex_gauge = not bool(self.arena2.show_bhex_gauge)
                if imgui.menu_item_simple(
                    "3D Superfluid Phase-Slip Threads", None, bool(self.arena2.show_phase_slips)
                ):
                    self.arena2.show_phase_slips = not bool(self.arena2.show_phase_slips)
            elif self._active_arena == 2:
                imgui.text_colored(COL_MUTED, "Galaxy / Cluster Scenario:")
                for gp_i, (_, gp_lbl) in enumerate(GALAXY_PRESETS):
                    if imgui.menu_item_simple(gp_lbl, None, int(self.arena3.preset_idx) == gp_i):
                        self._clear_replay_backup(2)
                        if gp_i == 5:
                            gals = get_sparc_175_cached()
                            self.arena3.load_sparc_galaxy(
                                gals[self.selected_sparc_idx % len(gals)],
                                mod=self.working_mod,
                                compiled=self.compiled,
                            )
                        else:
                            self.arena3.reset_galaxy(gp_i, mod=self.working_mod, compiled=self.compiled)
                        self._arena3_needs_field_sync = False
                        self._poll_memory_telemetry(force=True)
                        self._reset_frame_pacing_clock()
                imgui.separator()
                imgui.text_colored(COL_MUTED, "Particle Count:")
                for pcnt, plbl in (
                    (250_000, "250,000 Particles"),
                    (1_000_000, "1,000,000 Particles (Default)"),
                    (2_000_000, "2,000,000 Particles (RTX 5080)"),
                ):
                    if imgui.menu_item_simple(plbl, None, int(self.arena3.particle_count) == pcnt):
                        self._clear_replay_backup(2)
                        self.arena3.set_particle_count(pcnt, mod=self.working_mod, compiled=self.compiled)
                        self._arena3_needs_field_sync = False
                        self._poll_memory_telemetry(force=True)
                        self._reset_frame_pacing_clock()
                imgui.separator()
                if imgui.menu_item_simple(
                    "3D Volumetric Lensing Contours", None, bool(self.arena3.show_lensing)
                ):
                    self.arena3.show_lensing = not bool(self.arena3.show_lensing)
            else:
                if imgui.menu_item_simple("Auto-Scrub Cosmic Time", None, bool(self.arena4.auto_scrub)):
                    self.arena4.auto_scrub = not bool(self.arena4.auto_scrub)
                if imgui.menu_item_simple(
                    "Landauer Memory Creases", None, bool(self.arena4.show_creases)
                ):
                    self.arena4.show_creases = not bool(self.arena4.show_creases)
                imgui.separator()
                if imgui.menu_item_simple("Reset Observer Slice (|tau| = 2.65)"):
                    self.arena4.tau_now = 2.65
            imgui.end_menu()

        if imgui.menu_item_simple("Reset 3D Camera View", "Home"):
            self.reset_active_camera()
        _rec_rect("reset_camera")

        imgui.separator()

        # 5. Display & Performance
        if imgui.begin_menu("UI Scale"):
            auto_pct = int(
                round(
                    (
                        self._compute_auto_ui_scale(self._mon_scale, self._mon_width)
                        if self._visible
                        else 1.0
                    )
                    * 100.0
                )
            )
            if imgui.menu_item_simple(f"Auto ({auto_pct}%)", "Ctrl+0", self._ui_scale_auto):
                self.reset_ui_scale_auto()
            for stop in UI_SCALE_STOPS:
                stop_pct = int(round(stop * 100.0))
                is_ui_sel = (not self._ui_scale_auto) and (abs(self.ui_scale - stop) < 0.03)
                if imgui.menu_item_simple(f"{stop_pct}%", None, is_ui_sel):
                    self.set_ui_scale(stop, keep_auto=False)
            imgui.end_menu()

        if imgui.begin_menu("3D Render Scale"):
            for r_val, r_lbl in RENDER_SCALE_OPTIONS:
                if imgui.menu_item_simple(r_lbl, None, abs(self.render_scale - r_val) < 0.04):
                    self.set_render_scale(r_val)
            imgui.end_menu()

        if imgui.begin_menu("FPS Cap"):
            for f_val, f_lbl in FPS_CAP_OPTIONS:
                if imgui.menu_item_simple(f_lbl, None, self.fps_cap == f_val):
                    self.set_fps_cap(f_val, explicit_user_choice=True)
            imgui.end_menu()

        if imgui.menu_item_simple(
            "Hibernate Inactive Arenas in DDR5 RAM", None, self.ddr5_hibernate_enabled
        ):
            self._set_ddr5_hibernate_enabled(not self.ddr5_hibernate_enabled)
        _rec_rect("toggle_hibernate")

        imgui.end_popup()
        self.is_context_menu_open = bool(imgui.is_popup_open("##StudioContextMenu"))
        if not self.is_context_menu_open:
            self._last_context_menu_rect = None

    def _draw_center_viewport(
        self, x0: float, view_w: float, panel_h: float, top_bar_h: float, dt: float
    ) -> None:
        s = self.ui_scale
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        imgui.set_next_window_pos(imgui.ImVec2(x0, top_bar_h), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(view_w, panel_h), imgui.Cond_.always)
        imgui.begin("##Center3DViewport", None, flags)
        imgui.pop_style_var()

        avail = imgui.get_content_region_avail()
        vw = max(int(avail.x), 64)
        vh = max(int(avail.y), 64)
        assert self._view_tex is not None

        # Display zero-copy OpenGL FBO texture (already rendered at center_w*render_scale x panel_h*render_scale)
        imgui.image(
            imgui.ImTextureRef(self._view_tex.glo),
            imgui.ImVec2(float(vw), float(vh)),
            imgui.ImVec2(0.0, 1.0),
            imgui.ImVec2(1.0, 0.0),
        )

        # Interactive 60–120+ FPS Camera Input with High-DPI normalized mouse deltas!
        if imgui.is_item_hovered() and not self.is_context_menu_open:
            io = imgui.get_io()
            cam = self.cameras[self._active_arena]
            inv_s = 1.0 / max(s, 0.5)
            if imgui.is_mouse_dragging(imgui.MouseButton_.left):
                cam.orbit_drag(io.mouse_delta.x * inv_s, io.mouse_delta.y * inv_s)
            rmb_pan = imgui.is_mouse_dragging(imgui.MouseButton_.right) and (
                self._rmb_drag_dist > 5.0 * s
            )
            if rmb_pan or imgui.is_mouse_dragging(imgui.MouseButton_.middle):
                cam.pan_drag(io.mouse_delta.x * inv_s, io.mouse_delta.y * inv_s)
            if abs(io.mouse_wheel) > 1e-4:
                cam.zoom_scroll(io.mouse_wheel)

            if not io.want_text_input:
                cam.process_wasd(
                    dt=dt,
                    w=imgui.is_key_down(imgui.Key.w),
                    s=imgui.is_key_down(imgui.Key.s),
                    a=imgui.is_key_down(imgui.Key.a),
                    d=imgui.is_key_down(imgui.Key.d),
                    q=imgui.is_key_down(imgui.Key.q),
                    e=imgui.is_key_down(imgui.Key.e),
                    sprint=bool(
                        imgui.is_key_down(imgui.Key.left_shift)
                        or imgui.is_key_down(imgui.Key.right_shift)
                    ),
                )

        # Scaled, clip-rect-protected HUD overlay inside top-left of 3D viewport
        dl = imgui.get_window_draw_list()
        top_line = f"{ARENA_TITLES[self._active_arena]}  |  {self.working_mod.name}"
        sub_line = "Camera: Left-Drag Orbit | Right-Drag Pan | Right-Click Menu | WASD+QE Fly"
        if self.compiled is not None:
            sub_line = (
                f"2PN Shadow = {self.compiled.shadow_radius_rg:.3f} r_g ({self.compiled.shadow_vs_gr_pct:+.2f}% vs GR)  |  "
                f"LMB: Orbit  RMB-Drag: Pan  RMB: Menu  WASD: Fly"
            )
        pad_x = 10.0 * s
        pad_y = 7.0 * s
        gap_y = 4.0 * s
        lh = imgui.get_text_line_height()
        req_w = max(imgui.calc_text_size(top_line).x, imgui.calc_text_size(sub_line).x) + pad_x * 2.0 + 4.0 * s
        box_w = max(min(req_w, view_w - 24.0 * s), 64.0)
        box_h = pad_y * 2.0 + lh * 2.0 + gap_y
        p_min = imgui.ImVec2(x0 + 12.0 * s, top_bar_h + 12.0 * s)
        p_max = imgui.ImVec2(p_min.x + box_w, p_min.y + box_h)
        rounding = 6.0 * s
        dl.add_rect_filled(p_min, p_max, imgui.get_color_u32(imgui.ImVec4(0.06, 0.10, 0.17, 0.82)), rounding)
        dl.add_rect(p_min, p_max, imgui.get_color_u32(imgui.ImVec4(0.28, 0.54, 0.82, 0.70)), rounding)
        dl.push_clip_rect(p_min, p_max, True)
        dl.add_text(
            imgui.ImVec2(p_min.x + pad_x, p_min.y + pad_y),
            imgui.get_color_u32(COL_ACCENT),
            top_line,
        )
        dl.add_text(
            imgui.ImVec2(p_min.x + pad_x, p_min.y + pad_y + lh + gap_y),
            imgui.get_color_u32(COL_MUTED),
            sub_line,
        )
        dl.pop_clip_rect()

        # Live ● REC HUD indicator badge in top-right of 3D viewport when recording
        if self.is_recording_video:
            dur_r = max(int(self._video_recorder.duration_s), 0)
            rmm, rss = divmod(dur_r, 60)
            rec_hud = (
                f"● REC {self._video_recorder.fps} FPS ({self._video_recorder.encoder_short_label})  "
                f"{rmm:02d}:{rss:02d}"
            )
            rw = imgui.calc_text_size(rec_hud).x + pad_x * 2.0
            rh = lh + pad_y * 1.6
            r_max_x = x0 + view_w - 12.0 * s
            r_min_x = max(r_max_x - rw, p_max.x + 8.0 * s)
            if r_max_x - r_min_x >= 80.0 * s:
                rp_min = imgui.ImVec2(r_min_x, top_bar_h + 12.0 * s)
                rp_max = imgui.ImVec2(r_max_x, rp_min.y + rh)
                dl.add_rect_filled(
                    rp_min,
                    rp_max,
                    imgui.get_color_u32(imgui.ImVec4(0.36, 0.06, 0.10, 0.88)),
                    rounding,
                )
                dl.add_rect(
                    rp_min,
                    rp_max,
                    imgui.get_color_u32(imgui.ImVec4(0.96, 0.26, 0.34, 0.92)),
                    rounding,
                )
                dl.push_clip_rect(rp_min, rp_max, True)
                dl.add_text(
                    imgui.ImVec2(rp_min.x + pad_x, rp_min.y + pad_y * 0.8),
                    imgui.get_color_u32(imgui.ImVec4(1.0, 0.42, 0.48, 1.0)),
                    rec_hud,
                )
                dl.pop_clip_rect()

        # Host comprehensive Right-Click Context Menu in consistent window ID scope
        self._draw_studio_context_menu()

        imgui.end()

    def _draw_right_panel(self, x0: float, right_w: float, panel_h: float, top_bar_h: float) -> None:
        s = self.ui_scale
        flags = imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse
        imgui.set_next_window_pos(imgui.ImVec2(x0, top_bar_h), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(right_w, panel_h), imgui.Cond_.always)
        imgui.begin("Live Observational Pass/Fail Crucible", None, flags)

        rep = self.crucible
        if rep is not None:
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(6.0 * s, 4.0 * s))
            imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(6.0 * s, 3.0 * s))
            for b in rep.badges:
                bc = badge_color(b.status)
                imgui.begin_child(
                    f"##cb_{b.badge_id}",
                    imgui.ImVec2(-1.0, 0.0),
                    imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
                )
                title_str = f"[{b.status}] {b.dataset_name}"
                title_w = imgui.calc_text_size(title_str).x
                imgui.text_colored(bc, title_str)
                mw = imgui.calc_text_size(b.headline_metric).x
                avail_line = imgui.get_window_width() - mw - 10.0 * s
                if avail_line > title_w + 16.0 * s:
                    imgui.same_line(avail_line)
                imgui.text_colored(COL_ACCENT, b.headline_metric)
                imgui.push_text_wrap_pos(0.0)
                imgui.text_colored(COL_MUTED, b.detail_honest)
                imgui.pop_text_wrap_pos()
                imgui.end_child()
            imgui.pop_style_var(2)

            imgui.separator()
            # 1. SPARC-175 Galaxy Selector + 1-Click 3D Arena 3 Spawner!
            style = imgui.get_style()
            spawn_reserve = (
                imgui.calc_text_size("3D Spawn").x + style.frame_padding.x * 2.0 + style.item_spacing.x
            )
            imgui.set_next_item_width(-spawn_reserve)
            ch_gal, new_g = imgui.combo("##SparcGalSelect", self.selected_sparc_idx, self.sparc_names)
            if ch_gal:
                self.selected_sparc_idx = new_g
                self._refresh_crucible_fast()
                rep = self.crucible
                assert rep is not None
            imgui.same_line()
            if imgui.button("3D Spawn", imgui.ImVec2(-1.0, 0.0)):
                self._clear_replay_backup(2)
                self._switch_active_arena(2)
                gals = get_sparc_175_cached()
                self.arena3.load_sparc_galaxy(
                    gals[self.selected_sparc_idx % len(gals)],
                    mod=self.working_mod,
                    compiled=self.compiled,
                )
                self._arena3_needs_field_sync = False
                self._poll_memory_telemetry(force=True)
                self._reset_frame_pacing_clock()

            avail_plots_h = max(imgui.get_content_region_avail().y - 6.0 * s, 465.0 * s)
            plot_h = float(np.clip(avail_plots_h / 3.0 - style.item_spacing.y, 155.0 * s, 420.0 * s))
            w_scale = max(1.0, 0.85 * s)

            # Chart 1: SPARC Real Galaxy Rotation Curve V(R)
            if implot.begin_plot(f"SPARC: {rep.sparc_galaxy_name} V(R) [km/s]", imgui.ImVec2(-1.0, plot_h)):
                implot.setup_axes("Radius R [kpc]", "V [km/s]")
                implot.setup_legend(implot.Location_.south_east, implot.LegendFlags_.horizontal)
                r_arr = np.ascontiguousarray(rep.sparc_r_kpc, dtype=np.float64)
                vo_arr = np.ascontiguousarray(rep.sparc_v_obs, dtype=np.float64)
                vb_arr = np.ascontiguousarray(rep.sparc_v_bary, dtype=np.float64)
                vm_arr = np.ascontiguousarray(rep.sparc_v_mod, dtype=np.float64)
                rmax = max(float(np.max(r_arr)) * 1.06, 2.0)
                vmax = max(float(np.max(vo_arr)), float(np.max(vm_arr)), float(np.max(vb_arr)), 30.0) * 1.18
                implot.setup_axes_limits(0.0, rmax, 0.0, vmax, implot.Cond_.always)

                implot.plot_line(
                    "Baryons",
                    r_arr,
                    vb_arr,
                    spec=implot.Spec(line_color=imgui.ImVec4(0.58, 0.68, 0.82, 0.85), line_weight=1.7 * w_scale),
                )
                implot.plot_line(
                    "Active Mod",
                    r_arr,
                    vm_arr,
                    spec=implot.Spec(line_color=COL_WARN, line_weight=2.5 * w_scale),
                )
                implot.plot_scatter(
                    "SPARC Obs",
                    r_arr,
                    vo_arr,
                    spec=implot.Spec(
                        marker_fill_color=COL_FORECAST,
                        marker_line_color=COL_FORECAST,
                        marker_size=3.8 * w_scale,
                    ),
                )
                implot.end_plot()

            # Chart 2: Real CAMB Boltzmann Planck 2018 CMB TT Acoustic Spectrum
            if implot.begin_plot("Planck 2018 CAMB TT Spectrum D_l [uK^2]", imgui.ImVec2(-1.0, plot_h)):
                implot.setup_axes("Multipole l", "D_l [uK^2]")
                implot.setup_legend(implot.Location_.north_east, implot.LegendFlags_.horizontal)
                ell_a = np.ascontiguousarray(rep.cmb_ell, dtype=np.float64)
                obs_a = np.ascontiguousarray(rep.cmb_dl_obs, dtype=np.float64)
                mod_a = np.ascontiguousarray(rep.cmb_dl_mod, dtype=np.float64)
                dmax = max(float(np.max(obs_a)), float(np.max(mod_a)), 1000.0) * 1.22
                implot.setup_axes_limits(2.0, 1000.0, 0.0, dmax, implot.Cond_.always)

                implot.plot_line(
                    "Planck 2018",
                    ell_a,
                    obs_a,
                    spec=implot.Spec(line_color=COL_FORECAST, line_weight=1.8 * w_scale),
                )
                implot.plot_line(
                    "Active Mod",
                    ell_a,
                    mod_a,
                    spec=implot.Spec(
                        line_color=COL_PASS if rep.cmb_r32_ratio > 0.85 else COL_FAIL,
                        line_weight=2.2 * w_scale,
                    ),
                )
                implot.end_plot()

            # Chart 3: Dec 2, 2026 Gaia DR4 Wide-Binary |z_gal| Velocity Boost
            wb = rep.wb_curves
            if wb and implot.begin_plot("Gaia DR4 Wide-Binary Boost vs log10(s/AU)", imgui.ImVec2(-1.0, plot_h)):
                implot.setup_axes("log10(s [AU])", "Boost [%]")
                implot.setup_legend(implot.Location_.north_west, implot.LegendFlags_.horizontal)
                log_s = np.ascontiguousarray(np.log10(np.asarray(wb["r_au"], dtype=np.float64)))
                b_d = np.ascontiguousarray(wb["boost_disk_pct"], dtype=np.float64)
                b_h = np.ascontiguousarray(wb["boost_halo_pct"], dtype=np.float64)
                bmax = max(float(np.max(b_h)), float(np.max(b_d)), 8.0) * 1.28
                bmin = min(float(np.min(b_d)), -1.5)
                implot.setup_axes_limits(float(log_s[0]), float(log_s[-1]), bmin, bmax, implot.Cond_.always)

                implot.plot_line(
                    "Disk |z|~0",
                    log_s,
                    b_d,
                    spec=implot.Spec(line_color=COL_FORECAST, line_weight=2.1 * w_scale),
                )
                implot.plot_line(
                    "Halo |z|=6 kpc",
                    log_s,
                    b_h,
                    spec=implot.Spec(line_color=COL_WARN, line_weight=2.5 * w_scale),
                )
                implot.end_plot()

        imgui.end()

    def _draw_shader_inspector_window(
        self, win_w: float, win_h: float, left_w: float = 0.0, center_w: float = 0.0
    ) -> None:
        if not self.show_shader_modal or self.compiled is None:
            return
        s = self.ui_scale
        if center_w >= 480.0 * s:
            modal_w = min(780.0 * s, center_w - 24.0 * s, win_w * 0.85)
            pos_x = left_w + max((center_w - modal_w) * 0.5, 12.0 * s)
        else:
            modal_w = min(780.0 * s, win_w * 0.85)
            pos_x = max((win_w - modal_w) * 0.5, 16.0 * s)
        modal_h = min(560.0 * s, win_h * 0.82)
        pos_y = max((win_h - modal_h) * 0.5, float(round(52.0 * s)) + 12.0 * s)
        imgui.set_next_window_pos(imgui.ImVec2(pos_x, pos_y), imgui.Cond_.appearing)
        imgui.set_next_window_size(imgui.ImVec2(modal_w, modal_h), imgui.Cond_.appearing)
        opened, self.show_shader_modal = imgui.begin(
            "Generated Symbolic Euler-Lagrange & GPU Shaders (GLSL 430 + WGSL)",
            self.show_shader_modal,
        )
        if opened:
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(COL_PASS, self.compiled.euler_lagrange_phi1_str)
            imgui.text_colored(COL_FORECAST, self.compiled.optical_series_str)
            imgui.pop_text_wrap_pos()
            if imgui.begin_tab_bar("##ShaderLangTabs"):
                if imgui.begin_tab_item("OpenGL 4.6 Compute (GLSL 430)")[0]:
                    if self.mono_font is not None:
                        imgui.push_font(self.mono_font, float(self.mono_font.legacy_size))
                    imgui.input_text_multiline(
                        "##glsl_view",
                        self.compiled.glsl_compute_source,
                        imgui.ImVec2(-1.0, -1.0),
                        imgui.InputTextFlags_.read_only,
                    )
                    if self.mono_font is not None:
                        imgui.pop_font()
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Cross-Platform Vulkan/DX12 (WGSL)")[0]:
                    if self.mono_font is not None:
                        imgui.push_font(self.mono_font, float(self.mono_font.legacy_size))
                    imgui.input_text_multiline(
                        "##wgsl_view",
                        self.compiled.wgsl_compute_source,
                        imgui.ImVec2(-1.0, -1.0),
                        imgui.InputTextFlags_.read_only,
                    )
                    if self.mono_font is not None:
                        imgui.pop_font()
                    imgui.end_tab_item()
                imgui.end_tab_bar()
        imgui.end()

    # =========================================================================
    # MAIN EVENT & RENDER LOOP
    # =========================================================================
    def render_single_frame(self, capture_out: Optional[Path] = None) -> None:
        self._in_render_frame = True
        self._one_shot_stall_in_frame = False
        try:
            glfw.poll_events()

            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            if fb_w <= 0 or fb_h <= 0:
                # Window is minimized: sleep briefly to avoid spinning CPU/GPU and skip 3D passes
                if self._visible:
                    time.sleep(0.016)
                now_min = time.perf_counter()
                self._prev_wall = now_min
                self._prev_swap_wall = now_min
                return

            if self._pending_theme_apply:
                apply_luminous_sapphire_theme(self.ui_scale)
                self._pending_theme_apply = False

            now = time.perf_counter()
            sim_dt = float(np.clip(now - self._prev_wall, 1e-4, 0.05))
            self._prev_wall = now
            wall_t = now - self._start_wall

            # 1. Precompute High-DPI panel layout & size offscreen FBO BEFORE rendering active 3D Arena!
            s = self.ui_scale
            win_w_i, win_h_i = glfw.get_window_size(self.window)
            win_w = max(float(win_w_i), 640.0)
            win_h = max(float(win_h_i), 480.0)
            top_bar_h = float(round(52.0 * s))
            panel_h = max(win_h - top_bar_h, 200.0)

            left_w = float(np.clip(win_w * 0.265, 395.0 * s, 500.0 * s))
            right_w = float(np.clip(win_w * 0.288, 415.0 * s, 520.0 * s))
            max_side_sum = min(win_w * 0.76, max(win_w - 240.0, win_w * 0.50))
            if (left_w + right_w) > max_side_sum and (left_w + right_w) > 1e-3:
                ratio = max_side_sum / (left_w + right_w)
                left_w = float(round(left_w * ratio))
                right_w = float(round(right_w * ratio))
            center_w = max(win_w - left_w - right_w, 240.0)

            fbo_w = max(int(round(center_w * self.render_scale)), 64)
            fbo_h = max(int(round(panel_h * self.render_scale)), 64)
            self._ensure_viewport_fbo(fbo_w, fbo_h)

            # 2. Step GPU physics & render active 3D Arena into freshly-sized offscreen FBO (zero black flash!)
            self._step_and_render_arena(sim_dt, wall_t)
            self._poll_memory_telemetry(force=False)

            # 3. Start Dear ImGui frame & draw Studio UI
            imgui.backends.opengl3_new_frame()
            imgui.backends.glfw_new_frame()
            imgui.new_frame()
            self._in_imgui_frame = True
            try:
                self._handle_ui_scale_hotkeys()
                self._draw_top_bar(win_w, top_bar_h)
                self._draw_left_panel(left_w, panel_h, top_bar_h)
                self._draw_center_viewport(left_w, center_w, panel_h, top_bar_h, sim_dt)
                self._draw_right_panel(left_w + center_w, right_w, panel_h, top_bar_h)
                self._draw_shader_inspector_window(win_w, win_h, left_w, center_w)
                imgui.render()
            finally:
                self._in_imgui_frame = False

            # 4. Composite Studio UI onto main window backbuffer
            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            self.ctx.screen.use()
            GL.glViewport(0, 0, fb_w, fb_h)
            GL.glDisable(GL.GL_SCISSOR_TEST)
            GL.glClearColor(0.08, 0.12, 0.19, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

            if self._video_recorder.is_recording:
                now_rec = time.perf_counter()
                rec_dt = max(now_rec - self._last_rec_tick_wall, 0.0)
                self._last_rec_tick_wall = now_rec
                self._video_recorder.capture_frame_if_due(fb_w, fb_h, rec_dt)

            pending_snap = self._pending_screenshot_out
            self._pending_screenshot_out = None
            eff_capture = capture_out if capture_out is not None else pending_snap
            if eff_capture is not None:
                GL.glFinish()
                GL.glReadBuffer(GL.GL_BACK)
                self.save_window_screenshot(eff_capture)

            glfw.swap_buffers(self.window)

            # 5. Optional frame pacing (prevents needless 240Hz GPU saturation) + Honest Post-Swap Frame-Time EWMA!
            if self._visible and self.fps_cap > 0 and not self._one_shot_stall_in_frame:
                target_dt = 1.0 / float(self.fps_cap)
                rem = target_dt - (time.perf_counter() - self._prev_swap_wall)
                if rem > 0.0:
                    if rem > 0.0012:
                        self._sleep_precise(rem - 0.0008)
                    spin_deadline = time.perf_counter() + 0.0015
                    while (time.perf_counter() - self._prev_swap_wall) < target_dt:
                        if time.perf_counter() >= spin_deadline:
                            break

            post_swap = time.perf_counter()
            raw_dt = max(post_swap - self._prev_swap_wall, 1e-6)
            self._prev_swap_wall = post_swap
            if self._one_shot_stall_in_frame:
                self._prev_wall = post_swap
                self._one_shot_stall_in_frame = False
                if self._rendered_frames == 0:
                    self._rendered_frames = 1
                else:
                    self._rendered_frames += 1
            elif self._rendered_frames == 0:
                self._rendered_frames = 1
            else:
                self._rendered_frames += 1
                self._dt_smooth = 0.90 * self._dt_smooth + 0.10 * raw_dt
                self.fps_smooth = 1.0 / max(self._dt_smooth, 1e-5)
                self.frame_ms_smooth = self._dt_smooth * 1000.0
        finally:
            self._in_render_frame = False

    def run(
        self,
        max_frames: Optional[int] = None,
        screenshot_out: Optional[Path] = None,
    ) -> None:
        frame_idx = 0
        try:
            while not glfw.window_should_close(self.window):
                is_last = (max_frames is not None) and (frame_idx + 1 >= max_frames)
                cap = screenshot_out if is_last else None
                self.render_single_frame(capture_out=cap)
                frame_idx += 1
                if is_last:
                    break
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if getattr(self, "_shut_down", False):
            return
        self._shut_down = True

        rec = getattr(self, "_video_recorder", None)
        if rec is not None and getattr(rec, "is_recording", False):
            try:
                rec.stop()
            except Exception:
                pass

        self._replay_live_backup.clear()
        if hasattr(self, "lan_dispatcher"):
            shut_lan = getattr(self.lan_dispatcher, "shutdown", None)
            if callable(shut_lan):
                try:
                    shut_lan()
                except Exception:
                    pass

        for name in ("arena1", "arena2", "arena3", "arena4"):
            arena = getattr(self, name, None)
            if arena is not None:
                try:
                    arena.release()
                except Exception:
                    pass
        for obj in (getattr(self, "_view_fbo", None), getattr(self, "_view_tex", None)):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        try:
            imgui.backends.opengl3_shutdown()
            imgui.backends.glfw_shutdown()
            implot.destroy_context()
            imgui.destroy_context()
        except Exception:
            pass
        if getattr(self, "_hires_timer", None) is not None and sys.platform == "win32":
            try:
                ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(int(self._hires_timer)))
            except Exception:
                pass
            self._hires_timer = None
        if getattr(self, "_timer_period_active", False) and sys.platform == "win32":
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
            self._timer_period_active = False
        try:
            if getattr(self, "window", None) is not None:
                glfw.destroy_window(self.window)
            glfw.terminate()
        except Exception:
            pass
