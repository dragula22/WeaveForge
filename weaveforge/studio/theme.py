"""
Option A+ Luminous Sapphire Slate Theme (`weaveforge/studio/theme.py`)
======================================================================
Applies a cohesive, non-fatiguing mid-tone slate-sapphire scientific studio palette
to Dear ImGui and ImPlot (zero white flashbangs, zero pitch-black contrast crush),
paired with crisp Windows system typography (`Segoe UI` + `Consolas` code font).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

from imgui_bundle import imgui, implot


# Semantic RGBA color tuples for badges & ImPlot series
COL_PASS = imgui.ImVec4(0.20, 0.90, 0.60, 1.00)      # Emerald Mint
COL_WARN = imgui.ImVec4(0.98, 0.78, 0.24, 1.00)      # Solar Gold-Amber
COL_FAIL = imgui.ImVec4(0.98, 0.34, 0.42, 1.00)      # Crimson Alert
COL_FORECAST = imgui.ImVec4(0.32, 0.84, 1.00, 1.00)  # Luminous Cyan
COL_MUTED = imgui.ImVec4(0.66, 0.76, 0.88, 1.00)     # Soft Slate Text
COL_ACCENT = imgui.ImVec4(0.28, 0.78, 0.98, 1.00)    # Sapphire-Cyan Highlight


def badge_color(status: str) -> imgui.ImVec4:
    s = status.upper()
    if s == "PASS":
        return COL_PASS
    if s == "WARN":
        return COL_WARN
    if s == "FAIL":
        return COL_FAIL
    return COL_FORECAST


def load_studio_fonts() -> Tuple[object | None, object | None]:
    """Loads crisp Segoe UI (18.0px) and Consolas (16.0px) if available on Windows."""
    io = imgui.get_io()
    ui_font = None
    mono_font = None

    segoe = Path("C:/Windows/Fonts/segoeui.ttf")
    consola = Path("C:/Windows/Fonts/consola.ttf")

    if segoe.exists():
        ui_font = io.fonts.add_font_from_file_ttf(str(segoe), 18.0)
    else:
        ui_font = io.fonts.add_font_default()

    if consola.exists():
        mono_font = io.fonts.add_font_from_file_ttf(str(consola), 16.0)
    else:
        mono_font = ui_font

    return ui_font, mono_font


def apply_luminous_sapphire_theme(ui_scale: float = 1.0) -> None:
    """Configures Dear ImGui + ImPlot with the Option A+ Luminous Sapphire Slate aesthetic at `ui_scale`."""
    raw_scale = float(ui_scale) if math.isfinite(float(ui_scale)) else 1.0
    s = max(0.5, min(raw_scale, 4.0))

    # Reset fresh base styles first so repeated live UI-scale changes never compound!
    ctx = imgui.get_current_context()
    if ctx is not None:
        ctx.style = imgui.Style()
    implot.set_style(implot.Style())

    style = imgui.get_style()
    style.window_rounding = 6.0
    style.child_rounding = 6.0
    style.frame_rounding = 5.0
    style.popup_rounding = 6.0
    style.scrollbar_rounding = 6.0
    style.grab_rounding = 4.0
    style.tab_rounding = 5.0
    style.window_border_size = 1.0
    style.frame_border_size = 1.0
    style.scrollbar_size = 14.0
    style.item_spacing = imgui.ImVec2(8.0, 6.0)
    style.frame_padding = imgui.ImVec2(8.0, 5.0)
    style.window_padding = imgui.ImVec2(10.0, 10.0)

    c = style.set_color_
    Col = imgui.Col_

    c(Col.text, imgui.ImVec4(0.93, 0.96, 1.00, 1.00))
    c(Col.text_disabled, imgui.ImVec4(0.55, 0.65, 0.78, 1.00))
    c(Col.window_bg, imgui.ImVec4(0.105, 0.155, 0.235, 0.97))
    c(Col.child_bg, imgui.ImVec4(0.125, 0.182, 0.275, 0.92))
    c(Col.popup_bg, imgui.ImVec4(0.095, 0.145, 0.225, 0.98))
    c(Col.border, imgui.ImVec4(0.24, 0.37, 0.54, 0.65))
    c(Col.border_shadow, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))

    c(Col.frame_bg, imgui.ImVec4(0.075, 0.118, 0.185, 0.95))
    c(Col.frame_bg_hovered, imgui.ImVec4(0.14, 0.23, 0.36, 1.00))
    c(Col.frame_bg_active, imgui.ImVec4(0.18, 0.30, 0.46, 1.00))

    c(Col.title_bg, imgui.ImVec4(0.085, 0.130, 0.205, 1.00))
    c(Col.title_bg_active, imgui.ImVec4(0.130, 0.205, 0.320, 1.00))
    c(Col.menu_bar_bg, imgui.ImVec4(0.095, 0.145, 0.225, 1.00))

    c(Col.button, imgui.ImVec4(0.16, 0.34, 0.56, 0.95))
    c(Col.button_hovered, imgui.ImVec4(0.22, 0.46, 0.74, 1.00))
    c(Col.button_active, imgui.ImVec4(0.28, 0.56, 0.86, 1.00))

    c(Col.header, imgui.ImVec4(0.16, 0.28, 0.44, 0.90))
    c(Col.header_hovered, imgui.ImVec4(0.22, 0.38, 0.58, 0.95))
    c(Col.header_active, imgui.ImVec4(0.26, 0.46, 0.70, 1.00))

    c(Col.tab, imgui.ImVec4(0.12, 0.19, 0.30, 0.95))
    c(Col.tab_hovered, imgui.ImVec4(0.22, 0.42, 0.66, 1.00))
    c(Col.tab_selected, imgui.ImVec4(0.19, 0.38, 0.62, 1.00))

    c(Col.slider_grab, imgui.ImVec4(0.28, 0.74, 0.96, 1.00))
    c(Col.slider_grab_active, imgui.ImVec4(0.96, 0.78, 0.28, 1.00))
    c(Col.check_mark, imgui.ImVec4(0.26, 0.88, 0.66, 1.00))

    style.scale_all_sizes(s)
    style.window_border_size = 1.0
    style.frame_border_size = 1.0

    io = imgui.get_io()
    io.config_dpi_scale_fonts = False
    style.font_scale_dpi = 1.0
    style.font_scale_main = s

    # Matching ImPlot scientific chart styling scaled cleanly for High-DPI / 4K
    pstyle = implot.get_style()
    pstyle.plot_padding = imgui.ImVec2(8.0 * s, 8.0 * s)
    pstyle.label_padding = imgui.ImVec2(5.0 * s, 5.0 * s)
    pstyle.legend_padding = imgui.ImVec2(8.0 * s, 8.0 * s)
    pstyle.legend_inner_padding = imgui.ImVec2(5.0 * s, 4.0 * s)
    pstyle.major_tick_len = imgui.ImVec2(8.0 * s, 8.0 * s)
    pstyle.minor_tick_len = imgui.ImVec2(4.0 * s, 4.0 * s)
    pstyle.plot_min_size = imgui.ImVec2(180.0 * s, 120.0 * s)

    pc = pstyle.set_color_
    PCol = implot.Col_
    pc(PCol.frame_bg, imgui.ImVec4(0.085, 0.132, 0.205, 0.95))
    pc(PCol.plot_bg, imgui.ImVec4(0.070, 0.112, 0.178, 0.95))
    pc(PCol.plot_border, imgui.ImVec4(0.24, 0.38, 0.56, 0.70))
    pc(PCol.axis_grid, imgui.ImVec4(0.26, 0.40, 0.58, 0.30))
    pc(PCol.legend_bg, imgui.ImVec4(0.085, 0.135, 0.215, 0.88))
