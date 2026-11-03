"""
Cross-Platform `wgpu` (Vulkan / DX12 / Metal) 3D Compute Engine (`weaveforge/compute/wgpu_engine.py`)
====================================================================================================
Executes `SymPy`-generated `WGSL` 3D compute shaders natively on ANY modern GPU without CUDA:
  • Rig 1: NVIDIA GeForce RTX 5080 (DiscreteGPU via Vulkan / D3D12)
  • Rig 2: AMD Radeon RX 5700 XT 8GB (DiscreteGPU via Vulkan / D3D12 — ZERO CUDA dependency!)
  • Community: Intel Arc, AMD RDNA, NVIDIA, and Apple M-series Metal GPUs.

IMPORTANT WINDOWS DRIVER NOTE:
  Always select Vulkan or D3D12 adapters explicitly (`backend != 'OpenGL'`) so `wgpu` never
  perturbs any active GLFW/ModernGL OpenGL 4.6 viewport context on the same thread!
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import os
import struct
import sys
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

# Restrict wgpu_native to Vulkan / D3D12 / Metal so probing never creates dummy WGL contexts
os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan,D3D12,Metal")

import numpy as np
import wgpu

from weaveforge.core.frameworks import PhysicsMod
from weaveforge.core.symbolic_compiler import CompiledMasterAction, compile_master_action


@contextlib.contextmanager
def _preserve_wgl_context() -> Iterator[None]:
    """Saves and restores any active 64-bit Windows WGL OpenGL context on the calling thread."""
    if sys.platform != "win32":
        yield
        return
    opengl32 = ctypes.windll.opengl32
    wglGetCurrentDC = opengl32.wglGetCurrentDC
    wglGetCurrentDC.restype = ctypes.c_void_p
    wglGetCurrentContext = opengl32.wglGetCurrentContext
    wglGetCurrentContext.restype = ctypes.c_void_p
    wglMakeCurrent = opengl32.wglMakeCurrent
    wglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    wglMakeCurrent.restype = ctypes.c_int

    prev_dc = wglGetCurrentDC()
    prev_ctx = wglGetCurrentContext()
    try:
        yield
    finally:
        if prev_ctx and wglGetCurrentContext() != prev_ctx:
            wglMakeCurrent(prev_dc, prev_ctx)


def _enumerate_non_gl_adapters_isolated() -> List[wgpu.GPUAdapter]:
    """Enumerates wgpu adapters on a clean helper thread so no active OpenGL context is touched."""
    box: Dict[str, object] = {}

    def _probe() -> None:
        try:
            adapters = wgpu.gpu.enumerate_adapters_sync()
            box["adapters"] = [a for a in adapters if "via OpenGL" not in a.summary]
        except Exception as exc:
            box["error"] = exc

    with _preserve_wgl_context():
        t = threading.Thread(target=_probe, daemon=True)
        t.start()
        t.join()
    if "error" in box:
        raise RuntimeError(f"wgpu adapter enumeration failed: {box['error']}")
    return list(box.get("adapters", []))  # type: ignore[arg-type]


def compute_effective_gw_speed(
    mod: PhysicsMod,
    compiled: Optional[CompiledMasterAction] = None,
) -> float:
    """Computes the causal weave-tension wave speed `c_gw = |r_gw| * |c_w|` from the compiled Master Action."""
    r_gw = (
        abs(float(compiled.gw_speed_ratio_val))
        if (compiled is not None and math.isfinite(float(getattr(compiled, "gw_speed_ratio_val", 1.0))))
        else 1.0
    )
    return max(r_gw, 0.05) * max(abs(float(mod.parameters.c_w)), 0.05)


def compute_safe_symplectic_dt(
    dx: float,
    mod: PhysicsMod,
    compiled: Optional[CompiledMasterAction] = None,
    peak_rho_est: float = 2.8,
) -> float:
    """
    Adaptive Courant–Friedrichs–Lewy + Klein–Gordon mass + Skyrme & nonlinear potential stiffness controller:
    guarantees symplectic Störmer–Verlet stability even for stiff user Lagrangians (`g6 = 500`, `k_skyrme > 0`).
    """
    cs_sq = abs(compiled.sound_speed_sq) if compiled is not None and math.isfinite(compiled.sound_speed_sq) else 1.0
    c_gw_guard = (
        compute_effective_gw_speed(mod, compiled)
        if (
            abs(float(getattr(mod.parameters, "g_knot", 0.0))) > 1.0e-8
            and bool(getattr(mod.parameters, "retarded_weave_u", True))
        )
        else 0.0
    )
    c_eff = max(abs(float(mod.parameters.c_w)), c_gw_guard, math.sqrt(max(cs_sq, 0.05)), 0.25)

    rho_ref = max(float(peak_rho_est), 2.8)
    k_sk = abs(float(getattr(mod.parameters, "k_skyrme", 0.0)))
    if k_sk > 0.0 and (compiled is None or compiled.needs_full_hessian):
        grad_sq_est = 18.0 * rho_ref
        c_local = math.sqrt((c_eff**2) + 4.0 * k_sk * grad_sq_est)
    else:
        c_local = c_eff
    dt_cfl = 0.22 * dx / c_local

    pot_stiff = (
        abs(float(compiled.max_potential_stiffness))
        if compiled is not None and math.isfinite(compiled.max_potential_stiffness)
        else 0.0
    )
    m2 = max(float(mod.parameters.m_eff) ** 2, pot_stiff)
    stiff_nl = max(2.5 * abs(float(mod.parameters.g4)) * rho_ref, 2.5 * abs(float(mod.parameters.g6)) * (rho_ref**2))
    omega_max_sq = (12.0 * (c_local**2)) / max(dx * dx, 1e-8) + m2 + stiff_nl
    dt_stiff = 0.42 / math.sqrt(max(omega_max_sq, 1.0))
    return min(dt_cfl, dt_stiff)


WEAVE_ALPHA_KNOT: float = 1.75e-4
WEAVE_KAPPA_SCALE: float = 4.0 * math.pi * WEAVE_ALPHA_KNOT
_HE_KERNEL_FFT_CACHE: Dict[Tuple[int, float], np.ndarray] = {}


def solve_3d_weave_tension_u_cpu(
    rho_h_zyx: np.ndarray,
    box_size: float = 12.0,
    g_knot: float = 0.0,
) -> np.ndarray:
    """
    Solves the exact isolated open-space 3D Poisson equation `nabla^2 U = -kappa * rho_H`
    (`U -> +kappa * M / (4 * pi * r) > 0` as `r -> inf`) via Hockney-Eastwood `(2N)^3` padded
    FFT Green's convolution (`1 / (4 * pi * |x - x'|)`), with zero periodic image artifacts.
    """
    arr = np.asarray(rho_h_zyx, dtype=np.float32)
    n_grid = int(arr.shape[0])
    if abs(float(g_knot)) <= 1.0e-12 or arr.size == 0:
        return np.zeros((n_grid, n_grid, n_grid), dtype=np.float32)

    # Downsample smoothly to 64^3 if N > 96 to bound FFT RAM while keeping smooth long-range U(x)
    if n_grid > 96 and n_grid % 2 == 0:
        m = 64
        blk = n_grid // m
        coarse = arr.reshape(m, blk, m, blk, m, blk).mean(axis=(1, 3, 5))
        u_coarse = solve_3d_weave_tension_u_cpu(coarse, box_size=box_size, g_knot=g_knot)
        from scipy.ndimage import zoom

        return np.asarray(zoom(u_coarse, float(n_grid) / float(m), order=1), dtype=np.float32)

    dx = float(box_size) / float(n_grid)
    cache_key = (n_grid, round(dx, 9))
    g_k = _HE_KERNEL_FFT_CACHE.get(cache_key)
    p2 = 2 * n_grid
    if g_k is None:
        idx = np.arange(p2, dtype=np.float64)
        wrapped = np.where(idx <= n_grid, idx, idx - p2) * dx
        zz, yy, xx = np.meshgrid(wrapped, wrapped, wrapped, indexing="ij")
        rr = np.sqrt(xx * xx + yy * yy + zz * zz)
        # Exact self-potential of a uniform dx^3 cube at r=0 is 2.380077 / (4 * pi * dx)
        g_real = np.where(rr > 1.0e-12, 1.0 / (4.0 * math.pi * np.maximum(rr, 1.0e-12)), 2.380077 / (4.0 * math.pi * dx))
        g_k = np.fft.rfftn(g_real)
        if len(_HE_KERNEL_FFT_CACHE) > 8:
            _HE_KERNEL_FFT_CACHE.clear()
        _HE_KERNEL_FFT_CACHE[cache_key] = g_k

    rho_pad = np.zeros((p2, p2, p2), dtype=np.float64)
    rho_pad[:n_grid, :n_grid, :n_grid] = arr.astype(np.float64)
    kappa = WEAVE_KAPPA_SCALE * float(g_knot)
    u_pad = np.fft.irfftn(np.fft.rfftn(rho_pad, axes=(0, 1, 2)) * g_k, s=(p2, p2, p2), axes=(0, 1, 2))
    u_box = (kappa * (dx**3)) * u_pad[:n_grid, :n_grid, :n_grid]
    return np.clip(u_box, -0.35, 0.75).astype(np.float32)


def _eval_trefoil_pq_numpy(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    a0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates homogeneous complex polynomial pair `(P_re, P_im, Q_re, Q_im)` of a `T_(3,2)` Trefoil (`Q_H = +7`)."""
    r = np.sqrt(X * X + Y * Y + Z * Z)
    rn = r / float(a0)
    rn2 = rn * rn
    chi = 3.14159265 * np.exp(-0.6931472 * rn2 - 0.095 * rn2 * rn2)
    sc_over_r = np.where(r > 1.0e-5, np.sin(chi) / np.maximum(r, 1.0e-5), 0.0)
    z1_re = sc_over_r * X
    z1_im = sc_over_r * Y
    z2_re = sc_over_r * Z
    z2_im = np.cos(chi)

    z1_2_re = z1_re * z1_re - z1_im * z1_im
    z1_2_im = 2.0 * z1_re * z1_im
    z2_2_re = z2_re * z2_re - z2_im * z2_im
    z2_2_im = 2.0 * z2_re * z2_im

    q_re = (z1_2_re * z1_re - z1_2_im * z1_im) + z2_2_re
    q_im = (z1_2_re * z1_im + z1_2_im * z1_re) + z2_2_im
    p_re = 1.95 * (z1_2_re * z2_re - z1_2_im * z2_im)
    p_im = 1.95 * (z1_2_re * z2_im + z1_2_im * z2_re)
    return p_re, p_im, q_re, q_im


def build_3d_topological_initial_state(
    grid_n: int = 64,
    preset: str = "trefoil_knot",
    box_size: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs C-infinity smooth, periodic-boundary-clean 3D topological Hopfion / knot initial fields
    `(phi_init, vel_init)` of shape `(N, N, N, 4)` (`float32`), matching `_INIT_KNOT_GLSL` in Arena 1!
      • `trefoil_knot`:               Sutcliffe (2007, Eq. 3.7) `T_(3,2)` Trefoil Knot Rational Map (`Q_H = +7`)!
      • `z3_braid`:                   3-Ribbon `Z_3` Cyclic Generation Braid Link (Koide `theta0 = 2/9`, `Q_H = +9`!)
      • `hopfion_ring`:               Symmetric Orthogonal Linked 3D Hopfion Rings (`6_1^2` link, `Q_H = -6`!)
      • `packet_smash`:               Head-on 3D relativistic wave-packet collision (`Observer Channel A/B` interference)
      • `two_matter_trefoils`:        Two separated `+7 / +7` Matter Trefoils (`Q_H = +14`) at rest for `1/r^2` gravity!
      • `matter_antimatter_trefoils`: Separated `+7 / -7` Matter + Antimatter Trefoils (`Q_H = 0`) for gravity + annihilation!
    """
    idx_lin = (np.arange(grid_n, dtype=np.float32) + 0.5) / float(grid_n)
    coords = (idx_lin - 0.5) * float(box_size)
    Z, Y, X = np.meshgrid(coords, coords, coords, indexing="ij")  # matches (z * N + y) * N + x memory layout!

    phi = np.zeros((grid_n, grid_n, grid_n, 4), dtype=np.float32)
    vel = np.zeros((grid_n, grid_n, grid_n, 4), dtype=np.float32)
    v0 = 0.85

    if preset == "trefoil_knot":
        p_re, p_im, q_re, q_im = _eval_trefoil_pq_numpy(X, Y, Z, 0.195 * float(box_size))
        q2 = q_re * q_re + q_im * q_im
        p2 = p_re * p_re + p_im * p_im
        denom = np.maximum(p2 + q2, 1.0e-12)
        pqbar_re = p_re * q_re + p_im * q_im
        pqbar_im = p_im * q_re - p_re * q_im

        phi[..., 0] = v0 * (2.0 * pqbar_re / denom)
        phi[..., 1] = v0 * (2.0 * pqbar_im / denom)
        phi[..., 2] = v0 * (((q2 - p2) / denom) - 1.0)

    elif preset == "z3_braid":
        r_maj = 0.195 * float(box_size)
        r_braid = 0.080 * float(box_size)
        w0 = 0.068 * float(box_size)
        w_env = 0.098 * float(box_size)
        inv_wenv2 = 1.0 / (w_env * w_env)
        m_samp = 48
        s_grid = (2.0 * math.pi * np.arange(m_samp, dtype=np.float32)) / float(m_samp)

        w_sum_x = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
        w_sum_y = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
        hit_core = np.zeros((grid_n, grid_n, grid_n), dtype=bool)

        for st_i in range(3):
            off = 0.2222222 + 2.0943951 * float(st_i)
            best_d2 = np.full((grid_n, grid_n, grid_n), 1.0e20, dtype=np.float32)
            best_s = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
            for k in range(m_samp):
                s_val = float(s_grid[k])
                tw = s_val + off
                rad = r_maj + r_braid * math.cos(tw)
                cx = rad * math.cos(s_val)
                cy = rad * math.sin(s_val)
                cz = -r_braid * math.sin(tw)
                d2_k = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
                better = d2_k < best_d2
                best_d2 = np.where(better, d2_k, best_d2)
                best_s = np.where(better, s_val, best_s)

            for _ in range(2):
                tw = best_s + off
                rad = r_maj + r_braid * np.cos(tw)
                cs = np.cos(best_s)
                ss = np.sin(best_s)
                ctw = np.cos(tw)
                stw = np.sin(tw)
                bc_x = rad * cs
                bc_y = rad * ss
                bc_z = -r_braid * stw
                dc_x = -r_braid * stw * cs - rad * ss
                dc_y = -r_braid * stw * ss + rad * cs
                dc_z = -r_braid * ctw
                dc2 = np.maximum(dc_x * dc_x + dc_y * dc_y + dc_z * dc_z, 1.0e-6)
                ds = ((X - bc_x) * dc_x + (Y - bc_y) * dc_y + (Z - bc_z) * dc_z) / dc2
                best_s = best_s + np.clip(ds, -0.09, 0.09)

            tw = best_s + off
            rad = r_maj + r_braid * np.cos(tw)
            cs = np.cos(best_s)
            ss = np.sin(best_s)
            ctw = np.cos(tw)
            stw = np.sin(tw)
            bc_x = rad * cs
            bc_y = rad * ss
            bc_z = -r_braid * stw
            tx = -r_braid * stw * cs - rad * ss
            ty = -r_braid * stw * ss + rad * cs
            tz = -r_braid * ctw
            t_norm = np.maximum(np.sqrt(tx * tx + ty * ty + tz * tz), 1.0e-8)
            tx /= t_norm
            ty /= t_norm
            tz /= t_norm

            er_dot_t = cs * tx + ss * ty
            ux = cs - er_dot_t * tx
            uy = ss - er_dot_t * ty
            uz = -er_dot_t * tz
            u_norm = np.maximum(np.sqrt(ux * ux + uy * uy + uz * uz), 1.0e-8)
            ux /= u_norm
            uy /= u_norm
            uz /= u_norm

            vx = ty * uz - tz * uy
            vy = tz * ux - tx * uz
            vz = tx * uy - ty * ux

            dx_v = X - bc_x
            dy_v = Y - bc_y
            dz_v = Z - bc_z
            u_p = dx_v * ux + dy_v * uy + dz_v * uz
            v_p = dx_v * vx + dy_v * vy + dz_v * vz
            d2 = dx_v * dx_v + dy_v * dy_v + dz_v * dz_v
            xi = d2 * inv_wenv2
            a_amp = w0 * np.exp(-xi - 0.15 * xi * xi)
            psi_ang = -1.0 * best_s + off
            cpsi = np.cos(psi_ang)
            spsi = np.sin(psi_ang)
            qx = u_p * cpsi - v_p * spsi
            qy = u_p * spsi + v_p * cpsi

            core_m = d2 < 1.0e-10
            hit_core |= core_m
            safe_d2 = np.maximum(d2, 1.0e-10)
            w_sum_x += np.where(core_m, 0.0, (a_amp / safe_d2) * qx)
            w_sum_y += np.where(core_m, 0.0, (a_amp / safe_d2) * qy)

        w2 = w_sum_x * w_sum_x + w_sum_y * w_sum_y
        den = 1.0 + w2
        phi[..., 0] = np.where(hit_core, 0.0, v0 * (2.0 * w_sum_x / den))
        phi[..., 1] = np.where(hit_core, 0.0, v0 * (2.0 * w_sum_y / den))
        phi[..., 2] = np.where(hit_core, -2.0 * v0, v0 * (((1.0 - w2) / den) - 1.0))

    elif preset == "hopfion_ring":
        r_maj = 0.190 * float(box_size)
        x_off = 0.56 * r_maj
        w0 = 0.072 * float(box_size)
        w_env = 0.102 * float(box_size)
        inv_wenv2 = 1.0 / (w_env * w_env)

        px1 = X + x_off
        rc1_2 = px1 * px1 + Y * Y
        reg1 = rc1_2 / (rc1_2 + 0.18 * r_maj * r_maj)
        s1 = np.arctan2(Y, px1)
        cs1 = np.cos(s1)
        ss1 = np.sin(s1)
        d1x = X - (-x_off + r_maj * cs1)
        d1y = Y - (r_maj * ss1)
        d1z = Z
        d1_2 = d1x * d1x + d1y * d1y + d1z * d1z
        xi1 = d1_2 * inv_wenv2
        a1 = w0 * reg1 * np.exp(-xi1 - 0.15 * xi1 * xi1)
        u1p = d1x * cs1 + d1y * ss1
        v1p = -d1z
        psi1 = 2.0 * s1
        q1x = u1p * np.cos(psi1) - v1p * np.sin(psi1)
        q1y = u1p * np.sin(psi1) + v1p * np.cos(psi1)

        px2 = X - x_off
        rc2_2 = px2 * px2 + Z * Z
        reg2 = rc2_2 / (rc2_2 + 0.18 * r_maj * r_maj)
        s2 = np.arctan2(Z, px2)
        cs2 = np.cos(s2)
        ss2 = np.sin(s2)
        d2x = X - (x_off + r_maj * cs2)
        d2y = Y
        d2z = Z - (r_maj * ss2)
        d2_2 = d2x * d2x + d2y * d2y + d2z * d2z
        xi2 = d2_2 * inv_wenv2
        a2 = w0 * reg2 * np.exp(-xi2 - 0.15 * xi2 * xi2)
        u2p = d2x * cs2 + d2z * ss2
        v2p = d2y
        psi2 = 2.0 * s2
        q2x = u2p * np.cos(psi2) - v2p * np.sin(psi2)
        q2y = u2p * np.sin(psi2) + v2p * np.cos(psi2)

        hit_core = (d1_2 < 1.0e-10) | (d2_2 < 1.0e-10)
        sd1 = np.maximum(d1_2, 1.0e-10)
        sd2 = np.maximum(d2_2, 1.0e-10)
        wx = (a1 / sd1) * q1x + (a2 / sd2) * q2x
        wy = (a1 / sd1) * q1y + (a2 / sd2) * q2y
        w2 = wx * wx + wy * wy
        den = 1.0 + w2
        phi[..., 0] = np.where(hit_core, 0.0, v0 * (2.0 * wx / den))
        phi[..., 1] = np.where(hit_core, 0.0, v0 * (2.0 * wy / den))
        phi[..., 2] = np.where(hit_core, -2.0 * v0, v0 * (((1.0 - w2) / den) - 1.0))

    elif preset in ("two_matter_trefoils", "matter_antimatter_trefoils", "binary_trefoil_inspiral"):
        a0 = (0.130 if preset == "binary_trefoil_inspiral" else 0.132) * float(box_size)
        if preset in ("two_matter_trefoils", "binary_trefoil_inspiral"):
            x_off = (0.145 if preset == "binary_trefoil_inspiral" else 0.163) * float(box_size)
            p1_re, p1_im, q1_re, q1_im = _eval_trefoil_pq_numpy(Y, Z, X + x_off, a0)
            ct = -0.30901699
            st = 0.95105652
            yr = ct * Y - st * Z
            zr = st * Y + ct * Z
            p2_re, p2_im, q2_re, q2_im = _eval_trefoil_pq_numpy(yr, zr, X - x_off, a0)
            pt_re = (p1_re * q2_re - p1_im * q2_im) + (p2_re * q1_re - p2_im * q1_im)
            pt_im = (p1_re * q2_im + p1_im * q2_re) + (p2_re * q1_im + p2_im * q1_re)
        else:
            x_off = 0.138 * float(box_size)
            p1_re, p1_im, q1_re, q1_im = _eval_trefoil_pq_numpy(Y, Z, X + x_off, a0)
            p2_re, p2_im, q2_re, q2_im = _eval_trefoil_pq_numpy(Y, Z, x_off - X, a0)
            pt_re = (p1_re * q2_re - p1_im * q2_im) - (p2_re * q1_re - p2_im * q1_im)
            pt_im = (p1_re * q2_im + p1_im * q2_re) - (p2_re * q1_im + p2_im * q1_re)
        qt_re = q1_re * q2_re - q1_im * q2_im
        qt_im = q1_re * q2_im + q1_im * q2_re

        q2 = qt_re * qt_re + qt_im * qt_im
        p2 = pt_re * pt_re + pt_im * pt_im
        denom = np.maximum(p2 + q2, 1.0e-12)
        pqbar_re = pt_re * qt_re + pt_im * qt_im
        pqbar_im = pt_im * qt_re - pt_re * qt_im
        phi[..., 0] = v0 * (2.0 * pqbar_re / denom)
        phi[..., 1] = v0 * (2.0 * pqbar_im / denom)
        phi[..., 2] = v0 * (((q2 - p2) / denom) - 1.0)
        if preset == "binary_trefoil_inspiral":
            dx_init = float(box_size) / float(grid_n)
            vy_prof = 0.105 * np.tanh(X / 1.10)
            dphi_dy = (np.roll(phi[..., :3], -1, axis=1) - np.roll(phi[..., :3], 1, axis=1)) / (2.0 * dx_init)
            vel[..., :3] = -vy_prof[..., None] * dphi_dy

    else:  # packet_smash
        sep = 0.20 * box_size
        w0 = 0.065 * box_size
        k_kick = 3.2
        g_left = 2.15 * np.exp(-((X + sep) ** 2 + Y**2 + Z**2) / (2.0 * (w0**2)))
        g_right = 2.15 * np.exp(-((X - sep) ** 2 + Y**2 + Z**2) / (2.0 * (w0**2)))
        phi[..., 0] = g_left * np.cos(k_kick * X) + g_right * np.cos(-k_kick * X)
        phi[..., 1] = g_left * np.sin(k_kick * X) + g_right * np.sin(-k_kick * X)
        phi[..., 2] = 0.35 * (g_left - g_right)
        vel[..., 0] = k_kick * (g_left * np.sin(k_kick * X) - g_right * np.sin(-k_kick * X))
        vel[..., 1] = -k_kick * (g_left * np.cos(k_kick * X) - g_right * np.cos(-k_kick * X))

    phi[..., 3] = phi[..., 0] ** 2 + phi[..., 1] ** 2 + phi[..., 2] ** 2
    return phi, vel


_PRESET_NAME_TO_IDX: Dict[str, int] = {
    "trefoil_knot": 0,
    "z3_braid": 1,
    "hopfion_ring": 2,
    "packet_smash": 3,
    "two_matter_trefoils": 4,
    "matter_antimatter_trefoils": 5,
    "binary_trefoil_inspiral": 6,
}

_INIT_KNOT_WGSL = """
struct InitParams {
    N: u32,
    preset: u32,
    box_size: f32,
    pad0: f32,
};

@group(0) @binding(0) var<uniform> params: InitParams;
@group(0) @binding(1) var<storage, read_write> phi_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> vel_out: array<f32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cconj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

fn eval_trefoil_pq(pt: vec3<f32>, a0: f32) -> vec4<f32> {
    let r = length(pt);
    let rn = r / a0;
    let rn2 = rn * rn;
    let chi = 3.14159265 * exp(-0.6931472 * rn2 - 0.095 * rn2 * rn2);
    var sc_over_r: f32 = 0.0;
    if (r > 1e-5) {
        sc_over_r = sin(chi) / r;
    }
    let Z1 = vec2<f32>(sc_over_r * pt.x, sc_over_r * pt.y);
    let Z2 = vec2<f32>(sc_over_r * pt.z, cos(chi));
    let Z1_2 = cmul(Z1, Z1);
    let Z2_2 = cmul(Z2, Z2);
    let Q_map = cmul(Z1_2, Z1) + Z2_2;
    let P_map = 1.95 * cmul(Z1_2, Z2);
    return vec4<f32>(P_map.x, P_map.y, Q_map.x, Q_map.y);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.N;
    if (gid.x >= n || gid.y >= n || gid.z >= n) {
        return;
    }
    let idx = (gid.z * n + gid.y) * n + gid.x;
    let u_box = params.box_size;
    let uvw = (vec3<f32>(gid) + vec3<f32>(0.5)) / f32(n);
    let p = (uvw - vec3<f32>(0.5)) * u_box;

    var phi = vec3<f32>(0.0);
    var vel = vec3<f32>(0.0);
    let v0: f32 = 0.85;

    if (params.preset == 0u) {
        let a0 = 0.195 * u_box;
        let r = length(p);
        let rn = r / a0;
        let rn2 = rn * rn;
        let chi = 3.14159265 * exp(-0.6931472 * rn2 - 0.095 * rn2 * rn2);
        var sc_over_r: f32 = 0.0;
        if (r > 1e-5) {
            sc_over_r = sin(chi) / r;
        }
        let Z1 = vec2<f32>(sc_over_r * p.x, sc_over_r * p.y);
        let Z2 = vec2<f32>(sc_over_r * p.z, cos(chi));
        let Z1_2 = cmul(Z1, Z1);
        let Z2_2 = cmul(Z2, Z2);
        let Q_map = cmul(Z1_2, Z1) + Z2_2;
        let P_map = 1.95 * cmul(Z1_2, Z2);
        let denom = max(dot(P_map, P_map) + dot(Q_map, Q_map), 1e-12);
        let PQbar = cmul(P_map, cconj(Q_map));
        let n_vec = vec3<f32>(
            2.0 * PQbar.x / denom,
            2.0 * PQbar.y / denom,
            (dot(Q_map, Q_map) - dot(P_map, P_map)) / denom
        );
        phi = v0 * vec3<f32>(n_vec.x, n_vec.y, n_vec.z - 1.0);
    } else if (params.preset == 1u) {
        let r_maj = 0.195 * u_box;
        let r_braid = 0.080 * u_box;
        let w0 = 0.068 * u_box;
        let w_env = 0.098 * u_box;
        let inv_wenv2 = 1.0 / (w_env * w_env);
        var W_sum = vec2<f32>(0.0);
        var hit_core: bool = false;
        for (var st_i: i32 = 0; st_i < 3; st_i = st_i + 1) {
            let off = 0.2222222 + 2.0943951 * f32(st_i);
            var best_d2: f32 = 1e20;
            var best_s: f32 = 0.0;
            for (var k: i32 = 0; k < 48; k = k + 1) {
                let s_ang = 6.2831853 * f32(k) / 48.0;
                let tw = s_ang + off;
                let rad = r_maj + r_braid * cos(tw);
                let c_pt = vec3<f32>(rad * cos(s_ang), rad * sin(s_ang), -r_braid * sin(tw));
                let diff_v = p - c_pt;
                let d2 = dot(diff_v, diff_v);
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_s = s_ang;
                }
            }
            for (var it: i32 = 0; it < 2; it = it + 1) {
                let tw = best_s + off;
                let rad = r_maj + r_braid * cos(tw);
                let bc = vec3<f32>(rad * cos(best_s), rad * sin(best_s), -r_braid * sin(tw));
                let dc_ds = vec3<f32>(
                    -r_braid * sin(tw) * cos(best_s) - rad * sin(best_s),
                    -r_braid * sin(tw) * sin(best_s) + rad * cos(best_s),
                    -r_braid * cos(tw)
                );
                let ds = dot(p - bc, dc_ds) / max(dot(dc_ds, dc_ds), 1e-6);
                best_s = best_s + clamp(ds, -0.09, 0.09);
            }
            let tw = best_s + off;
            let rad = r_maj + r_braid * cos(tw);
            let best_c = vec3<f32>(rad * cos(best_s), rad * sin(best_s), -r_braid * sin(tw));
            let best_t = normalize(vec3<f32>(
                -r_braid * sin(tw) * cos(best_s) - rad * sin(best_s),
                -r_braid * sin(tw) * sin(best_s) + rad * cos(best_s),
                -r_braid * cos(tw)
            ));
            let er_xy = vec3<f32>(cos(best_s), sin(best_s), 0.0);
            let u_hat = normalize(er_xy - dot(er_xy, best_t) * best_t);
            let v_hat = cross(best_t, u_hat);
            let d_vec = p - best_c;
            let u_p = dot(d_vec, u_hat);
            let v_p = dot(d_vec, v_hat);
            let d2 = dot(d_vec, d_vec);
            let xi = d2 * inv_wenv2;
            let A_amp = w0 * exp(-xi - 0.15 * xi * xi);
            let psi_ang = -1.0 * best_s + off;
            let qx = u_p * cos(psi_ang) - v_p * sin(psi_ang);
            let qy = u_p * sin(psi_ang) + v_p * cos(psi_ang);
            if (d2 < 1e-10) {
                hit_core = true;
            } else {
                W_sum = W_sum + (A_amp / d2) * vec2<f32>(qx, qy);
            }
        }
        if (hit_core) {
            phi = vec3<f32>(0.0, 0.0, -2.0 * v0);
        } else {
            let w2 = dot(W_sum, W_sum);
            let den = 1.0 + w2;
            let n_vec = vec3<f32>(2.0 * W_sum.x / den, 2.0 * W_sum.y / den, (1.0 - w2) / den);
            phi = v0 * vec3<f32>(n_vec.x, n_vec.y, n_vec.z - 1.0);
        }
    } else if (params.preset == 2u) {
        let r_maj = 0.190 * u_box;
        let x_off = 0.56 * r_maj;
        let w0 = 0.072 * u_box;
        let w_env = 0.102 * u_box;
        let inv_wenv2 = 1.0 / (w_env * w_env);

        let px1 = p.x + x_off;
        let rc1_2 = px1 * px1 + p.y * p.y;
        let reg1 = rc1_2 / (rc1_2 + 0.18 * r_maj * r_maj);
        let s1 = atan2(p.y, px1);
        let c1 = vec3<f32>(-x_off + r_maj * cos(s1), r_maj * sin(s1), 0.0);
        let u1 = vec3<f32>(cos(s1), sin(s1), 0.0);
        let v1 = vec3<f32>(0.0, 0.0, -1.0);
        let d1v = p - c1;
        let d1_2 = dot(d1v, d1v);
        let xi1 = d1_2 * inv_wenv2;
        let A1 = w0 * reg1 * exp(-xi1 - 0.15 * xi1 * xi1);
        let u1p = dot(d1v, u1);
        let v1p = dot(d1v, v1);
        let psi1 = 2.0 * s1;
        let q1 = vec2<f32>(u1p * cos(psi1) - v1p * sin(psi1), u1p * sin(psi1) + v1p * cos(psi1));

        let px2 = p.x - x_off;
        let rc2_2 = px2 * px2 + p.z * p.z;
        let reg2 = rc2_2 / (rc2_2 + 0.18 * r_maj * r_maj);
        let s2 = atan2(p.z, px2);
        let c2 = vec3<f32>(x_off + r_maj * cos(s2), 0.0, r_maj * sin(s2));
        let u2 = vec3<f32>(cos(s2), 0.0, sin(s2));
        let v2 = vec3<f32>(0.0, 1.0, 0.0);
        let d2v = p - c2;
        let d2_2 = dot(d2v, d2v);
        let xi2 = d2_2 * inv_wenv2;
        let A2 = w0 * reg2 * exp(-xi2 - 0.15 * xi2 * xi2);
        let u2p = dot(d2v, u2);
        let v2p = dot(d2v, v2);
        let psi2 = 2.0 * s2;
        let q2 = vec2<f32>(u2p * cos(psi2) - v2p * sin(psi2), u2p * sin(psi2) + v2p * cos(psi2));

        if (d1_2 < 1e-10 || d2_2 < 1e-10) {
            phi = vec3<f32>(0.0, 0.0, -2.0 * v0);
        } else {
            let W_sum = (A1 / d1_2) * q1 + (A2 / d2_2) * q2;
            let w2 = dot(W_sum, W_sum);
            let den = 1.0 + w2;
            let n_vec = vec3<f32>(2.0 * W_sum.x / den, 2.0 * W_sum.y / den, (1.0 - w2) / den);
            phi = v0 * vec3<f32>(n_vec.x, n_vec.y, n_vec.z - 1.0);
        }
    } else if (params.preset == 4u || params.preset == 5u || params.preset == 6u) {
        let a0 = select(0.132, 0.130, params.preset == 6u) * u_box;
        var P_tot = vec2<f32>(0.0);
        var Q_tot = vec2<f32>(1.0, 0.0);
        if (params.preset == 4u || params.preset == 6u) {
            let x_off = select(0.163, 0.145, params.preset == 6u) * u_box;
            let pq1 = eval_trefoil_pq(vec3<f32>(p.y, p.z, p.x + x_off), a0);
            let yr = -0.30901699 * p.y - 0.95105652 * p.z;
            let zr =  0.95105652 * p.y - 0.30901699 * p.z;
            let pq2 = eval_trefoil_pq(vec3<f32>(yr, zr, p.x - x_off), a0);
            P_tot = cmul(pq1.xy, pq2.zw) + cmul(pq2.xy, pq1.zw);
            Q_tot = cmul(pq1.zw, pq2.zw);
        } else {
            let x_off = 0.138 * u_box;
            let pq1 = eval_trefoil_pq(vec3<f32>(p.y, p.z, p.x + x_off), a0);
            let pq2 = eval_trefoil_pq(vec3<f32>(p.y, p.z, x_off - p.x), a0);
            P_tot = cmul(pq1.xy, pq2.zw) - cmul(pq2.xy, pq1.zw);
            Q_tot = cmul(pq1.zw, pq2.zw);
        }
        let denom = max(dot(P_tot, P_tot) + dot(Q_tot, Q_tot), 1e-12);
        let PQbar = cmul(P_tot, cconj(Q_tot));
        let n_vec = vec3<f32>(
            2.0 * PQbar.x / denom,
            2.0 * PQbar.y / denom,
            (dot(Q_tot, Q_tot) - dot(P_tot, P_tot)) / denom
        );
        phi = v0 * vec3<f32>(n_vec.x, n_vec.y, n_vec.z - 1.0);
        if (params.preset == 6u) {
            let dx_i = u_box / f32(n);
            let x_off = 0.145 * u_box;
            let yp = p.y + dx_i;
            let ym = p.y - dx_i;
            let pq1p = eval_trefoil_pq(vec3<f32>(yp, p.z, p.x + x_off), a0);
            let pq2p = eval_trefoil_pq(vec3<f32>(-0.30901699 * yp - 0.95105652 * p.z, 0.95105652 * yp - 0.30901699 * p.z, p.x - x_off), a0);
            let Pp = cmul(pq1p.xy, pq2p.zw) + cmul(pq2p.xy, pq1p.zw);
            let Qp = cmul(pq1p.zw, pq2p.zw);
            let den_p = max(dot(Pp, Pp) + dot(Qp, Qp), 1e-12);
            let pqbp = cmul(Pp, cconj(Qp));
            let phi_yp = v0 * vec3<f32>(2.0 * pqbp.x / den_p, 2.0 * pqbp.y / den_p, ((dot(Qp, Qp) - dot(Pp, Pp)) / den_p) - 1.0);

            let pq1m = eval_trefoil_pq(vec3<f32>(ym, p.z, p.x + x_off), a0);
            let pq2m = eval_trefoil_pq(vec3<f32>(-0.30901699 * ym - 0.95105652 * p.z, 0.95105652 * ym - 0.30901699 * p.z, p.x - x_off), a0);
            let Pm = cmul(pq1m.xy, pq2m.zw) + cmul(pq2m.xy, pq1m.zw);
            let Qm = cmul(pq1m.zw, pq2m.zw);
            let den_m = max(dot(Pm, Pm) + dot(Qm, Qm), 1e-12);
            let pqbm = cmul(Pm, cconj(Qm));
            let phi_ym = v0 * vec3<f32>(2.0 * pqbm.x / den_m, 2.0 * pqbm.y / den_m, ((dot(Qm, Qm) - dot(Pm, Pm)) / den_m) - 1.0);

            let vy_orb = 0.105 * tanh(p.x / 1.10);
            vel = -vy_orb * ((phi_yp - phi_ym) / (2.0 * dx_i));
        }
    } else {
        let sep = 0.20 * u_box;
        let w0 = 0.065 * u_box;
        let k_kick = 3.2;
        let gL = 2.15 * exp(-((p.x + sep) * (p.x + sep) + p.y * p.y + p.z * p.z) / (2.0 * w0 * w0));
        let gR = 2.15 * exp(-((p.x - sep) * (p.x - sep) + p.y * p.y + p.z * p.z) / (2.0 * w0 * w0));
        phi.x = gL * cos(k_kick * p.x) + gR * cos(-k_kick * p.x);
        phi.y = gL * sin(k_kick * p.x) + gR * sin(-k_kick * p.x);
        phi.z = 0.35 * (gL - gR);
        vel.x = k_kick * (gL * sin(k_kick * p.x) - gR * sin(-k_kick * p.x));
        vel.y = -k_kick * (gL * cos(k_kick * p.x) - gR * cos(-k_kick * p.x));
    }

    let b = idx * 3u;
    phi_out[b] = phi.x;
    phi_out[b + 1u] = phi.y;
    phi_out[b + 2u] = phi.z;
    vel_out[b] = vel.x;
    vel_out[b + 1u] = vel.y;
    vel_out[b + 2u] = vel.z;
}
"""

_REDUCE_RHO_WGSL = """
struct ReduceParams {
    total_vox: u32,
    num_wg: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> phi_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> wg_stats: array<vec2<f32>>;

var<workgroup> s_max: array<f32, 256>;
var<workgroup> s_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid_v: vec3<u32>,
    @builtin(workgroup_id) wid_v: vec3<u32>,
) {
    let lid = lid_v.x;
    let wid = wid_v.x;
    let step_i = params.num_wg * 256u;
    var i = wid * 256u + lid;
    var loc_max: f32 = 0.0;
    var loc_sum: f32 = 0.0;

    loop {
        if (i >= params.total_vox) {
            break;
        }
        let b = i * 3u;
        let px = phi_in[b];
        let py = phi_in[b + 1u];
        let pz = phi_in[b + 2u];
        let rho = px * px + py * py + pz * pz;
        loc_max = max(loc_max, rho);
        loc_sum = loc_sum + rho;
        i = i + step_i;
    }

    s_max[lid] = loc_max;
    s_sum[lid] = loc_sum;
    workgroupBarrier();

    var s: u32 = 128u;
    loop {
        if (s == 0u) {
            break;
        }
        if (lid < s) {
            s_max[lid] = max(s_max[lid], s_max[lid + s]);
            s_sum[lid] = s_sum[lid] + s_sum[lid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (lid == 0u) {
        wg_stats[wid] = vec2<f32>(s_max[0], s_sum[0]);
    }
}
"""


_DEFAULT_HAM_WGSL = (
    "(0.5 * (dt_phi1*dt_phi1 + dt_phi2*dt_phi2 + dt_phi3*dt_phi3)"
    " + 0.5 * (c_w*c_w) * ("
    "dx_phi1*dx_phi1 + dy_phi1*dy_phi1 + dz_phi1*dz_phi1 + "
    "dx_phi2*dx_phi2 + dy_phi2*dy_phi2 + dz_phi2*dz_phi2 + "
    "dx_phi3*dx_phi3 + dy_phi3*dy_phi3 + dz_phi3*dz_phi3)"
    " + 0.5 * (m_eff*m_eff) * (phi1*phi1 + phi2*phi2 + phi3*phi3))"
)


def _build_step_tension_wave_wgsl(ham_wgsl_expr: str) -> str:
    """
    Emits a cross-platform WGSL compute shader advancing causal hyperbolic weave tension
    `d^2 U / dt^2 = c_gw^2 * (nabla^2 U + kappa * rho_{H,0})` with 1st-order spherical
    Sommerfeld / Bayliss-Turkel non-reflecting boundary conditions `(d_t + c_gw d_r)(r U) = 0`.
    """
    expr = ham_wgsl_expr.strip() if ham_wgsl_expr and ham_wgsl_expr.strip() else _DEFAULT_HAM_WGSL
    return f"""
struct UWaveParams {{
    N: u32,
    dx: f32,
    dt: f32,
    half_box: f32,
    kappa: f32,
    c_gw: f32,
    c_w: f32,
    m_eff: f32,
    g4: f32,
    g6: f32,
    k_skyrme: f32,
    pad0: f32,
}};

@group(0) @binding(0) var<uniform> uparams: UWaveParams;
@group(0) @binding(1) var<storage, read> phi_in: array<f32>;
@group(0) @binding(2) var<storage, read> vel_in: array<f32>;
@group(0) @binding(3) var<storage, read> u_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> ut_io: array<f32>;
@group(0) @binding(5) var<storage, read_write> u_out: array<f32>;

fn idx_wrap(x: i32, y: i32, z: i32, n: i32) -> u32 {{
    let xm = (x + n) % n;
    let ym = (y + n) % n;
    let zm = (z + n) % n;
    return u32((zm * n + ym) * n + xm);
}}

fn idx_clamp(x: i32, y: i32, z: i32, n: i32) -> u32 {{
    let xc = clamp(x, 0, n - 1);
    let yc = clamp(y, 0, n - 1);
    let zc = clamp(z, 0, n - 1);
    return u32((zc * n + yc) * n + xc);
}}

fn load_phi(idx: u32) -> vec3<f32> {{
    let b = idx * 3u;
    return vec3<f32>(phi_in[b], phi_in[b + 1u], phi_in[b + 2u]);
}}

fn load_vel(idx: u32) -> vec3<f32> {{
    let b = idx * 3u;
    return vec3<f32>(vel_in[b], vel_in[b + 1u], vel_in[b + 2u]);
}}

fn eval_ham_rest(c: vec3<f32>, v: vec3<f32>, gx: vec3<f32>, gy: vec3<f32>, gz: vec3<f32>) -> f32 {{
    let U: f32 = 0.0;
    let c_w: f32 = uparams.c_w;
    let m_eff: f32 = uparams.m_eff;
    let g4: f32 = uparams.g4;
    let g6: f32 = uparams.g6;
    let k_skyrme: f32 = uparams.k_skyrme;
    let phi1: f32 = c.x; let phi2: f32 = c.y; let phi3: f32 = c.z;
    let dt_phi1: f32 = v.x; let dt_phi2: f32 = v.y; let dt_phi3: f32 = v.z;
    let dx_phi1: f32 = gx.x; let dy_phi1: f32 = gy.x; let dz_phi1: f32 = gz.x;
    let dx_phi2: f32 = gx.y; let dy_phi2: f32 = gy.y; let dz_phi2: f32 = gz.y;
    let dx_phi3: f32 = gx.z; let dy_phi3: f32 = gy.z; let dz_phi3: f32 = gz.z;
    return {expr};
}}

fn sample_trilinear_u(fc: vec3<f32>, n: i32) -> f32 {{
    let i0 = vec3<i32>(floor(fc));
    let t = fc - vec3<f32>(i0);
    let c000 = u_in[idx_clamp(i0.x,     i0.y,     i0.z,     n)];
    let c100 = u_in[idx_clamp(i0.x + 1, i0.y,     i0.z,     n)];
    let c010 = u_in[idx_clamp(i0.x,     i0.y + 1, i0.z,     n)];
    let c110 = u_in[idx_clamp(i0.x + 1, i0.y + 1, i0.z,     n)];
    let c001 = u_in[idx_clamp(i0.x,     i0.y,     i0.z + 1, n)];
    let c101 = u_in[idx_clamp(i0.x + 1, i0.y,     i0.z + 1, n)];
    let c011 = u_in[idx_clamp(i0.x,     i0.y + 1, i0.z + 1, n)];
    let c111 = u_in[idx_clamp(i0.x + 1, i0.y + 1, i0.z + 1, n)];
    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n_u = uparams.N;
    if (gid.x >= n_u || gid.y >= n_u || gid.z >= n_u) {{
        return;
    }}
    let n = i32(n_u);
    let gx = i32(gid.x);
    let gy = i32(gid.y);
    let gz = i32(gid.z);
    let c_idx = u32((gz * n + gy) * n + gx);
    let h = uparams.dx;
    let u_c = u_in[c_idx];

    if (gx == 0 || gx == n - 1 || gy == 0 || gy == n - 1 || gz == 0 || gz == n - 1) {{
        let p_cell = (vec3<f32>(f32(gx), f32(gy), f32(gz)) + vec3<f32>(0.5)) * h - vec3<f32>(uparams.half_box);
        let r = max(length(p_cell), h);
        let r_in = max(r - h, 0.5 * h);
        let p_in = p_cell * (r_in / r);
        let fc = clamp(
            (p_in + vec3<f32>(uparams.half_box)) / h - vec3<f32>(0.5),
            vec3<f32>(0.0),
            vec3<f32>(f32(n - 1) - 1e-4)
        );
        let u_inner = sample_trilinear_u(fc, n);
        let u_target = (r_in / r) * u_inner;
        let alpha = (uparams.c_gw * uparams.dt) / h;
        let u_next = (u_c + alpha * u_target) / (1.0 + alpha);
        ut_io[c_idx] = (u_next - u_c) / max(uparams.dt, 1e-12);
        u_out[c_idx] = u_next;
        return;
    }}

    let inv_dx = 1.0 / h;
    let inv_dx2 = inv_dx * inv_dx;
    let c = load_phi(c_idx);
    let v = load_vel(c_idx);
    let xp = load_phi(idx_wrap(gx + 1, gy, gz, n));
    let xm = load_phi(idx_wrap(gx - 1, gy, gz, n));
    let yp = load_phi(idx_wrap(gx, gy + 1, gz, n));
    let ym = load_phi(idx_wrap(gx, gy - 1, gz, n));
    let zp = load_phi(idx_wrap(gx, gy, gz + 1, n));
    let zm = load_phi(idx_wrap(gx, gy, gz - 1, n));

    let hp = eval_ham_rest(c, v, (xp - c) * inv_dx, (yp - c) * inv_dx, (zp - c) * inv_dx);
    let hm = eval_ham_rest(c, v, (c - xm) * inv_dx, (c - ym) * inv_dx, (c - zm) * inv_dx);
    let rho_h0 = max(0.5 * (hp + hm), 0.0);

    let sum_nb = u_in[idx_clamp(gx - 1, gy, gz, n)] + u_in[idx_clamp(gx + 1, gy, gz, n)]
               + u_in[idx_clamp(gx, gy - 1, gz, n)] + u_in[idx_clamp(gx, gy + 1, gz, n)]
               + u_in[idx_clamp(gx, gy, gz - 1, n)] + u_in[idx_clamp(gx, gy, gz + 1, n)];
    let lap = (sum_nb - 6.0 * u_c) * inv_dx2;
    let acc_u = (uparams.c_gw * uparams.c_gw) * (lap + uparams.kappa * rho_h0);
    let ut_next = ut_io[c_idx] + uparams.dt * acc_u;
    let u_next = u_c + uparams.dt * ut_next;
    ut_io[c_idx] = ut_next;
    u_out[c_idx] = u_next;
}}
"""

_LIMIT_KEYS_FOR_COMPUTE = (
    "max-storage-buffer-binding-size",
    "max-buffer-size",
    "max-storage-buffers-per-shader-stage",
    "max-compute-workgroup-storage-size",
    "max-compute-invocations-per-workgroup",
    "max-compute-workgroups-per-dimension",
)


def _build_required_limits(adapter: wgpu.GPUAdapter) -> Dict[str, int]:
    """Requests the adapter's full storage-buffer and compute limits (up to `2^31 - 4` bytes per binding)."""
    out: Dict[str, int] = {}
    adapter_limits = getattr(adapter, "limits", None)
    if adapter_limits:
        for key in _LIMIT_KEYS_FOR_COMPUTE:
            if key in adapter_limits and adapter_limits[key] is not None:
                out[key] = int(adapter_limits[key])
    return out


class WgpuComputeEngine:
    """
    Hardware-portable Vulkan / DX12 / Metal 3D PDE & Parameter-Sweep Engine.
    Never touches OpenGL contexts (`backend != 'OpenGL'`), making it 100% safe both
    standalone on Rig 2 (`AMD RX 5700 XT`) and alongside `ModernGL` on Rig 1 (`RTX 5080`).
    """

    def __init__(
        self,
        prefer_vendor: Optional[str] = None,
        prefer_igpu_when_shared: bool = False,
    ) -> None:
        self.is_shared_igpu_mode: bool = False
        with _preserve_wgl_context():
            non_gl = _enumerate_non_gl_adapters_isolated()
            chosen = None
            if prefer_vendor:
                pv = prefer_vendor.lower()
                for a in non_gl:
                    if pv in a.summary.lower() and "Vulkan" in a.summary:
                        chosen = a
                        break
                if chosen is None:
                    for a in non_gl:
                        if pv in a.summary.lower():
                            chosen = a
                            break

            if chosen is None and prefer_igpu_when_shared:
                has_discrete = any("DiscreteGPU" in a.summary for a in non_gl)
                if has_discrete:
                    igpu_candidates = [
                        a
                        for a in non_gl
                        if "IntegratedGPU" in a.summary
                        and any(v in a.summary.lower() for v in ("amd", "radeon", "intel", "arc", "iris", "uhd"))
                    ]
                    for a in igpu_candidates:
                        if "Vulkan" in a.summary:
                            chosen = a
                            self.is_shared_igpu_mode = True
                            break
                    if chosen is None and igpu_candidates:
                        chosen = igpu_candidates[0]
                        self.is_shared_igpu_mode = True

            if chosen is None:
                for a in non_gl:
                    if "DiscreteGPU" in a.summary and "Vulkan" in a.summary:
                        chosen = a
                        break
            if chosen is None:
                for a in non_gl:
                    if "DiscreteGPU" in a.summary:
                        chosen = a
                        break
            if chosen is None:
                chosen = non_gl[0] if non_gl else wgpu.gpu.request_adapter_sync(power_preference="high-performance")

            self.adapter = chosen
            req_limits = _build_required_limits(chosen)
            try:
                self.device = chosen.request_device_sync(required_limits=req_limits)
            except Exception:
                self.device = chosen.request_device_sync()
            self.adapter_summary = chosen.summary
        self._pipeline_cache: Dict[str, wgpu.GPUComputePipeline] = {}

    @staticmethod
    def list_available_adapters() -> List[str]:
        return [a.summary for a in _enumerate_non_gl_adapters_isolated()]

    def _get_pipeline(self, wgsl_code: str) -> wgpu.GPUComputePipeline:
        pipe = self._pipeline_cache.get(wgsl_code)
        if pipe is None:
            shader = self.device.create_shader_module(code=wgsl_code)
            pipe = self.device.create_compute_pipeline(
                layout="auto",
                compute={"module": shader, "entry_point": "main"},
            )
            self._pipeline_cache[wgsl_code] = pipe
        return pipe

    def _reduce_rho_gpu(self, phi_buf: wgpu.GPUBuffer, total_vox: int, dx: float) -> Tuple[float, float]:
        """Exact parallel GPU reduction over all `total_vox` voxels (`<= 64 KB` readback even at `512³`)."""
        num_wg = min(max((total_vox + 255) // 256, 1), 8192)
        red_pipe = self._get_pipeline(_REDUCE_RHO_WGSL)
        uni_bytes = struct.pack("<4I", int(total_vox), int(num_wg), 0, 0)
        buf_red_uni = self.device.create_buffer_with_data(
            data=uni_bytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        buf_stats = self.device.create_buffer(
            size=num_wg * 8,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        try:
            bg = self.device.create_bind_group(
                layout=red_pipe.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": buf_red_uni}},
                    {"binding": 1, "resource": {"buffer": phi_buf}},
                    {"binding": 2, "resource": {"buffer": buf_stats}},
                ],
            )
            enc = self.device.create_command_encoder()
            cpass = enc.begin_compute_pass()
            cpass.set_pipeline(red_pipe)
            cpass.set_bind_group(0, bg)
            cpass.dispatch_workgroups(num_wg, 1, 1)
            cpass.end()
            self.device.queue.submit([enc.finish()])

            raw_stats = self.device.queue.read_buffer(buf_stats)
            stats = np.frombuffer(raw_stats, dtype=np.float32).reshape((num_wg, 2))
            peak = float(np.max(stats[:, 0]))
            l2 = float(np.sum(stats[:, 1], dtype=np.float64) * (dx**3))
            return peak, l2
        finally:
            buf_red_uni.destroy()
            buf_stats.destroy()

    def run_3d_knot_evolution(
        self,
        compiled: CompiledMasterAction,
        mod: PhysicsMod,
        grid_n: int = 64,
        n_steps: int = 80,
        preset: str = "trefoil_knot",
        box_size: float = 12.0,
        dt: Optional[float] = None,
        damping: float = 0.0,
        sponge: float = 0.0,
        phi_init_override: Optional[np.ndarray] = None,
        vel_init_override: Optional[np.ndarray] = None,
        readback_full_field: Optional[bool] = None,
        solve_tension_u: bool = False,
        u_field_override: Optional[np.ndarray] = None,
        zero_init_tension_u: bool = False,
        readback_tension_u: bool = False,
    ) -> Dict[str, object]:
        """
        Executes `n_steps` of the compiled 3D WGSL PDE kernel on GPU using packed 3-float (`12` B/voxel)
        storage buffers (`1,610,612,736` B = `1.50 GiB` at `512³`, strictly below `2^31 - 4`).
        """
        t0 = time.perf_counter()
        grid_n = int(grid_n)
        n_steps = int(n_steps)
        if grid_n < 4:
            raise ValueError(f"grid_n must be >= 4, got {grid_n}")
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")
        if self.is_shared_igpu_mode and grid_n > 256 and phi_init_override is None:
            grid_n = 256

        dx = float(box_size) / float(grid_n)
        total_vox = grid_n * grid_n * grid_n
        nbytes_vec3 = total_vox * 12
        if readback_full_field is None:
            readback_full_field = grid_n <= 128

        usage_storage = (
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC
        )
        wg_xy = (grid_n + 7) // 8
        wg_z = (grid_n + 3) // 4

        buf_a: Optional[wgpu.GPUBuffer] = None
        buf_b: Optional[wgpu.GPUBuffer] = None
        buf_vel: Optional[wgpu.GPUBuffer] = None
        buf_u: Optional[wgpu.GPUBuffer] = None
        buf_u_b: Optional[wgpu.GPUBuffer] = None
        buf_ut: Optional[wgpu.GPUBuffer] = None
        buf_u_uni: Optional[wgpu.GPUBuffer] = None
        buf_uni: Optional[wgpu.GPUBuffer] = None
        buf_init_uni: Optional[wgpu.GPUBuffer] = None
        phi0_3_cpu: Optional[np.ndarray] = None
        vel0_3_cpu: Optional[np.ndarray] = None

        try:
            if phi_init_override is not None and vel_init_override is not None:
                phi0_3 = np.ascontiguousarray(phi_init_override[..., :3], dtype=np.float32)
                vel0_3 = np.ascontiguousarray(vel_init_override[..., :3], dtype=np.float32)
                if phi0_3.shape != (grid_n, grid_n, grid_n, 3) or vel0_3.shape != (grid_n, grid_n, grid_n, 3):
                    raise ValueError(
                        f"Override fields must have shape ({grid_n}, {grid_n}, {grid_n}, 3|4); "
                        f"got {phi_init_override.shape} and {vel_init_override.shape}"
                    )
                phi0_3_cpu = phi0_3
                vel0_3_cpu = vel0_3
                init_rho = phi0_3[..., 0] ** 2 + phi0_3[..., 1] ** 2 + phi0_3[..., 2] ** 2
                init_peak = float(np.max(init_rho))
                init_l2 = float(np.sum(init_rho, dtype=np.float64) * (dx**3))

                buf_a = self.device.create_buffer_with_data(data=phi0_3.tobytes(), usage=usage_storage)
                buf_b = self.device.create_buffer(size=nbytes_vec3, usage=usage_storage)
                buf_vel = self.device.create_buffer_with_data(data=vel0_3.tobytes(), usage=usage_storage)
            else:
                buf_a = self.device.create_buffer(size=nbytes_vec3, usage=usage_storage)
                buf_b = self.device.create_buffer(size=nbytes_vec3, usage=usage_storage)
                buf_vel = self.device.create_buffer(size=nbytes_vec3, usage=usage_storage)

                preset_idx = _PRESET_NAME_TO_IDX.get(preset, 3)
                init_pipe = self._get_pipeline(_INIT_KNOT_WGSL)
                init_uni_bytes = struct.pack("<2I2f", grid_n, preset_idx, float(box_size), 0.0)
                buf_init_uni = self.device.create_buffer_with_data(
                    data=init_uni_bytes,
                    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                )
                bg_init = self.device.create_bind_group(
                    layout=init_pipe.get_bind_group_layout(0),
                    entries=[
                        {"binding": 0, "resource": {"buffer": buf_init_uni}},
                        {"binding": 1, "resource": {"buffer": buf_a}},
                        {"binding": 2, "resource": {"buffer": buf_vel}},
                    ],
                )
                enc_init = self.device.create_command_encoder()
                cpass_init = enc_init.begin_compute_pass()
                cpass_init.set_pipeline(init_pipe)
                cpass_init.set_bind_group(0, bg_init)
                cpass_init.dispatch_workgroups(wg_xy, wg_xy, wg_z)
                cpass_init.end()
                self.device.queue.submit([enc_init.finish()])

                init_peak, init_l2 = self._reduce_rho_gpu(buf_a, total_vox, dx)

            if dt is None:
                dt = compute_safe_symplectic_dt(dx, mod, compiled=compiled, peak_rho_est=init_peak)

            g_knot_val = float(getattr(mod.parameters, "g_knot", 0.0))
            retarded_active = False
            use_tension_u32 = 0
            nbytes_u = total_vox * 4
            if u_field_override is not None:
                u_arr_32 = np.ascontiguousarray(u_field_override, dtype=np.float32)
                if u_arr_32.shape != (grid_n, grid_n, grid_n):
                    raise ValueError(f"u_field_override must have shape ({grid_n}, {grid_n}, {grid_n}), got {u_arr_32.shape}")
                buf_u = self.device.create_buffer_with_data(data=u_arr_32.tobytes(), usage=usage_storage)
                use_tension_u32 = 1
            elif solve_tension_u and abs(g_knot_val) > 1.0e-12 and grid_n <= 128:
                if zero_init_tension_u:
                    buf_u = self.device.create_buffer_with_data(
                        data=np.zeros(total_vox, dtype=np.float32).tobytes(),
                        usage=usage_storage,
                    )
                else:
                    if phi0_3_cpu is None or vel0_3_cpu is None:
                        raw_p0 = self.device.queue.read_buffer(buf_a, size=nbytes_vec3)
                        raw_v0 = self.device.queue.read_buffer(buf_vel, size=nbytes_vec3)
                        phi0_3_cpu = np.frombuffer(raw_p0, dtype=np.float32).reshape((grid_n, grid_n, grid_n, 3))
                        vel0_3_cpu = np.frombuffer(raw_v0, dtype=np.float32).reshape((grid_n, grid_n, grid_n, 3))
                    p_c = np.moveaxis(phi0_3_cpu, -1, 0)
                    v_c = np.moveaxis(vel0_3_cpu, -1, 0)
                    grad_c = np.stack(
                        [
                            (np.roll(p_c, -1, axis=3) - np.roll(p_c, 1, axis=3)) / (2.0 * dx),
                            (np.roll(p_c, -1, axis=2) - np.roll(p_c, 1, axis=2)) / (2.0 * dx),
                            (np.roll(p_c, -1, axis=1) - np.roll(p_c, 1, axis=1)) / (2.0 * dx),
                        ],
                        axis=1,
                    )
                    assert compiled.hamiltonian_numpy_fn is not None
                    rho_h0 = compiled.hamiltonian_numpy_fn(
                        p_c,
                        v_c,
                        grad_c,
                        float(mod.parameters.c_w),
                        float(mod.parameters.m_eff),
                        float(mod.parameters.g4),
                        float(mod.parameters.g6),
                        float(mod.parameters.k_skyrme),
                    )
                    u_solved = solve_3d_weave_tension_u_cpu(rho_h0, box_size=box_size, g_knot=g_knot_val)
                    buf_u = self.device.create_buffer_with_data(data=u_solved.tobytes(), usage=usage_storage)
                use_tension_u32 = 1
                if bool(getattr(mod.parameters, "retarded_weave_u", True)) and n_steps > 0:
                    retarded_active = True
                    buf_u_b = self.device.create_buffer(size=nbytes_u, usage=usage_storage)
                    buf_ut = self.device.create_buffer_with_data(
                        data=np.zeros(total_vox, dtype=np.float32).tobytes(),
                        usage=usage_storage,
                    )
            else:
                buf_u = self.device.create_buffer_with_data(
                    data=np.zeros(4, dtype=np.float32).tobytes(),
                    usage=usage_storage,
                )

            pipe = self._get_pipeline(compiled.wgsl_compute_source)
            uniform_bytes = struct.pack(
                "<I10fI",
                int(grid_n),
                float(dx),
                float(dt),
                float(damping),
                float(mod.parameters.c_w),
                float(mod.parameters.m_eff),
                float(mod.parameters.g4),
                float(mod.parameters.g6),
                float(mod.parameters.k_skyrme),
                float(sponge),
                float(box_size) * 0.5,
                int(use_tension_u32),
            )
            buf_uni = self.device.create_buffer_with_data(
                data=uniform_bytes,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )

            u_wave_pipe = None
            bg_u_ab = None
            bg_u_ba = None
            if retarded_active:
                assert buf_u_b is not None and buf_ut is not None
                u_wave_src = _build_step_tension_wave_wgsl(getattr(compiled, "hamiltonian_wgsl_expr", ""))
                u_wave_pipe = self._get_pipeline(u_wave_src)
                c_gw_val = compute_effective_gw_speed(mod, compiled)
                u_uni_bytes = struct.pack(
                    "<I11f",
                    int(grid_n),
                    float(dx),
                    float(dt),
                    float(box_size) * 0.5,
                    float(WEAVE_KAPPA_SCALE * g_knot_val),
                    float(c_gw_val),
                    float(mod.parameters.c_w),
                    float(mod.parameters.m_eff),
                    float(mod.parameters.g4),
                    float(mod.parameters.g6),
                    float(mod.parameters.k_skyrme),
                    0.0,
                )
                buf_u_uni = self.device.create_buffer_with_data(
                    data=u_uni_bytes,
                    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                )
                u_layout = u_wave_pipe.get_bind_group_layout(0)
                bg_u_ab = self.device.create_bind_group(
                    layout=u_layout,
                    entries=[
                        {"binding": 0, "resource": {"buffer": buf_u_uni}},
                        {"binding": 1, "resource": {"buffer": buf_a}},
                        {"binding": 2, "resource": {"buffer": buf_vel}},
                        {"binding": 3, "resource": {"buffer": buf_u}},
                        {"binding": 4, "resource": {"buffer": buf_ut}},
                        {"binding": 5, "resource": {"buffer": buf_u_b}},
                    ],
                )
                bg_u_ba = self.device.create_bind_group(
                    layout=u_layout,
                    entries=[
                        {"binding": 0, "resource": {"buffer": buf_u_uni}},
                        {"binding": 1, "resource": {"buffer": buf_b}},
                        {"binding": 2, "resource": {"buffer": buf_vel}},
                        {"binding": 3, "resource": {"buffer": buf_u_b}},
                        {"binding": 4, "resource": {"buffer": buf_ut}},
                        {"binding": 5, "resource": {"buffer": buf_u}},
                    ],
                )

            bind_layout = pipe.get_bind_group_layout(0)
            u_for_ab = buf_u_b if (retarded_active and buf_u_b is not None) else buf_u
            u_for_ba = buf_u
            bg_ab = self.device.create_bind_group(
                layout=bind_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": buf_uni}},
                    {"binding": 1, "resource": {"buffer": buf_a}},
                    {"binding": 2, "resource": {"buffer": buf_vel}},
                    {"binding": 3, "resource": {"buffer": buf_b}},
                    {"binding": 4, "resource": {"buffer": u_for_ab}},
                ],
            )
            bg_ba = self.device.create_bind_group(
                layout=bind_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": buf_uni}},
                    {"binding": 1, "resource": {"buffer": buf_b}},
                    {"binding": 2, "resource": {"buffer": buf_vel}},
                    {"binding": 3, "resource": {"buffer": buf_a}},
                    {"binding": 4, "resource": {"buffer": u_for_ba}},
                ],
            )

            cur_in_a = True
            if n_steps > 0:
                encoder = self.device.create_command_encoder()
                cpass = encoder.begin_compute_pass()
                if retarded_active and u_wave_pipe is not None:
                    for _ in range(n_steps):
                        cpass.set_pipeline(u_wave_pipe)
                        cpass.set_bind_group(0, bg_u_ab if cur_in_a else bg_u_ba)
                        cpass.dispatch_workgroups(wg_xy, wg_xy, wg_z)
                        cpass.set_pipeline(pipe)
                        cpass.set_bind_group(0, bg_ab if cur_in_a else bg_ba)
                        cpass.dispatch_workgroups(wg_xy, wg_xy, wg_z)
                        cur_in_a = not cur_in_a
                else:
                    cpass.set_pipeline(pipe)
                    for _ in range(n_steps):
                        cpass.set_bind_group(0, bg_ab if cur_in_a else bg_ba)
                        cpass.dispatch_workgroups(wg_xy, wg_xy, wg_z)
                        cur_in_a = not cur_in_a
                cpass.end()
                self.device.queue.submit([encoder.finish()])

            final_buf = buf_a if cur_in_a else buf_b
            final_u_buf = (buf_u if cur_in_a else buf_u_b) if retarded_active else buf_u
            u_final_arr: Optional[np.ndarray] = None
            udot_final_arr: Optional[np.ndarray] = None
            if readback_tension_u and use_tension_u32 == 1 and final_u_buf is not None:
                raw_u = self.device.queue.read_buffer(final_u_buf, size=nbytes_u)
                u_final_arr = np.frombuffer(raw_u, dtype=np.float32).reshape((grid_n, grid_n, grid_n)).copy()
                if buf_ut is not None:
                    raw_ut = self.device.queue.read_buffer(buf_ut, size=nbytes_u)
                    udot_final_arr = np.frombuffer(raw_ut, dtype=np.float32).reshape((grid_n, grid_n, grid_n)).copy()
                else:
                    udot_final_arr = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)

            if readback_full_field:
                phi_final: Optional[np.ndarray] = np.empty((grid_n, grid_n, grid_n, 4), dtype=np.float32)
                vel_final: Optional[np.ndarray] = np.zeros((grid_n, grid_n, grid_n, 4), dtype=np.float32)
                phi_v4 = phi_final.reshape((total_vox, 4))
                vel_v4 = vel_final.reshape((total_vox, 4))

                chunk_vox = 4 * 1024 * 1024  # 48 MiB per staging slice (exact 12-byte voxel multiple)
                final_peak = 0.0
                sum_rho = 0.0
                for vox_start in range(0, total_vox, chunk_vox):
                    count_vox = min(chunk_vox, total_vox - vox_start)
                    off_b = vox_start * 12
                    sz_b = count_vox * 12
                    raw_out = self.device.queue.read_buffer(final_buf, buffer_offset=off_b, size=sz_b)
                    raw_vel = self.device.queue.read_buffer(buf_vel, buffer_offset=off_b, size=sz_b)
                    p_chunk = np.frombuffer(raw_out, dtype=np.float32).reshape((count_vox, 3))
                    v_chunk = np.frombuffer(raw_vel, dtype=np.float32).reshape((count_vox, 3))
                    rho_chunk = p_chunk[:, 0] ** 2 + p_chunk[:, 1] ** 2 + p_chunk[:, 2] ** 2
                    phi_v4[vox_start : vox_start + count_vox, :3] = p_chunk
                    phi_v4[vox_start : vox_start + count_vox, 3] = rho_chunk
                    vel_v4[vox_start : vox_start + count_vox, :3] = v_chunk
                    if count_vox > 0:
                        c_max = float(np.max(rho_chunk))
                        if c_max > final_peak:
                            final_peak = c_max
                        sum_rho += float(np.sum(rho_chunk, dtype=np.float64))

                final_l2 = sum_rho * (dx**3)
                mid_slice_xy = phi_final[grid_n // 2, :, :, 3].copy()
            else:
                final_peak, final_l2 = self._reduce_rho_gpu(final_buf, total_vox, dx)
                slab_bytes = grid_n * grid_n * 12
                slab_offset = (grid_n // 2) * slab_bytes
                raw_slab = self.device.queue.read_buffer(final_buf, buffer_offset=slab_offset, size=slab_bytes)
                slab_3 = np.frombuffer(raw_slab, dtype=np.float32).reshape((grid_n, grid_n, 3))
                mid_slice_xy = (slab_3[..., 0] ** 2 + slab_3[..., 1] ** 2 + slab_3[..., 2] ** 2).copy()
                phi_final = None
                vel_final = None

            retention_ratio = final_peak / max(init_peak, 1e-12)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            return {
                "adapter": self.adapter_summary,
                "grid_n": grid_n,
                "n_steps": n_steps,
                "dx": dx,
                "dt": dt,
                "storage_bytes_per_field": nbytes_vec3,
                "init_peak_rho": init_peak,
                "final_peak_rho": final_peak,
                "retention_ratio": retention_ratio,
                "init_l2_norm": init_l2,
                "final_l2_norm": final_l2,
                "phi_final": phi_final,
                "vel_final": vel_final,
                "u_final": u_final_arr,
                "udot_final": udot_final_arr,
                "mid_slice_xy": mid_slice_xy,
                "elapsed_ms": elapsed_ms,
            }
        finally:
            for buf_obj in (buf_a, buf_b, buf_vel, buf_u, buf_u_b, buf_ut, buf_u_uni, buf_uni, buf_init_uni):
                if buf_obj is not None:
                    buf_obj.destroy()

    def run_knot_stability_phase_sweep(
        self,
        base_mod: PhysicsMod,
        g4_values: List[float],
        g6_values: List[float],
        grid_n: int = 48,
        n_steps: int = 90,
    ) -> Dict[str, object]:
        """
        Sweeps a 2D `(g4, g6)` coupling grid on GPU via `wgpu` to map out where 3D Trefoil knots:
          • Disperse (`retention < 0.45`)
          • Lock into stable breathing 3D solitons (`0.55 <= retention <= 3.5`)
          • Collapse / false-vacuum boil (`retention > 3.5` or `final_l2 > 2.5 * init_l2`)
        """
        t0 = time.perf_counter()
        if not g4_values or not g6_values:
            return {
                "adapter": self.adapter_summary,
                "g4_values": list(g4_values),
                "g6_values": list(g6_values),
                "retention_grid": [],
                "elapsed_ms": 0.0,
            }

        mod_copy = base_mod.clone()
        compiled = compile_master_action(mod_copy)
        grid_n = int(grid_n)
        if self.is_shared_igpu_mode and grid_n > 256:
            grid_n = 256
        if grid_n <= 96:
            phi0, vel0 = build_3d_topological_initial_state(grid_n=grid_n, preset="trefoil_knot")
        else:
            phi0, vel0 = None, None

        retention_grid = np.zeros((len(g6_values), len(g4_values)), dtype=np.float32)
        for i_g6, v_g6 in enumerate(g6_values):
            for j_g4, v_g4 in enumerate(g4_values):
                mod_copy.parameters.g4 = float(v_g4)
                mod_copy.parameters.g6 = float(v_g6)
                res = self.run_3d_knot_evolution(
                    compiled,
                    mod_copy,
                    grid_n=grid_n,
                    n_steps=n_steps,
                    phi_init_override=phi0,
                    vel_init_override=vel0,
                    readback_full_field=False,
                )
                retention_grid[i_g6, j_g4] = float(res["retention_ratio"])

        return {
            "adapter": self.adapter_summary,
            "grid_n": int(grid_n),
            "g4_values": [float(v) for v in g4_values],
            "g6_values": [float(v) for v in g6_values],
            "retention_grid": retention_grid.tolist(),
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }
