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


def compute_safe_symplectic_dt(
    dx: float,
    mod: PhysicsMod,
    compiled: Optional[CompiledMasterAction] = None,
    peak_rho_est: float = 2.8,
) -> float:
    """
    Adaptive Courant–Friedrichs–Lewy + Klein–Gordon mass + nonlinear potential stiffness controller:
    guarantees symplectic Störmer–Verlet stability even for stiff user Lagrangians (`g6 = 500`, high `m_eff`).
    """
    cs_sq = abs(compiled.sound_speed_sq) if compiled is not None and math.isfinite(compiled.sound_speed_sq) else 1.0
    c_eff = max(abs(float(mod.parameters.c_w)), math.sqrt(max(cs_sq, 0.05)), 0.25)
    dt_cfl = 0.22 * dx / c_eff

    rho_ref = max(float(peak_rho_est), 2.8)
    m2 = float(mod.parameters.m_eff) ** 2
    stiff_nl = max(2.5 * abs(float(mod.parameters.g4)) * rho_ref, 2.5 * abs(float(mod.parameters.g6)) * (rho_ref**2))
    omega_max_sq = (12.0 * (c_eff**2)) / max(dx * dx, 1e-8) + m2 + stiff_nl
    dt_stiff = 0.42 / math.sqrt(max(omega_max_sq, 1.0))
    return min(dt_cfl, dt_stiff)


def build_3d_topological_initial_state(
    grid_n: int = 64,
    preset: str = "trefoil_knot",
    box_size: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs C-infinity smooth, periodic-boundary-clean 3D topological knot/braid initial fields
    `(phi_init, vel_init)` of shape `(N, N, N, 4)` (`float32`), matching `_INIT_KNOT_GLSL` in Arena 1!
      • `trefoil_knot`:  C-infinity line-integral 3D `(2, 3)` Torus Trefoil Knot (`boundary rho < 1e-5`)!
      • `z3_braid`:      3-Ribbon `Z_3` Cyclic Generation Braid (Koide `theta0 = 2/9` triad!)
      • `hopfion_ring`:  Linked 3D Hopfion / Superfluid Vortex Ring pair
      • `packet_smash`:  Head-on 3D relativistic wave-packet collision (`Observer Channel A/B` interference)
    """
    idx_lin = (np.arange(grid_n, dtype=np.float32) + 0.5) / float(grid_n)
    coords = (idx_lin - 0.5) * float(box_size)
    Z, Y, X = np.meshgrid(coords, coords, coords, indexing="ij")  # matches (z * N + y) * N + x memory layout!

    phi = np.zeros((grid_n, grid_n, grid_n, 4), dtype=np.float32)
    vel = np.zeros((grid_n, grid_n, grid_n, 4), dtype=np.float32)

    if preset == "trefoil_knot":
        # Exact C-infinity smooth line-integral (2, 3) Trefoil Knot (zero Voronoi seams, boundary rho < 1e-5!)
        n_samp = 96
        s_vals = np.linspace(0.0, 2.0 * math.pi, n_samp, endpoint=False, dtype=np.float32)
        scale_r = 0.112 * box_size
        tube_w = 0.052 * box_size
        inv_2w2 = 1.0 / (2.0 * tube_w * tube_w)

        cx = scale_r * (np.sin(s_vals) + 2.0 * np.sin(2.0 * s_vals))
        cy = scale_r * (np.cos(s_vals) - 2.0 * np.cos(2.0 * s_vals))
        cz = scale_r * (-1.35 * np.sin(3.0 * s_vals))
        cos3 = np.cos(3.0 * s_vals)
        sin3 = np.sin(3.0 * s_vals)
        sin2 = np.sin(2.0 * s_vals)

        sum_c = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
        sum_s = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
        sum_z = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)

        for k in range(n_samp):
            d2 = (X - cx[k]) ** 2 + (Y - cy[k]) ** 2 + (Z - cz[k]) ** 2
            w = np.exp(-d2 * inv_2w2)
            sum_c += w * cos3[k]
            sum_s += w * sin3[k]
            sum_z += w * sin2[k]

        norm_k = 0.37
        phi[..., 0] = norm_k * sum_c
        phi[..., 1] = norm_k * sum_s
        phi[..., 2] = 0.45 * norm_k * sum_z
        # Coleman Q-Ball internal phase rotation (omega0 = 2.25 < m_eff = 2.50!)
        omega0 = 2.25
        vel[..., 0] = -omega0 * phi[..., 1]
        vel[..., 1] = omega0 * phi[..., 0]
        vel[..., 2] = 0.35 * omega0 * phi[..., 2]

    elif preset == "z3_braid":
        theta0 = 2.0 / 9.0
        r_braid = 0.130 * box_size
        tube_w = 0.048 * box_size
        kz = 1.35 * (2.0 * math.pi) / box_size
        yn = Y / (0.33 * box_size)
        y_env = np.exp(-(yn**6))
        omega0 = 2.20
        for strand in range(3):
            ang = kz * Y + theta0 + (2.0 * math.pi * strand / 3.0)
            cx = r_braid * np.cos(ang)
            cz = r_braid * np.sin(ang)
            d2 = (X - cx) ** 2 + (Z - cz) ** 2
            env = 1.55 * y_env * np.exp(-d2 / (2.0 * (tube_w**2)))
            phi[..., strand] = env * (0.85 + 0.20 * np.cos(ang))
            vel[..., strand] = omega0 * env * np.sin(ang)

    elif preset == "hopfion_ring":
        r_maj = 0.165 * box_size
        tube_w = 0.046 * box_size
        d_ring1 = (np.sqrt((X + 0.075 * box_size) ** 2 + Y**2) - r_maj) ** 2 + Z**2
        d_ring2 = (np.sqrt((X - 0.075 * box_size) ** 2 + Z**2) - r_maj) ** 2 + Y**2
        a1 = 1.62 * np.exp(-d_ring1 / (2.0 * (tube_w**2)))
        a2 = 1.62 * np.exp(-d_ring2 / (2.0 * (tube_w**2)))
        ang1 = np.atan2(Y, X + 0.075 * box_size)
        ang2 = np.atan2(Z, X - 0.075 * box_size)
        phi[..., 0] = a1 * np.cos(2.0 * ang1) + a2 * np.cos(2.0 * ang2)
        phi[..., 1] = a1 * np.sin(2.0 * ang1) - a2 * np.sin(2.0 * ang2)
        phi[..., 2] = 0.65 * (a1 - a2)
        omega0 = 2.20
        vel[..., 0] = -omega0 * phi[..., 1]
        vel[..., 1] = omega0 * phi[..., 0]

    else:  # packet_smash
        sep = 0.20 * box_size
        w0 = 0.065 * box_size
        k_kick = 3.2
        g_left = 1.70 * np.exp(-((X + sep) ** 2 + Y**2 + Z**2) / (2.0 * (w0**2)))
        g_right = 1.70 * np.exp(-((X - sep) ** 2 + Y**2 + Z**2) / (2.0 * (w0**2)))
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

    if (params.preset == 0u) {
        let scale_r = 0.112 * u_box;
        let tube_w = 0.052 * u_box;
        let inv_2w2 = 1.0 / (2.0 * tube_w * tube_w);
        var sum_cs = vec2<f32>(0.0);
        var sum_z: f32 = 0.0;
        for (var k: i32 = 0; k < 96; k = k + 1) {
            let s = 6.2831853 * f32(k) / 96.0;
            let c = scale_r * vec3<f32>(
                sin(s) + 2.0 * sin(2.0 * s),
                cos(s) - 2.0 * cos(2.0 * s),
                -1.35 * sin(3.0 * s)
            );
            let diff = p - c;
            let w = exp(-dot(diff, diff) * inv_2w2);
            sum_cs = sum_cs + w * vec2<f32>(cos(3.0 * s), sin(3.0 * s));
            sum_z = sum_z + w * sin(2.0 * s);
        }
        let norm_k = 0.37;
        phi.x = norm_k * sum_cs.x;
        phi.y = norm_k * sum_cs.y;
        phi.z = 0.45 * norm_k * sum_z;
        let omega0 = 2.25;
        vel.x = -omega0 * phi.y;
        vel.y = omega0 * phi.x;
        vel.z = 0.35 * omega0 * phi.z;
    } else if (params.preset == 1u) {
        let theta0 = 0.2222222;
        let r_braid = 0.130 * u_box;
        let tube_w = 0.048 * u_box;
        let kz = 1.35 * 6.2831853 / u_box;
        let yn = p.y / (0.33 * u_box);
        let yn2 = yn * yn;
        let y_env = exp(-(yn2 * yn2 * yn2));
        let omega0 = 2.20;
        for (var s: i32 = 0; s < 3; s = s + 1) {
            let ang = kz * p.y + theta0 + 2.0943951 * f32(s);
            let cx = r_braid * cos(ang);
            let cz = r_braid * sin(ang);
            let dx = p.x - cx;
            let dz = p.z - cz;
            let d2 = dx * dx + dz * dz;
            let env = 1.55 * y_env * exp(-d2 / (2.0 * tube_w * tube_w));
            let pv = env * (0.85 + 0.20 * cos(ang));
            let vv = omega0 * env * sin(ang);
            if (s == 0) {
                phi.x = pv;
                vel.x = vv;
            } else if (s == 1) {
                phi.y = pv;
                vel.y = vv;
            } else {
                phi.z = pv;
                vel.z = vv;
            }
        }
    } else if (params.preset == 2u) {
        let r_maj = 0.165 * u_box;
        let tube_w = 0.046 * u_box;
        let dr1 = length(vec2<f32>(p.x + 0.075 * u_box, p.y)) - r_maj;
        let dr2 = length(vec2<f32>(p.x - 0.075 * u_box, p.z)) - r_maj;
        let d1 = dr1 * dr1 + p.z * p.z;
        let d2 = dr2 * dr2 + p.y * p.y;
        let a1 = 1.62 * exp(-d1 / (2.0 * tube_w * tube_w));
        let a2 = 1.62 * exp(-d2 / (2.0 * tube_w * tube_w));
        let ang1 = atan2(p.y, p.x + 0.075 * u_box);
        let ang2 = atan2(p.z, p.x - 0.075 * u_box);
        phi.x = a1 * cos(2.0 * ang1) + a2 * cos(2.0 * ang2);
        phi.y = a1 * sin(2.0 * ang1) - a2 * sin(2.0 * ang2);
        phi.z = 0.65 * (a1 - a2);
        let omega0 = 2.20;
        vel.x = -omega0 * phi.y;
        vel.y = omega0 * phi.x;
    } else {
        let sep = 0.20 * u_box;
        let w0 = 0.065 * u_box;
        let k_kick = 3.2;
        let gL = 1.70 * exp(-((p.x + sep) * (p.x + sep) + p.y * p.y + p.z * p.z) / (2.0 * w0 * w0));
        let gR = 1.70 * exp(-((p.x - sep) * (p.x - sep) + p.y * p.y + p.z * p.z) / (2.0 * w0 * w0));
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
        phi_init_override: Optional[np.ndarray] = None,
        vel_init_override: Optional[np.ndarray] = None,
        readback_full_field: Optional[bool] = None,
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
        buf_uni: Optional[wgpu.GPUBuffer] = None
        buf_init_uni: Optional[wgpu.GPUBuffer] = None

        try:
            if phi_init_override is not None and vel_init_override is not None:
                phi0_3 = np.ascontiguousarray(phi_init_override[..., :3], dtype=np.float32)
                vel0_3 = np.ascontiguousarray(vel_init_override[..., :3], dtype=np.float32)
                if phi0_3.shape != (grid_n, grid_n, grid_n, 3) or vel0_3.shape != (grid_n, grid_n, grid_n, 3):
                    raise ValueError(
                        f"Override fields must have shape ({grid_n}, {grid_n}, {grid_n}, 3|4); "
                        f"got {phi_init_override.shape} and {vel_init_override.shape}"
                    )
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

            pipe = self._get_pipeline(compiled.wgsl_compute_source)
            uniform_bytes = struct.pack(
                "<I7f",
                int(grid_n),
                float(dx),
                float(dt),
                float(damping),
                float(mod.parameters.c_w),
                float(mod.parameters.m_eff),
                float(mod.parameters.g4),
                float(mod.parameters.g6),
            )
            buf_uni = self.device.create_buffer_with_data(
                data=uniform_bytes,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )

            bind_layout = pipe.get_bind_group_layout(0)
            bg_ab = self.device.create_bind_group(
                layout=bind_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": buf_uni}},
                    {"binding": 1, "resource": {"buffer": buf_a}},
                    {"binding": 2, "resource": {"buffer": buf_vel}},
                    {"binding": 3, "resource": {"buffer": buf_b}},
                ],
            )
            bg_ba = self.device.create_bind_group(
                layout=bind_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": buf_uni}},
                    {"binding": 1, "resource": {"buffer": buf_b}},
                    {"binding": 2, "resource": {"buffer": buf_vel}},
                    {"binding": 3, "resource": {"buffer": buf_a}},
                ],
            )

            cur_in_a = True
            if n_steps > 0:
                encoder = self.device.create_command_encoder()
                cpass = encoder.begin_compute_pass()
                cpass.set_pipeline(pipe)
                for _ in range(n_steps):
                    cpass.set_bind_group(0, bg_ab if cur_in_a else bg_ba)
                    cpass.dispatch_workgroups(wg_xy, wg_xy, wg_z)
                    cur_in_a = not cur_in_a
                cpass.end()
                self.device.queue.submit([encoder.finish()])

            final_buf = buf_a if cur_in_a else buf_b

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
                "mid_slice_xy": mid_slice_xy,
                "elapsed_ms": elapsed_ms,
            }
        finally:
            for buf_obj in (buf_a, buf_b, buf_vel, buf_uni, buf_init_uni):
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
