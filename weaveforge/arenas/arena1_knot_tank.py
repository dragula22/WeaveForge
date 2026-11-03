"""
Arena 1 — The 3D Quantum & Topological Knot Tank (`weaveforge/arenas/arena1_knot_tank.py`)
==========================================================================================
Vertical Slice #1 & Micro/Topological Proving Ground:
  • Runs the exact `SymPy`-compiled 3D Relativistic Triad Wave / Soliton PDE on GPU at
    `128³` (Ultra-Fast), `256³` (Real-Time 60–120+ FPS on RTX 5080!), or `512³` (134M-Voxel Max Crunch!).
  • Zero-copy GPU pipeline:
      1. `_INIT_KNOT_GLSL` initializes C-infinity 3D Trefoil Knots, 3-Ribbon `Z_3` Generation Braids
         (`Koide theta0 = 2/9`), Linked Hopfions, or 3D Wave-Packet Collisions directly in VRAM in `<3 ms`!
      2. Auto-compiled `SymPy -> GLSL 430` compute shader advances `(phi1, phi2, phi3)` symplectically
         with an adaptive Courant + potential-stiffness timestep controller.
      3. Global 3D GPU parallel reduction kernel computes live 3D `peak_rho`, `3D L2 Norm`, AND
         `3D Symplectic Hamiltonian Energy H` conservation in `<0.06 ms`!
      4. Real-time 3D Volumetric Ray-Marcher renders glowing topological tubes with 3D gradient
         Fresnel lighting, Option A+ 'Luminous Sapphire Slate' studio framing, interactive slice
         cutaway, and 3 distinct color palettes (Phase Weave, Rainbow Phase, Thermal Energy Density)!
"""

from __future__ import annotations

from collections import deque
import math
from typing import Deque, Optional

import moderngl
import numpy as np
from OpenGL import GL

from weaveforge.compute.wgpu_engine import (
    WEAVE_KAPPA_SCALE,
    compute_effective_gw_speed,
    compute_safe_symplectic_dt,
)
from weaveforge.core.camera import Camera3D
from weaveforge.core.frameworks import PhysicsMod
from weaveforge.core.symbolic_compiler import CompiledMasterAction


_GL_GPU_MEM_INFO_CURRENT_AVAILABLE_NVX = 0x9049
_XFER_CHUNK_BYTES = 16 * 1024 * 1024  # 16 MiB reusable staging window (4,194,304 uint32 / 8,388,608 fp16)
_XFER_CHUNK_UINTS = _XFER_CHUNK_BYTES // 4
_XFER_CHUNK_FLOATS_FP16 = _XFER_CHUNK_UINTS * 2

_DEFAULT_HAMILTONIAN_GLSL = (
    "(0.5 * (dt_phi1*dt_phi1 + dt_phi2*dt_phi2 + dt_phi3*dt_phi3)"
    " + 0.5 * (c_w*c_w) * ("
    "dx_phi1*dx_phi1 + dy_phi1*dy_phi1 + dz_phi1*dz_phi1 + "
    "dx_phi2*dx_phi2 + dy_phi2*dy_phi2 + dz_phi2*dz_phi2 + "
    "dx_phi3*dx_phi3 + dy_phi3*dy_phi3 + dz_phi3*dz_phi3)"
    " + 0.5 * (m_eff*m_eff) * rho"
    " - 0.25 * g4 * (rho*rho)"
    " + (1.0 / 6.0) * g6 * (rho*rho*rho))"
)


def _copy_buffer_sub_data(
    src_buf: moderngl.Buffer,
    dst_stage_buf: moderngl.Buffer,
    src_offset: int,
    dst_offset: int,
    size_bytes: int,
) -> None:
    """Copies GPU buffer range into a `dynamic=True` staging buffer without demoting the source SSBO."""
    if size_bytes <= 0:
        return
    GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, int(src_buf.glo))
    GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, int(dst_stage_buf.glo))
    GL.glCopyBufferSubData(
        GL.GL_COPY_READ_BUFFER,
        GL.GL_COPY_WRITE_BUFFER,
        int(src_offset),
        int(dst_offset),
        int(size_bytes),
    )
    GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, 0)
    GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, 0)


def _query_available_vram_mb() -> Optional[float]:
    """Queries free dedicated GDDR7 VRAM (MiB) via `GL_NVX_gpu_memory_info` when supported."""
    try:
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass
        avail_kb = int(GL.glGetIntegerv(_GL_GPU_MEM_INFO_CURRENT_AVAILABLE_NVX))
        if GL.glGetError() != GL.GL_NO_ERROR or avail_kb <= 0:
            return None
        return float(avail_kb) / 1024.0
    except Exception:
        return None


def _unbind_arena1_gl_slots() -> None:
    """Unbinds indexed SSBO, Image3D, and Texture3D binding points so WDDM frees deleted GDDR7 VRAM immediately."""
    try:
        for slot in (0, 1, 2, 3):
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, slot, 0)
        GL.glBindImageTexture(0, 0, 0, GL.GL_TRUE, 0, GL.GL_WRITE_ONLY, GL.GL_RGBA16F)
        GL.glBindImageTexture(1, 0, 0, GL.GL_TRUE, 0, GL.GL_WRITE_ONLY, GL.GL_RG32F)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, 0)
        GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, 0)
    except Exception:
        pass


KNOT_PRESETS = [
    ("trefoil_knot", "3D (2,3) Trefoil Particle Knot"),
    ("z3_braid", "3-Ribbon Z_3 Generation Braid (Koide 2/9)"),
    ("hopfion_ring", "Linked 3D Hopfion Vortex Rings"),
    ("packet_smash", "3D Wave-Packet Collider (Channel A/B)"),
    ("two_matter_trefoils", "Two Matter Trefoils (+7 / +7): Emergent 1/r^2 Gravity & Scattering"),
    ("matter_antimatter_trefoils", "Matter + Antimatter Trefoils (+7 / -7): Gravitational Fall & Annihilation"),
    ("binary_trefoil_inspiral", "Binary Trefoil Inspiral (+7 / +7): Orbiting Quadrupole GW Ripples"),
]

KNOT_COLOR_MODES = [
    "Sapphire-Gold Phase Weave",
    "Topological Rainbow Phase",
    "Thermal Energy Density",
]


_INIT_KNOT_GLSL = """#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

layout(std430, binding = 0) writeonly buffer PhiBuf { float phi_out[]; };
layout(std430, binding = 1) writeonly buffer VelBuf { float vel_out[]; };
layout(rgba16f, binding = 0) writeonly uniform image3D u_vol_tex;

uniform int u_N;
uniform int u_preset;   // 0=trefoil, 1=z3_braid, 2=hopfion, 3=packet_smash
uniform float u_box;

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 cconj(vec2 a) {
    return vec2(a.x, -a.y);
}

vec4 eval_trefoil_pq(vec3 pt, float a0) {
    float r = length(pt);
    float rn = r / a0;
    float rn2 = rn * rn;
    float chi = 3.14159265 * exp(-0.6931472 * rn2 - 0.095 * rn2 * rn2);
    float sc_over_r = (r > 1e-5) ? (sin(chi) / r) : 0.0;
    vec2 Z1 = vec2(sc_over_r * pt.x, sc_over_r * pt.y);
    vec2 Z2 = vec2(sc_over_r * pt.z, cos(chi));
    vec2 Z1_2 = cmul(Z1, Z1);
    vec2 Z2_2 = cmul(Z2, Z2);
    vec2 Q = cmul(Z1_2, Z1) + Z2_2;
    vec2 P = 1.95 * cmul(Z1_2, Z2);
    return vec4(P.x, P.y, Q.x, Q.y);
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    if (gid.x >= u_N || gid.y >= u_N || gid.z >= u_N) return;

    int N = u_N;
    int idx = (gid.z * N + gid.y) * N + gid.x;
    vec3 uvw = (vec3(gid) + 0.5) / float(N);
    vec3 p = (uvw - 0.5) * u_box;

    vec3 phi = vec3(0.0);
    vec3 vel = vec3(0.0);
    const float v0 = 0.85;

    if (u_preset == 0) {
        // Sutcliffe (2007, Eq. 3.7) T_(3,2) Trefoil Knot Rational Map (exact Hopf charge Q_H = +7!)
        float a0 = 0.195 * u_box;
        float r = length(p);
        float rn = r / a0;
        float rn2 = rn * rn;
        float chi = 3.14159265 * exp(-0.6931472 * rn2 - 0.095 * rn2 * rn2);
        float sc_over_r = (r > 1e-5) ? (sin(chi) / r) : 0.0;
        vec2 Z1 = vec2(sc_over_r * p.x, sc_over_r * p.y);
        vec2 Z2 = vec2(sc_over_r * p.z, cos(chi));
        vec2 Z1_2 = cmul(Z1, Z1);
        vec2 Z2_2 = cmul(Z2, Z2);
        vec2 Q = cmul(Z1_2, Z1) + Z2_2;
        vec2 P = 1.95 * cmul(Z1_2, Z2);
        float denom = max(dot(P, P) + dot(Q, Q), 1e-12);
        vec2 PQbar = cmul(P, cconj(Q));
        vec3 n_vec = vec3(
            2.0 * PQbar.x / denom,
            2.0 * PQbar.y / denom,
            (dot(Q, Q) - dot(P, P)) / denom
        );
        phi = v0 * vec3(n_vec.x, n_vec.y, n_vec.z - 1.0);
    }
    else if (u_preset == 1) {
        // 3-Ribbon Z_3 Cyclic Generation Braid Link (Koide theta0 = 2/9 rad, exact Hopf charge Q_H = +9!)
        float r_maj = 0.195 * u_box;
        float r_braid = 0.080 * u_box;
        float w0 = 0.068 * u_box;
        float w_env = 0.098 * u_box;
        float inv_wenv2 = 1.0 / (w_env * w_env);
        const int M = 48;
        vec2 W_sum = vec2(0.0);
        bool hit_core = false;
        for (int st_i = 0; st_i < 3; ++st_i) {
            float off = 0.2222222 + 2.0943951 * float(st_i);
            float best_d2 = 1e20;
            float best_s = 0.0;
            for (int k = 0; k < M; ++k) {
                float s = 6.2831853 * float(k) / float(M);
                float tw = s + off;
                float rad = r_maj + r_braid * cos(tw);
                vec3 c = vec3(rad * cos(s), rad * sin(s), -r_braid * sin(tw));
                vec3 diff = p - c;
                float d2 = dot(diff, diff);
                if (d2 < best_d2) { best_d2 = d2; best_s = s; }
            }
            for (int it = 0; it < 2; ++it) {
                float tw = best_s + off;
                float rad = r_maj + r_braid * cos(tw);
                vec3 bc = vec3(rad * cos(best_s), rad * sin(best_s), -r_braid * sin(tw));
                vec3 dc_ds = vec3(
                    -r_braid * sin(tw) * cos(best_s) - rad * sin(best_s),
                    -r_braid * sin(tw) * sin(best_s) + rad * cos(best_s),
                    -r_braid * cos(tw)
                );
                float ds = dot(p - bc, dc_ds) / max(dot(dc_ds, dc_ds), 1e-6);
                best_s += clamp(ds, -0.09, 0.09);
            }
            float tw = best_s + off;
            float rad = r_maj + r_braid * cos(tw);
            vec3 best_c = vec3(rad * cos(best_s), rad * sin(best_s), -r_braid * sin(tw));
            vec3 best_t = normalize(vec3(
                -r_braid * sin(tw) * cos(best_s) - rad * sin(best_s),
                -r_braid * sin(tw) * sin(best_s) + rad * cos(best_s),
                -r_braid * cos(tw)
            ));
            vec3 er_xy = vec3(cos(best_s), sin(best_s), 0.0);
            vec3 u_hat = normalize(er_xy - dot(er_xy, best_t) * best_t);
            vec3 v_hat = cross(best_t, u_hat);
            vec3 d_vec = p - best_c;
            float u_p = dot(d_vec, u_hat);
            float v_p = dot(d_vec, v_hat);
            float d2 = dot(d_vec, d_vec);
            float xi = d2 * inv_wenv2;
            float A = w0 * exp(-xi - 0.15 * xi * xi);
            float psi = -1.0 * best_s + off;
            float qx = u_p * cos(psi) - v_p * sin(psi);
            float qy = u_p * sin(psi) + v_p * cos(psi);
            if (d2 < 1e-10) { hit_core = true; }
            else { W_sum += (A / d2) * vec2(qx, qy); }
        }
        if (hit_core) {
            phi = vec3(0.0, 0.0, -2.0 * v0);
        } else {
            float w2 = dot(W_sum, W_sum);
            float den = 1.0 + w2;
            vec3 n_vec = vec3(2.0 * W_sum.x / den, 2.0 * W_sum.y / den, (1.0 - w2) / den);
            phi = v0 * vec3(n_vec.x, n_vec.y, n_vec.z - 1.0);
        }
    }
    else if (u_preset == 2) {
        // Symmetric Orthogonal Linked 3D Hopfion Rings (6_1^2 link, exact Hopf charge Q_H = -6!)
        float r_maj = 0.190 * u_box;
        float x_off = 0.56 * r_maj;
        float w0 = 0.072 * u_box;
        float w_env = 0.102 * u_box;
        float inv_wenv2 = 1.0 / (w_env * w_env);

        float px1 = p.x + x_off;
        float rc1_2 = px1 * px1 + p.y * p.y;
        float reg1 = rc1_2 / (rc1_2 + 0.18 * r_maj * r_maj);
        float s1 = atan(p.y, px1);
        vec3 c1 = vec3(-x_off + r_maj * cos(s1), r_maj * sin(s1), 0.0);
        vec3 u1 = vec3(cos(s1), sin(s1), 0.0);
        vec3 v1 = vec3(0.0, 0.0, -1.0);
        vec3 d1v = p - c1;
        float d1_2 = dot(d1v, d1v);
        float xi1 = d1_2 * inv_wenv2;
        float A1 = w0 * reg1 * exp(-xi1 - 0.15 * xi1 * xi1);
        float u1p = dot(d1v, u1); float v1p = dot(d1v, v1);
        float psi1 = 2.0 * s1;
        vec2 q1 = vec2(u1p * cos(psi1) - v1p * sin(psi1), u1p * sin(psi1) + v1p * cos(psi1));

        float px2 = p.x - x_off;
        float rc2_2 = px2 * px2 + p.z * p.z;
        float reg2 = rc2_2 / (rc2_2 + 0.18 * r_maj * r_maj);
        float s2 = atan(p.z, px2);
        vec3 c2 = vec3(x_off + r_maj * cos(s2), 0.0, r_maj * sin(s2));
        vec3 u2 = vec3(cos(s2), 0.0, sin(s2));
        vec3 v2 = vec3(0.0, 1.0, 0.0);
        vec3 d2v = p - c2;
        float d2_2 = dot(d2v, d2v);
        float xi2 = d2_2 * inv_wenv2;
        float A2 = w0 * reg2 * exp(-xi2 - 0.15 * xi2 * xi2);
        float u2p = dot(d2v, u2); float v2p = dot(d2v, v2);
        float psi2 = 2.0 * s2;
        vec2 q2 = vec2(u2p * cos(psi2) - v2p * sin(psi2), u2p * sin(psi2) + v2p * cos(psi2));

        if (d1_2 < 1e-10 || d2_2 < 1e-10) {
            phi = vec3(0.0, 0.0, -2.0 * v0);
        } else {
            vec2 W_sum = (A1 / d1_2) * q1 + (A2 / d2_2) * q2;
            float w2 = dot(W_sum, W_sum);
            float den = 1.0 + w2;
            vec3 n_vec = vec3(2.0 * W_sum.x / den, 2.0 * W_sum.y / den, (1.0 - w2) / den);
            phi = v0 * vec3(n_vec.x, n_vec.y, n_vec.z - 1.0);
        }
    }
    else if (u_preset == 4 || u_preset == 5 || u_preset == 6) {
        // Projective Coaxial Two-Knot Superposition (Preset 4/6: +7/+7 Q_H=+14; Preset 5: +7/-7 Q_H=0!)
        float a0 = ((u_preset == 6) ? 0.130 : 0.132) * u_box;
        vec2 P_tot;
        vec2 Q_tot;
        if (u_preset == 4 || u_preset == 6) {
            float x_off = ((u_preset == 6) ? 0.145 : 0.163) * u_box;
            vec4 pq1 = eval_trefoil_pq(vec3(p.y, p.z, p.x + x_off), a0);
            float yr = -0.30901699 * p.y - 0.95105652 * p.z;
            float zr =  0.95105652 * p.y - 0.30901699 * p.z;
            vec4 pq2 = eval_trefoil_pq(vec3(yr, zr, p.x - x_off), a0);
            P_tot = cmul(pq1.xy, pq2.zw) + cmul(pq2.xy, pq1.zw);
            Q_tot = cmul(pq1.zw, pq2.zw);
        } else {
            float x_off = 0.138 * u_box;
            vec4 pq1 = eval_trefoil_pq(vec3(p.y, p.z, p.x + x_off), a0);
            vec4 pq2 = eval_trefoil_pq(vec3(p.y, p.z, x_off - p.x), a0);
            P_tot = cmul(pq1.xy, pq2.zw) - cmul(pq2.xy, pq1.zw);
            Q_tot = cmul(pq1.zw, pq2.zw);
        }
        float denom = max(dot(P_tot, P_tot) + dot(Q_tot, Q_tot), 1e-12);
        vec2 PQbar = cmul(P_tot, cconj(Q_tot));
        vec3 n_vec = vec3(
            2.0 * PQbar.x / denom,
            2.0 * PQbar.y / denom,
            (dot(Q_tot, Q_tot) - dot(P_tot, P_tot)) / denom
        );
        phi = v0 * vec3(n_vec.x, n_vec.y, n_vec.z - 1.0);
    }
    else {
        // Head-on 3D Relativistic Wave-Packet Collision & Interference Fringes
        float sep = 0.20 * u_box;
        float w0 = 0.065 * u_box;
        float k_kick = 3.2;
        float gL = 2.15 * exp(-((p.x + sep)*(p.x + sep) + p.y*p.y + p.z*p.z) / (2.0 * w0 * w0));
        float gR = 2.15 * exp(-((p.x - sep)*(p.x - sep) + p.y*p.y + p.z*p.z) / (2.0 * w0 * w0));
        phi.x = gL * cos(k_kick * p.x) + gR * cos(-k_kick * p.x);
        phi.y = gL * sin(k_kick * p.x) + gR * sin(-k_kick * p.x);
        phi.z = 0.35 * (gL - gR);
        vel.x =  k_kick * (gL * sin(k_kick * p.x) - gR * sin(-k_kick * p.x));
        vel.y = -k_kick * (gL * cos(k_kick * p.x) - gR * cos(-k_kick * p.x));
    }

    float rho = dot(phi, phi);
    int b = idx * 3;
    phi_out[b]     = phi.x;
    phi_out[b + 1] = phi.y;
    phi_out[b + 2] = phi.z;
    vel_out[b]     = vel.x;
    vel_out[b + 1] = vel.y;
    vel_out[b + 2] = vel.z;
    imageStore(u_vol_tex, gid, vec4(rho, phi.x, phi.y, phi.z));
}
"""

_ZERO_BUF_GLSL = """#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) writeonly buffer ZeroBuf { float buf_out[]; };
uniform uint u_num_floats;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < u_num_floats) {
        buf_out[idx] = 0.0;
    }
}
"""

_BOOST_ORBITAL_GLSL = """#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer PhiBuf { float phi_in[]; };
layout(std430, binding = 1) writeonly buffer VelBuf { float vel_out[]; };

uniform int u_N;
uniform float u_box;
uniform float u_v_orb;

vec3 load_p(int x, int y, int z, int N) {
    int xm = (x + N) % N;
    int ym = (y + N) % N;
    int zm = (z + N) % N;
    int b = ((zm * N + ym) * N + xm) * 3;
    return vec3(phi_in[b], phi_in[b + 1], phi_in[b + 2]);
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    if (gid.x >= u_N || gid.y >= u_N || gid.z >= u_N) return;
    int N = u_N;
    float dx = u_box / float(N);
    vec3 p_pos = ((vec3(gid) + 0.5) / float(N) - 0.5) * u_box;
    vec3 yp = load_p(gid.x, gid.y + 1, gid.z, N);
    vec3 ym = load_p(gid.x, gid.y - 1, gid.z, N);
    float vy = u_v_orb * tanh(p_pos.x / 1.10);
    vec3 v = -vy * ((yp - ym) / (2.0 * dx));
    int b = ((gid.z * N + gid.y) * N + gid.x) * 3;
    vel_out[b]     = v.x;
    vel_out[b + 1] = v.y;
    vel_out[b + 2] = v.z;
}
"""

_PACK_VOL_GLSL = """#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer PhiBuf { float phi_in[]; };
layout(rgba16f, binding = 0) writeonly uniform image3D u_vol_tex;

uniform int u_N;

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    if (gid.x >= u_N || gid.y >= u_N || gid.z >= u_N) return;
    int idx = (gid.z * u_N + gid.y) * u_N + gid.x;
    int b = idx * 3;
    vec3 p = vec3(phi_in[b], phi_in[b + 1], phi_in[b + 2]);
    float rho = dot(p, p);
    imageStore(u_vol_tex, gid, vec4(rho, p.x, p.y, p.z));
}
"""


_MG_LEVELS_U: tuple[int, ...] = (64, 32, 16, 8, 4)


def _build_deposit_u_rhs_glsl(ham_expr_glsl: str) -> str:
    """
    Deposits the rest Hamiltonian energy density `rho_{H,0}(x)` (`U = 0`) onto the `64³` Multigrid
    Level-0 RHS buffer (`rhs_fine = -kappa * rho_{H,0}`) using a symmetric `2x2x2` sub-lattice when `N >= 128`.
    """
    return f"""#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer PhiBuf {{ float phi_in[]; }};
layout(std430, binding = 1) readonly buffer VelBuf {{ float vel_in[]; }};
layout(std430, binding = 2) writeonly buffer RhsFineBuf {{ float rhs_fine[]; }};

uniform int u_N;
uniform int u_M;
uniform float u_dx;
uniform float u_kappa;
uniform float c_w;
uniform float m_eff;
uniform float g4;
uniform float g6;
uniform float k_skyrme;

int idx3(int x, int y, int z, int N) {{
    int xm = (x + N) % N;
    int ym = (y + N) % N;
    int zm = (z + N) % N;
    return (zm * N + ym) * N + xm;
}}

vec3 load_phi(int idx) {{
    int b = idx * 3;
    return vec3(phi_in[b], phi_in[b + 1], phi_in[b + 2]);
}}

vec3 load_vel(int idx) {{
    int b = idx * 3;
    return vec3(vel_in[b], vel_in[b + 1], vel_in[b + 2]);
}}

float eval_ham_rest(vec3 c, vec3 v, vec3 gx, vec3 gy, vec3 gz) {{
    const float U = 0.0;
    float phi1 = c.x; float phi2 = c.y; float phi3 = c.z;
    float dt_phi1 = v.x; float dt_phi2 = v.y; float dt_phi3 = v.z;
    float dx_phi1 = gx.x; float dy_phi1 = gy.x; float dz_phi1 = gz.x;
    float dx_phi2 = gx.y; float dy_phi2 = gy.y; float dz_phi2 = gz.y;
    float dx_phi3 = gx.z; float dy_phi3 = gy.z; float dz_phi3 = gz.z;
    float rho = dot(c, c);
    return {ham_expr_glsl};
}}

float sample_fine_rho_h(ivec3 fg, int N, float inv_dx) {{
    int c_idx = (fg.z * N + fg.y) * N + fg.x;
    vec3 c  = load_phi(c_idx);
    vec3 v  = load_vel(c_idx);
    vec3 xp = load_phi(idx3(fg.x + 1, fg.y, fg.z, N));
    vec3 xm = load_phi(idx3(fg.x - 1, fg.y, fg.z, N));
    vec3 yp = load_phi(idx3(fg.x, fg.y + 1, fg.z, N));
    vec3 ym = load_phi(idx3(fg.x, fg.y - 1, fg.z, N));
    vec3 zp = load_phi(idx3(fg.x, fg.y, fg.z + 1, N));
    vec3 zm = load_phi(idx3(fg.x, fg.y, fg.z - 1, N));
    float hp = eval_ham_rest(c, v, (xp - c) * inv_dx, (yp - c) * inv_dx, (zp - c) * inv_dx);
    float hm = eval_ham_rest(c, v, (c - xm) * inv_dx, (c - ym) * inv_dx, (c - zm) * inv_dx);
    return max(0.5 * (hp + hm), 0.0);
}}

void main() {{
    ivec3 g = ivec3(gl_GlobalInvocationID);
    if (g.x >= u_M || g.y >= u_M || g.z >= u_M) return;

    int N = u_N;
    int M = u_M;
    float inv_dx = 1.0 / u_dx;
    int cell = (g.z * M + g.y) * M + g.x;

    float rho_avg = 0.0;
    if (N >= 2 * M) {{
        int blk = N / M;
        int q0 = max(blk / 4, 0);
        int q1 = max((3 * blk) / 4, min(1, blk - 1));
        ivec3 base = g * blk;
        int offs[2] = int[2](q0, q1);
        for (int dz = 0; dz < 2; ++dz) {{
            for (int dy = 0; dy < 2; ++dy) {{
                for (int dx_i = 0; dx_i < 2; ++dx_i) {{
                    ivec3 fg = clamp(base + ivec3(offs[dx_i], offs[dy], offs[dz]), ivec3(0), ivec3(N - 1));
                    rho_avg += sample_fine_rho_h(fg, N, inv_dx);
                }}
            }}
        }}
        rho_avg *= 0.125;
    }} else {{
        ivec3 fg = clamp(ivec3(((vec3(g) + 0.5) / float(M)) * float(N)), ivec3(0), ivec3(N - 1));
        rho_avg = sample_fine_rho_h(fg, N, inv_dx);
    }}
    rhs_fine[cell] = -u_kappa * rho_avg;
}}
"""


_U_MOMENTS_GLSL = """#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer RhsFineBuf { float rhs_fine[]; };
layout(std430, binding = 1) writeonly buffer MomentsBuf { vec4 mom_out[]; };

uniform int u_M;
uniform float u_half_box;

shared vec4 s_left[256];
shared vec4 s_right[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint M = uint(u_M);
    uint total = M * M * M;
    float h = (2.0 * u_half_box) / float(u_M);
    float gm_vol = (h * h * h) / 12.566370614359172;

    vec4 acc_l = vec4(0.0);
    vec4 acc_r = vec4(0.0);
    for (uint i = tid; i < total; i += 256u) {
        float gm_i = (-rhs_fine[i]) * gm_vol;
        if (abs(gm_i) > 0.0) {
            uint x = i % M;
            uint tmp = i / M;
            uint y = tmp % M;
            uint z = tmp / M;
            vec3 p = (vec3(float(x), float(y), float(z)) + 0.5) * h - vec3(u_half_box);
            if (p.x < 0.0) {
                acc_l += vec4(p * gm_i, gm_i);
            } else {
                acc_r += vec4(p * gm_i, gm_i);
            }
        }
    }
    s_left[tid] = acc_l;
    s_right[tid] = acc_r;
    barrier();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_left[tid] += s_left[tid + s];
            s_right[tid] += s_right[tid + s];
        }
        barrier();
    }

    if (tid == 0u) {
        vec3 xl = (abs(s_left[0].w) > 1e-18) ? (s_left[0].xyz / s_left[0].w) : vec3(-0.25 * u_half_box, 0.0, 0.0);
        vec3 xr = (abs(s_right[0].w) > 1e-18) ? (s_right[0].xyz / s_right[0].w) : vec3(0.25 * u_half_box, 0.0, 0.0);
        mom_out[0] = vec4(xl, s_left[0].w);
        mom_out[1] = vec4(xr, s_right[0].w);
    }
}
"""


_U_MG_SMOOTH_GLSL = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer RhsBuf     { float rhs_in[]; };
layout(std430, binding = 1) readonly buffer UInBuf     { float u_in[]; };
layout(std430, binding = 2) writeonly buffer UOutBuf   { float u_out[]; };
layout(std430, binding = 3) readonly buffer MomentsBuf { vec4 mom_in[]; };

uniform int u_M;
uniform float u_half_box;
uniform int u_is_fine;
uniform int u_zero_init;

int cidx(int x, int y, int z, int M) {
    return (z * M + y) * M + x;
}

float wall_bc(vec3 p_wall) {
    if (u_is_fine == 0) return 0.0;
    vec4 mL = mom_in[0];
    vec4 mR = mom_in[1];
    float u_w = 0.0;
    if (abs(mL.w) > 1e-18) u_w += mL.w / max(length(p_wall - mL.xyz), 0.5);
    if (abs(mR.w) > 1e-18) u_w += mR.w / max(length(p_wall - mR.xyz), 0.5);
    return u_w;
}

void main() {
    ivec3 g = ivec3(gl_GlobalInvocationID);
    if (g.x >= u_M || g.y >= u_M || g.z >= u_M) return;

    int M = u_M;
    int id = cidx(g.x, g.y, g.z, M);
    float h = (2.0 * u_half_box) / float(M);
    vec3 p_cell = (vec3(g) + 0.5) * h - vec3(u_half_box);

    float sum_nb = 0.0;
    float diag = 0.0;
    float u_cur = (u_zero_init == 1) ? 0.0 : u_in[id];

    const ivec3 dirs[6] = ivec3[6](
        ivec3(-1, 0, 0), ivec3(1, 0, 0),
        ivec3(0, -1, 0), ivec3(0, 1, 0),
        ivec3(0, 0, -1), ivec3(0, 0, 1)
    );
    for (int d = 0; d < 6; ++d) {
        ivec3 nb = g + dirs[d];
        if (nb.x < 0 || nb.x >= M || nb.y < 0 || nb.y >= M || nb.z < 0 || nb.z >= M) {
            vec3 p_wall = p_cell + 0.5 * h * vec3(dirs[d]);
            sum_nb += 2.0 * wall_bc(p_wall);
            diag += 2.0;
        } else {
            sum_nb += (u_zero_init == 1) ? 0.0 : u_in[cidx(nb.x, nb.y, nb.z, M)];
            diag += 1.0;
        }
    }

    float h2_rhs = (h * h) * rhs_in[id];
    float u_jac = (sum_nb - h2_rhs) / diag;
    u_out[id] = mix(u_cur, u_jac, 0.6666667);
}
"""


_U_MG_RESTRICT_GLSL = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer FineRhsBuf    { float rhs_f[]; };
layout(std430, binding = 1) readonly buffer FineUBuf      { float u_f[]; };
layout(std430, binding = 2) writeonly buffer CoarseRhsBuf { float rhs_c[]; };
layout(std430, binding = 3) readonly buffer MomentsBuf    { vec4 mom_in[]; };

uniform int u_Mc;
uniform float u_half_box;
uniform int u_is_fine;

int fidx(int x, int y, int z, int Mf) {
    return (z * Mf + y) * Mf + x;
}

float wall_bc(vec3 p_wall) {
    if (u_is_fine == 0) return 0.0;
    vec4 mL = mom_in[0];
    vec4 mR = mom_in[1];
    float u_w = 0.0;
    if (abs(mL.w) > 1e-18) u_w += mL.w / max(length(p_wall - mL.xyz), 0.5);
    if (abs(mR.w) > 1e-18) u_w += mR.w / max(length(p_wall - mR.xyz), 0.5);
    return u_w;
}

float fine_residual(ivec3 gf, int Mf, float hf, float inv_hf2) {
    int id = fidx(gf.x, gf.y, gf.z, Mf);
    float u_c = u_f[id];
    vec3 p_cell = (vec3(gf) + 0.5) * hf - vec3(u_half_box);

    float sum_nb = 0.0;
    float diag = 0.0;
    const ivec3 dirs[6] = ivec3[6](
        ivec3(-1, 0, 0), ivec3(1, 0, 0),
        ivec3(0, -1, 0), ivec3(0, 1, 0),
        ivec3(0, 0, -1), ivec3(0, 0, 1)
    );
    for (int d = 0; d < 6; ++d) {
        ivec3 nb = gf + dirs[d];
        if (nb.x < 0 || nb.x >= Mf || nb.y < 0 || nb.y >= Mf || nb.z < 0 || nb.z >= Mf) {
            vec3 p_wall = p_cell + 0.5 * hf * vec3(dirs[d]);
            sum_nb += 2.0 * wall_bc(p_wall);
            diag += 2.0;
        } else {
            sum_nb += u_f[fidx(nb.x, nb.y, nb.z, Mf)];
            diag += 1.0;
        }
    }
    float lap = (sum_nb - diag * u_c) * inv_hf2;
    return rhs_f[id] - lap;
}

void main() {
    ivec3 gc = ivec3(gl_GlobalInvocationID);
    if (gc.x >= u_Mc || gc.y >= u_Mc || gc.z >= u_Mc) return;

    int cid = (gc.z * u_Mc + gc.y) * u_Mc + gc.x;
    int Mf = u_Mc * 2;
    float hf = (2.0 * u_half_box) / float(Mf);
    float inv_hf2 = 1.0 / (hf * hf);
    ivec3 base = gc * 2;

    float sum_r = 0.0;
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx_i = 0; dx_i < 2; ++dx_i) {
                sum_r += fine_residual(base + ivec3(dx_i, dy, dz), Mf, hf, inv_hf2);
            }
        }
    }
    rhs_c[cid] = 0.125 * sum_r;
}
"""


_U_MG_PROLONG_ADD_GLSL = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer CoarseErrBuf { float err_c[]; };
layout(std430, binding = 1) buffer FineUBuf              { float u_f[]; };

uniform int u_Mf;

float sample_coarse_err(int x, int y, int z, int Mc) {
    float sgn = 1.0;
    if (x < 0)        { x = 0;      sgn *= -1.0; }
    else if (x >= Mc) { x = Mc - 1; sgn *= -1.0; }
    if (y < 0)        { y = 0;      sgn *= -1.0; }
    else if (y >= Mc) { y = Mc - 1; sgn *= -1.0; }
    if (z < 0)        { z = 0;      sgn *= -1.0; }
    else if (z >= Mc) { z = Mc - 1; sgn *= -1.0; }
    return sgn * err_c[(z * Mc + y) * Mc + x];
}

void main() {
    ivec3 gf = ivec3(gl_GlobalInvocationID);
    if (gf.x >= u_Mf || gf.y >= u_Mf || gf.z >= u_Mf) return;

    int Mc = u_Mf / 2;
    vec3 fc = (vec3(gf) + 0.5) * 0.5 - vec3(0.5);
    ivec3 i0 = ivec3(floor(fc));
    vec3 t = fc - vec3(i0);

    float c000 = sample_coarse_err(i0.x,     i0.y,     i0.z,     Mc);
    float c100 = sample_coarse_err(i0.x + 1, i0.y,     i0.z,     Mc);
    float c010 = sample_coarse_err(i0.x,     i0.y + 1, i0.z,     Mc);
    float c110 = sample_coarse_err(i0.x + 1, i0.y + 1, i0.z,     Mc);
    float c001 = sample_coarse_err(i0.x,     i0.y,     i0.z + 1, Mc);
    float c101 = sample_coarse_err(i0.x + 1, i0.y,     i0.z + 1, Mc);
    float c011 = sample_coarse_err(i0.x,     i0.y + 1, i0.z + 1, Mc);
    float c111 = sample_coarse_err(i0.x + 1, i0.y + 1, i0.z + 1, Mc);

    float c00 = mix(c000, c100, t.x);
    float c10 = mix(c010, c110, t.x);
    float c01 = mix(c001, c101, t.x);
    float c11 = mix(c011, c111, t.x);
    float corr = mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);

    int fid = (gf.z * u_Mf + gf.y) * u_Mf + gf.x;
    u_f[fid] += corr;
}
"""


_U_WAVE_STEP_GLSL = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer RhsBuf   { float rhs_in[]; };
layout(std430, binding = 1) readonly buffer UInBuf   { float u_in[]; };
layout(std430, binding = 2) buffer UTIOBuf           { float ut_io[]; };
layout(std430, binding = 3) writeonly buffer UOutBuf { float u_out[]; };

uniform int u_M;
uniform float u_half_box;
uniform float u_dt;
uniform float u_c_gw;

int cidx_clamp(int x, int y, int z, int M) {
    int xc = clamp(x, 0, M - 1);
    int yc = clamp(y, 0, M - 1);
    int zc = clamp(z, 0, M - 1);
    return (zc * M + yc) * M + xc;
}

float sample_trilinear_u(vec3 fc, int M) {
    ivec3 i0 = ivec3(floor(fc));
    vec3 t = fc - vec3(i0);
    float c000 = u_in[cidx_clamp(i0.x,     i0.y,     i0.z,     M)];
    float c100 = u_in[cidx_clamp(i0.x + 1, i0.y,     i0.z,     M)];
    float c010 = u_in[cidx_clamp(i0.x,     i0.y + 1, i0.z,     M)];
    float c110 = u_in[cidx_clamp(i0.x + 1, i0.y + 1, i0.z,     M)];
    float c001 = u_in[cidx_clamp(i0.x,     i0.y,     i0.z + 1, M)];
    float c101 = u_in[cidx_clamp(i0.x + 1, i0.y,     i0.z + 1, M)];
    float c011 = u_in[cidx_clamp(i0.x,     i0.y + 1, i0.z + 1, M)];
    float c111 = u_in[cidx_clamp(i0.x + 1, i0.y + 1, i0.z + 1, M)];
    float c00 = mix(c000, c100, t.x);
    float c10 = mix(c010, c110, t.x);
    float c01 = mix(c001, c101, t.x);
    float c11 = mix(c011, c111, t.x);
    return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}

void main() {
    ivec3 g = ivec3(gl_GlobalInvocationID);
    if (g.x >= u_M || g.y >= u_M || g.z >= u_M) return;

    int M = u_M;
    int id = (g.z * M + g.y) * M + g.x;
    float h = (2.0 * u_half_box) / float(M);
    float u_c = u_in[id];

    // 1st-order spherical Bayliss-Turkel / Sommerfeld outflow (d_t + c_gw d_r)(r * U) = 0:
    // Exact zero-drift annihilator for static C / r monopoles AND non-reflecting for outgoing GWs!
    if (g.x == 0 || g.x == M - 1 || g.y == 0 || g.y == M - 1 || g.z == 0 || g.z == M - 1) {
        vec3 p_cell = (vec3(g) + 0.5) * h - vec3(u_half_box);
        float r = max(length(p_cell), h);
        float r_in = max(r - h, 0.5 * h);
        vec3 p_in = p_cell * (r_in / r);
        vec3 fc = clamp(
            (p_in + vec3(u_half_box)) / h - vec3(0.5),
            vec3(0.0),
            vec3(float(M - 1) - 1e-4)
        );
        float u_inner = sample_trilinear_u(fc, M);
        float u_target = (r_in / r) * u_inner;
        float alpha = (u_c_gw * u_dt) / h;
        float u_next = (u_c + alpha * u_target) / (1.0 + alpha);
        ut_io[id] = (u_next - u_c) / max(u_dt, 1e-12);
        u_out[id] = u_next;
        return;
    }

    float inv_h2 = 1.0 / (h * h);
    float sum_nb = u_in[cidx_clamp(g.x - 1, g.y, g.z, M)] + u_in[cidx_clamp(g.x + 1, g.y, g.z, M)]
                 + u_in[cidx_clamp(g.x, g.y - 1, g.z, M)] + u_in[cidx_clamp(g.x, g.y + 1, g.z, M)]
                 + u_in[cidx_clamp(g.x, g.y, g.z - 1, M)] + u_in[cidx_clamp(g.x, g.y, g.z + 1, M)];
    float lap = (sum_nb - 6.0 * u_c) * inv_h2;
    // rhs_in[id] = -kappa * rho_{H,0} => (lap - rhs_in[id]) = nabla^2 U + kappa * rho_{H,0}
    float acc_u = (u_c_gw * u_c_gw) * (lap - rhs_in[id]);
    float ut_next = ut_io[id] + u_dt * acc_u;
    float u_next = u_c + u_dt * ut_next;
    ut_io[id] = ut_next;
    u_out[id] = u_next;
}
"""


_PACK_TENSION_TEX_GLSL = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer UBuf  { float u_in[]; };
layout(std430, binding = 2) readonly buffer UTBuf { float ut_in[]; };
layout(rg32f, binding = 1) writeonly uniform image3D u_tension_img;

uniform int u_M;
uniform int u_zero_mode;
uniform int u_has_ut;

void main() {
    ivec3 g = ivec3(gl_GlobalInvocationID);
    if (g.x >= u_M || g.y >= u_M || g.z >= u_M) return;
    int id = (g.z * u_M + g.y) * u_M + g.x;
    float val = (u_zero_mode == 1) ? 0.0 : clamp(u_in[id], -0.35, 0.75);
    float ut_val = (u_zero_mode == 1 || u_has_ut == 0) ? 0.0 : ut_in[id];
    imageStore(u_tension_img, g, vec4(val, ut_val, 0.0, 0.0));
}
"""


def _build_reduce_telemetry_glsl(ham_expr_glsl: str) -> str:
    """
    Emits a fast 3D GPU parallel workgroup reduction shader (`64³` uniform lattice across the tank ->
    `1024` workgroups -> `16 KB` readback!) measuring global `peak_rho`, `3D L2 Norm`, AND `3D Hamiltonian H`!
    """
    return f"""#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer PhiBuf {{ float phi_in[]; }};
layout(std430, binding = 1) readonly buffer VelBuf {{ float vel_in[]; }};
layout(std430, binding = 2) writeonly buffer OutStatsBuf {{ vec4 wg_stats[]; }};

uniform int u_N;
uniform int u_stride;
uniform float u_dx;
uniform float c_w;
uniform float m_eff;
uniform float g4;
uniform float g6;
uniform float k_skyrme;
uniform int u_use_tension;
uniform int u_retarded_mode;
uniform float u_inv_kappa;
uniform float u_inv_cgw2;
uniform float u_box_size;
uniform sampler3D u_tension_tex;

shared float s_max_rho[256];
shared float s_sum_rho[256];
shared float s_sum_ham[256];

int idx3(int x, int y, int z, int N) {{
    int xm = (x + N) % N;
    int ym = (y + N) % N;
    int zm = (z + N) % N;
    return (zm * N + ym) * N + xm;
}}

vec3 load_phi(int idx) {{
    int b = idx * 3;
    return vec3(phi_in[b], phi_in[b + 1], phi_in[b + 2]);
}}

vec3 load_vel(int idx) {{
    int b = idx * 3;
    return vec3(vel_in[b], vel_in[b + 1], vel_in[b + 2]);
}}

float eval_ham_density(vec3 c, vec3 v, vec3 gx, vec3 gy, vec3 gz, float U) {{
    float phi1 = c.x; float phi2 = c.y; float phi3 = c.z;
    float dt_phi1 = v.x; float dt_phi2 = v.y; float dt_phi3 = v.z;
    float dx_phi1 = gx.x; float dy_phi1 = gy.x; float dz_phi1 = gz.x;
    float dx_phi2 = gx.y; float dy_phi2 = gy.y; float dz_phi2 = gz.y;
    float dx_phi3 = gx.z; float dy_phi3 = gy.z; float dz_phi3 = gz.z;
    float rho = dot(c, c);
    return {ham_expr_glsl};
}}

void main() {{
    uint lid = gl_LocalInvocationIndex;
    ivec3 sgid = ivec3(gl_GlobalInvocationID) * u_stride;
    int N = u_N;

    float loc_max = 0.0;
    float loc_rho = 0.0;
    float loc_ham = 0.0;

    if (sgid.x < N && sgid.y < N && sgid.z < N) {{
        int c_idx = (sgid.z * N + sgid.y) * N + sgid.x;
        float inv_dx = 1.0 / u_dx;

        vec3 c  = load_phi(c_idx);
        vec3 v  = load_vel(c_idx);
        vec3 xp = load_phi(idx3(sgid.x + 1, sgid.y, sgid.z, N));
        vec3 xm = load_phi(idx3(sgid.x - 1, sgid.y, sgid.z, N));
        vec3 yp = load_phi(idx3(sgid.x, sgid.y + 1, sgid.z, N));
        vec3 ym = load_phi(idx3(sgid.x, sgid.y - 1, sgid.z, N));
        vec3 zp = load_phi(idx3(sgid.x, sgid.y, sgid.z + 1, N));
        vec3 zm = load_phi(idx3(sgid.x, sgid.y, sgid.z - 1, N));

        vec3 gx_p = (xp - c) * inv_dx; vec3 gy_p = (yp - c) * inv_dx; vec3 gz_p = (zp - c) * inv_dx;
        vec3 gx_m = (c - xm) * inv_dx; vec3 gy_m = (c - ym) * inv_dx; vec3 gz_m = (c - zm) * inv_dx;

        float rho = dot(c, c);
        float H_val = 0.0;
        if (u_use_tension == 0) {{
            H_val = 0.5 * (
                eval_ham_density(c, v, gx_p, gy_p, gz_p, 0.0)
                + eval_ham_density(c, v, gx_m, gy_m, gz_m, 0.0)
            );
        }} else {{
            vec3 uvw = (vec3(sgid) + 0.5) / float(N);
            vec2 ut_tex = texture(u_tension_tex, uvw).rg;
            float u_c = clamp(ut_tex.r, -0.35, 0.75);
            float H_mat = 0.5 * (
                eval_ham_density(c, v, gx_p, gy_p, gz_p, u_c)
                + eval_ham_density(c, v, gx_m, gy_m, gz_m, u_c)
            );
            if (u_retarded_mode == 1 && u_inv_kappa > 0.0) {{
                float eps_u = 1.0 / 64.0;
                float inv_2hu = 32.0 / max(u_box_size, 1.0);
                vec3 grad_u = vec3(
                    texture(u_tension_tex, uvw + vec3(eps_u, 0.0, 0.0)).r - texture(u_tension_tex, uvw - vec3(eps_u, 0.0, 0.0)).r,
                    texture(u_tension_tex, uvw + vec3(0.0, eps_u, 0.0)).r - texture(u_tension_tex, uvw - vec3(0.0, eps_u, 0.0)).r,
                    texture(u_tension_tex, uvw + vec3(0.0, 0.0, eps_u)).r - texture(u_tension_tex, uvw - vec3(0.0, 0.0, eps_u)).r
                ) * inv_2hu;
                float u_dot = ut_tex.g;
                float H_wave = 0.5 * u_inv_kappa * ((u_dot * u_dot) * u_inv_cgw2 + dot(grad_u, grad_u));
                H_val = H_mat + H_wave;
            }} else {{
                float H_rest = 0.5 * (
                    eval_ham_density(c, v, gx_p, gy_p, gz_p, 0.0)
                    + eval_ham_density(c, v, gx_m, gy_m, gz_m, 0.0)
                );
                H_val = H_mat + 0.5 * u_c * H_rest;
            }}
        }}
        loc_max = rho;
        loc_rho = rho;
        loc_ham = H_val;
    }}

    s_max_rho[lid] = loc_max;
    s_sum_rho[lid] = loc_rho;
    s_sum_ham[lid] = loc_ham;
    barrier();

    for (uint s = 128u; s > 0u; s >>= 1u) {{
        if (lid < s) {{
            s_max_rho[lid] = max(s_max_rho[lid], s_max_rho[lid + s]);
            s_sum_rho[lid] += s_sum_rho[lid + s];
            s_sum_ham[lid] += s_sum_ham[lid + s];
        }}
        barrier();
    }}

    if (lid == 0u) {{
        uint wg_id = (gl_WorkGroupID.z * gl_NumWorkGroups.y + gl_WorkGroupID.y) * gl_NumWorkGroups.x + gl_WorkGroupID.x;
        wg_stats[wg_id] = vec4(s_max_rho[0], s_sum_rho[0], s_sum_ham[0], 1.0);
    }}
}}
"""


_FULLSCREEN_VERT = """#version 430
out vec2 v_uv;
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
"""

_RAYMARCH_KNOT_FRAG = """#version 430
in vec2 v_uv;
out vec4 frag_color;

uniform sampler3D u_vol_tex;
uniform sampler3D u_tension_tex;
uniform vec3 u_cam_eye;
uniform vec3 u_cam_fwd;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_fov_tan;
uniform float u_aspect;
uniform float u_box_half;
uniform float u_density_gain;
uniform float u_slice_x;       // < 0.99 slices open the 3D knot interior!
uniform int u_color_mode;      // 0 = Sapphire-Gold Phase Weave, 1 = Rainbow Phase, 2 = Thermal Energy Density
uniform float u_time;
uniform float u_rho_floor;
uniform int u_show_gw_ripples;
uniform float u_gw_vis_gain;

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec2 intersect_aabb(vec3 ro, vec3 rd, vec3 box_min, vec3 box_max) {
    vec3 inv_d = 1.0 / (rd + vec3(1e-8));
    vec3 t0s = (box_min - ro) * inv_d;
    vec3 t1s = (box_max - ro) * inv_d;
    vec3 tsmaller = min(t0s, t1s);
    vec3 tbigger  = max(t0s, t1s);
    float tmin = max(max(tsmaller.x, tsmaller.y), max(tsmaller.z, 0.0));
    float tmax = min(min(tbigger.x, tbigger.y), tbigger.z);
    return vec2(tmin, tmax);
}

vec3 phase_palette(float phase_rad, float rho, int mode) {
    if (mode == 2) {
        // Mode 2: True Thermal Energy Density Ramp (deep cobalt -> cyan -> solar amber -> white-gold core!)
        float t = clamp((rho - 0.12) / 2.25, 0.0, 1.0);
        vec3 c0 = vec3(0.05, 0.22, 0.85);
        vec3 c1 = vec3(0.06, 0.88, 0.96);
        vec3 c2 = vec3(1.35, 0.72, 0.10);
        vec3 c3 = vec3(1.85, 1.55, 0.88);
        if (t < 0.36) return mix(c0, c1, smoothstep(0.0, 0.36, t));
        if (t < 0.74) return mix(c1, c2, smoothstep(0.36, 0.74, t));
        return mix(c2, c3, smoothstep(0.74, 1.0, t));
    }
    if (mode == 1) {
        // Mode 1: Topological Rainbow Phase Wheel
        return (0.55 + 0.45 * cos(phase_rad + vec3(0.0, 2.094, 4.188))) * (0.85 + 0.18 * clamp(rho, 0.0, 2.0));
    }
    // Mode 0: Option A+ Sapphire-Cyan-Gold Phase Weave
    float h = fract((phase_rad + 3.14159265) / 6.2831853);
    vec3 c_sapphire = vec3(0.06, 0.44, 1.18);
    vec3 c_cyan     = vec3(0.05, 0.96, 0.92);
    vec3 c_gold     = vec3(1.28, 0.72, 0.08);
    vec3 col;
    if (h < 0.38) {
        col = mix(c_sapphire, c_cyan, smoothstep(0.0, 0.38, h));
    } else if (h < 0.72) {
        col = mix(c_cyan, c_gold, smoothstep(0.38, 0.72, h));
    } else {
        col = mix(c_gold, c_sapphire, smoothstep(0.72, 1.0, h));
    }
    return col * (0.85 + 0.18 * clamp(rho, 0.0, 2.0));
}

float box_corner_tick(vec3 p, float half_L) {
    vec3 d = abs(abs(p) - vec3(half_L));
    float max_d = max(d.x, max(d.y, d.z));
    if (max_d > 0.55) return 0.0;
    float e = min(max(d.x, d.y), min(max(d.y, d.z), max(d.x, d.z)));
    return (1.0 - smoothstep(0.012, 0.028, e)) * (1.0 - smoothstep(0.32, 0.55, max_d));
}

float box_back_wire(vec3 p, float half_L) {
    vec3 d = abs(abs(p) - vec3(half_L));
    float e = min(max(d.x, d.y), min(max(d.y, d.z), max(d.x, d.z)));
    return 1.0 - smoothstep(0.012, 0.032, e);
}

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(
        u_cam_fwd
        + ndc.x * u_aspect * u_fov_tan * u_cam_right
        + ndc.y * u_fov_tan * u_cam_up
    );
    vec3 ro = u_cam_eye;

    // Option A+ 'Luminous Sapphire Slate' studio radial gradient
    float rad_vig = length(ndc * vec2(0.85, 1.0));
    vec3 bg_inner = vec3(0.072, 0.132, 0.235);
    vec3 bg_outer = vec3(0.024, 0.046, 0.092);
    vec3 col_hdr = mix(bg_inner, bg_outer, clamp(rad_vig * 0.78, 0.0, 1.0));

    // Subtle 3D studio reference floor grid at y = -u_box_half
    float floor_y = -u_box_half;
    if (abs(rd.y) > 1e-4) {
        float t_fl = (floor_y - ro.y) / rd.y;
        if (t_fl > 0.0 && t_fl < 35.0) {
            vec3 fp = ro + t_fl * rd;
            if (abs(fp.x) <= u_box_half * 1.6 && abs(fp.z) <= u_box_half * 1.6) {
                vec2 g2 = abs(fract(fp.xz * 0.5) - 0.5) / fwidth(fp.xz * 0.5);
                float line = 1.0 - min(min(g2.x, g2.y), 1.0);
                float fade = exp(-0.08 * t_fl);
                col_hdr += vec3(0.10, 0.24, 0.42) * line * fade * 0.42;
            }
        }
    }

    vec3 bmin = vec3(-u_box_half);
    vec3 bmax = vec3(u_box_half);
    bmax.x = min(bmax.x, u_slice_x * u_box_half);

    vec2 hit = intersect_aabb(ro, rd, bmin, bmax);
    if (hit.y > hit.x) {
        // Back-wall wireframe behind volume (occluded cleanly by opaque knot filaments!)
        vec3 p_exit = ro + hit.y * rd;
        col_hdr += vec3(0.14, 0.36, 0.58) * box_back_wire(p_exit, u_box_half) * 0.28;

        const int STEPS = 152;
        float dt_ray = (2.0 * u_box_half * 1.732) / float(STEPS);
        float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        float t = hit.x + jitter * dt_ray;

        vec3 accum_rgb = vec3(0.0);
        float transmittance = 1.0;
        vec3 light_dir = normalize(vec3(0.55, 0.78, 0.32));
        float eps_uv = 1.0 / 128.0;

        float rho_cut = (u_rho_floor > 0.01) ? u_rho_floor : 1.15;
        float rho_span = (rho_cut > 0.5) ? 1.55 : 1.85;
        for (int i = 0; i < STEPS; ++i) {
            if (t > hit.y || transmittance < 0.012) break;
            vec3 pos = ro + t * rd;
            vec3 uvw = (pos / (2.0 * u_box_half)) + 0.5;
            vec4 sample_rgba = texture(u_vol_tex, uvw);
            float rho = sample_rgba.r;

            if (rho > rho_cut) {
                float gx = texture(u_vol_tex, uvw + vec3(eps_uv, 0.0, 0.0)).r
                         - texture(u_vol_tex, uvw - vec3(eps_uv, 0.0, 0.0)).r;
                float gy = texture(u_vol_tex, uvw + vec3(0.0, eps_uv, 0.0)).r
                         - texture(u_vol_tex, uvw - vec3(0.0, eps_uv, 0.0)).r;
                float gz = texture(u_vol_tex, uvw + vec3(0.0, 0.0, eps_uv)).r
                         - texture(u_vol_tex, uvw - vec3(0.0, 0.0, eps_uv)).r;
                vec3 grad = vec3(gx, gy, gz);
                float g_len = length(grad);
                vec3 normal = (g_len > 1e-4) ? (-grad / g_len) : (-rd);

                float ndotl = max(dot(normal, light_dir), 0.0);
                float fresnel = pow(1.0 - abs(dot(normal, -rd)), 2.6);
                vec3 half_v = normalize(light_dir - rd);
                float spec = pow(max(dot(normal, half_v), 0.0), 34.0) * smoothstep(0.08, 0.55, g_len);

                float ph_smooth = atan(sample_rgba.b, sample_rgba.g);
                vec3 base_c = phase_palette(ph_smooth, rho, u_color_mode);
                vec3 lit_c = base_c * (0.38 + 0.72 * ndotl)
                           + vec3(0.28, 0.78, 1.15) * (0.35 * fresnel)
                           + vec3(1.18, 0.96, 0.55) * (0.52 * spec);

                if (u_slice_x < 0.99 && abs(pos.x - u_slice_x * u_box_half) < dt_ray * 1.15) {
                    lit_c += vec3(1.15, 0.78, 0.22) * 0.85;
                }

                float dens_sculpt = pow(clamp((rho - rho_cut) / rho_span, 0.0, 1.35), 1.90);
                float sigma = dens_sculpt * u_density_gain * 3.9;
                float alpha = 1.0 - exp(-sigma * dt_ray);
                accum_rgb += transmittance * alpha * lit_c;
                transmittance *= (1.0 - alpha);
            } else if (u_show_gw_ripples == 1) {
                float u_dot = texture(u_tension_tex, uvw).g * u_gw_vis_gain;
                float abs_ud = abs(u_dot);
                if (abs_ud > 0.0011) {
                    float box_inf = max(abs(pos.x), max(abs(pos.y), abs(pos.z)));
                    float wall_fade = 1.0 - smoothstep(u_box_half * 0.80, u_box_half * 0.98, box_inf);
                    float r_center = length(pos);
                    float halo_mask = (1.0 - smoothstep(0.18 * rho_cut, 0.78 * rho_cut, rho))
                                    * (0.25 + 0.75 * smoothstep(1.35, 3.10, r_center));
                    float eq_sheet = 0.18 + 0.82 * exp(-0.55 * pos.z * pos.z);
                    float gw_norm = clamp((abs_ud - 0.0011) / 0.0115, 0.0, 1.0);
                    vec3 gw_col = (u_dot >= 0.0) ? vec3(1.35, 0.84, 0.18) : vec3(0.10, 0.86, 1.38);
                    float sigma_gw = pow(gw_norm, 1.35) * wall_fade * halo_mask * eq_sheet * 0.16;
                    float alpha_gw = 1.0 - exp(-sigma_gw * dt_ray);
                    accum_rgb += transmittance * alpha_gw * (gw_col * (0.75 + 0.85 * gw_norm));
                    transmittance *= (1.0 - 0.12 * alpha_gw);
                }
            }
            t += dt_ray;
        }
        col_hdr = col_hdr * transmittance + accum_rgb;

        // Subtle corner brackets ONLY near the 8 outer vertices (never draws diagonal stripes across the knot!)
        vec3 p_enter = ro + hit.x * rd;
        col_hdr += vec3(0.22, 0.62, 0.95) * box_corner_tick(p_enter, u_box_half) * 0.45;
    }

    vec3 mapped = aces_tonemap(col_hdr * 1.12);
    mapped = pow(mapped, vec3(1.0 / 2.2));
    frag_color = vec4(mapped, 1.0);
}
"""


_PACK_FP16_CHUNK_GLSL = """#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PhiBuf { float phi_in[]; };
layout(std430, binding = 1) writeonly buffer HalfBuf { uint half_out[]; };

uniform uint u_pair_offset;
uniform uint u_pair_count;
uniform uint u_num_floats;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= u_pair_count) return;
    uint gpair = u_pair_offset + idx;
    uint f0 = gpair * 2u;
    float a = phi_in[f0];
    float b = (f0 + 1u < u_num_floats) ? phi_in[f0 + 1u] : 0.0;
    half_out[idx] = packHalf2x16(vec2(a, b));
}
"""

_UNPACK_FP16_CHUNK_GLSL = """#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer HalfBuf { uint half_in[]; };
layout(std430, binding = 1) writeonly buffer PhiBuf { float phi_out[]; };

uniform uint u_pair_offset;
uniform uint u_pair_count;
uniform uint u_num_floats;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= u_pair_count) return;
    vec2 ab = unpackHalf2x16(half_in[idx]);
    uint gpair = u_pair_offset + idx;
    uint f0 = gpair * 2u;
    phi_out[f0] = ab.x;
    if (f0 + 1u < u_num_floats) {
        phi_out[f0 + 1u] = ab.y;
    }
}
"""


_DEFAULT_MOD_COMPILED_CACHE: Optional[tuple[PhysicsMod, CompiledMasterAction]] = None


def _get_default_mod_and_compiled(
    mod: Optional[PhysicsMod] = None,
) -> tuple[PhysicsMod, CompiledMasterAction]:
    """Returns a process-cached default `(tnap_weave, CompiledMasterAction)` so cold `Arena1KnotTank(ctx)` never stalls Frame 0."""
    global _DEFAULT_MOD_COMPILED_CACHE
    if mod is not None:
        from weaveforge.core.symbolic_compiler import compile_master_action

        return mod, compile_master_action(mod)
    if _DEFAULT_MOD_COMPILED_CACHE is None:
        from weaveforge.core.frameworks import get_mods_by_id
        from weaveforge.core.symbolic_compiler import compile_master_action

        def_mod = get_mods_by_id()["tnap_weave"]
        _DEFAULT_MOD_COMPILED_CACHE = (def_mod, compile_master_action(def_mod))
    base_mod, base_comp = _DEFAULT_MOD_COMPILED_CACHE
    return base_mod.clone(), base_comp


_PACK_REPLAY_PHASE_GLSL = """#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PhiBuf { float phi_in[]; };
layout(std430, binding = 1) readonly buffer VelBuf { float vel_in[]; };
layout(std430, binding = 2) writeonly buffer HalfOutBuf { uint half_out[]; };

uniform uint u_src_N;
uniform uint u_snap_N;
uniform uint u_downsample;
uniform uint u_snap_vox;
uniform uint u_vox_pairs;
uniform uint u_total_pairs;

vec2 clamp_fp16(vec2 v) {
    return clamp(v, vec2(-65500.0), vec2(65500.0));
}

void sample_vox_pv(uint vox, out vec3 p_out, out vec3 v_out) {
    if (vox >= u_snap_vox) {
        p_out = vec3(0.0);
        v_out = vec3(0.0);
        return;
    }
    if (u_downsample == 0u) {
        uint b = vox * 3u;
        p_out = vec3(phi_in[b], phi_in[b + 1u], phi_in[b + 2u]);
        v_out = vec3(vel_in[b], vel_in[b + 1u], vel_in[b + 2u]);
        return;
    }
    uint sn = u_snap_N;
    uint sx = vox % sn;
    uint tmp = vox / sn;
    uint sy = tmp % sn;
    uint sz = tmp / sn;

    uint fn = u_src_N;
    uint bx0 = ((sx * fn) / sn) % fn;
    uint bx1 = (bx0 + 1u) % fn;
    uint by0 = ((sy * fn) / sn) % fn;
    uint by1 = (by0 + 1u) % fn;
    uint bz0 = ((sz * fn) / sn) % fn;
    uint bz1 = (bz0 + 1u) % fn;

    uint yz00 = (bz0 * fn + by0) * fn;
    uint yz01 = (bz0 * fn + by1) * fn;
    uint yz10 = (bz1 * fn + by0) * fn;
    uint yz11 = (bz1 * fn + by1) * fn;

    uint i0 = (yz00 + bx0) * 3u;
    uint i1 = (yz00 + bx1) * 3u;
    uint i2 = (yz01 + bx0) * 3u;
    uint i3 = (yz01 + bx1) * 3u;
    uint i4 = (yz10 + bx0) * 3u;
    uint i5 = (yz10 + bx1) * 3u;
    uint i6 = (yz11 + bx0) * 3u;
    uint i7 = (yz11 + bx1) * 3u;

    p_out = vec3(
        phi_in[i0] + phi_in[i1] + phi_in[i2] + phi_in[i3] + phi_in[i4] + phi_in[i5] + phi_in[i6] + phi_in[i7],
        phi_in[i0 + 1u] + phi_in[i1 + 1u] + phi_in[i2 + 1u] + phi_in[i3 + 1u] + phi_in[i4 + 1u] + phi_in[i5 + 1u] + phi_in[i6 + 1u] + phi_in[i7 + 1u],
        phi_in[i0 + 2u] + phi_in[i1 + 2u] + phi_in[i2 + 2u] + phi_in[i3 + 2u] + phi_in[i4 + 2u] + phi_in[i5 + 2u] + phi_in[i6 + 2u] + phi_in[i7 + 2u]
    ) * 0.125;
    v_out = vec3(
        vel_in[i0] + vel_in[i1] + vel_in[i2] + vel_in[i3] + vel_in[i4] + vel_in[i5] + vel_in[i6] + vel_in[i7],
        vel_in[i0 + 1u] + vel_in[i1 + 1u] + vel_in[i2 + 1u] + vel_in[i3 + 1u] + vel_in[i4 + 1u] + vel_in[i5 + 1u] + vel_in[i6 + 1u] + vel_in[i7 + 1u],
        vel_in[i0 + 2u] + vel_in[i1 + 2u] + vel_in[i2 + 2u] + vel_in[i3 + 2u] + vel_in[i4 + 2u] + vel_in[i5 + 2u] + vel_in[i6 + 2u] + vel_in[i7 + 2u]
    ) * 0.125;
}

void main() {
    uint vp = gl_GlobalInvocationID.x;
    if (vp >= u_vox_pairs) return;
    vec3 p0, v0, p1, v1;
    sample_vox_pv(vp * 2u, p0, v0);
    sample_vox_pv(vp * 2u + 1u, p1, v1);

    uint out_p = vp * 3u;
    uint out_v = u_total_pairs + out_p;
    half_out[out_p] = packHalf2x16(clamp_fp16(vec2(p0.x, p0.y)));
    half_out[out_p + 1u] = packHalf2x16(clamp_fp16(vec2(p0.z, p1.x)));
    half_out[out_v] = packHalf2x16(clamp_fp16(vec2(v0.x, v0.y)));
    half_out[out_v + 1u] = packHalf2x16(clamp_fp16(vec2(v0.z, v1.x)));
    if (out_p + 2u < u_total_pairs) {
        half_out[out_p + 2u] = packHalf2x16(clamp_fp16(vec2(p1.y, p1.z)));
        half_out[out_v + 2u] = packHalf2x16(clamp_fp16(vec2(v1.y, v1.z)));
    }
}
"""

_UNPACK_REPLAY_PHASE_GLSL = """#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer HalfInBuf { uint half_in[]; };
layout(std430, binding = 1) writeonly buffer PhiOutBuf { float phi_out[]; };
layout(std430, binding = 2) writeonly buffer VelOutBuf { float vel_out[]; };

uniform int u_src_N;
uniform int u_dst_N;
uniform uint u_total_pairs;

float load_half_at(uint base_pair, uint scalar_idx) {
    vec2 ab = unpackHalf2x16(half_in[base_pair + (scalar_idx >> 1u)]);
    return ((scalar_idx & 1u) == 0u) ? ab.x : ab.y;
}

void load_coarse_pv(int x, int y, int z, int M, out vec3 p_val, out vec3 v_val) {
    int xm = (x % M + M) % M;
    int ym = (y % M + M) % M;
    int zm = (z % M + M) % M;
    uint s = uint(((zm * M + ym) * M + xm) * 3);
    p_val = vec3(
        load_half_at(0u, s),
        load_half_at(0u, s + 1u),
        load_half_at(0u, s + 2u)
    );
    v_val = vec3(
        load_half_at(u_total_pairs, s),
        load_half_at(u_total_pairs, s + 1u),
        load_half_at(u_total_pairs, s + 2u)
    );
}

void main() {
    ivec3 gf = ivec3(gl_GlobalInvocationID);
    if (gf.x >= u_dst_N || gf.y >= u_dst_N || gf.z >= u_dst_N) return;

    vec3 phi_res;
    vec3 vel_res;
    if (u_src_N == u_dst_N) {
        load_coarse_pv(gf.x, gf.y, gf.z, u_src_N, phi_res, vel_res);
    } else {
        vec3 fc = (vec3(gf) + 0.5) * (float(u_src_N) / float(u_dst_N)) - 0.5;
        ivec3 i0 = ivec3(floor(fc));
        vec3 t = fc - vec3(i0);

        vec3 p000, v000, p100, v100, p010, v010, p110, v110;
        vec3 p001, v001, p101, v101, p011, v011, p111, v111;
        load_coarse_pv(i0.x,     i0.y,     i0.z,     u_src_N, p000, v000);
        load_coarse_pv(i0.x + 1, i0.y,     i0.z,     u_src_N, p100, v100);
        load_coarse_pv(i0.x,     i0.y + 1, i0.z,     u_src_N, p010, v010);
        load_coarse_pv(i0.x + 1, i0.y + 1, i0.z,     u_src_N, p110, v110);
        load_coarse_pv(i0.x,     i0.y,     i0.z + 1, u_src_N, p001, v001);
        load_coarse_pv(i0.x + 1, i0.y,     i0.z + 1, u_src_N, p101, v101);
        load_coarse_pv(i0.x,     i0.y + 1, i0.z + 1, u_src_N, p011, v011);
        load_coarse_pv(i0.x + 1, i0.y + 1, i0.z + 1, u_src_N, p111, v111);

        vec3 p00 = mix(p000, p100, t.x);
        vec3 p10 = mix(p010, p110, t.x);
        vec3 p01 = mix(p001, p101, t.x);
        vec3 p11 = mix(p011, p111, t.x);
        phi_res = mix(mix(p00, p10, t.y), mix(p01, p11, t.y), t.z);

        vec3 v00 = mix(v000, v100, t.x);
        vec3 v10 = mix(v010, v110, t.x);
        vec3 v01 = mix(v001, v101, t.x);
        vec3 v11 = mix(v011, v111, t.x);
        vel_res = mix(mix(v00, v10, t.y), mix(v01, v11, t.y), t.z);
    }

    int out_b = ((gf.z * u_dst_N + gf.y) * u_dst_N + gf.x) * 3;
    phi_out[out_b]     = phi_res.x;
    phi_out[out_b + 1] = phi_res.y;
    phi_out[out_b + 2] = phi_res.z;
    vel_out[out_b]     = vel_res.x;
    vel_out[out_b + 1] = vel_res.y;
    vel_out[out_b + 2] = vel_res.z;
}
"""


class Arena1KnotTank:
    """
    GPU-accelerated 3D Quantum & Topological Knot Tank (`128³` / `256³` / `512³`).
    Directly executes `SymPy`-compiled `GLSL 430` compute shaders on the active OpenGL 4.6 context
    with zero-demotion GDDR7 SSBO residency, DDR5 System RAM hibernation, and DDR5 replay scrubbing!
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        initial_grid_n: int = 256,
        compiled: Optional[CompiledMasterAction] = None,
        mod: Optional[PhysicsMod] = None,
        lazy_gpu: bool = False,
        grid_n: Optional[int] = None,
    ) -> None:
        self.ctx = ctx
        self.grid_n = int(grid_n if grid_n is not None else initial_grid_n)
        self._box_size_override: Optional[float] = None
        self.preset_idx = 0
        self.density_gain = 2.65
        self.slice_x = 1.0
        self.color_mode = 0
        self.damping = 0.0
        self.sponge_strength = 4.5
        self.show_gw_ripples: bool = True
        self.gw_ripple_gain: float = 1.0
        self.substeps_per_frame = 2
        self.sim_time = 0.0
        self.step_count = 0
        self.last_dt = 0.01

        # Live 3D GPU telemetry (full-tank reduction)
        self.init_peak_rho = 1.0
        self.current_peak_rho = 1.0
        self.retention_ratio = 1.0
        self.init_l2_norm = 1.0
        self.l2_norm = 1.0
        self.init_hamiltonian = 0.0
        self.hamiltonian_energy = 0.0
        self.hamiltonian_drift_pct = 0.0
        self.allocated_vram_mb: float = 0.0
        self.is_hibernated: bool = bool(lazy_gpu)

        # DDR5 System RAM Hibernation & Time-Machine Replay Ring-Buffer state
        self.replay_enabled: bool = True
        self.replay_max_frames: int = 48
        self.replay_cursor: int = -1
        self._replay_frames: Deque[np.ndarray] = deque()
        self._replay_meta: Deque[tuple[float, int]] = deque()
        self._live_backup: Optional[dict[str, object]] = None
        self._hib_phi_ram: Optional[np.ndarray] = None
        self._hib_vel_ram: Optional[np.ndarray] = None
        self._hib_u_ram: Optional[np.ndarray] = None
        self._hib_ut_ram: Optional[np.ndarray] = None
        self._hib_u_warm: bool = False

        self._init_prog = self.ctx.compute_shader(_INIT_KNOT_GLSL)
        self._zero_buf_prog = self.ctx.compute_shader(_ZERO_BUF_GLSL)
        self._boost_orbital_prog = self.ctx.compute_shader(_BOOST_ORBITAL_GLSL)
        self._pack_prog = self.ctx.compute_shader(_PACK_VOL_GLSL)
        self._ref_relax_prog: Optional[moderngl.ComputeShader] = None
        self._pack_fp16_prog = self.ctx.compute_shader(_PACK_FP16_CHUNK_GLSL)
        self._unpack_fp16_prog = self.ctx.compute_shader(_UNPACK_FP16_CHUNK_GLSL)
        self._pack_replay_prog = self.ctx.compute_shader(_PACK_REPLAY_PHASE_GLSL)
        self._unpack_replay_prog = self.ctx.compute_shader(_UNPACK_REPLAY_PHASE_GLSL)
        self._default_reduce_prog = self.ctx.compute_shader(
            _build_reduce_telemetry_glsl(_DEFAULT_HAMILTONIAN_GLSL)
        )
        self._default_deposit_u_prog = self.ctx.compute_shader(
            _build_deposit_u_rhs_glsl(_DEFAULT_HAMILTONIAN_GLSL)
        )
        self._u_moments_prog = self.ctx.compute_shader(_U_MOMENTS_GLSL)
        self._u_mg_smooth_prog = self.ctx.compute_shader(_U_MG_SMOOTH_GLSL)
        self._u_mg_restrict_prog = self.ctx.compute_shader(_U_MG_RESTRICT_GLSL)
        self._u_mg_prolong_prog = self.ctx.compute_shader(_U_MG_PROLONG_ADD_GLSL)
        self._u_wave_step_prog = self.ctx.compute_shader(_U_WAVE_STEP_GLSL)
        self._pack_tension_prog = self.ctx.compute_shader(_PACK_TENSION_TEX_GLSL)
        self._ray_prog = self.ctx.program(
            vertex_shader=_FULLSCREEN_VERT,
            fragment_shader=_RAYMARCH_KNOT_FRAG,
        )
        self._empty_vao = self.ctx.vertex_array(self._ray_prog, [])

        self._pde_prog: Optional[moderngl.ComputeShader] = None
        self._compiled_glsl_src: str = ""
        self._reduce_prog: Optional[moderngl.ComputeShader] = None
        self._deposit_u_prog: Optional[moderngl.ComputeShader] = None
        self._compiled_ham_expr: str = ""
        self._ham_params_tuple: Optional[tuple[object, ...]] = None
        self._pending_ham_baseline_reset: bool = False
        self._active_mod: Optional[PhysicsMod] = mod
        self._active_compiled: Optional[CompiledMasterAction] = compiled

        self._mg_rhs_bufs: list[moderngl.Buffer] = []
        self._mg_u_bufs: list[moderngl.Buffer] = []
        self._mg_scratch_bufs: list[moderngl.Buffer] = []
        self._mg_ut_buf: Optional[moderngl.Buffer] = None
        self._mg_moments_buf: Optional[moderngl.Buffer] = None
        self._tension_tex: Optional[moderngl.Texture3D] = None
        self._u_warm_valid: bool = False
        self._zero_wave_init_active: bool = False
        self._external_tension_u_active: bool = False

        self._buf_a: Optional[moderngl.Buffer] = None
        self._buf_b: Optional[moderngl.Buffer] = None
        self._buf_vel: Optional[moderngl.Buffer] = None
        # GPU-resident reduction target (NEVER call .read() directly on _buf_stats!)
        self._buf_stats = self.ctx.buffer(reserve=2048 * 16)
        # Dedicated dynamic=True PCIe staging buffers for non-demoting glCopyBufferSubData readbacks
        self._stats_stage_buf = self.ctx.buffer(reserve=2048 * 16, dynamic=True)
        self._async_stats_bufs = [
            self.ctx.buffer(reserve=2048 * 16, dynamic=True),
            self.ctx.buffer(reserve=2048 * 16, dynamic=True),
        ]
        self._async_fences: list[object] = [None, None]
        self._async_meta: list[Optional[tuple[int, float, int]]] = [None, None]
        self._async_slot: int = 0
        self._last_telem_step: int = 0

        self._xfer_stage_buf: Optional[moderngl.Buffer] = None
        self._fp16_chunk_gpu_buf: Optional[moderngl.Buffer] = None
        self._replay_gpu_pack_buf: Optional[moderngl.Buffer] = None
        self._replay_stage_buf: Optional[moderngl.Buffer] = None
        self._pending_replay_fence: Optional[object] = None
        self._pending_replay_meta: Optional[tuple[int, int, int, float, int]] = None
        self._vol_tex: Optional[moderngl.Texture3D] = None
        self._cur_in_a = True

        if compiled is not None:
            self.update_compiled_action(compiled, mod=mod)
        elif not lazy_gpu:
            def_mod, def_comp = _get_default_mod_and_compiled(mod)
            self.update_compiled_action(def_comp, mod=def_mod)

        if not lazy_gpu:
            self._allocate_grid(self.grid_n)
            self.reset_field()
            self.flush_pending_replay_snapshot()

    @property
    def box_size(self) -> float:
        if self._box_size_override is not None:
            return self._box_size_override
        return 16.8 if self.preset_idx in (4, 5, 6) else 12.0

    @box_size.setter
    def box_size(self, val: Optional[float]) -> None:
        self._box_size_override = None if val is None else float(val)

    @property
    def energy_drift_pct(self) -> float:
        """Alias for `hamiltonian_drift_pct` so uniform arena telemetry queries never raise `AttributeError`."""
        return self.hamiltonian_drift_pct

    @property
    def ddr5_ram_mb(self) -> float:
        """Total DDR5 System RAM (MiB) currently held by hibernate snapshots and replay keyframes."""
        total_bytes = 0
        if self._hib_phi_ram is not None:
            total_bytes += int(self._hib_phi_ram.nbytes)
        if self._hib_vel_ram is not None:
            total_bytes += int(self._hib_vel_ram.nbytes)
        if self._hib_u_ram is not None:
            total_bytes += int(self._hib_u_ram.nbytes)
        if self._hib_ut_ram is not None:
            total_bytes += int(self._hib_ut_ram.nbytes)
        if self._live_backup is not None:
            phi_bk = self._live_backup.get("phi")
            vel_bk = self._live_backup.get("vel")
            u_bk = self._live_backup.get("u_grid")
            ut_bk = self._live_backup.get("ut_grid")
            if isinstance(phi_bk, np.ndarray):
                total_bytes += int(phi_bk.nbytes)
            if isinstance(vel_bk, np.ndarray):
                total_bytes += int(vel_bk.nbytes)
            if isinstance(u_bk, np.ndarray):
                total_bytes += int(u_bk.nbytes)
            if isinstance(ut_bk, np.ndarray):
                total_bytes += int(ut_bk.nbytes)
        for frm in self._replay_frames:
            total_bytes += int(frm.nbytes)
        return float(total_bytes) / (1024.0 * 1024.0)

    @property
    def replay_frames_count(self) -> int:
        """Number of DDR5 Time-Machine keyframes currently buffered (non-blocking harvest check)."""
        self._poll_async_telemetry(wait=False)
        self._poll_pending_replay_snapshot(wait=False)
        return len(self._replay_frames)

    def _clear_pending_replay_fence(self) -> None:
        fence = self._pending_replay_fence
        if fence is not None:
            try:
                GL.glDeleteSync(fence)
            except Exception:
                pass
        self._pending_replay_fence = None
        self._pending_replay_meta = None

    def _clear_async_fences(self) -> None:
        for i in (0, 1):
            fence = self._async_fences[i]
            if fence is not None:
                try:
                    GL.glDeleteSync(fence)
                except Exception:
                    pass
                self._async_fences[i] = None
            self._async_meta[i] = None

    def _ensure_xfer_buffers(self, need_fp16_gpu: bool = False) -> None:
        if self._xfer_stage_buf is None:
            self._xfer_stage_buf = self.ctx.buffer(reserve=_XFER_CHUNK_BYTES, dynamic=True)
        if need_fp16_gpu and self._fp16_chunk_gpu_buf is None:
            self._fp16_chunk_gpu_buf = self.ctx.buffer(reserve=_XFER_CHUNK_BYTES)

    def _ensure_replay_buffers(self, req_bytes: int) -> None:
        req = max(int(req_bytes), 4096)
        if self._replay_gpu_pack_buf is None or self._replay_gpu_pack_buf.size < req:
            if self._replay_gpu_pack_buf is not None:
                self._replay_gpu_pack_buf.release()
            self._replay_gpu_pack_buf = self.ctx.buffer(reserve=req)
        if self._replay_stage_buf is None or self._replay_stage_buf.size < req:
            if self._replay_stage_buf is not None:
                self._replay_stage_buf.release()
            self._replay_stage_buf = self.ctx.buffer(reserve=req, dynamic=True)

    def _release_xfer_buffers(self) -> None:
        self._clear_pending_replay_fence()
        for obj in (
            self._xfer_stage_buf,
            self._fp16_chunk_gpu_buf,
            self._replay_gpu_pack_buf,
            self._replay_stage_buf,
        ):
            if obj is not None:
                obj.release()
        self._xfer_stage_buf = None
        self._fp16_chunk_gpu_buf = None
        self._replay_gpu_pack_buf = None
        self._replay_stage_buf = None

    def _pack_tension_tex_gpu(self, zero_mode: bool = False, has_ut: bool = False) -> None:
        assert self._tension_tex is not None and len(self._mg_u_bufs) > 0 and self._mg_ut_buf is not None
        m0 = _MG_LEVELS_U[0]
        self._mg_u_bufs[0].bind_to_storage_buffer(0)
        self._mg_ut_buf.bind_to_storage_buffer(2)
        self._tension_tex.bind_to_image(1, read=False, write=True)
        self._set_uniform_safe(self._pack_tension_prog, "u_M", m0)
        self._set_uniform_safe(self._pack_tension_prog, "u_zero_mode", 1 if zero_mode else 0)
        self._set_uniform_safe(self._pack_tension_prog, "u_has_ut", 1 if has_ut else 0)
        wg = (m0 + 3) // 4
        self._pack_tension_prog.run(wg, wg, wg)
        self.ctx.memory_barrier()

    def _ensure_mg_resources(self) -> None:
        """Allocates the 5-level `64³ -> 4³` Multigrid + hyperbolic wave buffers and `GL_RG32F` 3D texture `(U, dU/dt)`."""
        if (
            self._tension_tex is not None
            and len(self._mg_u_bufs) == len(_MG_LEVELS_U)
            and self._mg_ut_buf is not None
        ):
            return
        self._release_mg_resources()
        for m in _MG_LEVELS_U:
            nbytes = m * m * m * 4
            self._mg_rhs_bufs.append(self.ctx.buffer(reserve=nbytes))
            self._mg_u_bufs.append(self.ctx.buffer(reserve=nbytes))
            self._mg_scratch_bufs.append(self.ctx.buffer(reserve=nbytes))
        m0 = _MG_LEVELS_U[0]
        self._mg_ut_buf = self.ctx.buffer(reserve=m0 * m0 * m0 * 4)
        self._zero_ssbo(self._mg_ut_buf, m0 * m0 * m0)
        self._mg_moments_buf = self.ctx.buffer(reserve=32)
        tex = self.ctx.texture3d((m0, m0, m0), components=2, dtype="f4")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        tex.repeat_z = False
        self._tension_tex = tex
        self._u_warm_valid = False
        self._pack_tension_tex_gpu(zero_mode=True, has_ut=False)

    def _release_mg_resources(self) -> None:
        for lst in (self._mg_rhs_bufs, self._mg_u_bufs, self._mg_scratch_bufs):
            for b in lst:
                if b is not None:
                    b.release()
            lst.clear()
        if self._mg_ut_buf is not None:
            self._mg_ut_buf.release()
            self._mg_ut_buf = None
        if self._mg_moments_buf is not None:
            self._mg_moments_buf.release()
            self._mg_moments_buf = None
        if self._tension_tex is not None:
            self._tension_tex.release()
            self._tension_tex = None
        self._u_warm_valid = False
        self._external_tension_u_active = False

    def _current_use_tension(self, mod: Optional[PhysicsMod] = None) -> tuple[int, float]:
        use_m = mod if mod is not None else self._active_mod
        g_k = float(getattr(use_m.parameters, "g_knot", 0.0)) if use_m is not None else 0.0
        if self._external_tension_u_active or abs(g_k) > 1e-8:
            return 1, g_k
        return 0, 0.0

    def _smooth_u_level(self, lvl: int, n_iters: int, zero_first: bool) -> None:
        assert self._mg_moments_buf is not None
        iters = max(2, int(n_iters) + (int(n_iters) % 2))
        m = _MG_LEVELS_U[lvl]
        wg = (m + 3) // 4
        self._mg_rhs_bufs[lvl].bind_to_storage_buffer(0)
        self._mg_moments_buf.bind_to_storage_buffer(3)
        self._set_uniform_safe(self._u_mg_smooth_prog, "u_M", m)
        self._set_uniform_safe(self._u_mg_smooth_prog, "u_half_box", float(self.box_size) * 0.5)
        self._set_uniform_safe(self._u_mg_smooth_prog, "u_is_fine", 1 if lvl == 0 else 0)
        for k in range(iters):
            if (k % 2) == 0:
                self._mg_u_bufs[lvl].bind_to_storage_buffer(1)
                self._mg_scratch_bufs[lvl].bind_to_storage_buffer(2)
            else:
                self._mg_scratch_bufs[lvl].bind_to_storage_buffer(1)
                self._mg_u_bufs[lvl].bind_to_storage_buffer(2)
            self._set_uniform_safe(
                self._u_mg_smooth_prog,
                "u_zero_init",
                1 if (k == 0 and zero_first) else 0,
            )
            self._u_mg_smooth_prog.run(wg, wg, wg)
            self.ctx.memory_barrier()

    def _solve_tension_u_gpu(
        self,
        mod: Optional[PhysicsMod] = None,
        n_vcycles: int = 2,
        zero_init_fine: bool = False,
    ) -> int:
        """
        Solves `nabla^2 U = -kappa * rho_{H,0}` on the `64³` GPU Multigrid hierarchy with exact
        left/right two-monopole open-space wall boundary conditions (`~0.25 ms`, zero CPU readback!).
        """
        if self.is_hibernated or self._buf_a is None or self._buf_b is None or self._buf_vel is None:
            return 0
        self._ensure_mg_resources()
        assert self._tension_tex is not None and self._mg_moments_buf is not None
        if self._external_tension_u_active:
            return 1

        use_t, g_k = self._current_use_tension(mod)
        if use_t == 0:
            if self._u_warm_valid:
                m0 = _MG_LEVELS_U[0]
                if self._mg_ut_buf is not None:
                    self._zero_ssbo(self._mg_ut_buf, m0 * m0 * m0)
                self._pack_tension_tex_gpu(zero_mode=True, has_ut=False)
                self._u_warm_valid = False
            return 0

        use_m = mod if mod is not None else self._active_mod
        p = use_m.parameters if use_m is not None else None
        dep_prog = self._deposit_u_prog if self._deposit_u_prog is not None else self._default_deposit_u_prog
        latest_buf = self._buf_a if self._cur_in_a else self._buf_b

        m0 = _MG_LEVELS_U[0]
        n = int(self.grid_n)
        dx = float(self.box_size) / float(n)
        half_box = float(self.box_size) * 0.5
        kappa = float(WEAVE_KAPPA_SCALE) * float(g_k)

        latest_buf.bind_to_storage_buffer(0)
        self._buf_vel.bind_to_storage_buffer(1)
        self._mg_rhs_bufs[0].bind_to_storage_buffer(2)
        self._set_uniform_safe(dep_prog, "u_N", n)
        self._set_uniform_safe(dep_prog, "u_M", m0)
        self._set_uniform_safe(dep_prog, "u_dx", dx)
        self._set_uniform_safe(dep_prog, "u_kappa", kappa)
        self._set_uniform_safe(dep_prog, "c_w", float(p.c_w) if p else 1.0)
        self._set_uniform_safe(dep_prog, "m_eff", float(p.m_eff) if p else 2.50)
        self._set_uniform_safe(dep_prog, "g4", float(p.g4) if p else 9.80)
        self._set_uniform_safe(dep_prog, "g6", float(p.g6) if p else 3.10)
        self._set_uniform_safe(dep_prog, "k_skyrme", float(getattr(p, "k_skyrme", 0.0)) if p else 1.55)
        wg0 = (m0 + 3) // 4
        dep_prog.run(wg0, wg0, wg0)
        self.ctx.memory_barrier()

        # Left/Right half-space monopole moments for open-space wall BCs (1 workgroup, 0 CPU sync!)
        self._mg_rhs_bufs[0].bind_to_storage_buffer(0)
        self._mg_moments_buf.bind_to_storage_buffer(1)
        self._set_uniform_safe(self._u_moments_prog, "u_M", m0)
        self._set_uniform_safe(self._u_moments_prog, "u_half_box", half_box)
        self._u_moments_prog.run(1, 1, 1)
        self.ctx.memory_barrier()

        last_lvl = len(_MG_LEVELS_U) - 1
        start_zero = bool(zero_init_fine or not self._u_warm_valid)
        eff_vcycles = max(8, int(n_vcycles)) if start_zero else max(1, int(n_vcycles))
        smooth_sweeps = 4 if start_zero else 2
        for v_it in range(eff_vcycles):
            for lvl in range(last_lvl):
                self._smooth_u_level(
                    lvl,
                    n_iters=smooth_sweeps,
                    zero_first=(start_zero if (lvl == 0 and v_it == 0) else (lvl > 0)),
                )
                mc = _MG_LEVELS_U[lvl + 1]
                wgc = (mc + 3) // 4
                self._mg_rhs_bufs[lvl].bind_to_storage_buffer(0)
                self._mg_u_bufs[lvl].bind_to_storage_buffer(1)
                self._mg_rhs_bufs[lvl + 1].bind_to_storage_buffer(2)
                self._mg_moments_buf.bind_to_storage_buffer(3)
                self._set_uniform_safe(self._u_mg_restrict_prog, "u_Mc", mc)
                self._set_uniform_safe(self._u_mg_restrict_prog, "u_half_box", half_box)
                self._set_uniform_safe(self._u_mg_restrict_prog, "u_is_fine", 1 if lvl == 0 else 0)
                self._u_mg_restrict_prog.run(wgc, wgc, wgc)
                self.ctx.memory_barrier()

            self._smooth_u_level(last_lvl, n_iters=24, zero_first=True)

            for lvl in range(last_lvl - 1, -1, -1):
                mf = _MG_LEVELS_U[lvl]
                wgf = (mf + 3) // 4
                self._mg_u_bufs[lvl + 1].bind_to_storage_buffer(0)
                self._mg_u_bufs[lvl].bind_to_storage_buffer(1)
                self._set_uniform_safe(self._u_mg_prolong_prog, "u_Mf", mf)
                self._u_mg_prolong_prog.run(wgf, wgf, wgf)
                self.ctx.memory_barrier()
                self._smooth_u_level(lvl, n_iters=smooth_sweeps, zero_first=False)

        if self._mg_ut_buf is not None:
            self._zero_ssbo(self._mg_ut_buf, m0 * m0 * m0)
        self._pack_tension_tex_gpu(zero_mode=False, has_ut=False)
        self._zero_wave_init_active = False
        self._u_warm_valid = True
        return 1

    def _step_tension_u_wave_gpu(
        self,
        mod: PhysicsMod,
        compiled: Optional[CompiledMasterAction],
        dt_u: float,
    ) -> int:
        """
        Advances causal hyperbolic weave tension `d^2 U / dt^2 = c_gw^2 * (nabla^2 U + kappa * rho_{H,0})`
        on the `64³` GPU grid with 1st-order spherical Bayliss-Turkel non-reflecting boundary conditions
        and packs `(U, dU/dt)` into `GL_RG32F` `_tension_tex` (`~0.06 ms` total, 0 CPU readback!).
        """
        if self.is_hibernated or self._buf_a is None or self._buf_b is None or self._buf_vel is None:
            return 0
        self._ensure_mg_resources()
        assert self._tension_tex is not None and self._mg_ut_buf is not None
        if self._external_tension_u_active:
            return 1

        use_t, g_k = self._current_use_tension(mod)
        if use_t == 0:
            return self._solve_tension_u_gpu(mod)
        if not self._u_warm_valid:
            self._solve_tension_u_gpu(mod, n_vcycles=4, zero_init_fine=True)

        p = mod.parameters
        dep_prog = self._deposit_u_prog if self._deposit_u_prog is not None else self._default_deposit_u_prog
        latest_buf = self._buf_a if self._cur_in_a else self._buf_b

        m0 = _MG_LEVELS_U[0]
        n = int(self.grid_n)
        dx = float(self.box_size) / float(n)
        half_box = float(self.box_size) * 0.5
        kappa = float(WEAVE_KAPPA_SCALE) * float(g_k)
        c_gw = compute_effective_gw_speed(mod, compiled)

        latest_buf.bind_to_storage_buffer(0)
        self._buf_vel.bind_to_storage_buffer(1)
        self._mg_rhs_bufs[0].bind_to_storage_buffer(2)
        self._set_uniform_safe(dep_prog, "u_N", n)
        self._set_uniform_safe(dep_prog, "u_M", m0)
        self._set_uniform_safe(dep_prog, "u_dx", dx)
        self._set_uniform_safe(dep_prog, "u_kappa", kappa)
        self._set_uniform_safe(dep_prog, "c_w", float(p.c_w))
        self._set_uniform_safe(dep_prog, "m_eff", float(p.m_eff))
        self._set_uniform_safe(dep_prog, "g4", float(p.g4))
        self._set_uniform_safe(dep_prog, "g6", float(p.g6))
        self._set_uniform_safe(dep_prog, "k_skyrme", float(getattr(p, "k_skyrme", 0.0)))
        wg0 = (m0 + 3) // 4
        dep_prog.run(wg0, wg0, wg0)
        self.ctx.memory_barrier()

        self._mg_rhs_bufs[0].bind_to_storage_buffer(0)
        self._mg_u_bufs[0].bind_to_storage_buffer(1)
        self._mg_ut_buf.bind_to_storage_buffer(2)
        self._mg_scratch_bufs[0].bind_to_storage_buffer(3)
        self._set_uniform_safe(self._u_wave_step_prog, "u_M", m0)
        self._set_uniform_safe(self._u_wave_step_prog, "u_half_box", half_box)
        self._set_uniform_safe(self._u_wave_step_prog, "u_dt", float(dt_u))
        self._set_uniform_safe(self._u_wave_step_prog, "u_c_gw", float(c_gw))
        self._u_wave_step_prog.run(wg0, wg0, wg0)
        self.ctx.memory_barrier()

        # Swap ping-pong fine U buffer so _mg_u_bufs[0] always holds the latest U(x, t)!
        self._mg_u_bufs[0], self._mg_scratch_bufs[0] = self._mg_scratch_bufs[0], self._mg_u_bufs[0]
        self._pack_tension_tex_gpu(zero_mode=False, has_ut=True)
        self._u_warm_valid = True
        return 1

    def zero_tension_wave_state(self) -> None:
        """Initializes flat weave tension `U(x, 0) = 0`, `dU/dt(x, 0) = 0` for causal light-cone turn-on tests."""
        if self.is_hibernated or self._buf_a is None:
            self.wake_from_ram()
        self._ensure_mg_resources()
        assert self._mg_ut_buf is not None
        m0 = _MG_LEVELS_U[0]
        self._zero_ssbo(self._mg_u_bufs[0], m0 * m0 * m0)
        self._zero_ssbo(self._mg_ut_buf, m0 * m0 * m0)
        self._pack_tension_tex_gpu(zero_mode=False, has_ut=True)
        self._external_tension_u_active = False
        self._zero_wave_init_active = True
        self._u_warm_valid = True

    def set_external_tension_u(self, u_zyx: Optional[np.ndarray]) -> None:
        """Uploads an explicit `(64, 64, 64)` `float32` tension field `U(x)` (or clears override when `None`)."""
        if self.is_hibernated or self._buf_a is None:
            self.wake_from_ram()
        self._ensure_mg_resources()
        assert self._tension_tex is not None and self._mg_ut_buf is not None
        m0 = _MG_LEVELS_U[0]
        if u_zyx is None:
            self._external_tension_u_active = False
            self._zero_wave_init_active = False
            self._u_warm_valid = False
            self._solve_tension_u_gpu(self._active_mod, n_vcycles=4, zero_init_fine=True)
            return
        arr = np.ascontiguousarray(u_zyx, dtype=np.float32)
        if arr.shape != (m0, m0, m0):
            raise ValueError(f"Expected external tension U shape ({m0}, {m0}, {m0}), got {arr.shape}")
        self._mg_u_bufs[0].write(arr)
        self._zero_ssbo(self._mg_ut_buf, m0 * m0 * m0)
        self._pack_tension_tex_gpu(zero_mode=False, has_ut=False)
        self._external_tension_u_active = True
        self._zero_wave_init_active = False
        self._u_warm_valid = False

    def readback_tension_u_grid(self) -> np.ndarray:
        """Reads back the active `64³` `(Z, Y, X)` `float32` Weave-tension field `U(x)` via PCIe staging."""
        m0 = _MG_LEVELS_U[0]
        use_t, _ = self._current_use_tension(self._active_mod)
        if self.is_hibernated or self._tension_tex is None or len(self._mg_u_bufs) == 0 or use_t == 0:
            return np.zeros((m0, m0, m0), dtype=np.float32)
        if not self._external_tension_u_active and not self._u_warm_valid:
            self._solve_tension_u_gpu(self._active_mod, n_vcycles=4, zero_init_fine=True)
        return self._copy_ssbo_to_ram_f32(self._mg_u_bufs[0], m0 * m0 * m0).reshape((m0, m0, m0))

    def readback_tension_udot_grid(self) -> np.ndarray:
        """Reads back the active `64³` `(Z, Y, X)` `float32` gravitational-wave strain rate `dU/dt(x, t)`."""
        m0 = _MG_LEVELS_U[0]
        use_t, _ = self._current_use_tension(self._active_mod)
        if self.is_hibernated or self._tension_tex is None or self._mg_ut_buf is None or use_t == 0:
            return np.zeros((m0, m0, m0), dtype=np.float32)
        return self._copy_ssbo_to_ram_f32(self._mg_ut_buf, m0 * m0 * m0).reshape((m0, m0, m0))

    def _copy_ssbo_to_ram_f32(self, src_buf: moderngl.Buffer, num_floats: int) -> np.ndarray:
        """Copies a GDDR7 SSBO into a contiguous DDR5 `float32` array in 16 MiB chunks without demotion."""
        self._ensure_xfer_buffers(need_fp16_gpu=False)
        assert self._xfer_stage_buf is not None
        out = np.empty(num_floats, dtype=np.float32)
        out_u8 = out.view(np.uint8)
        total_bytes = num_floats * 4
        offset = 0
        while offset < total_bytes:
            chunk = min(_XFER_CHUNK_BYTES, total_bytes - offset)
            _copy_buffer_sub_data(src_buf, self._xfer_stage_buf, offset, 0, chunk)
            raw = self._xfer_stage_buf.read(size=chunk)
            out_u8[offset : offset + chunk] = np.frombuffer(raw, dtype=np.uint8)
            offset += chunk
        return out

    def clear_replay_buffer(self) -> None:
        """Clears all buffered DDR5 Time-Machine keyframes and any paused live-scrub backup."""
        self._clear_pending_replay_fence()
        self._replay_frames.clear()
        self._replay_meta.clear()
        self._live_backup = None
        self.replay_cursor = -1

    def _replay_snap_n(self) -> int:
        """Downsamples `(128, 256] -> <= 128³` (`<= 24 MiB` total `(phi, vel)` fp16) while keeping `<= 128³` native."""
        if 128 < self.grid_n <= 256:
            return max(self.grid_n // 2, 64)
        return self.grid_n

    def _effective_replay_cap(self) -> int:
        if self.grid_n > 256:
            return 0
        cap = max(int(self.replay_max_frames), 0)
        if self.grid_n > 128:
            # 32 downsampled 128^3 (phi, vel) fp16 phase-space frames = 32 * 24 MiB = 768 MiB DDR5 ceiling
            return min(cap, 32)
        return cap

    def _unpack_fp16_1d_to_ssbo(self, arr_fp16: np.ndarray, dst_buf: moderngl.Buffer, num_floats: int) -> None:
        """Unpacks a DDR5 `float16` 1D array of length `num_floats` into a GDDR7 `float32` SSBO."""
        self._ensure_xfer_buffers(need_fp16_gpu=True)
        assert self._fp16_chunk_gpu_buf is not None
        contig = np.ascontiguousarray(arr_fp16, dtype=np.float16).reshape(-1)
        total_pairs = (num_floats + 1) // 2
        if contig.size < total_pairs * 2:
            padded = np.zeros(total_pairs * 2, dtype=np.float16)
            padded[: contig.size] = contig
            frame_u8 = padded.view(np.uint8)
        else:
            frame_u8 = contig[: total_pairs * 2].view(np.uint8)

        self._fp16_chunk_gpu_buf.bind_to_storage_buffer(0)
        dst_buf.bind_to_storage_buffer(1)
        self._set_uniform_safe(self._unpack_fp16_prog, "u_num_floats", num_floats)

        pair_offset = 0
        while pair_offset < total_pairs:
            pair_chunk = min(_XFER_CHUNK_UINTS, total_pairs - pair_offset)
            byte_src = pair_offset * 4
            byte_chunk = pair_chunk * 4
            self._fp16_chunk_gpu_buf.write(frame_u8[byte_src : byte_src + byte_chunk])
            self._set_uniform_safe(self._unpack_fp16_prog, "u_pair_offset", pair_offset)
            self._set_uniform_safe(self._unpack_fp16_prog, "u_pair_count", pair_chunk)
            wg_x = (pair_chunk + 255) // 256
            self._unpack_fp16_prog.run(wg_x, 1, 1)
            self.ctx.memory_barrier()
            pair_offset += pair_chunk

    def _poll_pending_replay_snapshot(self, wait: bool = False) -> None:
        """Harvests completed GPU->staging replay keyframe into DDR5 RAM with `0 ms` GPU pipeline stall."""
        fence = self._pending_replay_fence
        meta = self._pending_replay_meta
        if fence is None or meta is None or self._replay_stage_buf is None:
            return

        flags = int(GL.GL_SYNC_FLUSH_COMMANDS_BIT) if wait else 0
        timeout_ns = 250_000_000 if wait else 0
        res = GL.glClientWaitSync(fence, flags, timeout_ns)
        if wait and res not in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED):
            self.ctx.finish()
            res = GL.GL_CONDITION_SATISFIED

        if res not in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED):
            return

        snap_floats, total_pairs, total_bytes, snap_time, snap_step = meta
        GL.glDeleteSync(fence)
        self._pending_replay_fence = None
        self._pending_replay_meta = None

        cap = self._effective_replay_cap()
        if not self.replay_enabled or cap <= 0:
            return

        snap: Optional[np.ndarray] = None
        while len(self._replay_frames) >= cap:
            evicted = self._replay_frames.popleft()
            if (
                snap is None
                and total_pairs * 2 == snap_floats
                and evicted.dtype == np.float16
                and evicted.shape == (2, total_pairs * 2)
                and evicted.flags.c_contiguous
            ):
                snap = evicted
        while len(self._replay_meta) >= cap:
            self._replay_meta.popleft()

        if snap is None:
            snap = np.empty((2, total_pairs * 2), dtype=np.float16)
        self._replay_stage_buf.read_into(snap, size=total_bytes)
        if total_pairs * 2 != snap_floats:
            snap = snap[:, :snap_floats].copy()

        self._replay_frames.append(snap)
        self._replay_meta.append((snap_time, snap_step))

    def flush_pending_replay_snapshot(self) -> None:
        """Synchronously drains any in-flight async replay snapshot before scrubbing or hibernation."""
        self._poll_pending_replay_snapshot(wait=True)

    def _enqueue_replay_snapshot(self) -> None:
        """Dispatches GPU `(phi, vel)` `fp16` downsample/pack + staging DMA + `GL_FenceSync` without `.read()`."""
        if (
            not self.replay_enabled
            or self.grid_n > 256
            or self._pending_replay_fence is not None
            or self._buf_a is None
            or self._buf_b is None
            or self._buf_vel is None
        ):
            return
        cap = self._effective_replay_cap()
        if cap <= 0:
            return

        src_n = int(self.grid_n)
        snap_n = self._replay_snap_n()
        downsample = 1 if (snap_n < src_n) else 0
        snap_vox = snap_n * snap_n * snap_n
        vox_pairs = (snap_vox + 1) // 2
        snap_floats = snap_vox * 3
        total_pairs = (snap_floats + 1) // 2
        total_bytes = total_pairs * 8  # 2 arrays (phi, vel) * total_pairs * 4 bytes

        self._ensure_replay_buffers(total_bytes)
        assert self._replay_gpu_pack_buf is not None and self._replay_stage_buf is not None

        latest_buf = self._buf_a if self._cur_in_a else self._buf_b
        latest_buf.bind_to_storage_buffer(0)
        self._buf_vel.bind_to_storage_buffer(1)
        self._replay_gpu_pack_buf.bind_to_storage_buffer(2)

        self._set_uniform_safe(self._pack_replay_prog, "u_src_N", src_n)
        self._set_uniform_safe(self._pack_replay_prog, "u_snap_N", snap_n)
        self._set_uniform_safe(self._pack_replay_prog, "u_downsample", downsample)
        self._set_uniform_safe(self._pack_replay_prog, "u_snap_vox", snap_vox)
        self._set_uniform_safe(self._pack_replay_prog, "u_vox_pairs", vox_pairs)
        self._set_uniform_safe(self._pack_replay_prog, "u_total_pairs", total_pairs)

        wg_x = (vox_pairs + 255) // 256
        self._pack_replay_prog.run(wg_x, 1, 1)
        self.ctx.memory_barrier()

        _copy_buffer_sub_data(self._replay_gpu_pack_buf, self._replay_stage_buf, 0, 0, total_bytes)
        self._pending_replay_fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        GL.glFlush()
        self._pending_replay_meta = (
            snap_floats,
            total_pairs,
            total_bytes,
            float(self.sim_time),
            int(self.step_count),
        )

    def _capture_replay_snapshot(self) -> None:
        """Synchronous convenience wrapper around `_enqueue_replay_snapshot` + `flush_pending_replay_snapshot`."""
        self._poll_pending_replay_snapshot(wait=True)
        self._enqueue_replay_snapshot()
        self._poll_pending_replay_snapshot(wait=True)

    def load_replay_frame(self, idx: int) -> None:
        """Uploads DDR5 Time-Machine `(phi, vel)` keyframe `idx` onto GPU and refreshes `_vol_tex` + telemetry."""
        self.flush_pending_replay_snapshot()
        count = len(self._replay_frames)
        if count == 0:
            return
        i = int(idx)
        if i < 0:
            i += count
        i = max(0, min(i, count - 1))

        if self.is_hibernated or self._buf_a is None:
            self.wake_from_ram()

        assert (
            self._buf_a is not None
            and self._buf_b is not None
            and self._buf_vel is not None
            and self._vol_tex is not None
        )
        vox = self.grid_n * self.grid_n * self.grid_n
        num_floats = vox * 3

        # Snapshot exact float32 live state on first entry into replay scrub so Resume Live is 100% lossless!
        if self.replay_cursor < 0 and self._live_backup is None:
            self._poll_async_telemetry(wait=True)
            live_buf = self._buf_a if self._cur_in_a else self._buf_b
            m0_bk = _MG_LEVELS_U[0]
            u_bk = (
                self._copy_ssbo_to_ram_f32(self._mg_u_bufs[0], m0_bk * m0_bk * m0_bk)
                if (self._u_warm_valid and len(self._mg_u_bufs) > 0)
                else None
            )
            ut_bk = (
                self._copy_ssbo_to_ram_f32(self._mg_ut_buf, m0_bk * m0_bk * m0_bk)
                if (self._u_warm_valid and self._mg_ut_buf is not None)
                else None
            )
            self._live_backup = {
                "grid_n": int(self.grid_n),
                "phi": self._copy_ssbo_to_ram_f32(live_buf, num_floats),
                "vel": self._copy_ssbo_to_ram_f32(self._buf_vel, num_floats),
                "u_grid": u_bk,
                "ut_grid": ut_bk,
                "u_warm": bool(self._u_warm_valid),
                "sim_time": float(self.sim_time),
                "step_count": int(self.step_count),
                "init_hamiltonian": float(self.init_hamiltonian),
                "init_peak_rho": float(self.init_peak_rho),
                "init_l2_norm": float(self.init_l2_norm),
            }

        frame_fp16 = np.ascontiguousarray(self._replay_frames[i], dtype=np.float16)
        latest_buf = self._buf_a if self._cur_in_a else self._buf_b

        if frame_fp16.ndim == 2 and frame_fp16.shape[0] == 2:
            snap_floats = int(frame_fp16.shape[1])
            cand_vox = snap_floats // 3
            cand_n = int(round(cand_vox ** (1.0 / 3.0))) if cand_vox > 0 else 0
            total_pairs = (snap_floats + 1) // 2
            total_bytes = total_pairs * 8
            if cand_n >= 4 and (cand_n * cand_n * cand_n * 3 == snap_floats) and total_bytes <= 32 * 1024 * 1024:
                self._ensure_replay_buffers(total_bytes)
                assert self._replay_gpu_pack_buf is not None
                if total_pairs * 2 == snap_floats:
                    upload_u8 = frame_fp16.view(np.uint8)
                else:
                    padded = np.zeros((2, total_pairs * 2), dtype=np.float16)
                    padded[:, :snap_floats] = frame_fp16
                    upload_u8 = padded.view(np.uint8)
                self._replay_gpu_pack_buf.write(upload_u8)
                self._replay_gpu_pack_buf.bind_to_storage_buffer(0)
                latest_buf.bind_to_storage_buffer(1)
                self._buf_vel.bind_to_storage_buffer(2)
                self._set_uniform_safe(self._unpack_replay_prog, "u_src_N", cand_n)
                self._set_uniform_safe(self._unpack_replay_prog, "u_dst_N", int(self.grid_n))
                self._set_uniform_safe(self._unpack_replay_prog, "u_total_pairs", total_pairs)
                wg_xy_u = (self.grid_n + 7) // 8
                wg_z_u = (self.grid_n + 3) // 4
                self._unpack_replay_prog.run(wg_xy_u, wg_xy_u, wg_z_u)
                self.ctx.memory_barrier()
            elif snap_floats == num_floats:
                self._unpack_fp16_1d_to_ssbo(frame_fp16[0], latest_buf, num_floats)
                self._unpack_fp16_1d_to_ssbo(frame_fp16[1], self._buf_vel, num_floats)
            else:
                return
        elif frame_fp16.size == num_floats:
            self._unpack_fp16_1d_to_ssbo(frame_fp16, latest_buf, num_floats)
            self._zero_ssbo(self._buf_vel, num_floats)
        else:
            return

        if i < len(self._replay_meta):
            self.sim_time, self.step_count = self._replay_meta[i]

        n = self.grid_n
        wg_xy = (n + 7) // 8
        wg_z = (n + 3) // 4
        latest_buf.bind_to_storage_buffer(0)
        self._vol_tex.bind_to_image(0, read=False, write=True)
        self._pack_prog["u_N"].value = n
        self._pack_prog.run(wg_xy, wg_xy, wg_z)
        self.ctx.memory_barrier()

        self._solve_tension_u_gpu(self._active_mod, n_vcycles=4, zero_init_fine=True)
        self._sample_diagnostics(is_initial=False, reset_ham_baseline=False)
        self.replay_cursor = i

    def resume_live_from_replay(self) -> None:
        """Exits DDR5 Time-Machine scrub mode and restores the exact `float32` live `(phi, vel)` state."""
        bk = self._live_backup
        self._live_backup = None
        if bk is not None and int(bk.get("grid_n", -1)) == int(self.grid_n):
            if self.is_hibernated or self._buf_a is None:
                self.wake_from_ram()
            assert self._buf_a is not None and self._buf_vel is not None and self._vol_tex is not None
            self._buf_a.write(np.ascontiguousarray(bk["phi"], dtype=np.float32))
            self._buf_vel.write(np.ascontiguousarray(bk["vel"], dtype=np.float32))
            self._cur_in_a = True
            self.sim_time = float(bk["sim_time"])  # type: ignore[arg-type]
            self.step_count = int(bk["step_count"])  # type: ignore[arg-type]
            self.init_hamiltonian = float(bk["init_hamiltonian"])  # type: ignore[arg-type]
            self.init_peak_rho = float(bk["init_peak_rho"])  # type: ignore[arg-type]
            self.init_l2_norm = float(bk["init_l2_norm"])  # type: ignore[arg-type]

            n = self.grid_n
            wg_xy = (n + 7) // 8
            wg_z = (n + 3) // 4
            self._buf_a.bind_to_storage_buffer(0)
            self._vol_tex.bind_to_image(0, read=False, write=True)
            self._pack_prog["u_N"].value = n
            self._pack_prog.run(wg_xy, wg_xy, wg_z)
            self.ctx.memory_barrier()
            u_bk_arr = bk.get("u_grid")
            ut_bk_arr = bk.get("ut_grid")
            use_t_cur, _ = self._current_use_tension(self._active_mod)
            if bool(bk.get("u_warm")) and isinstance(u_bk_arr, np.ndarray) and use_t_cur == 1:
                self._ensure_mg_resources()
                assert self._tension_tex is not None and self._mg_ut_buf is not None
                m0 = _MG_LEVELS_U[0]
                self._mg_u_bufs[0].write(np.ascontiguousarray(u_bk_arr, dtype=np.float32))
                if isinstance(ut_bk_arr, np.ndarray) and ut_bk_arr.size == m0 * m0 * m0:
                    self._mg_ut_buf.write(np.ascontiguousarray(ut_bk_arr, dtype=np.float32))
                    has_ut_bk = True
                else:
                    self._zero_ssbo(self._mg_ut_buf, m0 * m0 * m0)
                    has_ut_bk = False
                self._pack_tension_tex_gpu(zero_mode=False, has_ut=has_ut_bk)
                self._u_warm_valid = True
            else:
                self._solve_tension_u_gpu(self._active_mod, n_vcycles=8, zero_init_fine=True)
            self._sample_diagnostics(is_initial=False, reset_ham_baseline=False)
        elif self.replay_cursor >= 0 and len(self._replay_frames) > 0:
            self.load_replay_frame(len(self._replay_frames) - 1)
            self._live_backup = None
        self.replay_cursor = -1

    def hibernate_to_ram(self) -> float:
        """
        Offloads the active 3D knot field and conjugate momenta out of GDDR7 VRAM into DDR5 System RAM
        via non-demoting `glCopyBufferSubData` staging, unbinding OpenGL slots so WDDM reclaims VRAM immediately.
        """
        if self.is_hibernated or self._buf_a is None or self._buf_b is None or self._buf_vel is None:
            self._clear_pending_replay_fence()
            self.is_hibernated = True
            self.allocated_vram_mb = 0.0
            return self.ddr5_ram_mb

        self.flush_pending_replay_snapshot()
        self._poll_async_telemetry(wait=True)
        self._clear_async_fences()


        vox = self.grid_n * self.grid_n * self.grid_n
        num_floats = vox * 3
        latest_buf = self._buf_a if self._cur_in_a else self._buf_b
        self._hib_phi_ram = self._copy_ssbo_to_ram_f32(latest_buf, num_floats)
        self._hib_vel_ram = self._copy_ssbo_to_ram_f32(self._buf_vel, num_floats)
        if self._u_warm_valid and len(self._mg_u_bufs) > 0:
            m0 = _MG_LEVELS_U[0]
            self._hib_u_ram = self._copy_ssbo_to_ram_f32(self._mg_u_bufs[0], m0 * m0 * m0)
            self._hib_ut_ram = (
                self._copy_ssbo_to_ram_f32(self._mg_ut_buf, m0 * m0 * m0)
                if self._mg_ut_buf is not None
                else None
            )
            self._hib_u_warm = True
        else:
            self._hib_u_ram = None
            self._hib_ut_ram = None
            self._hib_u_warm = False

        _unbind_arena1_gl_slots()
        self.ctx.finish()
        for obj in (self._buf_a, self._buf_b, self._buf_vel, self._vol_tex):
            if obj is not None:
                obj.release()
        self._buf_a = None
        self._buf_b = None
        self._buf_vel = None
        self._vol_tex = None
        self._release_mg_resources()
        self._release_xfer_buffers()
        self.ctx.finish()

        self.is_hibernated = True
        self.allocated_vram_mb = 0.0
        return self.ddr5_ram_mb

    def wake_from_ram(self) -> None:
        """Restores 3D knot buffers into GDDR7 VRAM from DDR5 System RAM (or initializes if lazy-booted)."""
        if not self.is_hibernated and self._buf_a is not None:
            return

        if self._active_compiled is None:
            def_mod, def_comp = _get_default_mod_and_compiled(self._active_mod)
            self.update_compiled_action(def_comp, mod=def_mod)

        saved_phi = self._hib_phi_ram
        saved_vel = self._hib_vel_ram
        saved_u = self._hib_u_ram
        saved_ut = self._hib_ut_ram
        saved_u_warm = self._hib_u_warm
        self._hib_phi_ram = None
        self._hib_vel_ram = None
        self._hib_u_ram = None
        self._hib_ut_ram = None
        self._hib_u_warm = False
        self.is_hibernated = False

        self._allocate_grid(self.grid_n)
        assert self._buf_a is not None and self._buf_vel is not None and self._vol_tex is not None

        vox = self.grid_n * self.grid_n * self.grid_n
        expected_floats = vox * 3
        if (
            saved_phi is not None
            and saved_vel is not None
            and saved_phi.size == expected_floats
            and saved_vel.size == expected_floats
        ):
            self._buf_a.write(np.ascontiguousarray(saved_phi, dtype=np.float32))
            self._buf_vel.write(np.ascontiguousarray(saved_vel, dtype=np.float32))
            self._cur_in_a = True

            n = self.grid_n
            wg_xy = (n + 7) // 8
            wg_z = (n + 3) // 4
            self._buf_a.bind_to_storage_buffer(0)
            self._vol_tex.bind_to_image(0, read=False, write=True)
            self._pack_prog["u_N"].value = n
            self._pack_prog.run(wg_xy, wg_xy, wg_z)
            self.ctx.memory_barrier()
            reset_base = bool(self._pending_ham_baseline_reset)
            self._pending_ham_baseline_reset = False
            use_t_cur, _ = self._current_use_tension(self._active_mod)
            if saved_u is not None and saved_u_warm and not reset_base and use_t_cur == 1:
                self._ensure_mg_resources()
                assert self._tension_tex is not None and self._mg_ut_buf is not None
                m0 = _MG_LEVELS_U[0]
                self._mg_u_bufs[0].write(np.ascontiguousarray(saved_u, dtype=np.float32))
                if saved_ut is not None and saved_ut.size == m0 * m0 * m0:
                    self._mg_ut_buf.write(np.ascontiguousarray(saved_ut, dtype=np.float32))
                    has_ut_hib = True
                else:
                    self._zero_ssbo(self._mg_ut_buf, m0 * m0 * m0)
                    has_ut_hib = False
                self._pack_tension_tex_gpu(zero_mode=False, has_ut=has_ut_hib)
                self._u_warm_valid = True
            else:
                self._solve_tension_u_gpu(self._active_mod, n_vcycles=8, zero_init_fine=True)
            self._sample_diagnostics(is_initial=False, reset_ham_baseline=reset_base)
        else:
            self._pending_ham_baseline_reset = False
            self.reset_field()

    def _allocate_grid(self, n: int) -> None:
        n = int(n)
        if n < 4:
            raise ValueError(f"Arena1 grid_n must be >= 4, got {n}")
        prev_n = self.grid_n if (4 <= self.grid_n < n or (self._buf_a is not None and self.grid_n != n)) else 0

        self._clear_async_fences()
        self._clear_pending_replay_fence()
        had_gpu_objs = any(x is not None for x in (self._buf_a, self._buf_b, self._buf_vel, self._vol_tex))
        if had_gpu_objs:
            _unbind_arena1_gl_slots()
            self.ctx.finish()
        for obj in (self._buf_a, self._buf_b, self._buf_vel, self._vol_tex):
            if obj is not None:
                obj.release()
        self._buf_a = None
        self._buf_b = None
        self._buf_vel = None
        self._vol_tex = None
        if n >= 384:
            self._release_xfer_buffers()
        if had_gpu_objs:
            self.ctx.finish()
        self.allocated_vram_mb = 0.0

        vox = n * n * n
        req_vram_mb = (vox * 44) / (1024.0 * 1024.0)
        # Packed 3-float SSBOs: 12 bytes/voxel => 1,610,612,736 B (1.50 GiB) at 512^3 (< 2^31 - 1 OpenGL limit!)
        nbytes_vec3 = vox * 12
        new_a: Optional[moderngl.Buffer] = None
        new_b: Optional[moderngl.Buffer] = None
        new_vel: Optional[moderngl.Buffer] = None
        new_tex: Optional[moderngl.Texture3D] = None
        try:
            if n >= 384:
                avail_mb = _query_available_vram_mb()
                if avail_mb is not None and (avail_mb - req_vram_mb) < 1200.0:
                    raise MemoryError(
                        f"GDDR7 headroom after {n}^3 ({req_vram_mb:.0f} MiB) would be "
                        f"{avail_mb - req_vram_mb:.0f} MiB (< 1200 MiB safety floor; available={avail_mb:.0f} MiB)"
                    )
            new_a = self.ctx.buffer(reserve=nbytes_vec3)
            new_b = self.ctx.buffer(reserve=nbytes_vec3)
            new_vel = self.ctx.buffer(reserve=nbytes_vec3)
            new_tex = self.ctx.texture3d((n, n, n), components=4, dtype="f2")
        except (moderngl.Error, MemoryError) as exc:
            _unbind_arena1_gl_slots()
            self.ctx.finish()
            for partial in (new_a, new_b, new_vel, new_tex):
                if partial is not None:
                    partial.release()
            self.ctx.finish()
            fallback_n = prev_n if (4 <= prev_n < n) else (128 if n > 128 else 0)
            if fallback_n >= 4:
                self._allocate_grid(fallback_n)
                self.reset_field()
                raise RuntimeError(
                    f"Insufficient GPU VRAM for {n}^3 grid; safely rolled back to {self.grid_n}^3 ({exc})"
                ) from exc
            raise

        self.grid_n = n
        self._buf_a = new_a
        self._buf_b = new_b
        self._buf_vel = new_vel
        self._vol_tex = new_tex
        self._vol_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._vol_tex.repeat_x = False
        self._vol_tex.repeat_y = False
        self._vol_tex.repeat_z = False
        self._ensure_mg_resources()
        self._cur_in_a = True
        self.is_hibernated = False
        # Honest exact VRAM footprint: 3 * 12B (fp32 scalar-packed vec3) + 1 * 8B (fp16 rgba) = 44 bytes/voxel
        self.allocated_vram_mb = req_vram_mb

    def set_grid_resolution(self, new_n: int) -> None:
        target_n = int(new_n)
        if target_n != self.grid_n or self.is_hibernated or self._buf_a is None:
            self._hib_phi_ram = None
            self._hib_vel_ram = None
            self._hib_u_ram = None
            self._hib_ut_ram = None
            self._hib_u_warm = False
            self.is_hibernated = False
            self.clear_replay_buffer()
            self._allocate_grid(target_n)
            self.reset_field()

    set_resolution = set_grid_resolution

    def update_compiled_action(self, compiled: CompiledMasterAction, mod: Optional[PhysicsMod] = None) -> None:
        self._active_compiled = compiled
        if mod is not None:
            self._active_mod = mod
        new_params: Optional[tuple[object, ...]] = None
        if self._active_mod is not None:
            mp = self._active_mod.parameters
            new_params = (
                float(mp.c_w),
                float(mp.m_eff),
                float(mp.g4),
                float(mp.g6),
                float(getattr(mp, "k_skyrme", 0.0)),
                float(getattr(mp, "g_knot", 0.0)),
                bool(getattr(mp, "retarded_weave_u", True)),
                float(getattr(compiled, "gw_speed_ratio_val", 1.0)),
            )

        if compiled.glsl_compute_source != self._compiled_glsl_src:
            new_prog = self.ctx.compute_shader(compiled.glsl_compute_source)
            if self._pde_prog is not None:
                self.ctx.finish()
                self._pde_prog.release()
            self._pde_prog = new_prog
            self._compiled_glsl_src = compiled.glsl_compute_source

        ham_expr_changed = (
            (compiled.hamiltonian_glsl_expr != self._compiled_ham_expr)
            or (self._reduce_prog is None)
            or (self._deposit_u_prog is None)
        )
        params_changed = (
            new_params is not None
            and self._ham_params_tuple is not None
            and new_params != self._ham_params_tuple
        )
        if new_params is not None:
            self._ham_params_tuple = new_params

        if ham_expr_changed:
            new_red = self.ctx.compute_shader(_build_reduce_telemetry_glsl(compiled.hamiltonian_glsl_expr))
            new_dep = self.ctx.compute_shader(_build_deposit_u_rhs_glsl(compiled.hamiltonian_glsl_expr))
            if self._reduce_prog is not None:
                self.ctx.finish()
                self._reduce_prog.release()
            if self._deposit_u_prog is not None:
                self.ctx.finish()
                self._deposit_u_prog.release()
            self._reduce_prog = new_red
            self._deposit_u_prog = new_dep
            self._compiled_ham_expr = compiled.hamiltonian_glsl_expr

        if ham_expr_changed or params_changed:
            # Refresh Hamiltonian baseline if GPU buffers are awake; otherwise defer until wake!
            if not self.is_hibernated and self._buf_a is not None:
                self._pending_ham_baseline_reset = False
                use_t_upd, _ = self._current_use_tension(self._active_mod)
                ret_upd = (
                    bool(getattr(self._active_mod.parameters, "retarded_weave_u", True))
                    if self._active_mod is not None
                    else True
                )
                if (
                    use_t_upd == 0
                    or (not ret_upd)
                    or (not self._u_warm_valid)
                    or (self.step_count == 0 and not self._zero_wave_init_active)
                ):
                    self._solve_tension_u_gpu(self._active_mod, n_vcycles=4, zero_init_fine=True)
                self._sample_diagnostics(is_initial=(self.step_count == 0), reset_ham_baseline=True)
            else:
                self._pending_ham_baseline_reset = True

    def reset_field(self, preset_idx: Optional[int] = None) -> None:
        if preset_idx is not None:
            self.preset_idx = int(preset_idx) % len(KNOT_PRESETS)

        if self.is_hibernated or self._buf_a is None:
            self._hib_phi_ram = None
            self._hib_vel_ram = None
            self._hib_u_ram = None
            self._hib_ut_ram = None
            self._hib_u_warm = False
            self.is_hibernated = False
            self._allocate_grid(self.grid_n)

        self._clear_async_fences()
        self.clear_replay_buffer()
        self._external_tension_u_active = False
        self._zero_wave_init_active = False
        self._u_warm_valid = False

        assert self._buf_a is not None and self._buf_vel is not None and self._vol_tex is not None
        n = self.grid_n
        self._buf_a.bind_to_storage_buffer(0)
        self._buf_vel.bind_to_storage_buffer(1)
        self._vol_tex.bind_to_image(0, read=False, write=True)

        self._init_prog["u_N"].value = n
        self._init_prog["u_preset"].value = self.preset_idx
        self._init_prog["u_box"].value = float(self.box_size)

        wg_xy = (n + 7) // 8
        wg_z = (n + 3) // 4
        self._init_prog.run(wg_xy, wg_xy, wg_z)
        self.ctx.memory_barrier()

        self._cur_in_a = True
        self._relax_initial_hopfion_t0()
        if self.preset_idx == 6:
            latest_b = self._buf_a if self._cur_in_a else self._buf_b
            latest_b.bind_to_storage_buffer(0)
            self._buf_vel.bind_to_storage_buffer(1)
            self._set_uniform_safe(self._boost_orbital_prog, "u_N", n)
            self._set_uniform_safe(self._boost_orbital_prog, "u_box", float(self.box_size))
            self._set_uniform_safe(self._boost_orbital_prog, "u_v_orb", 0.105)
            self._boost_orbital_prog.run(wg_xy, wg_xy, wg_z)
            self.ctx.memory_barrier()
        self._solve_tension_u_gpu(self._active_mod, n_vcycles=4, zero_init_fine=True)
        self.sim_time = 0.0
        self.step_count = 0
        self._pending_ham_baseline_reset = False
        self._sample_diagnostics(is_initial=True, reset_ham_baseline=True)
        if self.replay_enabled and self.grid_n <= 256:
            self._enqueue_replay_snapshot()

    def _zero_ssbo(self, buf: moderngl.Buffer, num_floats: int) -> None:
        buf.bind_to_storage_buffer(0)
        self._zero_buf_prog["u_num_floats"].value = int(num_floats)
        self._zero_buf_prog.run((num_floats + 255) // 256, 1, 1)
        self.ctx.memory_barrier()

    def _relax_initial_hopfion_t0(self) -> None:
        """
        Jennings (2015) t = 0 relaxation onto the discrete Faddeev-Skyrme manifold using the reference
        TNaP action, followed by exact GPU zeroing of `_buf_vel` (`vel_0 === 0`), so all physics mods
        (`tnap_weave`, `lambda_cdm`, etc.) start from the bit-for-bit identical `t = 0` initial state!
        """
        if self.preset_idx not in (0, 1, 2, 4, 5, 6) or self.grid_n < 48:
            return
        assert (
            self._buf_a is not None
            and self._buf_b is not None
            and self._buf_vel is not None
            and self._vol_tex is not None
        )
        self._ensure_mg_resources()
        assert self._tension_tex is not None
        ref_mod, ref_comp = _get_default_mod_and_compiled()
        if self._ref_relax_prog is None:
            self._ref_relax_prog = self.ctx.compute_shader(ref_comp.glsl_compute_source)

        n_loc = int(self.grid_n)
        steps = 420 if n_loc <= 128 else (120 if n_loc <= 256 else 16)
        mp = ref_mod.parameters
        dx_loc = float(self.box_size) / float(n_loc)
        dt_loc = 1.45 * compute_safe_symplectic_dt(dx_loc, ref_mod, compiled=ref_comp, peak_rho_est=2.95)

        self._tension_tex.use(location=1)
        self._set_uniform_safe(self._ref_relax_prog, "u_N", n_loc)
        self._set_uniform_safe(self._ref_relax_prog, "u_dx", dx_loc)
        self._set_uniform_safe(self._ref_relax_prog, "u_dt", dt_loc)
        self._set_uniform_safe(self._ref_relax_prog, "u_damping", 1.65)
        self._set_uniform_safe(self._ref_relax_prog, "c_w", float(mp.c_w))
        self._set_uniform_safe(self._ref_relax_prog, "m_eff", float(mp.m_eff))
        self._set_uniform_safe(self._ref_relax_prog, "g4", float(mp.g4))
        self._set_uniform_safe(self._ref_relax_prog, "g6", float(mp.g6))
        self._set_uniform_safe(self._ref_relax_prog, "k_skyrme", float(getattr(mp, "k_skyrme", 1.55)))
        self._set_uniform_safe(self._ref_relax_prog, "u_sponge", float(self.sponge_strength))
        self._set_uniform_safe(self._ref_relax_prog, "u_box_half", float(self.box_size) * 0.5)
        self._set_uniform_safe(self._ref_relax_prog, "u_use_tension", 0)
        self._set_uniform_safe(self._ref_relax_prog, "u_tension_tex", 1)

        wg_xy = (n_loc + 7) // 8
        wg_z = (n_loc + 3) // 4
        self._buf_vel.bind_to_storage_buffer(1)
        for _ in range(steps):
            if self._cur_in_a:
                self._buf_a.bind_to_storage_buffer(0)
                self._buf_b.bind_to_storage_buffer(2)
            else:
                self._buf_b.bind_to_storage_buffer(0)
                self._buf_a.bind_to_storage_buffer(2)
            self._ref_relax_prog.run(wg_xy, wg_xy, wg_z)
            self.ctx.memory_barrier()
            self._cur_in_a = not self._cur_in_a

        self._zero_ssbo(self._buf_vel, n_loc * n_loc * n_loc * 3)
        latest = self._buf_a if self._cur_in_a else self._buf_b
        latest.bind_to_storage_buffer(0)
        self._vol_tex.bind_to_image(0, read=False, write=True)
        self._pack_prog["u_N"].value = n_loc
        self._pack_prog.run(wg_xy, wg_xy, wg_z)
        self.ctx.memory_barrier()

    def _set_uniform_safe(self, prog: moderngl.ComputeShader, name: str, val: object) -> None:
        if name in prog:
            prog[name].value = val

    def _apply_stats_bytes(
        self,
        raw: bytes,
        num_wg: int,
        dv: float,
        is_initial: bool = False,
        reset_ham_baseline: bool = False,
    ) -> None:
        stats = np.frombuffer(raw, dtype=np.float32).reshape((num_wg, 4))
        peak = float(np.max(stats[:, 0]))
        vol_l2 = float(np.sum(stats[:, 1]) * dv)
        vol_ham = float(np.sum(stats[:, 2]) * dv)

        if is_initial:
            self.init_peak_rho = max(peak, 1e-6)
            self.init_l2_norm = max(vol_l2, 1e-12)
        if is_initial or reset_ham_baseline:
            self.init_hamiltonian = vol_ham
        self.current_peak_rho = peak
        self.retention_ratio = peak / max(self.init_peak_rho, 1e-6)
        self.l2_norm = vol_l2
        self.hamiltonian_energy = vol_ham
        if abs(self.init_hamiltonian) > 1e-6:
            self.hamiltonian_drift_pct = ((vol_ham - self.init_hamiltonian) / abs(self.init_hamiltonian)) * 100.0
        else:
            self.hamiltonian_drift_pct = 0.0

    def _dispatch_reduction_gpu(self) -> tuple[int, float]:
        assert self._buf_a is not None and self._buf_b is not None and self._buf_vel is not None
        self._ensure_mg_resources()
        assert self._tension_tex is not None
        n = self.grid_n
        dx = float(self.box_size) / float(n)
        latest_buf = self._buf_a if self._cur_in_a else self._buf_b
        red_prog = self._reduce_prog if self._reduce_prog is not None else self._default_reduce_prog
        use_t, _ = self._current_use_tension(self._active_mod)

        stride = max((n + 63) // 64, 1)
        eff_n = (n + stride - 1) // stride
        wg_x = (eff_n + 7) // 8
        wg_y = (eff_n + 7) // 8
        wg_z = (eff_n + 3) // 4
        num_wg = wg_x * wg_y * wg_z

        latest_buf.bind_to_storage_buffer(0)
        self._buf_vel.bind_to_storage_buffer(1)
        self._buf_stats.bind_to_storage_buffer(2)
        self._tension_tex.use(location=1)

        p = self._active_mod.parameters if self._active_mod is not None else None
        use_t, g_k = self._current_use_tension(self._active_mod)
        kappa = float(WEAVE_KAPPA_SCALE) * float(g_k)
        ret_mode = (
            1
            if (
                use_t == 1
                and not self._external_tension_u_active
                and p is not None
                and bool(getattr(p, "retarded_weave_u", True))
                and abs(kappa) > 1e-9
            )
            else 0
        )
        c_gw = (
            compute_effective_gw_speed(self._active_mod, self._active_compiled)
            if self._active_mod is not None
            else 1.0
        )
        self._set_uniform_safe(red_prog, "u_N", n)
        self._set_uniform_safe(red_prog, "u_stride", stride)
        self._set_uniform_safe(red_prog, "u_dx", dx)
        self._set_uniform_safe(red_prog, "c_w", float(p.c_w) if p else 1.0)
        self._set_uniform_safe(red_prog, "m_eff", float(p.m_eff) if p else 2.50)
        self._set_uniform_safe(red_prog, "g4", float(p.g4) if p else 9.80)
        self._set_uniform_safe(red_prog, "g6", float(p.g6) if p else 3.10)
        self._set_uniform_safe(red_prog, "k_skyrme", float(getattr(p, "k_skyrme", 0.0)) if p else 0.22)
        self._set_uniform_safe(red_prog, "u_use_tension", use_t)
        self._set_uniform_safe(red_prog, "u_retarded_mode", ret_mode)
        self._set_uniform_safe(red_prog, "u_inv_kappa", (1.0 / abs(kappa)) if abs(kappa) > 1e-9 else 0.0)
        self._set_uniform_safe(red_prog, "u_inv_cgw2", 1.0 / max(c_gw * c_gw, 1e-8))
        self._set_uniform_safe(red_prog, "u_box_size", float(self.box_size))
        self._set_uniform_safe(red_prog, "u_tension_tex", 1)

        red_prog.run(wg_x, wg_y, wg_z)
        self.ctx.memory_barrier()
        dv = (dx * stride) ** 3
        return num_wg, dv

    def _consume_async_slot(self, slot: int, wait: bool) -> None:
        fence = self._async_fences[slot]
        meta = self._async_meta[slot]
        if fence is None or meta is None:
            return
        flags = int(GL.GL_SYNC_FLUSH_COMMANDS_BIT) if wait else 0
        timeout_ns = 100_000_000 if wait else 0
        res = GL.glClientWaitSync(fence, flags, timeout_ns)
        if wait or res in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED):
            num_wg, dv, snap_step = meta
            raw = self._async_stats_bufs[slot].read(size=num_wg * 16)
            if snap_step >= self._last_telem_step:
                self._apply_stats_bytes(raw, num_wg, dv, is_initial=False, reset_ham_baseline=False)
                self._last_telem_step = snap_step
            GL.glDeleteSync(fence)
            self._async_fences[slot] = None
            self._async_meta[slot] = None

    def _poll_async_telemetry(self, wait: bool = False) -> None:
        # self._async_slot is the older slot (next to be overwritten); (1 - self._async_slot) is the newer slot
        older = self._async_slot
        newer = 1 - older
        self._consume_async_slot(older, wait=wait)
        self._consume_async_slot(newer, wait=wait)

    def _enqueue_async_diagnostics(self) -> None:
        slot = self._async_slot
        if self._async_fences[slot] is not None:
            self._consume_async_slot(slot, wait=True)
        num_wg, dv = self._dispatch_reduction_gpu()
        _copy_buffer_sub_data(self._buf_stats, self._async_stats_bufs[slot], 0, 0, num_wg * 16)
        self._async_fences[slot] = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        GL.glFlush()
        self._async_meta[slot] = (num_wg, dv, int(self.step_count))
        self._async_slot = 1 - slot

    def step(
        self,
        mod: Optional[PhysicsMod] = None,
        compiled: Optional[CompiledMasterAction] = None,
        n_substeps: Optional[int] = None,
    ) -> None:
        if self.is_hibernated or self._buf_a is None:
            self.wake_from_ram()

        self._poll_async_telemetry(wait=False)
        if self.replay_cursor >= 0:
            self._clear_pending_replay_fence()
            while len(self._replay_frames) > self.replay_cursor + 1:
                self._replay_frames.pop()
            while len(self._replay_meta) > self.replay_cursor + 1:
                self._replay_meta.pop()
            self.replay_cursor = -1
        self._live_backup = None

        use_mod = mod if mod is not None else self._active_mod
        use_comp = compiled if compiled is not None else self._active_compiled
        if use_mod is None or use_comp is None:
            def_mod, def_comp = _get_default_mod_and_compiled(use_mod)
            if use_mod is None:
                use_mod = def_mod
            if use_comp is None:
                use_comp = def_comp

        self._active_mod = use_mod
        self.update_compiled_action(use_comp, mod=use_mod)
        if self._pde_prog is None:
            return

        assert (
            self._buf_a is not None
            and self._buf_b is not None
            and self._buf_vel is not None
            and self._vol_tex is not None
        )

        n = self.grid_n
        dx = float(self.box_size) / float(n)
        rho_guard = self.current_peak_rho if self.current_peak_rho > 4.5 else max(self.init_peak_rho, 2.8)
        dt = compute_safe_symplectic_dt(dx, use_mod, compiled=use_comp, peak_rho_est=rho_guard)
        self.last_dt = dt
        steps = int(n_substeps if n_substeps is not None else self.substeps_per_frame)

        use_t, _ = self._current_use_tension(use_mod)
        retarded_on = (
            use_t == 1
            and bool(getattr(use_mod.parameters, "retarded_weave_u", True))
            and (not self._external_tension_u_active)
        )
        if use_t == 0 or self._external_tension_u_active:
            use_t = self._solve_tension_u_gpu(use_mod)
        elif not retarded_on:
            use_t = self._solve_tension_u_gpu(
                use_mod,
                n_vcycles=(2 if self._u_warm_valid else 4),
                zero_init_fine=(not self._u_warm_valid),
            )
        elif not self._u_warm_valid:
            use_t = self._solve_tension_u_gpu(use_mod, n_vcycles=4, zero_init_fine=True)

        assert self._tension_tex is not None
        self._tension_tex.use(location=1)

        self._set_uniform_safe(self._pde_prog, "u_N", n)
        self._set_uniform_safe(self._pde_prog, "u_dx", dx)
        self._set_uniform_safe(self._pde_prog, "u_dt", dt)
        self._set_uniform_safe(self._pde_prog, "u_damping", float(self.damping))
        self._set_uniform_safe(self._pde_prog, "c_w", float(use_mod.parameters.c_w))
        self._set_uniform_safe(self._pde_prog, "m_eff", float(use_mod.parameters.m_eff))
        self._set_uniform_safe(self._pde_prog, "g4", float(use_mod.parameters.g4))
        self._set_uniform_safe(self._pde_prog, "g6", float(use_mod.parameters.g6))
        self._set_uniform_safe(self._pde_prog, "k_skyrme", float(getattr(use_mod.parameters, "k_skyrme", 0.0)))
        self._set_uniform_safe(self._pde_prog, "u_sponge", float(self.sponge_strength))
        self._set_uniform_safe(self._pde_prog, "u_box_half", float(self.box_size) * 0.5)
        self._set_uniform_safe(self._pde_prog, "u_use_tension", use_t)
        self._set_uniform_safe(self._pde_prog, "u_tension_tex", 1)

        wg_xy = (n + 7) // 8
        wg_z = (n + 3) // 4

        self._buf_vel.bind_to_storage_buffer(1)
        prev_step_count = self.step_count

        for sub_i in range(steps):
            if retarded_on:
                if n <= 64 or steps == 1:
                    self._step_tension_u_wave_gpu(use_mod, use_comp, dt_u=dt)
                    self._buf_vel.bind_to_storage_buffer(1)
                    self._tension_tex.use(location=1)
                elif (sub_i % 2) == 0:
                    dt_u_pair = (2.0 * dt) if (sub_i + 1 < steps) else dt
                    self._step_tension_u_wave_gpu(use_mod, use_comp, dt_u=dt_u_pair)
                    self._buf_vel.bind_to_storage_buffer(1)
                    self._tension_tex.use(location=1)
            elif use_t == 1 and (not self._external_tension_u_active) and sub_i > 0 and (sub_i % 16) == 0:
                self._solve_tension_u_gpu(use_mod, n_vcycles=2, zero_init_fine=False)
                self._buf_vel.bind_to_storage_buffer(1)
                self._tension_tex.use(location=1)

            if self._cur_in_a:
                self._buf_a.bind_to_storage_buffer(0)
                self._buf_b.bind_to_storage_buffer(2)
            else:
                self._buf_b.bind_to_storage_buffer(0)
                self._buf_a.bind_to_storage_buffer(2)
            self._pde_prog.run(wg_xy, wg_xy, wg_z)
            self.ctx.memory_barrier()
            self._cur_in_a = not self._cur_in_a
            self.sim_time += dt
            self.step_count += 1

        # Pack latest 3D field into trilinear GL_RGBA16F Texture3D for zero-copy ray-marching!
        latest_buf = self._buf_a if self._cur_in_a else self._buf_b
        latest_buf.bind_to_storage_buffer(0)
        self._vol_tex.bind_to_image(0, read=False, write=True)
        self._pack_prog["u_N"].value = n
        self._pack_prog.run(wg_xy, wg_xy, wg_z)
        self.ctx.memory_barrier()

        if (self.step_count // 8) > (prev_step_count // 8):
            self._enqueue_async_diagnostics()

        want_enqueue = (
            self.replay_enabled
            and self.grid_n <= 256
            and (
                (len(self._replay_frames) == 0 and self._pending_replay_fence is None)
                or (self.step_count // 32) > (prev_step_count // 32)
            )
        )
        if self._pending_replay_fence is not None and (
            want_enqueue
            or (self._pending_replay_meta is not None and (self.step_count - self._pending_replay_meta[4]) >= 4)
        ):
            GL.glFlush()
            self._poll_pending_replay_snapshot(wait=False)
        if want_enqueue and self._pending_replay_fence is None:
            self._enqueue_replay_snapshot()

    def _sample_diagnostics(self, is_initial: bool = False, reset_ham_baseline: bool = False) -> None:
        """
        Executes a fast 3D GPU parallel workgroup reduction (`<= 16 KB` readback) across the entire 3D tank
        and copies into `self._stats_stage_buf` (`dynamic=True`) via `glCopyBufferSubData` — ZERO SSBO demotion!
        """
        if self.is_hibernated or self._buf_a is None or self._buf_b is None or self._buf_vel is None:
            return
        self._clear_async_fences()
        num_wg, dv = self._dispatch_reduction_gpu()
        _copy_buffer_sub_data(self._buf_stats, self._stats_stage_buf, 0, 0, num_wg * 16)
        raw = self._stats_stage_buf.read(size=num_wg * 16)
        self._apply_stats_bytes(raw, num_wg, dv, is_initial=is_initial, reset_ham_baseline=reset_ham_baseline)
        self._last_telem_step = int(self.step_count)

    def render(self, fbo: moderngl.Framebuffer, camera: Camera3D, wall_time: float) -> None:
        if self.is_hibernated or self._vol_tex is None:
            self.wake_from_ram()
        self._poll_async_telemetry(wait=False)
        assert self._vol_tex is not None
        fbo.use()
        w, h = fbo.size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        eye, fwd, right, up = camera.basis_vectors()
        aspect = float(w) / max(float(h), 1.0)
        fov_tan = math.tan(math.radians(camera.fov_deg) * 0.5)

        self._ensure_mg_resources()
        assert self._tension_tex is not None
        use_t_r, _ = self._current_use_tension(self._active_mod)
        ret_r = (
            bool(getattr(self._active_mod.parameters, "retarded_weave_u", True))
            if self._active_mod is not None
            else True
        )
        show_gw = 1 if (self.show_gw_ripples and use_t_r == 1 and ret_r) else 0

        self._vol_tex.use(location=0)
        self._tension_tex.use(location=1)
        for uname, uval in (
            ("u_vol_tex", 0),
            ("u_tension_tex", 1),
            ("u_cam_eye", tuple(float(v) for v in eye)),
            ("u_cam_fwd", tuple(float(v) for v in fwd)),
            ("u_cam_right", tuple(float(v) for v in right)),
            ("u_cam_up", tuple(float(v) for v in up)),
            ("u_fov_tan", float(fov_tan)),
            ("u_aspect", float(aspect)),
            ("u_box_half", float(self.box_size * 0.5)),
            ("u_density_gain", float(self.density_gain)),
            ("u_slice_x", float(self.slice_x)),
            ("u_color_mode", int(self.color_mode)),
            ("u_time", float(wall_time)),
            ("u_rho_floor", 0.72 if self.preset_idx in (5, 6) else (0.18 if self.preset_idx == 3 else 1.15)),
            ("u_show_gw_ripples", show_gw),
            ("u_gw_vis_gain", float(self.gw_ripple_gain)),
        ):
            if uname in self._ray_prog:
                self._ray_prog[uname].value = uval

        self._empty_vao.render(moderngl.TRIANGLES, vertices=3)
        self._poll_async_telemetry(wait=False)
        if (
            self._pending_replay_fence is not None
            and self._pending_replay_meta is not None
            and self.step_count > self._pending_replay_meta[4]
        ):
            GL.glFlush()
            self._poll_pending_replay_snapshot(wait=False)

    def release(self) -> None:
        self._clear_async_fences()
        self._clear_pending_replay_fence()
        _unbind_arena1_gl_slots()
        self.ctx.finish()
        self._release_mg_resources()
        self._release_xfer_buffers()
        for obj in (
            self._buf_a,
            self._buf_b,
            self._buf_vel,
            self._buf_stats,
            self._stats_stage_buf,
            self._async_stats_bufs[0],
            self._async_stats_bufs[1],
            self._vol_tex,
            self._pde_prog,
            self._ref_relax_prog,
            self._reduce_prog,
            self._default_reduce_prog,
            self._deposit_u_prog,
            self._default_deposit_u_prog,
            self._u_moments_prog,
            self._u_mg_smooth_prog,
            self._u_mg_restrict_prog,
            self._u_mg_prolong_prog,
            self._u_wave_step_prog,
            self._pack_tension_prog,
            self._pack_fp16_prog,
            self._unpack_fp16_prog,
            self._pack_replay_prog,
            self._unpack_replay_prog,
            self._zero_buf_prog,
            self._boost_orbital_prog,
            self._init_prog,
            self._pack_prog,
            self._empty_vao,
            self._ray_prog,
        ):
            if obj is not None:
                obj.release()
        self._buf_a = None
        self._buf_b = None
        self._buf_vel = None
        self._vol_tex = None
        self.allocated_vram_mb = 0.0
        self._hib_phi_ram = None
        self._hib_vel_ram = None
        self._hib_u_ram = None
        self._hib_ut_ram = None
        self._hib_u_warm = False
        self.clear_replay_buffer()

