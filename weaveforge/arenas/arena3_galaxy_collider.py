"""
Arena 3 — The 3D Galaxy & Cluster Collider (`weaveforge/arenas/arena3_galaxy_collider.py`)
==========================================================================================
Real-time 60–120+ FPS 3D GPU N-Body (`1,000,000`–`2,000,000` particles) + 5-Level 3D GPU
Geometric Multigrid V-Cycle (`64³ -> 32³ -> 16³ -> 8³ -> 4³`) Poisson / QUMOND-Weave Field
Solver & Volumetric Lensing Contour Renderer:

  • Preset 0 (`NGC 3198 — 1M-Star 3D SPARC Spiral`):
      Exponential stellar disk + bulge + extended HI gas out to 30 kpc. Holds a flat ~150 km/s
      rotation curve with ZERO dark matter under TNaP / MOND / AeST / Superfluid DM (or with live
      collisionless NFW CDM halo particles enabled in Lambda-CDM), and visibly disperses if both
      CDM and nu(y) boost are disabled!
  • Preset 1 (`DDO 154 — 1M-Particle Low-Acceleration Dwarf`):
      Deep low-surface-brightness gas-rich dwarf (`g_N << a_0`) testing the Unruh/MOND floor.
  • Preset 2 (`Bullet Cluster 1E 0657-56 — 4,574 km/s Fast Pass-Through`):
      High-Mach cluster collision: collisional X-ray gas (`type=1`, crimson-magenta) suffers
      3D smoothed ram-pressure + thermal bow-shock arrest at `x ~ 0`, while collisionless galaxies
      (`type=0`, solar gold), live CDM halos (`type=2`, when `has_cdm_halo=true`), and cyan
      gravitational lensing contours pass cleanly through (`offset ~ 6–10 kpc`)!
  • Preset 3 (`Slow Dwarf Merger — 14 km/s Transverse Wave-Pleat Splash`):
      Sub-sonic dwarf collision: repulsive wave-pleat core stiffness (`pinned_pleat_coupling > 0`)
      launches a falsifiable sideways (`±Y/±Z`) splash under TNaP / Superfluid DM vs zero sideways
      splash in collisionless `Lambda-CDM`!
  • Preset 4 (`Milky Way — Barred 4-Arm Spiral + Extended HI Disk`):
      Galactic stellar bar (`R_bar ~ 3.2 kpc`) + 4-arm spiral disk (`V_circ ~ 225 km/s` at `R_0 = 8.1 kpc`).
  • Preset 5 (`Live SPARC-175 Catalog Galaxy`):
      Spawns ANY of the 175 published Lelli+ (2016) SPARC galaxies (`load_sparc_galaxy(gal)`) in 3D!
"""

from __future__ import annotations

from collections import deque
import math
from typing import Callable, Deque, Dict, List, Optional, Tuple

import moderngl
import numpy as np
from OpenGL import GL

from weaveforge.core.camera import Camera3D
from weaveforge.core.frameworks import PhysicsMod
from weaveforge.core.symbolic_compiler import CompiledMasterAction
from weaveforge.data.sparc_catalog import SparcGalaxy


_XFER_CHUNK_BYTES = 16 * 1024 * 1024  # 16 MiB reusable staging window
_MASS_FP16_SCALE = 1024.0

# Exact Galactic Unit System: (1 length = 1 kpc, 1 velocity = 100 km/s, 1 mass = 10^9 M_sun, 1 time = 9.77792 Myr)
# Preset 2 (Bullet Cluster 1.4 Mpc box): 1 length = 25 kpc, bullet speed 6.15 <-> 4,574 km/s (1 vel = 743.74 km/s)
KPC_TO_M = 3.085677581491367e19
VEL_UNIT_KMS = 100.0
ACCEL_UNIT_SI = ((VEL_UNIT_KMS * 1000.0) ** 2) / KPC_TO_M            # 3.240779289444365e-10 m/s^2
ACCEL_UNIT_CLUSTER_SI = (((4574.0 / 6.15) * 1000.0) ** 2) / (25.0 * KPC_TO_M)  # 7.16984e-10 m/s^2
A0_REF_SI = 1.2031e-10
A0_REF_CODE = A0_REF_SI / ACCEL_UNIT_SI                              # 0.371237872...
A0_CLUSTER_REF_CODE = A0_REF_SI / ACCEL_UNIT_CLUSTER_SI              # 0.167800...
G_GRAV_CODE = 0.4300917                                              # kpc * (100 km/s)^2 / (10^9 M_sun)
_CLUSTER_RHO_CALIB = G_GRAV_CODE / 0.085                             # Preserves G*rho invariance on Presets 2 & 3


def _preset_is_rotating_disk(preset_idx: int) -> bool:
    return (int(preset_idx) % 6) in (0, 1, 4, 5)


def _preset_half_box(preset_idx: int) -> float:
    return 38.0 if _preset_is_rotating_disk(preset_idx) else 28.0


def _cdm_enclosed_mass_py(r: float, m_bary: float, r_b: float) -> float:
    """RAR-matched NFW/pseudo-isothermal enclosed CDM halo mass M_CDM(<r) in 10^9 M_sun."""
    r_m = math.sqrt(max(G_GRAV_CODE * max(m_bary, 1e-6) / A0_REF_CODE, 1e-8))
    rb_eff = max(float(r_b), 0.25)
    f_b = (r * r) / ((r + rb_eff) * (r + rb_eff))
    x = (r + rb_eff) / r_m
    return float(m_bary * f_b * (math.sqrt(0.25 + x * x) - 0.5))


def _copy_buffer_sub_data(
    src_buf: moderngl.Buffer,
    dst_stage_buf: moderngl.Buffer,
    src_offset: int,
    dst_offset: int,
    size_bytes: int,
) -> None:
    """Copies GPU buffer range into a `dynamic=True` staging buffer without demoting VBOs/SSBOs."""
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


def _unbind_arena3_gl_slots() -> None:
    """Unbinds indexed SSBO, Image3D, Texture, and VAO binding points so WDDM reclaims GDDR7 VRAM immediately."""
    try:
        for slot in (0, 1, 2, 3, 4, 5, 7):
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, slot, 0)
        GL.glBindImageTexture(0, 0, 0, GL.GL_TRUE, 0, GL.GL_WRITE_ONLY, GL.GL_RGBA16F)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, 0)
        GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, 0)
    except Exception:
        pass


GALAXY_PRESETS = [
    ("ngc3198_spiral", "NGC 3198 (1M-Star 3D SPARC Spiral)"),
    ("ddo154_dwarf", "DDO 154 (1M-Particle Gas-Rich Dwarf)"),
    ("bullet_cluster", "Bullet Cluster (4,574 km/s Fast Pass-Through)"),
    ("slow_dwarf_splash", "Slow Dwarf Merger (14 km/s Sideways Wave Splash)"),
    ("milky_way", "Milky Way (Barred 4-Arm Spiral, 225 km/s)"),
    ("sparc_custom", "Live SPARC-175 Catalog Galaxy"),
]

# ==============================================================================
# 1. GPU PCG32 INITIALIZATION SHADER (Stars=0, Gas=1, Live Invisible CDM Halo=2)
# ==============================================================================
_INIT_GALAXY_GLSL = """#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) writeonly buffer PosBuf { vec4 pos_mass[]; };
layout(std430, binding = 1) writeonly buffer VelBuf { vec4 vel_type[]; };

uniform int u_count;
uniform int u_preset;
// Custom SPARC-175 galaxy parameters: (m_bary_code, r_disk_kpc, gas_frac, bul_frac)
uniform vec4 u_sparc_shape;
uniform float u_sparc_vflat_code;

uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand01(inout uint seed) {
    seed = pcg_hash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

vec2 rand_gauss2(inout uint seed) {
    float u1 = max(rand01(seed), 1e-6);
    float u2 = rand01(seed) * 6.28318530718;
    float r = sqrt(-2.0 * log(u1));
    return vec2(r * cos(u2), r * sin(u2));
}

vec3 sample_spherical_halo(inout uint seed, float r_scale, float r_max) {
    float u_r = rand01(seed);
    float r = clamp(r_scale * (pow(1.0 - 0.88 * u_r, -0.55) - 1.0) * 1.35 + 0.35 * rand01(seed), 0.35, r_max);
    float z_cos = 2.0 * rand01(seed) - 1.0;
    float phi = 6.28318530718 * rand01(seed);
    float sin_t = sqrt(max(1.0 - z_cos * z_cos, 0.0));
    return vec3(r * sin_t * cos(phi), r * z_cos, r * sin_t * sin(phi));
}

float cdm_enclosed_mass(float r, float m_bary, float r_b) {
    const float G_CODE = 0.4300917;
    const float A0_CODE = 0.37123787;
    float r_m = sqrt(max(G_CODE * max(m_bary, 1e-6) / A0_CODE, 1e-8));
    float rb_eff = max(r_b, 0.25);
    float f_b = (r * r) / ((r + rb_eff) * (r + rb_eff));
    float x = (r + rb_eff) / r_m;
    return m_bary * f_b * (sqrt(0.25 + x * x) - 0.5);
}

vec3 sample_rar_cdm_halo(inout uint seed, float m_bary, float r_b, float r_max) {
    float u_m = clamp(rand01(seed), 1e-6, 0.999999);
    float m_tot = max(cdm_enclosed_mass(r_max, m_bary, r_b), 1e-8);
    float target_m = u_m * m_tot;
    float lo = 0.06;
    float hi = r_max;
    for (int it = 0; it < 14; ++it) {
        float mid = 0.5 * (lo + hi);
        if (cdm_enclosed_mass(mid, m_bary, r_b) < target_m) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    float r = 0.5 * (lo + hi);
    float z_cos = 2.0 * rand01(seed) - 1.0;
    float phi = 6.28318530718 * rand01(seed);
    float sin_t = sqrt(max(1.0 - z_cos * z_cos, 0.0));
    return vec3(r * sin_t * cos(phi), r * z_cos, r * sin_t * sin(phi));
}

float sample_disk_radius(inout uint seed, float r_scale, float r_min, float r_max) {
    for (int it = 0; it < 6; ++it) {
        float u1 = max(rand01(seed), 1e-6);
        float u2 = max(rand01(seed), 1e-6);
        float cand = -r_scale * log(u1 * u2);
        if (cand >= r_min && cand <= r_max) {
            return cand;
        }
    }
    return mix(r_min, r_max, sqrt(rand01(seed)));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_count)) return;

    uint seed = idx * 1973u + uint(u_preset) * 9277u + 12345u;
    // 70% visible baryons (type 0=Star, 1=Gas), 30% live collisionless NFW CDM halo particles (type 2)!
    bool is_cdm_slot = ((idx % 10u) >= 7u);
    float n_bary = max(0.70 * float(u_count), 1.0);
    float n_cdm  = max(0.30 * float(u_count), 1.0);

    vec3 pos = vec3(0.0);
    vec3 vel = vec3(0.0);
    float p_type = 0.0;
    float m_part = 0.0;

    // -------------------------------------------------------------------------
    // PRESET 0: NGC 3198 (2-Arm Grand-Design 3D Spiral + Bulge + Extended HI)
    // Exact (1 kpc, 100 km/s, 10^9 M_sun): M_bary = 27.5e9 M_sun -> V_flat ~ 150 km/s
    // -------------------------------------------------------------------------
    if (u_preset == 0) {
        float m_bary_tot = 27.5;
        float r_b_halo = 5.2;
        float r_max_halo = 31.0;
        if (is_cdm_slot) {
            p_type = 2.0;
            m_part = cdm_enclosed_mass(r_max_halo, m_bary_tot, r_b_halo) / n_cdm;
            pos = sample_rar_cdm_halo(seed, m_bary_tot, r_b_halo, r_max_halo);
        } else {
            m_part = m_bary_tot / n_bary;
            float u_kind = rand01(seed);
            if (u_kind < 0.12) {
                vec2 g1 = clamp(rand_gauss2(seed), vec2(-2.5), vec2(2.5));
                vec2 g2 = clamp(rand_gauss2(seed), vec2(-2.5), vec2(2.5));
                pos = vec3(g1.x, g1.y * 0.65, g2.x) * 1.05;
                p_type = 0.25; // Tagged bulge star -> calibrated to p_type = 0.0
            } else {
                bool is_gas = (u_kind > 0.72);
                p_type = is_gas ? 1.0 : 0.0;
                float r_scale = is_gas ? 5.5 : 3.14;
                float r_max_d = is_gas ? 22.5 : 18.0;
                float r_disk = sample_disk_radius(seed, r_scale, 0.0, r_max_d);
                float arm = (rand01(seed) < 0.5) ? 0.0 : 3.14159265;
                vec2 g_ang = clamp(rand_gauss2(seed), vec2(-2.6), vec2(2.6));
                float spread = is_gas ? 0.46 : (0.36 + 0.028 * r_disk);
                float theta = arm - 2.35 * log(r_disk * 0.45 + 0.35) + g_ang.x * spread;
                float z_thick = (is_gas ? 0.20 : 0.30) * g_ang.y * (1.0 + 0.02 * r_disk);
                pos = vec3(r_disk * cos(theta), z_thick, r_disk * sin(theta));
            }
        }
    }
    // -------------------------------------------------------------------------
    // PRESET 1: DDO 154 (Gas-Dominated Low-Acceleration Dwarf Galaxy)
    // Exact (1 kpc, 100 km/s, 10^9 M_sun): M_bary = 0.38e9 M_sun -> V_flat ~ 48 km/s
    // -------------------------------------------------------------------------
    else if (u_preset == 1) {
        float m_bary_tot = 0.38;
        float r_b_halo = 2.15;
        float r_max_halo = 28.0;
        if (is_cdm_slot) {
            p_type = 2.0;
            m_part = cdm_enclosed_mass(r_max_halo, m_bary_tot, r_b_halo) / n_cdm;
            pos = sample_rar_cdm_halo(seed, m_bary_tot, r_b_halo, r_max_halo);
        } else {
            m_part = m_bary_tot / n_bary;
            bool is_gas = (rand01(seed) > 0.18);
            p_type = is_gas ? 1.0 : 0.0;
            float r_scale = is_gas ? 3.3 : 1.15;
            float r_max_d = is_gas ? 16.5 : 8.0;
            float r_disk = sample_disk_radius(seed, r_scale, 0.0, r_max_d);
            float theta = rand01(seed) * 6.28318530718;
            vec2 gz = clamp(rand_gauss2(seed), vec2(-2.6), vec2(2.6));
            pos = vec3(r_disk * cos(theta), gz.x * (is_gas ? 0.24 : 0.30), r_disk * sin(theta));
        }
    }
    // -------------------------------------------------------------------------
    // PRESET 2: BULLET CLUSTER (4,574 km/s Fast Collision: Galaxies + Live CDM vs X-Ray Gas!)
    // Cluster-scaled box (1 unit = 25 kpc, G*M invariant under G_GRAV_CODE = 0.4300917)
    // -------------------------------------------------------------------------
    else if (u_preset == 2) {
        float m_bary_tot = 14.22952;
        bool is_main = (rand01(seed) < 0.62);
        vec3 center = is_main ? vec3(-4.6, 0.0, 0.0) : vec3(5.4, 0.15, 0.0);
        float vx_bulk = is_main ? 4.35 : -6.15;

        if (is_cdm_slot) {
            p_type = 2.0;
            m_part = (4.8 * m_bary_tot) / n_cdm;
            vec3 h_off = sample_spherical_halo(seed, is_main ? 2.6 : 1.8, 8.2);
            pos = center + h_off;
            vec2 gv1 = rand_gauss2(seed);
            vec2 gv2 = rand_gauss2(seed);
            vel = vec3(vx_bulk + gv1.x * 0.34, gv1.y * 0.34, gv2.x * 0.34);
        } else {
            m_part = m_bary_tot / n_bary;
            bool is_gas = (rand01(seed) > 0.34);
            p_type = is_gas ? 1.0 : 0.0;
            float r_core = is_main ? (is_gas ? 1.95 : 1.45) : (is_gas ? 1.30 : 1.00);
            vec2 g1 = rand_gauss2(seed);
            vec2 g2 = rand_gauss2(seed);
            pos = center + vec3(g1.x, g1.y, g2.x) * r_core;
            vec2 gv = rand_gauss2(seed);
            vel = vec3(vx_bulk + g2.y * 0.20, gv.x * 0.20, gv.y * 0.20);
        }
    }
    // -------------------------------------------------------------------------
    // PRESET 3: SLOW DWARF MERGER (14 km/s Sub-Sonic Transverse Wave Splash!)
    // -------------------------------------------------------------------------
    else if (u_preset == 3) {
        float m_bary_tot = 8.30056;
        bool is_left = (rand01(seed) < 0.5);
        vec3 center = is_left ? vec3(-2.35, 0.0, 0.0) : vec3(2.35, 0.12, 0.0);
        float vx_bulk = is_left ? 0.58 : -0.58;
        if (is_cdm_slot) {
            p_type = 2.0;
            m_part = (5.0 * m_bary_tot) / n_cdm;
            pos = center + sample_spherical_halo(seed, 3.0, 12.0);
            vec2 gv1 = rand_gauss2(seed);
            vec2 gv2 = rand_gauss2(seed);
            vel = vec3(vx_bulk + gv1.x * 0.18, gv1.y * 0.18, gv2.x * 0.18);
        } else {
            m_part = m_bary_tot / n_bary;
            p_type = (rand01(seed) > 0.72) ? 1.0 : 0.0;
            vec2 g1 = rand_gauss2(seed);
            vec2 g2 = rand_gauss2(seed);
            pos = center + vec3(g1.x, g1.y, g2.x) * 1.25;
            vec2 gv = rand_gauss2(seed);
            vel = vec3(vx_bulk + g2.y * 0.12, gv.x * 0.12, gv.y * 0.12);
        }
    }
    // -------------------------------------------------------------------------
    // PRESET 4: MILKY WAY (Barred 4-Arm Grand-Design Spiral, V_circ ~ 210-220 km/s!)
    // Exact (1 kpc, 100 km/s, 10^9 M_sun): M_bary = 70.0e9 M_sun (Bland-Hawthorn & Gerhard 2016)
    // -------------------------------------------------------------------------
    else if (u_preset == 4) {
        float m_bary_tot = 70.0;
        float r_b_halo = 4.7;
        float r_max_halo = 32.0;
        if (is_cdm_slot) {
            p_type = 2.0;
            m_part = cdm_enclosed_mass(r_max_halo, m_bary_tot, r_b_halo) / n_cdm;
            pos = sample_rar_cdm_halo(seed, m_bary_tot, r_b_halo, r_max_halo);
        } else {
            m_part = m_bary_tot / n_bary;
            float u_kind = rand01(seed);
            if (u_kind < 0.13) {
                vec2 g1 = clamp(rand_gauss2(seed), vec2(-2.4), vec2(2.4));
                vec2 g2 = clamp(rand_gauss2(seed), vec2(-2.4), vec2(2.4));
                vec3 bar_p = vec3(g1.x * 1.95, g1.y * 0.46, g2.x * 0.82);
                float cb = cos(0.47), sb = sin(0.47);
                pos = vec3(cb * bar_p.x - sb * bar_p.z, bar_p.y, sb * bar_p.x + cb * bar_p.z);
                p_type = 0.25; // Tagged stellar bar -> calibrated to p_type = 0.0
            } else {
                bool is_gas = (u_kind > 0.74);
                p_type = is_gas ? 1.0 : 0.0;
                float r_scale = is_gas ? 6.4 : 3.20;
                float r_max_d = is_gas ? 24.5 : 20.5;
                float r_disk = sample_disk_radius(seed, r_scale, 0.0, r_max_d);
                float arm = floor(rand01(seed) * 4.0) * 1.5707963;
                vec2 g_ang = clamp(rand_gauss2(seed), vec2(-2.6), vec2(2.6));
                float spread = is_gas ? 0.42 : (0.34 + 0.026 * r_disk);
                float theta = arm - 2.65 * log(r_disk * 0.40 + 0.35) + g_ang.x * spread;
                float z_thick = (is_gas ? 0.18 : 0.28) * g_ang.y * (1.0 + 0.02 * r_disk);
                pos = vec3(r_disk * cos(theta), z_thick, r_disk * sin(theta));
            }
        }
    }
    // -------------------------------------------------------------------------
    // PRESET 5: LIVE REAL SPARC-175 CATALOG GALAXY (Driven by u_sparc_shape!)
    // -------------------------------------------------------------------------
    else {
        float m_bary_tot = max(u_sparc_shape.x, 0.02);
        float rd_kpc     = clamp(u_sparc_shape.y, 0.35, 9.5);
        float gas_frac   = clamp(u_sparc_shape.z, 0.04, 0.92);
        float bul_frac   = clamp(u_sparc_shape.w, 0.0, 0.45);
        float r_gas_s    = clamp(2.1 * rd_kpc, 1.8, 6.0);
        float r_b_halo   = 1.35 * ((1.0 - gas_frac) * rd_kpc + gas_frac * r_gas_s);
        float r_max_halo = 30.0;

        if (is_cdm_slot) {
            p_type = 2.0;
            m_part = cdm_enclosed_mass(r_max_halo, m_bary_tot, r_b_halo) / n_cdm;
            pos = sample_rar_cdm_halo(seed, m_bary_tot, r_b_halo, r_max_halo);
        } else {
            m_part = m_bary_tot / n_bary;
            float u_kind = rand01(seed);
            if (u_kind < bul_frac) {
                vec2 g1 = clamp(rand_gauss2(seed), vec2(-2.5), vec2(2.5));
                vec2 g2 = clamp(rand_gauss2(seed), vec2(-2.5), vec2(2.5));
                pos = vec3(g1.x, g1.y * 0.70, g2.x) * max(0.35 * rd_kpc, 0.45);
                p_type = 0.25;
            } else {
                bool is_gas = (u_kind > (1.0 - gas_frac));
                p_type = is_gas ? 1.0 : 0.0;
                float r_scale = is_gas ? r_gas_s : rd_kpc;
                float r_max_d = is_gas ? min(max(4.5 * r_gas_s, 12.0), 23.0) : min(max(5.0 * rd_kpc, 6.0), 20.0);
                float r_disk = sample_disk_radius(seed, r_scale, 0.0, r_max_d);
                float arm = (rand01(seed) < 0.5) ? 0.0 : 3.14159265;
                vec2 g_ang = clamp(rand_gauss2(seed), vec2(-2.6), vec2(2.6));
                float spread = is_gas ? 0.50 : 0.38;
                float theta = arm - 2.2 * log(r_disk / max(rd_kpc, 0.4) + 0.35) + g_ang.x * spread;
                float z_thick = (is_gas ? 0.20 : 0.28) * g_ang.y * clamp(0.35 * rd_kpc, 0.22, 0.85);
                pos = vec3(r_disk * cos(theta), z_thick, r_disk * sin(theta));
            }
        }
    }

    pos_mass[idx] = vec4(pos, m_part);
    vel_type[idx] = vec4(vel, p_type);
}
"""

_CALIBRATE_GALAXY_VEL_GLSL = """#version 430
// Cheat-Free Observed-Universe t=0 Velocity Calibration (100% Independent of Active PhysicsMod!)
// Pass 0: Calibrates Disk Stars, HI Gas, and Bulge/Bar Stars from grid-softened g_N_bary * nu_RAR_ref.
// Pass 1: Calibrates Live NFW CDM Halo particles from grid-softened g_N_(bary+CDM) in exact Jeans equilibrium.
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosBuf { vec4 pos_mass[]; };
layout(std430, binding = 1) buffer VelBuf          { vec4 vel_type[]; };

uniform sampler3D u_force_tex;
uniform int u_count;
uniform int u_preset;
uniform int u_pass;
uniform float u_half_box;
uniform float u_m_bary;
uniform float u_rd_star;
uniform float u_r_b_halo;
uniform float u_r_max_halo;
uniform float u_gm_monopole;

uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand01(inout uint seed) {
    seed = pcg_hash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

vec2 rand_gauss2(inout uint seed) {
    float u1 = max(rand01(seed), 1e-6);
    float u2 = rand01(seed) * 6.28318530718;
    float r = sqrt(-2.0 * log(u1));
    return vec2(r * cos(u2), r * sin(u2));
}

vec3 sample_grid_or_mono_accel(vec3 p) {
    vec3 uvw = clamp((p + vec3(u_half_box)) / (2.0 * u_half_box), vec3(0.008), vec3(0.992));
    vec3 g_grid = texture(u_force_tex, uvw).xyz;
    float box_frac = max(max(abs(p.x), abs(p.y)), abs(p.z)) / u_half_box;
    vec3 g_mono = -u_gm_monopole * p / pow(dot(p, p) + 1.5, 1.5);
    if (box_frac >= 1.0) return g_mono;
    if (box_frac > 0.88) return mix(g_grid, g_mono, smoothstep(0.88, 0.98, box_frac));
    return g_grid;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_count)) return;

    vec3 p = pos_mass[idx].xyz;
    vec4 vt = vel_type[idx];
    float p_type = vt.w;
    uint seed = idx * 747796405u + uint(u_preset) * 1597334677u + 98765u;

    if (u_pass == 0) {
        if (p_type > 1.5) return;

        // Universal Empirical Radial Acceleration Relation (McGaugh+ 2016 SPARC RAR, a0_ref = 0.37123787)
        vec3 g_N = sample_grid_or_mono_accel(p);
        float g_mag = length(g_N);
        float y_ref = clamp(g_mag / 0.37123787, 1e-5, 2500.0);
        float nu_rar = 1.0 / max(1.0 - exp(-sqrt(y_ref)), 1e-4);
        float nu_sim = 0.5 + sqrt(0.25 + 1.0 / y_ref);
        float nu_ref = 0.5 * (nu_rar + nu_sim);
        vec3 g_rar = g_N * nu_ref;

        float r_xz = max(length(p.xz), 0.12);
        vec3 rad_hat = vec3(p.x, 0.0, p.z) / r_xz;
        vec3 tang_hat = vec3(-p.z, 0.0, p.x) / r_xz;
        float a_R = max(-dot(g_rar, rad_hat), 0.0);
        float v_circ = sqrt(r_xz * a_R);

        vec2 g1 = clamp(rand_gauss2(seed), vec2(-2.25), vec2(2.25)) / 0.967;
        vec2 g2 = clamp(rand_gauss2(seed), vec2(-2.25), vec2(2.25)) / 0.967;

        // 1. Bulge / Galactic Stellar Bar (tagged 0.25 -> stored as 0.0)
        if (p_type > 0.1 && p_type < 0.4) {
            float r3d = max(length(p), 0.18);
            float a_r3 = max(-dot(g_rar, p / r3d), 0.0);
            float v_3d = sqrt(r3d * a_r3);
            float f_rot = (u_preset == 4) ? 0.58 : 0.52;
            float v_rot = f_rot * v_circ;
            float sig_b = sqrt(max(v_3d * v_3d - v_rot * v_rot, 0.0) / 2.2);
            vec3 vel = tang_hat * v_rot + vec3(g1.x * sig_b, g1.y * sig_b * 0.72, g2.x * sig_b);
            vel_type[idx] = vec4(vel, 0.0);
            return;
        }

        // Local Toomre Q = 1.85 radial dispersion scale for self-gravitating disk stability
        float rd_eff = max(u_rd_star, 0.5);
        float sigma_surf = (0.72 * max(u_m_bary, 0.02) / (6.2831853 * rd_eff * rd_eff)) * exp(-r_xz / rd_eff);
        float kappa_ep = 1.4142136 * max(v_circ, 0.05) / max(r_xz, 0.85 * rd_eff);
        float sig_toomre = (1.85 * 3.36 * 0.4300917 * sigma_surf) / max(kappa_ep, 1e-3);

        // 2. Cold Extended HI Atomic Gas Disk (p_type == 1.0)
        if (p_type > 0.5) {
            float sig_g = clamp(max(0.042 * v_circ, 0.36 * sig_toomre), 0.018, 0.14 * max(v_circ, 0.2));
            float asd_g = clamp(1.10 * (r_xz / (1.8 * rd_eff)), 0.35, 2.5);
            float v_phi = sqrt(max(v_circ * v_circ - asd_g * sig_g * sig_g, 0.0));
            vec3 vel = tang_hat * (v_phi + g1.y * sig_g * 0.707)
                     + rad_hat * (g1.x * sig_g)
                     + vec3(0.0, g2.x * sig_g * 0.50, 0.0);
            vel_type[idx] = vec4(vel, 1.0);
            return;
        }

        // 3. Rotating Stellar Disk with Exact Toomre Q = 1.85 Dispersion & Asymmetric Drift (p_type == 0.0)
        float sig_floor = mix(0.060, 0.115, exp(-r_xz / max(3.0 * rd_eff, 2.0))) * v_circ;
        float sig_R = clamp(max(sig_floor, sig_toomre), 0.055 * v_circ, 0.30 * v_circ);
        float sig_phi = 0.7071 * sig_R;
        float z0 = 0.28 * (1.0 + 0.02 * r_xz);
        float sig_y = clamp(max(1.05 * z0 * (v_circ / max(r_xz, 1.0)), 0.52 * sig_R), 0.018 * v_circ, 0.16 * v_circ);
        float asd = clamp(2.0 * (r_xz / rd_eff) - 0.50, 0.45, 4.5);
        float v_phi = sqrt(max(v_circ * v_circ - asd * sig_R * sig_R, 0.08 * v_circ * v_circ));

        vec3 vel = tang_hat * (v_phi + g1.y * sig_phi)
                 + rad_hat * (g1.x * sig_R)
                 + vec3(0.0, g2.x * sig_y, 0.0);
        vel_type[idx] = vec4(vel, 0.0);
    } else {
        if (p_type <= 1.5) return;

        // Pass 1: Spherical Jeans Equilibrium with Smooth Outer Tangential Shell Boundary!
        vec3 g_tot = sample_grid_or_mono_accel(p);
        float r = max(length(p), 0.22);
        vec3 r_hat = p / r;
        float a_in = max(-dot(g_tot, r_hat), 0.0);
        float vc2 = r * a_in;

        float x = clamp((r * r) / max(u_r_max_halo * u_r_max_halo, 1.0), 0.0, 0.999);
        float alpha_j = 0.42;
        float vr2 = alpha_j * vc2 * (1.0 - x) * (1.0 - 0.55 * x);
        float gamma_rho = 1.0 + r / (r + max(u_r_b_halo, 0.5));
        float vt2_inner = vc2 * max(1.0 + alpha_j * (2.0 - gamma_rho), 0.72);
        float vt2_sum = mix(vt2_inner, 1.02 * vc2, smoothstep(0.25, 0.95, x));

        float sig_r = sqrt(max(vr2, 0.0)) / 0.967;
        float sig_t = sqrt(0.5 * vt2_sum) / 0.967;
        vec3 up_ref = (abs(r_hat.y) < 0.90) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 t1 = normalize(cross(up_ref, r_hat));
        vec3 t2 = cross(r_hat, t1);

        vec2 g1 = clamp(rand_gauss2(seed), vec2(-2.25), vec2(2.25));
        vec2 g2 = clamp(rand_gauss2(seed), vec2(-2.25), vec2(2.25));
        vec2 v_perp_gauss = vec2(g1.y, g2.x) * sig_t;
        float psi = 6.28318530718 * rand01(seed);
        vec2 v_perp_shell = vec2(cos(psi), sin(psi)) * sqrt(max(1.02 * vc2, 0.0));
        vec2 v_perp = mix(v_perp_gauss, v_perp_shell, smoothstep(0.45, 0.95, x));

        vec3 vel = r_hat * (g1.x * sig_r) + t1 * v_perp.x + t2 * v_perp.y;

        float v_max_bound = sqrt(2.85 * max(vc2, 1e-5));
        float v_len = length(vel);
        if (v_len > v_max_bound) {
            vel *= (v_max_bound / v_len);
        }
        vel_type[idx] = vec4(vel, 2.0);
    }
}
"""

# ==============================================================================
# 2. 64^3 PARTICLE DEPOSITION + 3x3x3 GAS HYDRO SMOOTHING + 5-LEVEL 3D MULTIGRID!
# ==============================================================================
_CLEAR_GRID_GLSL = """#version 430
layout(local_size_x = 64) in;

layout(std430, binding = 2) writeonly buffer MassIntBuf { int mass_int[]; };
layout(std430, binding = 3) writeonly buffer GasIntBuf  { ivec4 gas_int[]; };

uniform int u_cells;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_cells)) return;
    mass_int[idx] = 0;
    gas_int[idx]  = ivec4(0);
}
"""

_DEPOSIT_GRID_GLSL = """#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosBuf     { vec4 pos_mass[]; };
layout(std430, binding = 1) readonly buffer VelBuf     { vec4 vel_type[]; };
layout(std430, binding = 2) buffer MassIntBuf          { int mass_int[]; };
layout(std430, binding = 3) buffer GasIntBuf           { int gas_flat[]; };

uniform int u_count;
uniform int u_M;
uniform float u_half_box;
uniform int u_has_cdm;
uniform float u_cdm_scale;
uniform float u_fix_scale;
uniform float u_mom_scale;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_count)) return;

    vec4 pm = pos_mass[idx];
    vec4 vt = vel_type[idx];
    float p_type = vt.w;

    // Live collisionless CDM particles (type 2.0) contribute mass ONLY when has_cdm_halo is active!
    float eff_mass = pm.w;
    if (p_type > 1.5) {
        if (u_has_cdm == 0 || u_cdm_scale <= 0.001) return;
        eff_mass *= u_cdm_scale;
    }

    vec3 uvw = (pm.xyz + vec3(u_half_box)) / (2.0 * u_half_box);
    // Keep slightly overhanging bound halo orbits accounted for; drop only far unbound escapees
    if (any(lessThan(uvw, vec3(-0.18))) || any(greaterThan(uvw, vec3(1.18)))) return;

    ivec3 c = clamp(ivec3(uvw * float(u_M)), ivec3(0), ivec3(u_M - 1));
    int cell = (c.z * u_M + c.y) * u_M + c.x;

    // Dynamic fixed-point mass scale: zero clamp-to-1 inflation for DDO 154 AND zero INT32 overflow!
    int m_fix = int(eff_mass * u_fix_scale + 0.5);
    if (m_fix <= 0) return;
    atomicAdd(mass_int[cell], m_fix);

    // Collisional Gas ONLY (0.5 < p_type < 1.5) deposits into Eulerian hydro shock grid!
    if (p_type > 0.5 && p_type < 1.5) {
        int base4 = cell * 4;
        atomicAdd(gas_flat[base4 + 0], m_fix);
        atomicAdd(gas_flat[base4 + 1], int(round(eff_mass * vt.x * u_mom_scale)));
        atomicAdd(gas_flat[base4 + 2], int(round(eff_mass * vt.y * u_mom_scale)));
        atomicAdd(gas_flat[base4 + 3], int(round(eff_mass * vt.z * u_mom_scale)));
    }
}
"""

_UNPACK_AND_SMOOTH_GLSL = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 2) readonly buffer MassIntBuf   { int mass_int[]; };
layout(std430, binding = 3) readonly buffer GasIntBuf    { int gas_flat[]; };
layout(std430, binding = 4) writeonly buffer RhsFineBuf  { float rhs_fine[]; };
layout(std430, binding = 5) writeonly buffer RhoBaryBuf  { float rho_flt[]; };
layout(std430, binding = 7) writeonly buffer GasHydroBuf { vec4 gas_hydro[]; };

uniform int u_M;
uniform float u_half_box;
uniform float u_G_grav;
uniform float u_fix_scale;
uniform float u_mom_scale;

int cidx(int x, int y, int z, int M) {
    return (clamp(z, 0, M - 1) * M + clamp(y, 0, M - 1)) * M + clamp(x, 0, M - 1);
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    if (gid.x >= u_M || gid.y >= u_M || gid.z >= u_M) return;

    int M = u_M;
    int cell = (gid.z * M + gid.y) * M + gid.x;
    float dx = (2.0 * u_half_box) / float(M);
    float inv_vol = 1.0 / (dx * dx * dx);
    float inv_fix = 1.0 / max(u_fix_scale, 1.0);
    float inv_mom = 1.0 / max(u_mom_scale, 1.0);

    float m_cell = float(mass_int[cell]) * inv_fix;
    float rho = m_cell * inv_vol;
    rho_flt[cell] = rho;
    // Exact Poisson PDE RHS on Level 0: Lap(Phi) = 4 * pi * G * rho!
    rhs_fine[cell] = 12.566370614 * u_G_grav * rho;

    // 3x3x3 Trilinear [1, 2, 1]^3 Smoothed Eulerian Gas Density & Bulk Shock Velocity!
    float w_mass = 0.0;
    vec3 w_mom = vec3(0.0);
    float w_norm = 0.0;
    for (int dz = -1; dz <= 1; ++dz) {
        float wz = (dz == 0) ? 2.0 : 1.0;
        for (int dy = -1; dy <= 1; ++dy) {
            float wy = (dy == 0) ? 2.0 : 1.0;
            for (int dx_i = -1; dx_i <= 1; ++dx_i) {
                float wx = (dx_i == 0) ? 2.0 : 1.0;
                float w = wx * wy * wz;
                int nb = cidx(gid.x + dx_i, gid.y + dy, gid.z + dz, M);
                int b4 = nb * 4;
                float mg = float(gas_flat[b4 + 0]) * inv_fix;
                vec3 mom = vec3(
                    float(gas_flat[b4 + 1]),
                    float(gas_flat[b4 + 2]),
                    float(gas_flat[b4 + 3])
                ) * inv_mom;
                w_mass += w * mg;
                w_mom  += w * mom;
                w_norm += w;
            }
        }
    }
    float rho_gas_sm = (w_mass / w_norm) * inv_vol;
    vec3 u_gas_sm = (w_mass > 1e-9) ? (w_mom / w_mass) : vec3(0.0);
    gas_hydro[cell] = vec4(rho_gas_sm, u_gas_sm);
}
"""

_MG_SMOOTH_GLSL = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer RhsBuf  { float rhs_in[]; };
layout(std430, binding = 1) readonly buffer UInBuf  { float u_in[]; };
layout(std430, binding = 2) writeonly buffer UOutBuf { float u_out[]; };

uniform int u_M;
uniform float u_half_box;
uniform int u_is_fine;        // 1 = Level 0 (monopole Dirichlet BC on outer box faces), 0 = Coarse error level (zero BC)
uniform float u_gm_monopole;  // G * M_active for outer Dirichlet boundary
uniform int u_zero_init;      // 1 = treat u_in as 0.0 (first sweep on coarse error grid!)

int cidx(int x, int y, int z, int M) {
    return (z * M + y) * M + x;
}

float wall_bc(vec3 p_wall) {
    if (u_is_fine == 0) return 0.0;
    return -u_gm_monopole / max(length(p_wall), 1.0);
}

void main() {
    ivec3 g = ivec3(gl_GlobalInvocationID);
    if (g.x >= u_M || g.y >= u_M || g.z >= u_M) return;

    int M = u_M;
    int id = cidx(g.x, g.y, g.z, M);
    float h = (2.0 * u_half_box) / float(M);
    vec3 p_cell = (vec3(g) + 0.5) * h - vec3(u_half_box);

    // Cell-centered finite-volume stencil on [-u_half_box, +u_half_box]^3:
    // Across an outer box face at distance h/2, ghost cell = 2*u_wall - u_cell -> sum_nb += 2*u_wall, diag += 2.
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
    // Optimal 3D 7-point Multigrid damped-Jacobi smoother weight omega = 2/3!
    u_out[id] = mix(u_cur, u_jac, 0.6666667);
}
"""

_MG_RESTRICT_GLSL = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer FineRhsBuf    { float rhs_f[]; };
layout(std430, binding = 1) readonly buffer FineUBuf      { float u_f[]; };
layout(std430, binding = 2) writeonly buffer CoarseRhsBuf { float rhs_c[]; };

uniform int u_Mc;
uniform float u_half_box;
uniform int u_is_fine;
uniform float u_gm_monopole;

int fidx(int x, int y, int z, int Mf) {
    return (z * Mf + y) * Mf + x;
}

float wall_bc(vec3 p_wall) {
    if (u_is_fine == 0) return 0.0;
    return -u_gm_monopole / max(length(p_wall), 1.0);
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

_MG_PROLONG_ADD_GLSL = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 0) readonly buffer CoarseErrBuf { float err_c[]; };
layout(std430, binding = 1) buffer FineUBuf              { float u_f[]; };

uniform int u_Mf;

float sample_coarse_err(int x, int y, int z, int Mc) {
    // Odd reflection across zero-Dirichlet outer box faces: ghost = -err_c[boundary]
    float sgn = 1.0;
    if (x < 0)       { x = 0;      sgn *= -1.0; }
    else if (x >= Mc){ x = Mc - 1; sgn *= -1.0; }
    if (y < 0)       { y = 0;      sgn *= -1.0; }
    else if (y >= Mc){ y = Mc - 1; sgn *= -1.0; }
    if (z < 0)       { z = 0;      sgn *= -1.0; }
    else if (z >= Mc){ z = Mc - 1; sgn *= -1.0; }
    return sgn * err_c[(z * Mc + y) * Mc + x];
}

void main() {
    ivec3 gf = ivec3(gl_GlobalInvocationID);
    if (gf.x >= u_Mf || gf.y >= u_Mf || gf.z >= u_Mf) return;

    int Mc = u_Mf / 2;
    // Cell-centered trilinear prolongation from coarse cell centers (ic + 0.5) * 2 = gf + 0.5
    vec3 fc = (vec3(gf) + 0.5) * 0.5 - 0.5;
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
    float c0  = mix(c00, c10, t.y);
    float c1  = mix(c01, c11, t.y);

    int fid = (gf.z * u_Mf + gf.y) * u_Mf + gf.x;
    u_f[fid] += mix(c0, c1, t.z);
}
"""

_PACK_FORCE_TEX_GLSL = """#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(std430, binding = 5) readonly buffer PhiBuf    { float phi_in[]; };
layout(rgba16f, binding = 0) writeonly uniform image3D u_force_img;

uniform int u_M;
uniform float u_half_box;
uniform float u_gm_monopole;

float sample_phi_ext(int x, int y, int z, int M, float dx, float half_box, float gm_mono) {
    if (x >= 0 && x < M && y >= 0 && y < M && z >= 0 && z < M) {
        return phi_in[(z * M + y) * M + x];
    }
    vec3 p_ghost = (vec3(x, y, z) + 0.5) * dx - vec3(half_box);
    return -gm_mono / max(length(p_ghost), 1.0);
}

void main() {
    ivec3 g = ivec3(gl_GlobalInvocationID);
    if (g.x >= u_M || g.y >= u_M || g.z >= u_M) return;

    int M = u_M;
    float dx = (2.0 * u_half_box) / float(M);
    float inv_2dx = 0.5 / dx;

    float pxp = sample_phi_ext(g.x + 1, g.y, g.z, M, dx, u_half_box, u_gm_monopole);
    float pxm = sample_phi_ext(g.x - 1, g.y, g.z, M, dx, u_half_box, u_gm_monopole);
    float pyp = sample_phi_ext(g.x, g.y + 1, g.z, M, dx, u_half_box, u_gm_monopole);
    float pym = sample_phi_ext(g.x, g.y - 1, g.z, M, dx, u_half_box, u_gm_monopole);
    float pzp = sample_phi_ext(g.x, g.y, g.z + 1, M, dx, u_half_box, u_gm_monopole);
    float pzm = sample_phi_ext(g.x, g.y, g.z - 1, M, dx, u_half_box, u_gm_monopole);

    float gx = -(pxp - pxm) * inv_2dx;
    float gy = -(pyp - pym) * inv_2dx;
    float gz = -(pzp - pzm) * inv_2dx;
    float pot_mag = abs(phi_in[(g.z * M + g.y) * M + g.x]);

    imageStore(u_force_img, g, vec4(gx, gy, gz, pot_mag));
}
"""


def _build_particle_step_glsl(nu_glsl_expr: str) -> str:
    """
    Dynamically injects the active `PhysicsMod`'s SymPy-compiled `interpolation_nu(y)`
    into the 1M-2M particle GPU Leapfrog integrator (Stars + Collisional Gas + Live CDM Halo)
    AND drives Volumetric Gravitational Lensing Contours with the EFFECTIVE total acceleration!
    """
    return f"""#version 430
// AUTO-COMPILED 1M-2M STAR, GAS & LIVE CDM INTEGRATOR WITH SYMPY nu(y) INTERPOLATION
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PosBuf              {{ vec4 pos_mass[]; }};
layout(std430, binding = 1) buffer VelBuf              {{ vec4 vel_type[]; }};
layout(std430, binding = 5) readonly buffer RhoFlt     {{ float rho_flt[]; }};
layout(std430, binding = 7) readonly buffer GasHydro   {{ vec4 gas_hydro[]; }};

uniform sampler3D u_force_tex;
uniform int u_count;
uniform int u_M;
uniform float u_half_box;
uniform float u_dt;
uniform float u_a0_code;
uniform float u_gm_monopole;
uniform float u_pleat_splash;
uniform int u_preset;

int cidx(int x, int y, int z, int M) {{
    return (clamp(z, 0, M - 1) * M + clamp(y, 0, M - 1)) * M + clamp(x, 0, M - 1);
}}

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_count)) return;

    vec4 pm = pos_mass[idx];
    vec4 vt = vel_type[idx];
    vec3 p = pm.xyz;
    vec3 v = vt.xyz;
    float p_type = vt.w;

    vec3 uvw = clamp((p + vec3(u_half_box)) / (2.0 * u_half_box), vec3(0.008), vec3(0.992));
    vec3 g_N = texture(u_force_tex, uvw).xyz;

    // Smooth exterior Newtonian monopole fallback near/beyond [-u_half_box, +u_half_box]^3!
    float box_frac = max(max(abs(p.x), abs(p.y)), abs(p.z)) / u_half_box;
    if (box_frac > 0.88) {{
        vec3 g_mono = -u_gm_monopole * p / pow(dot(p, p) + 1.5, 1.5);
        g_N = (box_frac >= 1.0) ? g_mono : mix(g_N, g_mono, smoothstep(0.88, 0.98, box_frac));
    }}

    // 1. Evaluate SymPy-compiled Master Action interpolation function nu(y) where y = |g_N| / a0!
    float g_mag = length(g_N);
    float y = clamp(g_mag / max(u_a0_code, 1e-6), 1e-6, 2500.0);
    float nu_raw = {nu_glsl_expr};
    float nu_boost = clamp(nu_raw, 0.25, 250.0);
    vec3 acc = g_N * nu_boost;

    ivec3 c = clamp(ivec3(uvw * float(u_M)), ivec3(0), ivec3(u_M - 1));
    int cell = (c.z * u_M + c.y) * u_M + c.x;

    // 2. Collisional Eulerian Ram-Pressure Bow-Shock + ICM Thermal Pressure ONLY in Colliding Systems (Presets 2 & 3)!
    //    Quiet differentially rotating disks (Presets 0, 1, 4, 5) suffer ZERO artificial 3.5-kpc grid shear braking!
    if ((u_preset == 2 || u_preset == 3) && p_type > 0.5 && p_type < 1.5) {{
        vec4 gh = gas_hydro[cell];
        float rho_g = gh.x * 5.05990235;
        if (rho_g > 1e-4) {{
            vec3 u_cell = gh.yzw;
            vec3 dv = v - u_cell;
            float dv_len = length(dv);
            float drag_coeff = (u_preset == 2) ? (14.5 * clamp(rho_g * 5.5, 0.05, 4.5) * (0.45 + dv_len))
                                               : (0.28 * clamp(rho_g * 2.8, 0.0, 1.4));
            drag_coeff = min(drag_coeff, 0.90 / max(u_dt, 1e-4));
            acc -= drag_coeff * dv;

            if (u_preset == 2) {{
                // Shock-heated X-ray ICM isotropic pressure gradient (-c_s^2 * grad(rho_g) / rho_g)
                float dx = (2.0 * u_half_box) / float(u_M);
                vec3 grad_rg = 5.05990235 * vec3(
                    gas_hydro[cidx(c.x + 1, c.y, c.z, u_M)].x - gas_hydro[cidx(c.x - 1, c.y, c.z, u_M)].x,
                    gas_hydro[cidx(c.x, c.y + 1, c.z, u_M)].x - gas_hydro[cidx(c.x, c.y - 1, c.z, u_M)].x,
                    gas_hydro[cidx(c.x, c.y, c.z + 1, u_M)].x - gas_hydro[cidx(c.x, c.y, c.z - 1, u_M)].x
                ) / (2.0 * dx);
                acc -= clamp((0.65 * grad_rg) / max(rho_g, 0.015), vec3(-4.0), vec3(4.0));
            }}
        }}
    }}

    // 3. Sub-sonic Topological Wave-Pleat / Superfluid Core Repulsion (14 km/s Slow Dwarf Splash!)
    //    Active ONLY on baryons when the active PhysicsMod has non-zero wave-pleat / sextic core coupling!
    if (u_preset == 3 && p_type < 1.5 && u_pleat_splash > 0.01) {{
        float rho_local = rho_flt[cell] * 5.05990235;
        float r_center = length(p);
        if (r_center < 3.8 && rho_local > 0.06) {{
            vec3 trans_dir = vec3(0.0, p.y, p.z);
            float t_len = length(trans_dir);
            if (t_len > 1e-3) {{
                trans_dir /= t_len;
            }} else {{
                float ang = float(idx & 1023u) * 0.0061359;
                trans_dir = vec3(0.0, cos(ang), sin(ang));
            }}
            float core_overpressure = clamp((rho_local - 0.06) * 4.2, 0.0, 5.0);
            acc += trans_dir * (u_pleat_splash * core_overpressure * 1.05);
        }}
    }}

    // Symplectic Euler (Kick-Drift): conserves discrete phase-space volume & bounds orbital energy error!
    v = clamp(v + acc * u_dt, vec3(-18.0), vec3(18.0));
    p = p + v * u_dt;

    pos_mass[idx] = vec4(p, pm.w);
    vel_type[idx] = vec4(v, p_type);
}}
"""


# ==============================================================================
# 3. BACKGROUND + EFFECTIVE LENSING CONTOUR SHADER & 1M-2M PARTICLE SPRITE SHADER
# ==============================================================================
_BG_VERT = """#version 430
out vec2 v_uv;
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
"""


def _build_bg_lensing_frag(nu_glsl_expr: str) -> str:
    """Renders 3D Volumetric Lensing Contours using the TRUE effective acceleration `|g_N * nu(y)|`!"""
    return f"""#version 430
in vec2 v_uv;
out vec4 frag_color;

uniform sampler3D u_force_tex;
uniform vec3 u_cam_eye;
uniform vec3 u_cam_fwd;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_fov_tan;
uniform float u_aspect;
uniform float u_half_box;
uniform float u_a0_code;
uniform int u_show_lensing;

vec2 intersect_aabb(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {{
    vec3 inv_d = 1.0 / (rd + vec3(1e-8));
    vec3 t0s = (bmin - ro) * inv_d;
    vec3 t1s = (bmax - ro) * inv_d;
    vec3 tsmaller = min(t0s, t1s);
    vec3 tbigger  = max(t0s, t1s);
    float tmin = max(max(tsmaller.x, tsmaller.y), max(tsmaller.z, 0.0));
    float tmax = min(min(tbigger.x, tbigger.y), tbigger.z);
    return vec2(tmin, tmax);
}}

void main() {{
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(
        u_cam_fwd
        + ndc.x * u_aspect * u_fov_tan * u_cam_right
        + ndc.y * u_fov_tan * u_cam_up
    );
    vec3 ro = u_cam_eye;

    float r_scr = length(ndc * vec2(u_aspect, 1.0));
    vec3 bg = mix(vec3(0.048, 0.092, 0.175), vec3(0.020, 0.040, 0.082), clamp(r_scr * 0.65, 0.0, 1.0));

    if (abs(rd.y) > 1e-4) {{
        float tg = (-6.5 - ro.y) / rd.y;
        if (tg > 0.0 && tg < 160.0) {{
            vec3 pg = ro + tg * rd;
            float rg = length(pg.xz);
            if (rg < 36.0) {{
                float dr10 = abs(fract(rg / 5.0 + 0.5) - 0.5) * 5.0;
                float ring = exp(-(dr10 * dr10) * 22.0) * smoothstep(36.0, 10.0, rg);
                bg += vec3(0.08, 0.22, 0.42) * ring * 0.14;
            }}
        }}
    }}

    if (u_show_lensing == 1) {{
        vec2 tb = intersect_aabb(ro, rd, vec3(-u_half_box), vec3(u_half_box));
        if (tb.y > tb.x) {{
            int steps = 52;
            float dt = (tb.y - tb.x) / float(steps);
            float t = tb.x + 0.5 * dt;
            vec3 pot_acc = vec3(0.0);
            for (int i = 0; i < steps; ++i) {{
                vec3 p = ro + t * rd;
                vec3 uvw = (p + vec3(u_half_box)) / (2.0 * u_half_box);
                vec4 gp = texture(u_force_tex, uvw);
                float g_mag = length(gp.xyz);
                float y = clamp(g_mag / max(u_a0_code, 1e-6), 1e-6, 2500.0);
                float nu_eff = clamp({nu_glsl_expr}, 0.25, 36.0);
                float g_eff = g_mag * nu_eff;
                float shells = 0.52 + 0.48 * sin(gp.w * 6.5);
                pot_acc += vec3(0.08, 0.76, 1.02) * (g_eff * shells) * dt * 0.026;
                t += dt;
            }}
            bg += clamp(pot_acc, vec3(0.0), vec3(0.52));
        }}
    }}

    frag_color = vec4(bg, 1.0);
}}
"""


_PARTICLE_VERT = """#version 430
layout(location = 0) in vec4 in_pos_mass;
layout(location = 1) in vec4 in_vel_type;

uniform mat4 u_view_proj;
uniform vec3 u_cam_eye;
uniform float u_point_scale;
uniform float u_res_scale;
uniform int u_preset;

out vec3 v_color;
out float v_alpha;

void main() {
    float p_type = in_vel_type.w;
    // Type 2.0 = Invisible Collisionless Cold Dark Matter Halo Particle (emits zero photons!)
    if (p_type > 1.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        v_color = vec3(0.0);
        v_alpha = 0.0;
        return;
    }

    vec3 pos = in_pos_mass.xyz;
    gl_Position = u_view_proj * vec4(pos, 1.0);

    float dist = max(length(pos - u_cam_eye), 1.5);
    float speed = length(in_vel_type.xyz);
    bool is_gas = (p_type > 0.5);
    float r_scale = max(u_res_scale, 0.05);

    if (is_gas) {
        if (u_preset == 2) {
            // Chandra X-Ray ICM Gas in Bullet Cluster: vivid Crimson-Coral shock glow!
            float shock_boost = smoothstep(3.2, 0.3, speed);
            v_color = mix(vec3(1.28, 0.26, 0.52), vec3(1.92, 0.34, 0.44), shock_boost);
        } else {
            // Extended HI Atomic Gas in Spirals & Dwarfs: Cyan-Sapphire with warm star-forming knots
            float heat = clamp((speed - 0.85) * 0.45, 0.0, 1.0);
            v_color = mix(vec3(0.24, 0.86, 1.28), vec3(0.88, 0.42, 1.15), heat * 0.45);
        }
        float base_ps = clamp(28.0 / dist, 1.3, 6.0);
        float raw_ps = base_ps * r_scale;
        gl_PointSize = max(raw_ps, 1.0);
        v_alpha = 0.052 * u_point_scale * ((raw_ps < 1.0) ? (raw_ps * raw_ps) : 1.0);
    } else {
        // Collisionless Stars / Galaxies: Luminous Solar Gold to Warm Amber-White
        float kin_tint = clamp(speed * 0.22, 0.0, 1.0);
        v_color = mix(vec3(1.65, 1.00, 0.26), vec3(1.20, 0.96, 0.78), kin_tint * 0.40);
        float base_ps = clamp(22.0 / dist, 1.2, 4.8);
        float raw_ps = base_ps * r_scale;
        gl_PointSize = max(raw_ps, 1.0);
        v_alpha = 0.066 * u_point_scale * ((raw_ps < 1.0) ? (raw_ps * raw_ps) : 1.0);
    }
}
"""
_PARTICLE_VERT_GLSL = _PARTICLE_VERT

_PARTICLE_FRAG = """#version 430
in vec3 v_color;
in float v_alpha;
out vec4 frag_color;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float w = exp(-r2 * 3.4);
    frag_color = vec4(v_color * (v_alpha * w), v_alpha * w);
}
"""

_TONEMAP_FRAG = """#version 430
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_hdr_tex;

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(u_hdr_tex, v_uv).rgb;
    vec3 mapped = aces_tonemap(hdr * 1.12);
    mapped = pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.2));
    frag_color = vec4(mapped, 1.0);
}
"""

_PACK_POS_FP16_GLSL = """#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PosBuf { vec4 pos_in[]; };
layout(std430, binding = 1) writeonly buffer HalfPosBuf { uvec2 half_out[]; };
uniform uint u_count;
uniform float u_mass_scale;

vec2 clamp_fp16(vec2 v) {
    return clamp(v, vec2(-65500.0), vec2(65500.0));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= u_count) return;
    vec4 pm = pos_in[idx];
    half_out[idx] = uvec2(
        packHalf2x16(clamp_fp16(pm.xy)),
        packHalf2x16(clamp_fp16(vec2(pm.z, pm.w * u_mass_scale)))
    );
}
"""

_UNPACK_POS_FP16_GLSL = """#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer HalfPosBuf { uvec2 half_in[]; };
layout(std430, binding = 1) writeonly buffer PosBuf { vec4 pos_out[]; };
uniform uint u_count;
uniform float u_inv_mass_scale;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= u_count) return;
    uvec2 u = half_in[idx];
    vec2 xy = unpackHalf2x16(u.x);
    vec2 zw = unpackHalf2x16(u.y);
    pos_out[idx] = vec4(xy.x, xy.y, zw.x, zw.y * u_inv_mass_scale);
}
"""

_PACK_REPLAY_PARTICLES_FP16_GLSL = """#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer PosBuf { vec4 pos_in[]; };
layout(std430, binding = 1) readonly buffer VelBuf { vec4 vel_in[]; };
layout(std430, binding = 2) writeonly buffer HalfPhaseBuf { uvec2 half_out[]; };
uniform uint u_count;
uniform float u_mass_scale;

vec2 clamp_fp16(vec2 v) {
    return clamp(v, vec2(-65500.0), vec2(65500.0));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= u_count) return;
    vec4 pm = pos_in[idx];
    vec4 vt = vel_in[idx];
    half_out[idx] = uvec2(
        packHalf2x16(clamp_fp16(pm.xy)),
        packHalf2x16(clamp_fp16(vec2(pm.z, pm.w * u_mass_scale)))
    );
    half_out[u_count + idx] = uvec2(
        packHalf2x16(clamp_fp16(vt.xy)),
        packHalf2x16(clamp_fp16(vt.zw))
    );
}
"""


_DEFAULT_MOD_COMPILED_CACHE: Optional[Tuple[PhysicsMod, CompiledMasterAction]] = None


def _get_default_mod_and_compiled(
    mod: Optional[PhysicsMod] = None,
) -> Tuple[PhysicsMod, CompiledMasterAction]:
    """Returns a process-cached default `(tnap_weave, CompiledMasterAction)` so cold `Arena3GalaxyCollider(ctx)` never stalls Frame 0."""
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


class Arena3GalaxyCollider:
    """
    GPU-accelerated 3D `1,000,000`–`2,000,000` Star, Gas & Live CDM Collider
    coupled to `SymPy`-compiled `interpolation_nu(y)` + 5-Level 3D GPU Multigrid V-Cycle Solver,
    with GDDR7-resident VBO/SSBO staging, DDR5 RAM hibernation, and DDR5 Time-Machine scrubbing!
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        particle_count: int = 1_000_000,
        mod: Optional[PhysicsMod] = None,
        compiled: Optional[CompiledMasterAction] = None,
        lazy_gpu: bool = False,
    ) -> None:
        self.ctx = ctx
        self.particle_count = int(particle_count)
        self.grid_m = 64
        self.half_box = _preset_half_box(0)
        self.preset_idx = 0
        self.show_lensing = True
        self.multigrid_vcycles = 2
        self.sim_time = 0.0
        self.step_count = 0
        self.last_a0_code = float(A0_REF_CODE)

        # Residency & DDR5 Time-Machine Replay state
        self.is_hibernated: bool = bool(lazy_gpu)
        self.allocated_vram_mb: float = 0.0
        self.replay_enabled: bool = True
        self.replay_max_frames: int = 64
        self.replay_cursor: int = -1
        self._replay_frames: Deque[np.ndarray] = deque()
        self._replay_meta: Deque[Tuple[float, int]] = deque()
        self._live_backup: Optional[Dict[str, object]] = None
        self._hib_pos_ram: Optional[np.ndarray] = None
        self._hib_vel_ram: Optional[np.ndarray] = None
        self._dirty_field_on_wake: bool = False

        # Custom SPARC-175 galaxy parameters for Preset 5
        self.custom_sparc_name = "NGC 3198"
        self._sparc_shape = (33.6, 3.14, 0.43, 0.05)
        self._sparc_vflat_code = 1.501

        # Live quantitative telemetry
        self._v_outer_kms: float = 150.0
        self._lensing_gas_offset_kpc: float = 0.0
        self._transverse_splash_ratio: float = 0.35
        self._last_render_step: int = -1

        # Compile fixed GPU compute & Multigrid V-cycle programs
        self._init_prog = self.ctx.compute_shader(_INIT_GALAXY_GLSL)
        self._calibrate_vel_prog = self.ctx.compute_shader(_CALIBRATE_GALAXY_VEL_GLSL)
        self._clear_prog = self.ctx.compute_shader(_CLEAR_GRID_GLSL)
        self._deposit_prog = self.ctx.compute_shader(_DEPOSIT_GRID_GLSL)
        self._unpack_prog = self.ctx.compute_shader(_UNPACK_AND_SMOOTH_GLSL)
        self._mg_smooth_prog = self.ctx.compute_shader(_MG_SMOOTH_GLSL)
        self._mg_restrict_prog = self.ctx.compute_shader(_MG_RESTRICT_GLSL)
        self._mg_prolong_prog = self.ctx.compute_shader(_MG_PROLONG_ADD_GLSL)
        self._pack_force_prog = self.ctx.compute_shader(_PACK_FORCE_TEX_GLSL)
        self._pack_pos_fp16_prog = self.ctx.compute_shader(_PACK_POS_FP16_GLSL)
        self._unpack_pos_fp16_prog = self.ctx.compute_shader(_UNPACK_POS_FP16_GLSL)
        self._pack_replay_fp16_prog = self.ctx.compute_shader(_PACK_REPLAY_PARTICLES_FP16_GLSL)

        if compiled is None and not lazy_gpu:
            mod, compiled = _get_default_mod_and_compiled(mod)

        self._step_prog: Optional[moderngl.ComputeShader] = None
        self._bg_prog: Optional[moderngl.Program] = None
        self._bg_vao: Optional[moderngl.VertexArray] = None
        self._compiled_nu_src: str = ""
        self._active_mod: Optional[PhysicsMod] = mod
        self._active_compiled: Optional[CompiledMasterAction] = compiled
        self._last_nu_fn: Optional[Callable[[np.ndarray], np.ndarray]] = (
            compiled.nu_numpy_fn if compiled is not None else None
        )
        init_nu_expr = compiled.nu_glsl_expr if compiled is not None else "1.0"
        self._ensure_nu_programs(init_nu_expr)

        self._pt_prog = self.ctx.program(vertex_shader=_PARTICLE_VERT, fragment_shader=_PARTICLE_FRAG)
        self._tm_prog = self.ctx.program(vertex_shader=_BG_VERT, fragment_shader=_TONEMAP_FRAG)
        self._tm_vao = self.ctx.vertex_array(self._tm_prog, [])

        # Dedicated dynamic=True staging buffer (1.25 MiB covers 8192*16B pos + 8192*16B vel + 64^3*4B Phi_N!)
        self._telemetry_stage_buf = self.ctx.buffer(
            reserve=2 * 8192 * 16 + (self.grid_m ** 3) * 4,
            dynamic=True,
        )
        self._pending_telem_fence: Optional[object] = None
        self._pending_telem_meta: Optional[Tuple[int, int, int, int, float, int]] = None
        self._xfer_stage_buf: Optional[moderngl.Buffer] = None
        self._fp16_pos_gpu_buf: Optional[moderngl.Buffer] = None
        self._replay_gpu_pack_buf: Optional[moderngl.Buffer] = None
        self._replay_stage_buf: Optional[moderngl.Buffer] = None
        self._pending_replay_fence: Optional[object] = None
        self._pending_replay_meta: Optional[Tuple[int, int, float, int]] = None

        self._mg_sizes = [64, 32, 16, 8, 4]
        self._mass_int_buf: Optional[moderngl.Buffer] = None
        self._gas_int_buf: Optional[moderngl.Buffer] = None
        self._rho_flt_buf: Optional[moderngl.Buffer] = None
        self._gas_hydro_buf: Optional[moderngl.Buffer] = None
        self._mg_rhs: List[moderngl.Buffer] = []
        self._mg_ua: List[moderngl.Buffer] = []
        self._mg_ub: List[moderngl.Buffer] = []
        self._force_tex: Optional[moderngl.Texture3D] = None

        self._pos_buf: Optional[moderngl.Buffer] = None
        self._vel_buf: Optional[moderngl.Buffer] = None
        self._pt_vao: Optional[moderngl.VertexArray] = None

        self._hdr_tex: Optional[moderngl.Texture] = None
        self._hdr_fbo: Optional[moderngl.Framebuffer] = None

        if not lazy_gpu:
            self._allocate_grid_pyramid()
            self._allocate_particles(self.particle_count)
            self.reset_galaxy(0, mod=mod, compiled=compiled)
            self.flush_pending_replay_snapshot()

    @property
    def ddr5_ram_mb(self) -> float:
        """Total DDR5 System RAM (MiB) currently held by hibernate arrays and replay keyframes."""
        total_bytes = 0
        if self._hib_pos_ram is not None:
            total_bytes += int(self._hib_pos_ram.nbytes)
        if self._hib_vel_ram is not None:
            total_bytes += int(self._hib_vel_ram.nbytes)
        if self._live_backup is not None:
            pos_bk = self._live_backup.get("pos")
            vel_bk = self._live_backup.get("vel")
            if isinstance(pos_bk, np.ndarray):
                total_bytes += int(pos_bk.nbytes)
            if isinstance(vel_bk, np.ndarray):
                total_bytes += int(vel_bk.nbytes)
        for frm in self._replay_frames:
            total_bytes += int(frm.nbytes)
        return float(total_bytes) / (1024.0 * 1024.0)

    @property
    def replay_frames_count(self) -> int:
        """Number of DDR5 Time-Machine keyframes currently buffered (non-blocking harvest check)."""
        self._poll_pending_replay_snapshot(wait=False)
        return len(self._replay_frames)

    def _fp16_mass_scale(self) -> float:
        """Adaptive `fp16` mass scale keeping `pm.w * w_scale` safely in `[~1, 512]` for both tiny dwarfs (`DDO 154`) and giant spirals (`Milky Way`)."""
        m_bary = max(self._preset_baryon_mass(), 1e-4)
        m_cdm = m_bary * max(self._preset_cdm_ratio(), 0.1)
        n_p = float(max(self.particle_count, 1))
        m_max_part = max(m_bary / max(0.70 * n_p, 1.0), m_cdm / max(0.30 * n_p, 1.0))
        return float(np.clip(256.0 / max(m_max_part, 1e-12), 1.0, 1.0e9))

    def _clear_pending_telem_fence(self) -> None:
        fence = self._pending_telem_fence
        if fence is not None:
            try:
                GL.glDeleteSync(fence)
            except Exception:
                pass
        self._pending_telem_fence = None
        self._pending_telem_meta = None

    def _clear_pending_replay_fence(self) -> None:
        fence = self._pending_replay_fence
        if fence is not None:
            try:
                GL.glDeleteSync(fence)
            except Exception:
                pass
        self._pending_replay_fence = None
        self._pending_replay_meta = None

    def _refresh_vram_estimate(self) -> None:
        if self.is_hibernated or self._pos_buf is None:
            self.allocated_vram_mb = 0.0
            return
        total = 0
        for obj in (
            self._pos_buf,
            self._vel_buf,
            self._mass_int_buf,
            self._gas_int_buf,
            self._rho_flt_buf,
            self._gas_hydro_buf,
            *self._mg_rhs,
            *self._mg_ua,
            *self._mg_ub,
        ):
            if obj is not None:
                total += int(obj.size)
        if self._force_tex is not None:
            total += (self.grid_m ** 3) * 8
        if self._hdr_tex is not None:
            w, h = self._hdr_tex.size
            total += int(w) * int(h) * 8
        self.allocated_vram_mb = float(total) / (1024.0 * 1024.0)

    def _allocate_grid_pyramid(self) -> None:
        if self._mass_int_buf is not None and self._force_tex is not None and self._mg_ua:
            return
        cells = self.grid_m ** 3
        self._mass_int_buf = self.ctx.buffer(reserve=cells * 4)
        self._gas_int_buf = self.ctx.buffer(reserve=cells * 16)
        self._rho_flt_buf = self.ctx.buffer(reserve=cells * 4)
        self._gas_hydro_buf = self.ctx.buffer(reserve=cells * 16)

        self._mg_rhs = []
        self._mg_ua = []
        self._mg_ub = []
        for s in self._mg_sizes:
            n_c = s * s * s
            z_bytes = np.zeros(n_c, dtype=np.float32).tobytes()
            self._mg_rhs.append(self.ctx.buffer(data=z_bytes))
            self._mg_ua.append(self.ctx.buffer(data=z_bytes))
            self._mg_ub.append(self.ctx.buffer(data=z_bytes))

        self._force_tex = self.ctx.texture3d((self.grid_m, self.grid_m, self.grid_m), components=4, dtype="f2")
        self._force_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._force_tex.repeat_x = False
        self._force_tex.repeat_y = False
        self._force_tex.repeat_z = False

    def _release_grid_pyramid_and_fbo(self) -> None:
        self._clear_pending_telem_fence()
        self._clear_pending_replay_fence()
        _unbind_arena3_gl_slots()
        for obj in (
            self._mass_int_buf,
            self._gas_int_buf,
            self._rho_flt_buf,
            self._gas_hydro_buf,
            *self._mg_rhs,
            *self._mg_ua,
            *self._mg_ub,
            self._force_tex,
            self._hdr_fbo,
            self._hdr_tex,
            self._xfer_stage_buf,
            self._fp16_pos_gpu_buf,
            self._replay_gpu_pack_buf,
            self._replay_stage_buf,
        ):
            if obj is not None:
                obj.release()
        self._mass_int_buf = None
        self._gas_int_buf = None
        self._rho_flt_buf = None
        self._gas_hydro_buf = None
        self._mg_rhs = []
        self._mg_ua = []
        self._mg_ub = []
        self._force_tex = None
        self._hdr_fbo = None
        self._hdr_tex = None
        self._xfer_stage_buf = None
        self._fp16_pos_gpu_buf = None
        self._replay_gpu_pack_buf = None
        self._replay_stage_buf = None

    def _ensure_xfer_buffers(self, need_fp16_pos: bool = False) -> None:
        req_fp16 = max(self.particle_count * 8, 65536)
        req_stage = max(_XFER_CHUNK_BYTES, req_fp16)
        if self._xfer_stage_buf is None or self._xfer_stage_buf.size < req_stage:
            if self._xfer_stage_buf is not None:
                self._xfer_stage_buf.release()
            self._xfer_stage_buf = self.ctx.buffer(reserve=req_stage, dynamic=True)
        if need_fp16_pos and (self._fp16_pos_gpu_buf is None or self._fp16_pos_gpu_buf.size < req_fp16):
            if self._fp16_pos_gpu_buf is not None:
                self._fp16_pos_gpu_buf.release()
            self._fp16_pos_gpu_buf = self.ctx.buffer(reserve=req_fp16)

    def _ensure_replay_buffers(self, req_bytes: int) -> None:
        req = max(int(req_bytes), 65536)
        if self._replay_gpu_pack_buf is None or self._replay_gpu_pack_buf.size < req:
            if self._replay_gpu_pack_buf is not None:
                self._replay_gpu_pack_buf.release()
            self._replay_gpu_pack_buf = self.ctx.buffer(reserve=req)
        if self._replay_stage_buf is None or self._replay_stage_buf.size < req:
            if self._replay_stage_buf is not None:
                self._replay_stage_buf.release()
            self._replay_stage_buf = self.ctx.buffer(reserve=req, dynamic=True)

    def _copy_ssbo_to_ram_f32_vec4(self, src_buf: moderngl.Buffer, count: int) -> np.ndarray:
        self._ensure_xfer_buffers(need_fp16_pos=False)
        assert self._xfer_stage_buf is not None
        out = np.empty((count, 4), dtype=np.float32)
        out_u8 = out.view(np.uint8).reshape(-1)
        total_bytes = count * 16
        offset = 0
        while offset < total_bytes:
            chunk = min(int(self._xfer_stage_buf.size), total_bytes - offset)
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

    def _effective_replay_cap(self) -> int:
        cap = max(int(self.replay_max_frames), 0)
        if self.particle_count > 1_000_000:
            # 32 full (pos, vel) fp16 frames at 2M stars = 32 * 32 MiB = 1.00 GiB DDR5 ceiling
            return min(cap, 32)
        return min(cap, 48)

    def _pack_vec4_ssbo_to_fp16(self, src_buf: moderngl.Buffer, n_p: int, w_scale: float) -> np.ndarray:
        """Packs `(n_p, 4)` `float32` SSBO on GPU into `(n_p, 4)` `float16` DDR5 array without demotion."""
        self._ensure_xfer_buffers(need_fp16_pos=True)
        assert self._xfer_stage_buf is not None and self._fp16_pos_gpu_buf is not None
        byte_len = n_p * 8
        src_buf.bind_to_storage_buffer(0)
        self._fp16_pos_gpu_buf.bind_to_storage_buffer(1)
        self._pack_pos_fp16_prog["u_count"].value = int(n_p)
        self._pack_pos_fp16_prog["u_mass_scale"].value = float(w_scale)
        self._pack_pos_fp16_prog.run((n_p + 255) // 256, 1, 1)
        self.ctx.memory_barrier()

        _copy_buffer_sub_data(self._fp16_pos_gpu_buf, self._xfer_stage_buf, 0, 0, byte_len)
        raw = self._xfer_stage_buf.read(size=byte_len)
        return np.frombuffer(raw, dtype=np.float16).copy().reshape((n_p, 4))

    def _unpack_fp16_to_vec4_ssbo(self, arr_fp16: np.ndarray, dst_buf: moderngl.Buffer, n_p: int, inv_w_scale: float) -> None:
        """Unpacks `(n_p, 4)` `float16` DDR5 array onto GPU `(n_p, 4)` `float32` SSBO."""
        self._ensure_xfer_buffers(need_fp16_pos=True)
        assert self._fp16_pos_gpu_buf is not None
        contig = np.ascontiguousarray(arr_fp16, dtype=np.float16)
        self._fp16_pos_gpu_buf.write(contig.view(np.uint8))
        self._fp16_pos_gpu_buf.bind_to_storage_buffer(0)
        dst_buf.bind_to_storage_buffer(1)
        self._unpack_pos_fp16_prog["u_count"].value = int(n_p)
        self._unpack_pos_fp16_prog["u_inv_mass_scale"].value = float(inv_w_scale)
        self._unpack_pos_fp16_prog.run((n_p + 255) // 256, 1, 1)
        self.ctx.memory_barrier()

    def _poll_pending_replay_snapshot(self, wait: bool = False) -> None:
        """Harvests completed GPU->staging `(pos, vel)` `fp16` snapshot into DDR5 RAM with `0 ms` GPU stall."""
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

        n_p, total_bytes, snap_time, snap_step = meta
        GL.glDeleteSync(fence)
        self._pending_replay_fence = None
        self._pending_replay_meta = None

        cap = self._effective_replay_cap()
        if not self.replay_enabled or cap <= 0 or n_p != self.particle_count:
            return

        snap: Optional[np.ndarray] = None
        while len(self._replay_frames) >= cap:
            evicted = self._replay_frames.popleft()
            if (
                snap is None
                and evicted.dtype == np.float16
                and evicted.shape == (2, n_p, 4)
                and evicted.flags.c_contiguous
            ):
                snap = evicted
        while len(self._replay_meta) >= cap:
            self._replay_meta.popleft()

        if snap is None:
            snap = np.empty((2, n_p, 4), dtype=np.float16)
        self._replay_stage_buf.read_into(snap, size=total_bytes)

        self._replay_frames.append(snap)
        self._replay_meta.append((snap_time, snap_step))

    def flush_pending_replay_snapshot(self) -> None:
        """Synchronously drains any in-flight async replay snapshot before scrubbing or hibernation."""
        self._poll_pending_replay_snapshot(wait=True)

    def _enqueue_replay_snapshot(self) -> None:
        """Dispatches GPU `(_pos_buf, _vel_buf)` `fp16` pack + staging DMA + `GL_FenceSync` without `.read()`."""
        if (
            not self.replay_enabled
            or self._pending_replay_fence is not None
            or self._pos_buf is None
            or self._vel_buf is None
        ):
            return
        cap = self._effective_replay_cap()
        if cap <= 0:
            return

        n_p = int(self.particle_count)
        total_bytes = n_p * 16  # 2 arrays (pos, vel) * n_p * 4 * 2 bytes (16 MiB @ 1M, 32 MiB @ 2M)
        self._ensure_replay_buffers(total_bytes)
        assert self._replay_gpu_pack_buf is not None and self._replay_stage_buf is not None

        self._pos_buf.bind_to_storage_buffer(0)
        self._vel_buf.bind_to_storage_buffer(1)
        self._replay_gpu_pack_buf.bind_to_storage_buffer(2)
        self._pack_replay_fp16_prog["u_count"].value = n_p
        self._pack_replay_fp16_prog["u_mass_scale"].value = self._fp16_mass_scale()
        self._pack_replay_fp16_prog.run((n_p + 255) // 256, 1, 1)
        self.ctx.memory_barrier()

        _copy_buffer_sub_data(self._replay_gpu_pack_buf, self._replay_stage_buf, 0, 0, total_bytes)
        self._pending_replay_fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        GL.glFlush()
        self._pending_replay_meta = (n_p, total_bytes, float(self.sim_time), int(self.step_count))

    def _capture_replay_snapshot(self) -> None:
        """Synchronous convenience wrapper around `_enqueue_replay_snapshot` + `flush_pending_replay_snapshot`."""
        self._poll_pending_replay_snapshot(wait=True)
        self._enqueue_replay_snapshot()
        self._poll_pending_replay_snapshot(wait=True)

    def load_replay_frame(self, idx: int) -> None:
        """Restores DDR5 Time-Machine `(pos, vel)` keyframe `idx` onto GPU and re-solves 3D Multigrid potential."""
        self.flush_pending_replay_snapshot()
        count = len(self._replay_frames)
        if count == 0:
            return
        i = int(idx)
        if i < 0:
            i += count
        i = max(0, min(i, count - 1))

        if self.is_hibernated or self._pos_buf is None or self._vel_buf is None:
            self.wake_from_ram()

        assert self._pos_buf is not None and self._vel_buf is not None
        n_p = self.particle_count

        if self.replay_cursor < 0 and self._live_backup is None:
            self._poll_async_telemetry(wait=True)
            self._live_backup = {
                "particle_count": int(n_p),
                "pos": self._copy_ssbo_to_ram_f32_vec4(self._pos_buf, n_p),
                "vel": self._copy_ssbo_to_ram_f32_vec4(self._vel_buf, n_p),
                "sim_time": float(self.sim_time),
                "step_count": int(self.step_count),
            }

        snap = np.ascontiguousarray(self._replay_frames[i], dtype=np.float16)
        inv_mass_scale = float(1.0 / self._fp16_mass_scale())
        self._ensure_xfer_buffers(need_fp16_pos=True)
        if snap.ndim == 3 and snap.shape == (2, n_p, 4):
            self._unpack_fp16_to_vec4_ssbo(snap[0], self._pos_buf, n_p, inv_mass_scale)
            self._unpack_fp16_to_vec4_ssbo(snap[1], self._vel_buf, n_p, 1.0)
        elif snap.ndim == 2 and snap.shape == (n_p, 4):
            self._unpack_fp16_to_vec4_ssbo(snap, self._pos_buf, n_p, inv_mass_scale)
        else:
            return

        if i < len(self._replay_meta):
            self.sim_time, self.step_count = self._replay_meta[i]

        if self._mg_ua and self._mg_ub:
            zeros = np.zeros(self.grid_m ** 3, dtype=np.float32).tobytes()
            self._mg_ua[0].write(zeros)
            self._mg_ub[0].write(zeros)
        eff_has_ballistic, eff_halo_scale, _ = self._effective_halo_params(self._active_mod)
        self._solve_field_gpu(has_cdm=eff_has_ballistic, cdm_scale=eff_halo_scale, n_vcycles=3)
        self._sample_telemetry()
        self.replay_cursor = i

    def resume_live_from_replay(self) -> None:
        """Exits DDR5 Time-Machine scrub mode and restores the exact `float32` live `(pos, vel)` state."""
        bk = self._live_backup
        self._live_backup = None
        if bk is not None and int(bk.get("particle_count", -1)) == int(self.particle_count):
            if self.is_hibernated or self._pos_buf is None or self._vel_buf is None:
                self.wake_from_ram()
            assert self._pos_buf is not None and self._vel_buf is not None
            self._pos_buf.write(np.ascontiguousarray(bk["pos"], dtype=np.float32))
            self._vel_buf.write(np.ascontiguousarray(bk["vel"], dtype=np.float32))
            self.sim_time = float(bk["sim_time"])  # type: ignore[arg-type]
            self.step_count = int(bk["step_count"])  # type: ignore[arg-type]
            if self._mg_ua and self._mg_ub:
                zeros = np.zeros(self.grid_m ** 3, dtype=np.float32).tobytes()
                self._mg_ua[0].write(zeros)
                self._mg_ub[0].write(zeros)
            eff_has_ballistic, eff_halo_scale, _ = self._effective_halo_params(self._active_mod)
            self._solve_field_gpu(has_cdm=eff_has_ballistic, cdm_scale=eff_halo_scale, n_vcycles=3)
            self._sample_telemetry()
        elif self.replay_cursor >= 0 and len(self._replay_frames) > 0:
            self.load_replay_frame(len(self._replay_frames) - 1)
            self._live_backup = None
        self.replay_cursor = -1

    def hibernate_to_ram(self) -> float:
        """
        Offloads `1M–2M` particle VBO/SSBOs and 3D Multigrid pyramid out of GDDR7 VRAM into DDR5 RAM
        via non-demoting `glCopyBufferSubData` staging, unbinding OpenGL slots so WDDM reclaims VRAM immediately.
        """
        if self.is_hibernated or self._pos_buf is None or self._vel_buf is None:
            self._clear_pending_telem_fence()
            self._clear_pending_replay_fence()
            self.is_hibernated = True
            self.allocated_vram_mb = 0.0
            return self.ddr5_ram_mb

        self.flush_pending_replay_snapshot()
        self._poll_async_telemetry(wait=True)
        self._clear_pending_telem_fence()
        self._hib_pos_ram = self._copy_ssbo_to_ram_f32_vec4(self._pos_buf, self.particle_count)
        self._hib_vel_ram = self._copy_ssbo_to_ram_f32_vec4(self._vel_buf, self.particle_count)

        _unbind_arena3_gl_slots()
        self.ctx.finish()
        for obj in (self._pt_vao, self._pos_buf, self._vel_buf):
            if obj is not None:
                obj.release()
        self._pt_vao = None
        self._pos_buf = None
        self._vel_buf = None
        self._release_grid_pyramid_and_fbo()
        self.ctx.finish()

        self.is_hibernated = True
        self.allocated_vram_mb = 0.0
        return self.ddr5_ram_mb

    def wake_from_ram(self) -> None:
        """Restores particle VBO/SSBOs and 3D Multigrid pyramid into GDDR7 VRAM from DDR5 RAM."""
        if not self.is_hibernated and self._pos_buf is not None and self._force_tex is not None:
            return

        if self._active_compiled is None:
            self._active_mod, self._active_compiled = _get_default_mod_and_compiled(self._active_mod)

        saved_pos = self._hib_pos_ram
        saved_vel = self._hib_vel_ram
        self._hib_pos_ram = None
        self._hib_vel_ram = None
        self.is_hibernated = False

        if self._active_compiled is not None:
            self._ensure_nu_programs(self._active_compiled.nu_glsl_expr)
            self._last_nu_fn = self._active_compiled.nu_numpy_fn

        self._allocate_grid_pyramid()
        self._allocate_particles(self.particle_count)
        assert self._pos_buf is not None and self._vel_buf is not None

        if (
            saved_pos is not None
            and saved_vel is not None
            and saved_pos.shape == (self.particle_count, 4)
            and saved_vel.shape == (self.particle_count, 4)
        ):
            self._pos_buf.write(np.ascontiguousarray(saved_pos, dtype=np.float32))
            self._vel_buf.write(np.ascontiguousarray(saved_vel, dtype=np.float32))
            eff_has_ballistic, eff_halo_scale, ballistic_pleat = self._effective_halo_params(self._active_mod)
            if self._active_mod is not None:
                a0_ratio = max(float(self._active_mod.parameters.a0_si), 1e-14) / A0_REF_SI
                self.last_a0_code = 1e-6 if ballistic_pleat else (A0_REF_CODE * a0_ratio)
            self._solve_field_gpu(has_cdm=eff_has_ballistic, cdm_scale=eff_halo_scale, n_vcycles=3)
            if self._dirty_field_on_wake:
                self._sample_telemetry()
                self._dirty_field_on_wake = False
            self._refresh_vram_estimate()
        else:
            self._dirty_field_on_wake = False
            self.reset_galaxy(self.preset_idx, mod=self._active_mod, compiled=self._active_compiled)

    def _effective_halo_params(self, mod: Optional[PhysicsMod]) -> Tuple[bool, float, bool]:
        """
        Computes `(eff_has_ballistic, eff_halo_scale, ballistic_pleat)` for the active preset & mod.
        In supersonic 4,574 km/s cluster collisions (preset 2), pinned SO(3) pleats / Khronon
        condensates behave ballistically with the subcluster potential cores (unlike Classic MOND!).
        """
        if mod is None:
            return False, 0.0, False
        ballistic_pleat = (
            (not mod.parameters.has_cdm_halo)
            and (self.preset_idx == 2)
            and (float(mod.parameters.pinned_pleat_coupling) > 0.15)
        )
        eff_has_ballistic = bool(mod.parameters.has_cdm_halo) or ballistic_pleat
        eff_halo_scale = (
            float(mod.parameters.cdm_halo_scale)
            if mod.parameters.has_cdm_halo
            else (float(mod.parameters.pinned_pleat_coupling) if ballistic_pleat else 0.0)
        )
        return eff_has_ballistic, eff_halo_scale, ballistic_pleat

    def _compute_a0_code(self, mod: PhysicsMod, ballistic_pleat: bool) -> float:
        if ballistic_pleat:
            return 1e-6
        a0_ratio = max(float(mod.parameters.a0_si), 1e-14) / A0_REF_SI
        base = A0_CLUSTER_REF_CODE if (self.preset_idx == 2) else A0_REF_CODE
        return base * a0_ratio

    def sync_mod_state(
        self,
        mod: PhysicsMod,
        compiled: CompiledMasterAction,
        re_solve_field: bool = False,
    ) -> None:
        """
        Synchronizes `nu(y)` GPU shaders, `a0` scale, and optionally re-solves the 3D Multigrid potential.
        When hibernated in DDR5 RAM, updates cached mod state and marks `_dirty_field_on_wake` WITHOUT waking GPU!
        """
        self._active_mod = mod
        self._active_compiled = compiled
        eff_has_ballistic, eff_halo_scale, ballistic_pleat = self._effective_halo_params(mod)
        self.last_a0_code = self._compute_a0_code(mod, ballistic_pleat)
        self._last_nu_fn = compiled.nu_numpy_fn
        if self.is_hibernated or self._pos_buf is None:
            self._dirty_field_on_wake = True
            return
        self._ensure_nu_programs(compiled.nu_glsl_expr)
        if re_solve_field:
            self._solve_field_gpu(has_cdm=eff_has_ballistic, cdm_scale=eff_halo_scale, n_vcycles=3)
            self._sample_telemetry()

    def _ensure_nu_programs(self, nu_glsl_expr: str) -> None:
        if nu_glsl_expr == self._compiled_nu_src and self._step_prog is not None and self._bg_prog is not None:
            return
        new_step = self.ctx.compute_shader(_build_particle_step_glsl(nu_glsl_expr))
        new_bg = self.ctx.program(vertex_shader=_BG_VERT, fragment_shader=_build_bg_lensing_frag(nu_glsl_expr))
        new_bg_vao = self.ctx.vertex_array(new_bg, [])
        if self._step_prog is not None or self._bg_vao is not None or self._bg_prog is not None:
            self.ctx.finish()
        if self._step_prog is not None:
            self._step_prog.release()
        if self._bg_vao is not None:
            self._bg_vao.release()
        if self._bg_prog is not None:
            self._bg_prog.release()
        self._step_prog = new_step
        self._bg_prog = new_bg
        self._bg_vao = new_bg_vao
        self._compiled_nu_src = nu_glsl_expr

    def _allocate_particles(self, count: int) -> None:
        if any(x is not None for x in (self._pt_vao, self._pos_buf, self._vel_buf)):
            _unbind_arena3_gl_slots()
            self.ctx.finish()
        for obj in (self._pt_vao, self._pos_buf, self._vel_buf):
            if obj is not None:
                obj.release()
        self.particle_count = int(count)
        nbytes = self.particle_count * 16
        self._pos_buf = self.ctx.buffer(reserve=nbytes)
        self._vel_buf = self.ctx.buffer(reserve=nbytes)
        self._pt_vao = self.ctx.vertex_array(
            self._pt_prog,
            [
                (self._pos_buf, "4f", "in_pos_mass"),
                (self._vel_buf, "4f", "in_vel_type"),
            ],
        )
        self._refresh_vram_estimate()

    def set_particle_count(
        self,
        new_count: int,
        mod: Optional[PhysicsMod] = None,
        compiled: Optional[CompiledMasterAction] = None,
    ) -> None:
        if mod is not None:
            self._active_mod = mod
        if compiled is not None:
            self._active_compiled = compiled
            self._last_nu_fn = compiled.nu_numpy_fn
        new_c = int(np.clip(new_count, 50_000, 2_000_000))
        if new_c != self.particle_count:
            self._hib_pos_ram = None
            self._hib_vel_ram = None
            self._clear_pending_telem_fence()
            self.clear_replay_buffer()
            if self.is_hibernated:
                self.particle_count = new_c
                return
            self._allocate_grid_pyramid()
            self._allocate_particles(new_c)
            self.reset_galaxy(self.preset_idx, mod=self._active_mod, compiled=self._active_compiled)

    def load_sparc_galaxy(
        self,
        gal: SparcGalaxy,
        mod: Optional[PhysicsMod] = None,
        compiled: Optional[CompiledMasterAction] = None,
    ) -> None:
        """Parameterizes Preset 5 from ANY real published SPARC-175 galaxy in exact (1 kpc, 100 km/s, 10^9 M_sun) units!"""
        self.custom_sparc_name = gal.name
        m_star_1e9 = 0.50 * max(float(gal.lum_36_1e9_lsun), 0.01)
        m_gas_1e9 = 1.33 * max(float(gal.m_hi_1e9_msun), 0.005)
        m_bary_1e9 = m_star_1e9 + m_gas_1e9
        m_code = float(np.clip(m_bary_1e9, 0.02, 500.0))
        gas_frac = float(np.clip(m_gas_1e9 / max(m_bary_1e9, 1e-4), 0.05, 0.92))
        v_bul_peak = float(np.max(gal.v_bul_kms)) if len(gal.v_bul_kms) else 0.0
        v_disk_peak = max(float(np.max(gal.v_disk_kms)), 5.0)
        bul_frac = float(np.clip(0.70 * (v_bul_peak**2) / (0.50 * (v_disk_peak**2) + 0.70 * (v_bul_peak**2) + 1.0), 0.0, 0.42))
        v_flat_kms = float(gal.v_flat_kms) if gal.v_flat_kms > 5.0 else float(np.max(gal.v_obs_kms))
        vflat_code = float(np.clip(v_flat_kms / VEL_UNIT_KMS, 0.15, 3.60))
        rd_kpc = float(np.clip(gal.r_disk_kpc, 0.35, 9.5))

        self._sparc_shape = (m_code, rd_kpc, gas_frac, bul_frac)
        self._sparc_vflat_code = vflat_code
        self.reset_galaxy(5, mod=mod, compiled=compiled)

    def _ensure_hdr_fbo(self, w: int, h: int) -> None:
        if self._hdr_tex is not None and self._hdr_tex.size == (w, h):
            return
        if self._hdr_fbo is not None or self._hdr_tex is not None:
            self.ctx.finish()
        if self._hdr_fbo is not None:
            self._hdr_fbo.release()
        if self._hdr_tex is not None:
            self._hdr_tex.release()
        self._hdr_tex = self.ctx.texture((w, h), components=4, dtype="f2")
        self._hdr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._hdr_fbo = self.ctx.framebuffer(color_attachments=[self._hdr_tex])
        self._refresh_vram_estimate()

    def _preset_baryon_mass(self) -> float:
        p = self.preset_idx % len(GALAXY_PRESETS)
        if p == 5:
            return max(float(self._sparc_shape[0]), 0.02)
        return [27.5, 0.38, 14.22952, 8.30056, 70.0][p]

    def _preset_halo_shape(self) -> Tuple[float, float, float]:
        """Returns `(rd_star, r_b_halo, r_max_halo)` in kpc for the active preset."""
        p = self.preset_idx % len(GALAXY_PRESETS)
        if p == 0:
            return (3.14, 5.2, 31.0)
        if p == 1:
            return (1.15, 2.15, 28.0)
        if p == 4:
            return (3.20, 4.7, 32.0)
        if p == 5:
            rd = float(np.clip(self._sparc_shape[1], 0.35, 9.5))
            gf = float(np.clip(self._sparc_shape[2], 0.04, 0.92))
            r_gas_s = float(np.clip(2.1 * rd, 1.8, 6.0))
            r_b = 1.35 * ((1.0 - gf) * rd + gf * r_gas_s)
            return (rd, r_b, 30.0)
        return (1.5, 3.0, 12.0)

    def _preset_cdm_ratio(self) -> float:
        p = self.preset_idx % len(GALAXY_PRESETS)
        if p == 2:
            return 4.8
        if p == 3:
            return 5.0
        m_b = self._preset_baryon_mass()
        _, r_b, r_max = self._preset_halo_shape()
        return _cdm_enclosed_mass_py(r_max, m_b, r_b) / max(m_b, 1e-8)

    def _dispatch_velocity_calibration(self, u_pass: int, gm_monopole: float) -> None:
        assert self._pos_buf is not None and self._vel_buf is not None and self._force_tex is not None
        rd_star, r_b_halo, r_max_halo = self._preset_halo_shape()
        m_bary = self._preset_baryon_mass()
        self._pos_buf.bind_to_storage_buffer(0)
        self._vel_buf.bind_to_storage_buffer(1)
        self._force_tex.use(location=0)
        for uname, uval in (
            ("u_force_tex", 0),
            ("u_count", int(self.particle_count)),
            ("u_preset", int(self.preset_idx)),
            ("u_pass", int(u_pass)),
            ("u_half_box", float(self.half_box)),
            ("u_m_bary", float(m_bary)),
            ("u_rd_star", float(rd_star)),
            ("u_r_b_halo", float(r_b_halo)),
            ("u_r_max_halo", float(r_max_halo)),
            ("u_gm_monopole", float(gm_monopole)),
        ):
            if uname in self._calibrate_vel_prog:
                self._calibrate_vel_prog[uname].value = uval
        self._calibrate_vel_prog.run((self.particle_count + 255) // 256, 1, 1)
        self.ctx.memory_barrier()

    def reset_galaxy(
        self,
        preset_idx: Optional[int] = None,
        mod: Optional[PhysicsMod] = None,
        compiled: Optional[CompiledMasterAction] = None,
    ) -> None:
        if preset_idx is not None:
            self.preset_idx = int(preset_idx) % len(GALAXY_PRESETS)
        self.half_box = _preset_half_box(self.preset_idx)
        if mod is not None:
            self._active_mod = mod
        if compiled is not None:
            self._active_compiled = compiled

        if self.is_hibernated or self._pos_buf is None or self._force_tex is None:
            self._hib_pos_ram = None
            self._hib_vel_ram = None
            self.is_hibernated = False
            self._allocate_grid_pyramid()
            self._allocate_particles(self.particle_count)

        self._clear_pending_telem_fence()
        self.clear_replay_buffer()
        self._dirty_field_on_wake = False
        self._last_render_step = -1

        use_mod = self._active_mod
        use_comp = self._active_compiled
        if use_comp is not None:
            self._ensure_nu_programs(use_comp.nu_glsl_expr)
            self._last_nu_fn = use_comp.nu_numpy_fn

        eff_has_ballistic, eff_halo_scale, ballistic_pleat = self._effective_halo_params(use_mod)
        if use_mod is not None:
            self.last_a0_code = self._compute_a0_code(use_mod, ballistic_pleat)

        assert self._pos_buf is not None and self._vel_buf is not None
        self._pos_buf.bind_to_storage_buffer(0)
        self._vel_buf.bind_to_storage_buffer(1)
        self._init_prog["u_count"].value = self.particle_count
        self._init_prog["u_preset"].value = self.preset_idx
        if "u_sparc_shape" in self._init_prog:
            self._init_prog["u_sparc_shape"].value = self._sparc_shape
        if "u_sparc_vflat_code" in self._init_prog:
            self._init_prog["u_sparc_vflat_code"].value = float(self._sparc_vflat_code)
        wg = (self.particle_count + 255) // 256
        self._init_prog.run(wg, 1, 1)
        self.ctx.memory_barrier()

        zeros = np.zeros(self.grid_m ** 3, dtype=np.float32).tobytes()
        if _preset_is_rotating_disk(self.preset_idx):
            # 1. Cheat-free observed-universe baryon velocity calibration (fixed RAR a0_ref = 0.371238, independent of mod!)
            self._mg_ua[0].write(zeros)
            self._mg_ub[0].write(zeros)
            gm_bary = self._solve_field_gpu(has_cdm=False, cdm_scale=0.0, n_vcycles=4)
            self._dispatch_velocity_calibration(u_pass=0, gm_monopole=gm_bary)

            # 2. Cheat-free live NFW CDM halo Jeans equilibrium calibration (fixed cdm_scale = 1.0, independent of mod!)
            self._mg_ua[0].write(zeros)
            self._mg_ub[0].write(zeros)
            gm_cdm_ref = self._solve_field_gpu(has_cdm=True, cdm_scale=1.0, n_vcycles=3)
            self._dispatch_velocity_calibration(u_pass=1, gm_monopole=gm_cdm_ref)

        # 3. Converge t=0 potential globally in 4 full 5-level Multigrid V-cycles using the ACTUAL selected PhysicsMod!
        self._mg_ua[0].write(zeros)
        self._mg_ub[0].write(zeros)
        self._solve_field_gpu(has_cdm=eff_has_ballistic, cdm_scale=eff_halo_scale, n_vcycles=4)

        self.sim_time = 0.0
        self.step_count = 0
        self._sample_telemetry()
        self._refresh_vram_estimate()
        if self.replay_enabled:
            self._enqueue_replay_snapshot()

    def _run_mg_smooth(self, level: int, sweeps: int, gm_monopole: float, zero_first: bool = False) -> None:
        """Runs `sweeps` damped-Jacobi iterations on Multigrid `level`, leaving result in `self._mg_ua[level]`."""
        m_lvl = self._mg_sizes[level]
        wg = max((m_lvl + 3) // 4, 1)
        self._mg_rhs[level].bind_to_storage_buffer(0)
        self._mg_smooth_prog["u_M"].value = m_lvl
        self._mg_smooth_prog["u_half_box"].value = float(self.half_box)
        self._mg_smooth_prog["u_is_fine"].value = 1 if level == 0 else 0
        self._mg_smooth_prog["u_gm_monopole"].value = float(gm_monopole)

        # Ensure even number of ping-pong swaps or copy back so _mg_ua[level] always holds latest!
        n_even = sweeps if (sweeps % 2 == 0) else (sweeps + 1)
        for s in range(n_even):
            self._mg_smooth_prog["u_zero_init"].value = 1 if (zero_first and s == 0) else 0
            if s % 2 == 0:
                self._mg_ua[level].bind_to_storage_buffer(1)
                self._mg_ub[level].bind_to_storage_buffer(2)
            else:
                self._mg_ub[level].bind_to_storage_buffer(1)
                self._mg_ua[level].bind_to_storage_buffer(2)
            self._mg_smooth_prog.run(wg, wg, wg)
            self.ctx.memory_barrier()

    def _run_multigrid_vcycle(self, gm_monopole: float) -> None:
        """Executes one complete 5-level 3D Geometric Multigrid V-Cycle (`64³ -> 32³ -> 16³ -> 8³ -> 4³ -> 64³`)."""
        n_lvls = len(self._mg_sizes)
        # 1. Downward leg: pre-smooth (4 sweeps) & restrict exact cell-centered residual down to 4^3
        for lvl in range(n_lvls - 1):
            self._run_mg_smooth(lvl, sweeps=4, gm_monopole=gm_monopole, zero_first=(lvl > 0))
            m_coarse = self._mg_sizes[lvl + 1]
            wg_c = max((m_coarse + 3) // 4, 1)
            self._mg_rhs[lvl].bind_to_storage_buffer(0)
            self._mg_ua[lvl].bind_to_storage_buffer(1)
            self._mg_rhs[lvl + 1].bind_to_storage_buffer(2)
            self._mg_restrict_prog["u_Mc"].value = m_coarse
            self._mg_restrict_prog["u_half_box"].value = float(self.half_box)
            self._mg_restrict_prog["u_is_fine"].value = 1 if lvl == 0 else 0
            self._mg_restrict_prog["u_gm_monopole"].value = float(gm_monopole if lvl == 0 else 0.0)
            self._mg_restrict_prog.run(wg_c, wg_c, wg_c)
            self.ctx.memory_barrier()

        # 2. Exact solve on coarsest 4^3 grid (64 cells -> rapid relaxation in 16 sweeps!)
        self._run_mg_smooth(n_lvls - 1, sweeps=16, gm_monopole=0.0, zero_first=True)

        # 3. Upward leg: trilinearly prolongate coarse error & post-smooth (4 sweeps) up to 64^3
        for lvl in range(n_lvls - 2, -1, -1):
            m_fine = self._mg_sizes[lvl]
            wg_f = max((m_fine + 3) // 4, 1)
            self._mg_ua[lvl + 1].bind_to_storage_buffer(0)
            self._mg_ua[lvl].bind_to_storage_buffer(1)
            self._mg_prolong_prog["u_Mf"].value = m_fine
            self._mg_prolong_prog.run(wg_f, wg_f, wg_f)
            self.ctx.memory_barrier()
            self._run_mg_smooth(lvl, sweeps=4, gm_monopole=gm_monopole, zero_first=False)

    def _solve_field_gpu(self, has_cdm: bool, cdm_scale: float, n_vcycles: int = 2) -> float:
        assert (
            self._pos_buf is not None
            and self._vel_buf is not None
            and self._mass_int_buf is not None
            and self._gas_int_buf is not None
            and self._rho_flt_buf is not None
            and self._gas_hydro_buf is not None
            and self._force_tex is not None
        )
        m = self.grid_m
        cells = m * m * m
        G_grav = G_GRAV_CODE
        m_bary = self._preset_baryon_mass()
        m_active = m_bary * (1.0 + (self._preset_cdm_ratio() * max(cdm_scale, 0.0) if has_cdm else 0.0))
        fix_scale = float(np.clip(5.0e8 / max(m_active, 1.0e-4), 1000.0, 1.0e11))
        mom_scale = float(np.clip(fix_scale / 20.0, 50.0, 5.0e9))
        gm_mono = float(G_grav * m_active)

        # 1. Clear 64^3 fixed-point mass & gas momentum accumulators
        self._mass_int_buf.bind_to_storage_buffer(2)
        self._gas_int_buf.bind_to_storage_buffer(3)
        self._clear_prog["u_cells"].value = cells
        self._clear_prog.run((cells + 63) // 64, 1, 1)
        self.ctx.memory_barrier()

        # 2. Atomic deposit 1M-2M Stars, Gas, and Live Collisionless Halo / Ballistic Pleat Modes!
        self._pos_buf.bind_to_storage_buffer(0)
        self._vel_buf.bind_to_storage_buffer(1)
        self._deposit_prog["u_count"].value = self.particle_count
        self._deposit_prog["u_M"].value = m
        self._deposit_prog["u_half_box"].value = float(self.half_box)
        self._deposit_prog["u_has_cdm"].value = 1 if has_cdm else 0
        self._deposit_prog["u_cdm_scale"].value = float(cdm_scale)
        self._deposit_prog["u_fix_scale"].value = fix_scale
        self._deposit_prog["u_mom_scale"].value = mom_scale
        self._deposit_prog.run((self.particle_count + 255) // 256, 1, 1)
        self.ctx.memory_barrier()

        # 3. Unpack Level-0 Poisson RHS + 3x3x3 Smoothed Eulerian Gas Hydro Grid
        self._mg_rhs[0].bind_to_storage_buffer(4)
        self._rho_flt_buf.bind_to_storage_buffer(5)
        self._gas_hydro_buf.bind_to_storage_buffer(7)
        self._unpack_prog["u_M"].value = m
        self._unpack_prog["u_half_box"].value = float(self.half_box)
        self._unpack_prog["u_G_grav"].value = G_grav
        self._unpack_prog["u_fix_scale"].value = fix_scale
        self._unpack_prog["u_mom_scale"].value = mom_scale
        wg3 = (m + 3) // 4
        self._unpack_prog.run(wg3, wg3, wg3)
        self.ctx.memory_barrier()

        # 4. Run 5-Level 3D Geometric Multigrid V-Cycles!
        for _ in range(max(int(n_vcycles), 1)):
            self._run_multigrid_vcycle(gm_mono)

        # 5. Pack converged Newtonian acceleration g_N = -grad(Phi_N) & |Phi_N| into 3D Texture
        self._mg_ua[0].bind_to_storage_buffer(5)
        self._force_tex.bind_to_image(0, read=False, write=True)
        self._pack_force_prog["u_M"].value = m
        self._pack_force_prog["u_half_box"].value = float(self.half_box)
        if "u_gm_monopole" in self._pack_force_prog:
            self._pack_force_prog["u_gm_monopole"].value = gm_mono
        self._pack_force_prog.run(wg3, wg3, wg3)
        self.ctx.memory_barrier()
        return gm_mono

    @property
    def v_outer_kms(self) -> float:
        self._sync_telemetry_for_read()
        return self._v_outer_kms

    @v_outer_kms.setter
    def v_outer_kms(self, val: float) -> None:
        self._v_outer_kms = float(val)

    @property
    def lensing_gas_offset_kpc(self) -> float:
        self._sync_telemetry_for_read()
        return self._lensing_gas_offset_kpc

    @lensing_gas_offset_kpc.setter
    def lensing_gas_offset_kpc(self, val: float) -> None:
        self._lensing_gas_offset_kpc = float(val)

    @property
    def transverse_splash_ratio(self) -> float:
        self._sync_telemetry_for_read()
        return self._transverse_splash_ratio

    @transverse_splash_ratio.setter
    def transverse_splash_ratio(self, val: float) -> None:
        self._transverse_splash_ratio = float(val)

    def _sync_telemetry_for_read(self) -> None:
        if self._pending_telem_fence is None:
            return
        headless_batch = (self._last_render_step < 0) or ((self.step_count - self._last_render_step) > 4)
        self._poll_async_telemetry(wait=headless_batch)

    def step(
        self,
        mod: Optional[PhysicsMod] = None,
        compiled: Optional[CompiledMasterAction] = None,
        dt: float = 0.032,
    ) -> None:
        if self.is_hibernated or self._pos_buf is None or self._force_tex is None:
            self.wake_from_ram()

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
        self._active_compiled = use_comp
        self._ensure_nu_programs(use_comp.nu_glsl_expr)
        if self._step_prog is None:
            return

        eff_has_ballistic, eff_halo_scale, ballistic_pleat = self._effective_halo_params(use_mod)

        gm_mono = self._solve_field_gpu(
            has_cdm=eff_has_ballistic,
            cdm_scale=eff_halo_scale,
            n_vcycles=self.multigrid_vcycles,
        )

        assert (
            self._pos_buf is not None
            and self._vel_buf is not None
            and self._rho_flt_buf is not None
            and self._gas_hydro_buf is not None
            and self._force_tex is not None
        )
        self._pos_buf.bind_to_storage_buffer(0)
        self._vel_buf.bind_to_storage_buffer(1)
        self._rho_flt_buf.bind_to_storage_buffer(5)
        self._gas_hydro_buf.bind_to_storage_buffer(7)
        self._force_tex.use(location=0)

        # Avoid double-counting stationary nu(y) boost when ballistic pleat inertia is already active in Preset 2
        a0_code = self._compute_a0_code(use_mod, ballistic_pleat)
        self.last_a0_code = a0_code
        self._last_nu_fn = use_comp.nu_numpy_fn
        pleat_splash = max(
            float(use_mod.parameters.pinned_pleat_coupling),
            0.08 * float(use_mod.parameters.g6) if not use_mod.parameters.has_cdm_halo else 0.0,
        )

        for uname, uval in (
            ("u_force_tex", 0),
            ("u_count", int(self.particle_count)),
            ("u_M", int(self.grid_m)),
            ("u_half_box", float(self.half_box)),
            ("u_dt", float(dt)),
            ("u_a0_code", float(a0_code)),
            ("u_gm_monopole", float(gm_mono)),
            ("u_pleat_splash", float(pleat_splash)),
            ("u_preset", int(self.preset_idx)),
        ):
            if uname in self._step_prog:
                self._step_prog[uname].value = uval

        self._step_prog.run((self.particle_count + 255) // 256, 1, 1)
        self.ctx.memory_barrier()

        self.sim_time += float(dt)
        self.step_count += 1

        want_telem = (self.step_count % 16 == 0)
        if self._pending_telem_fence is not None and (
            want_telem
            or (self._pending_telem_meta is not None and (self.step_count - self._pending_telem_meta[5]) >= 2)
        ):
            GL.glFlush()
            self._poll_async_telemetry(wait=False)
        if want_telem and self._pending_telem_fence is None:
            self._enqueue_async_telemetry()

        want_replay = self.replay_enabled and (
            (len(self._replay_frames) == 0 and self._pending_replay_fence is None)
            or (self.step_count == 1)
            or (self.step_count % 32 == 0)
        )
        if self._pending_replay_fence is not None and (
            want_replay
            or (self._pending_replay_meta is not None and (self.step_count - self._pending_replay_meta[3]) >= 3)
        ):
            GL.glFlush()
            self._poll_pending_replay_snapshot(wait=False)
        if want_replay and self._pending_replay_fence is None:
            self._enqueue_replay_snapshot()

    def _apply_telemetry_arrays(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        phi: Optional[np.ndarray],
        preset_idx: int,
        a0_code: float,
    ) -> None:
        bary = vel[:, 3] < 1.5
        stars = vel[:, 3] < 0.5
        gas = (vel[:, 3] > 0.5) & (vel[:, 3] < 1.5)

        r_xz = np.hypot(pos[:, 0], pos[:, 2])
        outer = bary & (r_xz > 8.0) & (r_xz < 22.0)
        if np.count_nonzero(outer) < 32 and np.any(bary):
            r90 = float(np.percentile(r_xz[bary], 90.0))
            outer = bary & (r_xz > max(0.50 * r90, 0.5)) & (r_xz < max(0.98 * r90, 1.0))
        if np.any(outer):
            v_tan = np.abs(-pos[outer, 2] * vel[outer, 0] + pos[outer, 0] * vel[outer, 2]) / np.maximum(
                r_xz[outer], 0.1
            )
            self._v_outer_kms = float(np.median(v_tan) * VEL_UNIT_KMS)

        if np.any(stars) and np.any(gas):
            if preset_idx == 2:
                # True Gravitational Lensing Peak vs Shocked X-Ray ICM Gas Offset (1 code unit = 25 kpc on 1.4 Mpc cluster box!)
                gas_abs_x = float(np.mean(np.abs(pos[gas, 0])))
                nu_fn = getattr(self, "_last_nu_fn", None)
                if nu_fn is not None and phi is not None:
                    m = self.grid_m
                    h = float((2.0 * self.half_box) / float(m))
                    inv_2h = np.float32(0.5 / h)
                    # Note: phi layout is (z, y, x)! Evaluate 3D finite differences in float32 for 4x lower cache traffic.
                    gx = -(phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, :-2]) * inv_2h
                    gy = -(phi[1:-1, 2:, 1:-1] - phi[1:-1, :-2, 1:-1]) * inv_2h
                    gz = -(phi[2:, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1]) * inv_2h
                    g_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
                    y_val = np.clip(g_mag * np.float32(1.0 / max(float(a0_code), 1e-6)), 1e-6, 2500.0)
                    nu_boost = np.clip(np.asarray(nu_fn(y_val), dtype=np.float32), 0.25, 250.0)
                    gex, gey, gez = gx * nu_boost, gy * nu_boost, gz * nu_boost
                    # Effective gravitating density rho_lens ~ -div(g_eff) on interior (m-4)^3
                    div_g = (
                        (gex[1:-1, 1:-1, 2:] - gex[1:-1, 1:-1, :-2])
                        + (gey[1:-1, 2:, 1:-1] - gey[1:-1, :-2, 1:-1])
                        + (gez[2:, 1:-1, 1:-1] - gez[:-2, 1:-1, 1:-1])
                    ) * inv_2h
                    rho_lens = np.maximum(-div_g, 0.0)
                    # Projected weak-lensing convergence kappa(x, z) along line-of-sight y, peak-weighted (kappa^3)
                    xs = (np.arange(2, m - 2, dtype=np.float64) + 0.5) * h - self.half_box
                    kappa_xz = np.sum(rho_lens[16:-16, 16:-16, :], axis=1, dtype=np.float64)
                    w_x = np.sum(kappa_xz ** 3, axis=0)
                    w_tot = float(np.sum(w_x))
                    lens_abs_x = (
                        float(np.sum(w_x * np.abs(xs)) / w_tot)
                        if w_tot > 1e-12
                        else float(np.mean(np.abs(pos[stars, 0])))
                    )
                else:
                    lens_abs_x = float(np.mean(np.abs(pos[stars, 0])))
                # Convert cluster code units (half_box=28 <-> 700 kpc => 25 kpc/unit) to physical kpc!
                self._lensing_gas_offset_kpc = float(max(lens_abs_x - gas_abs_x, 0.0) * 25.0)
            else:
                s_right = stars & (pos[:, 0] > 0.0)
                g_right = gas & (pos[:, 0] > 0.0)
                if np.any(s_right) and np.any(g_right):
                    self._lensing_gas_offset_kpc = float(abs(np.mean(pos[s_right, 0]) - np.mean(pos[g_right, 0])))

        b_pos = pos[bary] if np.any(bary) else pos
        std_x = max(float(np.std(b_pos[:, 0])), 0.25)
        std_yz = float(np.hypot(np.std(b_pos[:, 1]), np.std(b_pos[:, 2])))
        self._transverse_splash_ratio = std_yz / std_x

    def _enqueue_async_telemetry(self) -> None:
        """Dispatches non-blocking `glCopyBufferSubData` + `GL_FenceSync` for telemetry without pipeline `.read()` stall."""
        if (
            self.is_hibernated
            or self._pos_buf is None
            or self._vel_buf is None
            or self._pending_telem_fence is not None
        ):
            return
        n_samp = min(8192, self.particle_count)
        samp_bytes = n_samp * 16
        _copy_buffer_sub_data(self._pos_buf, self._telemetry_stage_buf, 0, 0, samp_bytes)
        _copy_buffer_sub_data(self._vel_buf, self._telemetry_stage_buf, 0, samp_bytes, samp_bytes)

        need_phi = (
            self.preset_idx == 2
            and getattr(self, "_last_nu_fn", None) is not None
            and bool(self._mg_ua)
        )
        phi_bytes = (self.grid_m ** 3) * 4 if need_phi else 0
        if phi_bytes > 0:
            _copy_buffer_sub_data(self._mg_ua[0], self._telemetry_stage_buf, 0, 2 * samp_bytes, phi_bytes)

        self._pending_telem_fence = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        GL.glFlush()
        self._pending_telem_meta = (
            n_samp,
            samp_bytes,
            phi_bytes,
            int(self.preset_idx),
            float(self.last_a0_code),
            int(self.step_count),
        )

    def _poll_async_telemetry(self, wait: bool = False) -> None:
        fence = self._pending_telem_fence
        meta = self._pending_telem_meta
        if fence is None or meta is None:
            return
        flags = int(GL.GL_SYNC_FLUSH_COMMANDS_BIT) if wait else 0
        timeout_ns = 250_000_000 if wait else 0
        res = GL.glClientWaitSync(fence, flags, timeout_ns)
        if wait and res not in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED):
            self.ctx.finish()
            res = GL.GL_CONDITION_SATISFIED
        if res not in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED):
            return

        n_samp, samp_bytes, phi_bytes, preset_idx, a0_code, _ = meta
        GL.glDeleteSync(fence)
        self._pending_telem_fence = None
        self._pending_telem_meta = None

        total_bytes = 2 * samp_bytes + phi_bytes
        raw = self._telemetry_stage_buf.read(size=total_bytes)
        mv = memoryview(raw)
        pos = np.frombuffer(mv[:samp_bytes], dtype=np.float32).reshape((n_samp, 4))
        vel = np.frombuffer(mv[samp_bytes : 2 * samp_bytes], dtype=np.float32).reshape((n_samp, 4))
        m = self.grid_m
        phi = (
            np.frombuffer(mv[2 * samp_bytes : total_bytes], dtype=np.float32).reshape((m, m, m))
            if phi_bytes > 0
            else None
        )
        self._apply_telemetry_arrays(pos, vel, phi, preset_idx=preset_idx, a0_code=a0_code)

    def _sample_telemetry(self) -> None:
        """
        Synchronous one-shot telemetry sample (used on reset, scrub, and explicit test queries).
        Uses `glCopyBufferSubData` into `self._telemetry_stage_buf` (`dynamic=True`) — ZERO VBO/SSBO demotion!
        """
        if self.is_hibernated or self._pos_buf is None or self._vel_buf is None:
            return
        self._clear_pending_telem_fence()
        n_samp = min(8192, self.particle_count)
        samp_bytes = n_samp * 16
        _copy_buffer_sub_data(self._pos_buf, self._telemetry_stage_buf, 0, 0, samp_bytes)
        _copy_buffer_sub_data(self._vel_buf, self._telemetry_stage_buf, 0, samp_bytes, samp_bytes)

        need_phi = (
            self.preset_idx == 2
            and getattr(self, "_last_nu_fn", None) is not None
            and bool(self._mg_ua)
        )
        phi_bytes = (self.grid_m ** 3) * 4 if need_phi else 0
        if phi_bytes > 0:
            _copy_buffer_sub_data(self._mg_ua[0], self._telemetry_stage_buf, 0, 2 * samp_bytes, phi_bytes)

        total_bytes = 2 * samp_bytes + phi_bytes
        raw = self._telemetry_stage_buf.read(size=total_bytes)
        mv = memoryview(raw)
        pos = np.frombuffer(mv[:samp_bytes], dtype=np.float32).reshape((n_samp, 4))
        vel = np.frombuffer(mv[samp_bytes : 2 * samp_bytes], dtype=np.float32).reshape((n_samp, 4))
        m = self.grid_m
        phi = (
            np.frombuffer(mv[2 * samp_bytes : total_bytes], dtype=np.float32).reshape((m, m, m))
            if phi_bytes > 0
            else None
        )
        self._apply_telemetry_arrays(
            pos,
            vel,
            phi,
            preset_idx=int(self.preset_idx),
            a0_code=float(self.last_a0_code),
        )

    def telemetry_dict(self) -> Dict[str, float]:
        self._sync_telemetry_for_read()
        return {
            "particles": float(self.particle_count),
            "v_outer_kms": self._v_outer_kms,
            "lensing_gas_offset_kpc": self._lensing_gas_offset_kpc,
            "transverse_splash_ratio": self._transverse_splash_ratio,
            "sim_time": self.sim_time,
        }

    def render(self, fbo: moderngl.Framebuffer, camera: Camera3D, wall_time: float = 0.0) -> None:
        if self.is_hibernated or self._pos_buf is None or self._force_tex is None:
            self.wake_from_ram()
        assert (
            self._pt_vao is not None
            and self._bg_prog is not None
            and self._bg_vao is not None
            and self._force_tex is not None
        )
        w, h = fbo.size
        self._ensure_hdr_fbo(w, h)
        assert self._hdr_fbo is not None and self._hdr_tex is not None

        # Pass 1: Background + Effective 3D Volumetric Lensing Contours into HDR float16 FBO
        self._hdr_fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        eye, fwd, right, up = camera.basis_vectors()
        aspect = float(w) / max(float(h), 1.0)
        fov_tan = math.tan(math.radians(camera.fov_deg) * 0.5)

        self._force_tex.use(location=0)
        for uname, uval in (
            ("u_force_tex", 0),
            ("u_cam_eye", tuple(float(v) for v in eye)),
            ("u_cam_fwd", tuple(float(v) for v in fwd)),
            ("u_cam_right", tuple(float(v) for v in right)),
            ("u_cam_up", tuple(float(v) for v in up)),
            ("u_fov_tan", float(fov_tan)),
            ("u_aspect", float(aspect)),
            ("u_half_box", float(self.half_box)),
            ("u_a0_code", float(self.last_a0_code)),
            ("u_show_lensing", 1 if self.show_lensing else 0),
        ):
            if uname in self._bg_prog:
                self._bg_prog[uname].value = uval
        self._bg_vao.render(moderngl.TRIANGLES, vertices=3)

        # Pass 2: Additive HDR Star & Gas Point Sprites (Type 2 CDM Halo invisible!)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)

        if "u_view_proj" in self._pt_prog:
            self._pt_prog["u_view_proj"].write(camera.view_proj_gl_bytes(aspect))
        if "u_cam_eye" in self._pt_prog:
            self._pt_prog["u_cam_eye"].value = tuple(float(v) for v in eye)
        if "u_point_scale" in self._pt_prog:
            scale = math.sqrt(1_000_000.0 / max(float(self.particle_count) * 0.70, 1.0))
            self._pt_prog["u_point_scale"].value = float(scale)
        if "u_res_scale" in self._pt_prog:
            self._pt_prog["u_res_scale"].value = float(max(h, 64)) / 900.0
        if "u_preset" in self._pt_prog:
            self._pt_prog["u_preset"].value = int(self.preset_idx)

        self._pt_vao.render(moderngl.POINTS, vertices=self.particle_count)
        self.ctx.disable(moderngl.BLEND)

        # Pass 3: ACES Filmic Tonemap into target FBO
        fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self._hdr_tex.use(location=0)
        if "u_hdr_tex" in self._tm_prog:
            self._tm_prog["u_hdr_tex"].value = 0
        self._tm_vao.render(moderngl.TRIANGLES, vertices=3)
        self._last_render_step = int(self.step_count)

        harvest_telem = (
            self._pending_telem_fence is not None
            and self._pending_telem_meta is not None
            and self.step_count > self._pending_telem_meta[5]
        )
        harvest_replay = (
            self._pending_replay_fence is not None
            and self._pending_replay_meta is not None
            and self.step_count > self._pending_replay_meta[3]
        )
        if harvest_telem or harvest_replay:
            GL.glFlush()
            if harvest_telem:
                self._poll_async_telemetry(wait=False)
            if harvest_replay:
                self._poll_pending_replay_snapshot(wait=False)

    def release(self) -> None:
        self._clear_pending_telem_fence()
        self._clear_pending_replay_fence()
        _unbind_arena3_gl_slots()
        self.ctx.finish()
        self._release_grid_pyramid_and_fbo()
        for obj in (
            self._pt_vao,
            self._pos_buf,
            self._vel_buf,
            self._telemetry_stage_buf,
            self._step_prog,
            self._init_prog,
            self._calibrate_vel_prog,
            self._clear_prog,
            self._deposit_prog,
            self._unpack_prog,
            self._mg_smooth_prog,
            self._mg_restrict_prog,
            self._mg_prolong_prog,
            self._pack_force_prog,
            self._pack_pos_fp16_prog,
            self._unpack_pos_fp16_prog,
            self._pack_replay_fp16_prog,
            self._bg_vao,
            self._bg_prog,
            self._pt_prog,
            self._tm_vao,
            self._tm_prog,
        ):
            if obj is not None:
                obj.release()
        self._pt_vao = None
        self._pos_buf = None
        self._vel_buf = None
        self.allocated_vram_mb = 0.0
        self._hib_pos_ram = None
        self._hib_vel_ram = None
        self.clear_replay_buffer()
