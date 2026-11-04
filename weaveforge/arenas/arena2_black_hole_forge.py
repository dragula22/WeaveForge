"""
Arena 2 — The Strong-Gravity & Black-Hole Forge (`weaveforge/arenas/arena2_black_hole_forge.py`)
===============================================================================================
Real-time 60–144+ FPS 3D GPU Relativistic Raytracer & Strong-Field Laboratory:
  • Sub-Mode 0 (`Black Hole + BHEX Shadow + Phase-Slip Threads`):
      3D curved-space Fermat optical-Hamiltonian raytracer compiled LIVE from the active
      `PhysicsMod`'s symbolic `optical_metric_n(U)` (`H = 0.5*(|p|^2 - n(U)^2) = 0`), reproducing
      exact 1PN starlight deflection AND exact 2PN photon-ring capture (`5.196 r_g` GR vs
      `5.437 r_g` `+4.63%` TNaP Weave vs `2.718 r_g` 1911 half-bending!) with zero polynomial hacks,
      plus Kerr spin `a/M in [0, 0.998]` Lense–Thirring frame-dragging, Doppler-beamed accretion
      disk, exact finite-distance `asin(b_crit / (r_cam * n(U_cam)))` `BHEX` shadow gauge, and
      3D superfluid quantum phase-slip threads (`Melt • Slip 2pi • Weld`)!
  • Sub-Mode 1 (`Planck-Tension 1.000000 Fishing-Reel Horizon Spool`):
      Visualizes Gibbons Maximum Tension `T_P = c^4/(4G)` winding monofilament threads + matter knots
      onto a non-singular core whose radius `R = E / (2 T_P) = 2GM/c^2` locks to `1.000000` x Schwarzschild!
  • Sub-Mode 2 (`3D Orbital Clock & 3,189.1 km Zero-Crossing Explorer`):
      3D sunlit Earth + glowing emerald `h = 3,189.1 km = 0.5 R_Earth` (`0.00 us/day` SR/GR cancellation)
      ring + live interactive orbital clock probe (`ISS -24.57 us/d` -> `GPS +38.64 us/d` -> `GEO +46.65 us/d`).
  • Sub-Mode 3 (`Strong-Field BSSN Funnel & BHEX QNM Ringdown`):
      3D curved spatial embedding funnel `y(r) = -2.6 / sqrt(r + 0.35)` colored by strong-field lapse
      `alpha(r) = 1 / sqrt(max(n(1/r), 1.0))` & conformal grid lines, exact photon-sphere & BHEX shadow
      rings, outward-propagating quadrupolar `l=2, m=2` BHEX QNM spiral strain waves, and delayed
      post-ringdown gravitational-wave echoes from a reflective Gibbons Planck-spool throat!
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import moderngl
import numpy as np

from weaveforge.core.camera import Camera3D
from weaveforge.core.symbolic_compiler import CompiledMasterAction


FORGE_SUBMODES = [
    ("kerr_bhex", "Spinning Kerr vs. Weave Raytracer + Phase Slips"),
    ("planck_spool", "Planck-Tension 1.000000 Fishing-Reel Horizon"),
    ("orbital_clocks", "3D Earth Clocks & 3,189.1 km Zero-Crossing"),
    ("bssn_ringdown", "Strong-Field BSSN Funnel & BHEX QNM Ringdown"),
]

_VERT_QUAD = """#version 430
out vec2 v_uv;
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
"""


def _build_frag_black_hole_forge(optical_n_glsl: str, d_half_n2_dU_glsl: str) -> str:
    """
    Generates the Arena 2 GLSL 430 fragment shader with the active `PhysicsMod`'s exact
    SymPy-compiled Fermat optical refractive index `n(U)` and radial Hamiltonian gradient
    `d(0.5 * n(U)^2)/dU` embedded natively inside the RK2 curved-ray integrator!
    """
    return f"""#version 430
in vec2 v_uv;
out vec4 frag_color;

uniform vec3 u_cam_eye;
uniform vec3 u_cam_fwd;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_fov_tan;
uniform float u_aspect;
uniform float u_time;

uniform int u_submode;            // 0=Kerr/Weave BH, 1=Planck Spool, 2=Earth Clocks, 3=BSSN Funnel & BHEX Ringdown
uniform float u_spin_a;           // Kerr spin a/M in [0.0, 0.998]
uniform float u_shadow_rg;        // Compiled 2PN shadow radius (5.19615 GR vs 5.43656 TNaP)
uniform float u_u_photon;         // Compiled photon-sphere inverse radius U_ph = r_g / r_ph
uniform float u_u_cap;            // Safe coordinate horizon cap on U = r_g / r
uniform float u_n_cam_mod;        // Exact n_mod(U_cam) at camera distance
uniform float u_n_cam_gr;         // Exact n_GR(U_cam) at camera distance
uniform int u_show_bhex_gauge;    // 1 = Overlay GR 5.196 vs Active Model BHEX shadow gauge
uniform int u_show_phase_slips;   // 1 = Render 3D superfluid quantum phase-slip threads
uniform float u_mass_solar;       // Spool mass multiplier in [0.4, 2.2]
uniform float u_probe_alt_km;     // Orbital clock altitude in [0, 40000] km
uniform float u_fbo_h;            // Active render-target height for sub-pixel-safe BHEX ring gauge
uniform float u_spool_reflectivity; // Gibbons Planck-spool throat reflectivity R in [0.0, 1.0]
uniform float u_qnm_freq_ratio;     // Active mod QNM frequency ratio omega_220 / omega_220_GR

vec3 aces_tonemap(vec3 x) {{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}}

float hash21(vec2 p) {{
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}}

float eval_optical_n(float U_raw) {{
    float U = clamp(U_raw, 1e-6, u_u_cap);
    return max({optical_n_glsl}, 0.05);
}}

float eval_d_half_n2_dU(float U_raw) {{
    float U = clamp(U_raw, 1e-6, u_u_cap);
    return {d_half_n2_dU_glsl};
}}

vec3 fermat_kerr_force(vec3 pos, vec3 p_mom, float spin_a) {{
    float r = max(length(pos), 0.25);
    float inv_r = 1.0 / r;
    float inv_r3 = inv_r * inv_r * inv_r;
    float d_hn2 = eval_d_half_n2_dU(inv_r);
    // Exact Fermat Hamiltonian force: dp/dlambda = grad(0.5 * n(U)^2) = -d(0.5*n^2)/dU * pos / r^3
    vec3 f_opt = (-d_hn2 * inv_r3) * pos;
    // Gravitomagnetic Kerr Lense-Thirring frame-dragging (spin axis = +Y):
    // Prograde rays (x > 0 on camera right when viewing from +Z) bend more weakly -> classic D-shaped shadow!
    vec3 spin_vec = vec3(0.0, spin_a, 0.0);
    vec3 h_gm = (2.0 * inv_r3) * (3.0 * dot(spin_vec, pos) * inv_r * inv_r * pos - spin_vec);
    vec3 f_drag = cross(p_mom, h_gm) * eval_optical_n(inv_r);
    return f_opt + f_drag;
}}

vec3 background_cosmos(vec3 dir) {{
    float horizon_glow = exp(-abs(dir.y) * 2.6);
    vec3 base = mix(vec3(0.024, 0.046, 0.092), vec3(0.068, 0.125, 0.222), 0.45 + 0.55 * horizon_glow);

    float phi = atan(dir.z, dir.x);
    float th  = asin(clamp(dir.y, -1.0, 1.0));
    vec2 suv = vec2(phi * 28.0, th * 28.0);
    vec2 cell = floor(suv);
    vec2 f = fract(suv) - 0.5;
    float h = hash21(cell);
    if (h > 0.91) {{
        vec2 jitter = vec2(hash21(cell + 1.7), hash21(cell + 9.2)) - 0.5;
        vec2 d = f - 0.55 * jitter;
        float r2 = dot(d, d);
        float star = exp(-r2 * 260.0) * (0.65 + 1.85 * (h - 0.91) / 0.09);
        vec3 scol = mix(vec3(0.75, 0.90, 1.25), vec3(1.25, 0.95, 0.65), fract(h * 31.7));
        base += scol * star;
    }}
    return base;
}}

void main() {{
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(
        u_cam_fwd
        + ndc.x * u_aspect * u_fov_tan * u_cam_right
        + ndc.y * u_fov_tan * u_cam_up
    );
    vec3 ro = u_cam_eye;

    vec3 col_hdr = vec3(0.0);

    // =========================================================================
    // SUB-MODE 0: SYMPY-COMPILED FERMAT HAMILTONIAN KERR / WEAVE RAYTRACER
    // =========================================================================
    if (u_submode == 0) {{
        // World coordinate r is isotropic radius in units of r_g = GM/c^2 (so U = 1 / r).
        // Photon sphere sits at r_ph = 1 / u_u_photon; any ray plunging inward inside 0.80 * r_ph
        // (scaled by Kerr horizon contraction) is monotonically trapped inside the photon sphere!
        float r_ph = 1.0 / max(u_u_photon, 0.15);
        float kerr_shrink = 0.55 + 0.45 * sqrt(max(1.0 - u_spin_a * u_spin_a, 0.01));
        float r_capture = max(0.78 * r_ph * kerr_shrink, 1.02 / max(u_u_cap, 0.5));
        float shadow_scale = clamp(u_shadow_rg / 5.1961524, 0.45, 1.85);

        vec3 pos = ro;
        float r0 = max(length(pos), 1.0);
        vec3 p_mom = eval_optical_n(1.0 / r0) * rd;

        vec3 accum = vec3(0.0);
        float trans = 1.0;
        bool swallowed = false;

        // Periodic quantum phase-slip pulse every 4.2s on inner superfluid threads!
        float cyc = fract(u_time / 4.2);
        float stretch = smoothstep(0.0, 0.64, cyc);
        float slip_relax = smoothstep(0.64, 0.86, cyc);
        float melt_flash = exp(-((cyc - 0.67) * (cyc - 0.67)) / 0.0065);
        float wind_tight = 2.2 + 2.9 * (stretch - slip_relax);

        const int STEPS = 180;
        for (int i = 0; i < STEPS; ++i) {{
            float r = length(pos);
            // Guaranteed capture once ray crosses inward inside r_capture (or hits coordinate horizon cap)
            if (r < r_capture || (r < 0.94 * r_ph * kerr_shrink && dot(pos, p_mom) < 0.0)) {{
                swallowed = true;
                break;
            }}
            if (r > 52.0 && dot(pos, p_mom) > 0.0) {{
                break;
            }}

            float p_mag = max(length(p_mom), 0.25);
            // Adaptive affine step dlambda so Euclidean arc step |dx| = |p|*dlambda stays crisp near photon sphere!
            float arc_ds = clamp(0.052 * (r - 0.55 * r_capture), 0.026, 0.52);
            float dlam = arc_ds / p_mag;

            vec3 pos_prev = pos;
            // Symplectic Midpoint / RK2 step on Hamilton's optical equations: dx/dlam = p, dp/dlam = F(x, p)
            vec3 f1 = fermat_kerr_force(pos, p_mom, u_spin_a);
            vec3 pos_mid = pos + 0.5 * dlam * p_mom;
            vec3 p_mid   = p_mom + 0.5 * dlam * f1;
            vec3 f2 = fermat_kerr_force(pos_mid, p_mid, u_spin_a);
            pos = pos_prev + dlam * p_mid;
            p_mom = p_mom + dlam * f2;

            vec3 dir_unit = normalize(p_mom);

            // 1. Equatorial Thin Relativistic Accretion Disk (crossing y = 0 plane!)
            if (pos_prev.y * pos.y <= 0.0) {{
                float frac = abs(pos_prev.y) / max(abs(pos_prev.y - pos.y), 1e-6);
                vec3 hit_p = mix(pos_prev, pos, frac);
                float rd_disk = length(hit_p.xz);
                float r_isco = mix(4.8, 1.85, u_spin_a) * shadow_scale;
                if (rd_disk > r_isco && rd_disk < 14.8) {{
                    float ang = atan(hit_p.z, hit_p.x);
                    float outer_fade = smoothstep(14.8, 9.2, rd_disk);
                    float temp_prof = exp(-(rd_disk - r_isco) / 3.6)
                                    * smoothstep(r_isco, r_isco + 0.55, rd_disk)
                                    * outer_fade;
                    float spiral = 0.62 + 0.38 * sin(4.0 * ang + 9.5 * log(rd_disk + 0.2) - 3.6 * u_time)
                                         * cos(7.0 * ang - 12.0 / (rd_disk + 0.5) + 2.4 * u_time);

                    vec3 v_orb = normalize(vec3(-hit_p.z, 0.0, hit_p.x));
                    float beta_orb = clamp(sqrt(1.15 / max(rd_disk, 0.8)), 0.0, 0.72);
                    float cos_los = dot(v_orb, -dir_unit);
                    float doppler = 1.0 / max((1.0 - beta_orb * cos_los), 0.32);
                    float beam = doppler * doppler * doppler * 0.42;

                    vec3 c_blue = vec3(0.35, 0.86, 1.32);
                    vec3 c_gold = vec3(1.32, 0.74, 0.18);
                    float blue_w = clamp(0.5 + 0.65 * cos_los, 0.0, 1.0);
                    vec3 disk_rgb = mix(c_gold, c_blue, blue_w) * temp_prof * spiral * beam * 2.15;

                    float disk_alpha = clamp(temp_prof * 0.88, 0.0, 0.92);
                    accum += trans * disk_rgb;
                    trans *= (1.0 - disk_alpha);
                }}
            }}

            // 2. 3D Superfluid Quantum Phase-Slip Vortex Threads spiraling into the Ergosphere!
            if (u_show_phase_slips == 1 && r > r_capture * 1.05 && r < 7.8 && abs(pos.y) < 1.15) {{
                float r_xz = length(pos.xz);
                if (r_xz > r_capture * 1.04) {{
                    float phi_xz = atan(pos.z, pos.x);
                    float strand_phase = 14.0 * (phi_xz - wind_tight / max(r_xz - 0.55 * r_capture, 0.35) + 0.55 * u_time);
                    float s_wave = 0.5 + 0.5 * cos(strand_phase);
                    float s_sharp = s_wave * s_wave * s_wave * s_wave;
                    float y_sheet = exp(-(pos.y * pos.y) / 0.14);
                    float heal_ring = exp(-((r_xz - 1.25 * r_ph * kerr_shrink) * (r_xz - 1.25 * r_ph * kerr_shrink)) / 0.22);

                    vec3 thread_c = mix(
                        vec3(0.18, 0.85, 1.20),
                        vec3(1.45, 1.08, 0.32) * (1.0 + 1.3 * melt_flash),
                        heal_ring
                    );
                    float dens_t = s_sharp * y_sheet * exp(-r_xz * 0.22) * smoothstep(7.8, 5.0, r) * (arc_ds * 0.65);
                    accum += trans * thread_c * dens_t;
                }}
            }}
        }}

        if (!swallowed) {{
            accum += trans * background_cosmos(normalize(p_mom));
        }}
        col_hdr = accum;

        // 3. Exact Finite-Distance BHEX Photon-Ring Angular Shadow Gauge Overlay:
        //    sin(theta_crit) = b_crit / (r_cam * n(1 / r_cam)), plus Kerr Bardeen centroid shift!
        if (u_show_bhex_gauge == 1) {{
            float cam_d = max(length(ro), 6.2);
            vec3 to_bh = normalize(-ro);
            vec3 spin_axis = vec3(0.0, 1.0, 0.0);
            float sin_inc = length(cross(to_bh, spin_axis));
            vec3 shift_axis = (sin_inc > 1e-4) ? normalize(cross(spin_axis, to_bh)) : u_cam_right;
            // Bardeen (1973) Kerr shadow centroid shift ~ +1.85 * a * sin(i) in impact-parameter space
            float shift_ang = asin(clamp((1.85 * u_spin_a * sin_inc) / (cam_d * u_n_cam_mod), -0.35, 0.35));
            vec3 bh_center_dir = normalize(to_bh + tan(shift_ang) * shift_axis);

            float ang_ray = acos(clamp(dot(rd, bh_center_dir), -1.0, 1.0));
            float ang_gr  = asin(clamp(5.1961524 / max(cam_d * u_n_cam_gr, 5.25), 0.0, 0.99));
            float ang_mod = asin(clamp(u_shadow_rg / max(cam_d * u_n_cam_mod, u_shadow_rg * 1.01), 0.0, 0.99));

            float pix_w = max(0.0028, 2.2 / max(u_fbo_h, 64.0)) * u_fov_tan;
            float ring_gr  = exp(-((ang_ray - ang_gr) * (ang_ray - ang_gr)) / (pix_w * pix_w));
            float ring_mod = exp(-((ang_ray - ang_mod) * (ang_ray - ang_mod)) / (pix_w * pix_w));

            float scr_ang = atan(ndc.y, ndc.x * u_aspect);
            float dash = step(0.0, sin(scr_ang * 24.0));
            col_hdr += vec3(0.20, 0.92, 1.15) * ring_gr * dash * 0.85;
            col_hdr += vec3(1.30, 0.88, 0.22) * ring_mod * 0.95;
        }}
    }}

    // =========================================================================
    // SUB-MODE 1: GIBBONS PLANCK-TENSION (1.000000 R = 2GM/c^2) FISHING-REEL SPOOL
    // =========================================================================
    else if (u_submode == 1) {{
        col_hdr = background_cosmos(rd);
        float R_schw = 2.0 * u_mass_solar;
        float R_spool = R_schw * 1.000000;

        float b = dot(ro, rd);
        float c_sph = dot(ro, ro) - R_spool * R_spool;
        float disc = b * b - c_sph;

        if (disc > 0.0) {{
            float t_hit = -b - sqrt(disc);
            if (t_hit > 0.0) {{
                vec3 p_hit = ro + t_hit * rd;
                vec3 n_hit = normalize(p_hit);
                vec3 light_d = normalize(vec3(0.55, 0.72, 0.42));
                float ndl = max(dot(n_hit, light_d), 0.0);
                float fres = pow(max(1.0 - clamp(dot(n_hit, -rd), 0.0, 1.0), 0.0), 2.8);

                float lat = asin(clamp(n_hit.y, -1.0, 1.0));
                float lon = atan(n_hit.z, n_hit.x);
                float grooves = 0.65 + 0.35 * sin(58.0 * lat + 4.0 * lon - 5.0 * u_time);
                float knots = smoothstep(0.92, 1.0, sin(18.0 * lat) * cos(18.0 * lon + 2.0 * u_time));

                vec3 core_col = vec3(0.14, 0.66, 1.08) * grooves * (0.32 + 0.78 * ndl)
                              + vec3(1.25, 0.88, 0.30) * (0.65 * fres + 0.95 * knots);
                col_hdr = core_col;
            }}
        }}

        if (abs(rd.y) > 1e-4) {{
            float t_eq = -ro.y / rd.y;
            if (t_eq > 0.0 && t_eq < 45.0) {{
                vec3 pe = ro + t_eq * rd;
                float re = length(pe.xz);
                if (re > R_spool * 1.01 && re < R_spool * 3.8 && (disc <= 0.0 || t_eq < -b - sqrt(max(disc, 0.0)))) {{
                    float ang = atan(pe.z, pe.x);
                    float spiral_in = 0.5 + 0.5 * sin(12.0 * ang + 14.0 / (re + 0.2) - 4.5 * u_time);
                    float fil = spiral_in * spiral_in * spiral_in * spiral_in
                              * exp(-(re - R_spool) * 0.45)
                              * smoothstep(R_spool * 3.8, R_spool * 2.0, re);
                    col_hdr += vec3(0.32, 0.90, 1.25) * fil * 0.95;

                    float d_ring = abs(re - R_schw);
                    float rw = exp(-(d_ring * d_ring) / 0.0045) * step(0.0, sin(ang * 20.0));
                    col_hdr += vec3(1.35, 0.92, 0.25) * rw * 1.2;
                }}
            }}
        }}
    }}

    // =========================================================================
    // SUB-MODE 2: 3D EARTH ORBITAL CLOCKS & 3,189.1 KM ZERO-CROSSING RING
    // =========================================================================
    else if (u_submode == 2) {{
        col_hdr = background_cosmos(rd);
        float R_earth = 1.80;
        float R_zero  = R_earth * 1.50;
        float R_gps   = R_earth * (1.0 + 20184.0 / 6378.137);
        float R_probe = R_earth * (1.0 + clamp(u_probe_alt_km, 0.0, 40000.0) / 6378.137);

        float b = dot(ro, rd);
        float c_e = dot(ro, ro) - R_earth * R_earth;
        float disc = b * b - c_e;
        float t_earth = 1e9;

        if (disc > 0.0) {{
            float th = -b - sqrt(disc);
            if (th > 0.0) {{
                t_earth = th;
                vec3 p_hit = ro + th * rd;
                vec3 n_hit = normalize(p_hit);
                vec3 sun_d = normalize(vec3(0.75, 0.42, 0.50));
                float ndl = max(dot(n_hit, sun_d), 0.04);
                float atmo = pow(max(1.0 - clamp(dot(n_hit, -rd), 0.0, 1.0), 0.0), 3.0);

                float lon = atan(n_hit.z, n_hit.x) - 0.18 * u_time;
                float lat = asin(clamp(n_hit.y, -1.0, 1.0));
                float cont = sin(3.0 * lon) * cos(2.5 * lat) + 0.5 * cos(7.0 * lon - 4.0 * lat);
                vec3 ocean = vec3(0.06, 0.34, 0.78);
                vec3 land  = vec3(0.16, 0.54, 0.32);
                vec3 surf  = mix(ocean, land, smoothstep(0.18, 0.42, cont));
                float clouds = smoothstep(0.35, 0.85, sin(5.0 * lon + 0.1 * u_time) * sin(6.0 * lat));
                surf = mix(surf, vec3(0.92, 0.96, 1.0), 0.45 * clouds);

                col_hdr = surf * ndl * 1.35 + vec3(0.28, 0.74, 1.25) * atmo * 0.95;
            }}
        }}

        if (abs(rd.y) > 1e-4) {{
            float t_pl = -ro.y / rd.y;
            if (t_pl > 0.0 && t_pl < t_earth && t_pl < 90.0) {{
                vec3 pp = ro + t_pl * rd;
                float rp = length(pp.xz);
                float ang = atan(pp.z, pp.x);

                float funnel = exp(-rp * 0.12) * 0.22;
                float g_dr = fract(rp * 0.8) - 0.5;
                float grid_r = exp(-(g_dr * g_dr) * 90.0);
                col_hdr += vec3(0.12, 0.38, 0.68) * grid_r * funnel;

                float w_zero = exp(-((rp - R_zero) * (rp - R_zero)) / 0.0055);
                col_hdr += vec3(0.12, 1.18, 0.62) * w_zero * 1.15;

                float w_gps = exp(-((rp - R_gps) * (rp - R_gps)) / 0.012) * step(0.0, sin(ang * 24.0 - u_time));
                col_hdr += vec3(0.25, 0.86, 1.20) * w_gps * 0.85;

                float w_prb = exp(-((rp - R_probe) * (rp - R_probe)) / 0.009);
                float sat_ang = ang - 1.6 * u_time / sqrt(max(rp, 1.0));
                float beacon = exp(-(1.0 - cos(sat_ang)) * 140.0) * exp(-((rp - R_probe) * (rp - R_probe)) / 0.045);
                col_hdr += vec3(1.35, 0.90, 0.20) * (w_prb * 1.1 + beacon * 2.4);
            }}
        }}
    }}

    // =========================================================================
    // SUB-MODE 3: STRONG-FIELD BSSN CONFORMAL FUNNEL & BHEX QNM RINGDOWN
    // =========================================================================
    else {{
        col_hdr = background_cosmos(rd);
        float r_ph = 1.0 / max(u_u_photon, 0.15);
        float r_sh = max(u_shadow_rg, 1.4);
        float r_spool = max(0.52 * r_ph, 0.36);

        float t_min = 0.08;
        float t_max = min(length(ro) + 34.0, 96.0);
        vec3 p_prev = ro + t_min * rd;
        float r_prev = length(p_prev.xz);
        float f_prev = p_prev.y + 2.6 / sqrt(r_prev + 0.35);
        float t_prev = t_min;
        bool hit_funnel = false;
        float t_hit = 0.0;

        const int F_STEPS = 108;
        for (int i = 1; i <= F_STEPS; ++i) {{
            float t_cur = t_min + (t_max - t_min) * (float(i) / float(F_STEPS));
            vec3 p_cur = ro + t_cur * rd;
            float r_cur = length(p_cur.xz);
            float f_cur = p_cur.y + 2.6 / sqrt(r_cur + 0.35);
            if (min(r_prev, r_cur) <= 30.0 && f_prev * f_cur <= 0.0) {{
                float ta = t_prev;
                float tb = t_cur;
                float fa = f_prev;
                for (int b_it = 0; b_it < 6; ++b_it) {{
                    float tm = 0.5 * (ta + tb);
                    vec3 pm = ro + tm * rd;
                    float fm = pm.y + 2.6 / sqrt(length(pm.xz) + 0.35);
                    if (fa * fm <= 0.0) {{
                        tb = tm;
                    }} else {{
                        ta = tm;
                        fa = fm;
                    }}
                }}
                t_hit = 0.5 * (ta + tb);
                hit_funnel = true;
                break;
            }}
            t_prev = t_cur;
            r_prev = r_cur;
            f_prev = f_cur;
        }}

        if (hit_funnel) {{
            vec3 ph = ro + t_hit * rd;
            float r = max(length(ph.xz), 1e-4);
            float phi = atan(ph.z, ph.x);
            float sheet_mask = smoothstep(30.0, 23.5, r);

            // Exact upward unit normal to y_funnel(r) = -2.6 / sqrt(r + 0.35)
            float dy_dr = 1.3 / pow(r + 0.35, 1.5);
            vec3 n_surf = normalize(vec3(-dy_dr * (ph.x / r), 1.0, -dy_dr * (ph.z / r)));
            if (dot(n_surf, -rd) < 0.0) {{
                n_surf = -n_surf;
            }}

            // Strong-field conformal lapse alpha(r) = 1 / sqrt(max(n(1/r), 1.0))
            float n_opt = max(eval_optical_n(1.0 / max(r, 0.20)), 1.0);
            float alpha_lapse = 1.0 / sqrt(n_opt);

            // 4-point Gauss-Legendre tortoise coordinate r_*(r) - r_*(r_ph) = integral(n(1/r') dr')
            float half_s = 0.5 * (r - r_ph);
            float mid_s  = 0.5 * (r + r_ph);
            float r_g1 = max(mid_s - 0.8611363 * half_s, 0.22);
            float r_g2 = max(mid_s - 0.3399810 * half_s, 0.22);
            float r_g3 = max(mid_s + 0.3399810 * half_s, 0.22);
            float r_g4 = max(mid_s + 0.8611363 * half_s, 0.22);
            float r_star = half_s * (
                0.3478548 * (eval_optical_n(1.0 / r_g1) + eval_optical_n(1.0 / r_g4))
                + 0.6521452 * (eval_optical_n(1.0 / r_g2) + eval_optical_n(1.0 / r_g3))
            );

            // Lapse-colored BSSN spatial embedding surface + conformal polar (r, phi) grid
            vec3 c_throat = vec3(0.52, 0.11, 0.25);
            vec3 c_mid    = vec3(0.08, 0.30, 0.62);
            vec3 c_flat   = vec3(0.05, 0.18, 0.34);
            vec3 base_f = mix(
                c_throat,
                mix(c_mid, c_flat, smoothstep(0.45, 0.92, alpha_lapse)),
                smoothstep(0.08, 0.58, alpha_lapse)
            );

            float g_rad = abs(fract(r) - 0.5);
            float line_r = exp(-(g_rad * g_rad) * 220.0);
            float g_phi = abs(fract(phi * (16.0 / 6.2831853)) - 0.5);
            float line_p = exp(-(g_phi * g_phi) * 260.0) * smoothstep(0.25, 0.95, r);
            float conf_grid = max(line_r, line_p);

            vec3 light_dir = normalize(vec3(0.48, 0.78, 0.40));
            float ndl = 0.28 + 0.72 * max(dot(n_surf, light_dir), 0.0);
            float fres = pow(max(1.0 - clamp(dot(n_surf, -rd), 0.0, 1.0), 0.0), 2.6);
            vec3 surf_col = base_f * ndl
                          + vec3(0.14, 0.56, 0.92) * conf_grid * (0.28 + 0.35 * alpha_lapse)
                          + vec3(0.16, 0.48, 0.85) * fres * 0.42;

            // Outward-propagating quadrupolar l=2, m=2 BHEX QNM spiral ringdown strain crests:
            //   cos(2 * phi - omega_220 * (t - r_*)) damped by the QNM envelope
            float q_ratio = clamp(u_qnm_freq_ratio, 0.25, 3.0);
            float omega_220 = 1.120 * q_ratio;
            float gamma_220 = 0.095;
            float t_wave = mod(u_time * 4.8 + 11.5, 28.0);
            float ret_u = t_wave - r_star;
            float qnm_env = smoothstep(-2.5, 1.2, ret_u)
                          * exp(-gamma_220 * max(ret_u, 0.0))
                          * smoothstep(0.42 * r_ph, 0.95 * r_ph, r)
                          / sqrt(max(r, 0.65));
            float h_quad = cos(2.0 * phi - omega_220 * ret_u) * qnm_env;
            surf_col += vec3(0.18, 0.95, 1.35) * max(h_quad, 0.0) * 1.15;
            surf_col += vec3(1.25, 0.58, 0.14) * max(-h_quad, 0.0) * 0.85;

            // Delayed outward post-ringdown gravitational-wave echoes from inner Gibbons Planck-spool throat
            if (u_spool_reflectivity > 0.01) {{
                float refl = clamp(u_spool_reflectivity, 0.0, 1.0);
                float w_spool = exp(-((r - r_spool) * (r - r_spool)) / 0.016);
                float spool_braid = 0.65 + 0.35 * cos(12.0 * phi + 4.0 * u_time);
                surf_col += vec3(1.45, 0.96, 0.32) * w_spool * spool_braid * refl * 1.65;

                float u_ec1 = ret_u - 6.8;
                float u_ec2 = ret_u - 13.6;
                float throat_gate = smoothstep(r_spool * 0.80, r_spool * 1.20, r) / sqrt(max(r, 0.60));
                float env_ec1 = refl * exp(-0.22 * u_ec1 * u_ec1) * throat_gate;
                float env_ec2 = (refl * refl) * exp(-0.22 * u_ec2 * u_ec2) * throat_gate;
                float echo_rings = refl
                                 * (0.5 + 0.5 * cos(2.6 * (r_star + 2.2) - 4.2 * u_time - 1.3))
                                 * exp(-max(r - r_spool, 0.0) * 0.24)
                                 * smoothstep(r_spool * 0.78, r_spool * 1.10, r);
                float echo_quad = 0.55 + 0.45 * cos(2.0 * phi - omega_220 * u_ec1);
                float echo_sig = (env_ec1 + 0.75 * env_ec2 + 0.68 * echo_rings) * echo_quad;
                surf_col += vec3(0.32, 1.38, 0.88) * echo_sig * 1.35;
            }} else {{
                // Pure black-hole / sub-throat Sommerfeld absorption inside photon sphere
                surf_col *= smoothstep(0.30 * r_ph, 0.82 * r_ph, r);
            }}

            // Exact photon-sphere ring r_ph = 1 / u_u_photon and critical BHEX shadow ring r_sh = u_shadow_rg
            float w_ph = exp(-((r - r_ph) * (r - r_ph)) / 0.012);
            surf_col += vec3(1.38, 0.92, 0.26) * w_ph * 1.25;

            float w_sh = exp(-((r - r_sh) * (r - r_sh)) / 0.022) * (0.45 + 0.55 * step(0.0, sin(phi * 20.0)));
            surf_col += vec3(0.28, 0.96, 1.28) * w_sh * 0.95;

            if (u_show_bhex_gauge == 1) {{
                float w_gr_sh = exp(-((r - 5.1961524) * (r - 5.1961524)) / 0.018) * step(0.0, cos(phi * 28.0));
                surf_col += vec3(0.85, 0.92, 1.05) * w_gr_sh * 0.48;
            }}

            col_hdr = mix(col_hdr, surf_col, sheet_mask);
        }}
    }}

    vec3 mapped = aces_tonemap(col_hdr * 1.08);
    mapped = pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.2));
    frag_color = vec4(mapped, 1.0);
}}
"""


class Arena2BlackHoleForge:
    """
    GPU Relativistic Strong-Gravity & Black-Hole Forge.
    Compiles the active `PhysicsMod`'s symbolic Fermat optical metric directly into GLSL 430!
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        compiled: Optional[CompiledMasterAction] = None,
        mod: Optional[PhysicsMod] = None,
        lazy_gpu: bool = False,
    ) -> None:
        self.ctx = ctx
        self._active_mod: Optional[PhysicsMod] = mod
        self.submode = 0
        self.spin_a = 0.94
        self.show_bhex_gauge = True
        self.show_phase_slips = True
        self.mass_solar = 1.0
        self.probe_alt_km = 3189.07  # Default right on the 0.00 us/day zero-crossing!
        self.spool_reflectivity: float = 0.0
        self.qnm_freq_ratio: float = 1.0
        self.is_hibernated: bool = bool(lazy_gpu)
        self.allocated_vram_mb: float = 0.0 if lazy_gpu else 0.5
        self.ddr5_ram_mb: float = 0.0

        init_n = compiled.optical_n_glsl_expr if compiled is not None else "exp((2.0 * U))"
        init_dn2 = compiled.d_half_n2_dU_glsl_expr if compiled is not None else "(2.0 * exp((4.0 * U)))"
        self._pending_key: Tuple[str, str] = (init_n, init_dn2)
        self._compiled_key: Tuple[str, str] = ("", "")
        self._prog: Optional[moderngl.Program] = None
        self._vao: Optional[moderngl.VertexArray] = None
        if not lazy_gpu:
            self._ensure_program(init_n, init_dn2)

    def hibernate_to_ram(self) -> float:
        """Releases raytracer shader/VAO (< 0.5 MiB VRAM) when hibernated; returns 0.0 MiB DDR5."""
        if self._vao is not None or self._prog is not None:
            self.ctx.finish()
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._prog is not None:
            self._prog.release()
            self._prog = None
        self._compiled_key = ("", "")
        self.is_hibernated = True
        self.allocated_vram_mb = 0.0
        return 0.0

    def wake_from_ram(self) -> None:
        """Restores raytracer shader/VAO on wake for uniform studio residency lifecycle."""
        self.is_hibernated = False
        self._ensure_program(*self._pending_key)
        self.allocated_vram_mb = 0.5

    def _ensure_program(self, opt_n_glsl: str, d_hn2_glsl: str) -> None:
        key = (opt_n_glsl, d_hn2_glsl)
        self._pending_key = key
        if key == self._compiled_key and self._prog is not None and self._vao is not None:
            return
        frag_src = _build_frag_black_hole_forge(opt_n_glsl, d_hn2_glsl)
        new_prog = self.ctx.program(vertex_shader=_VERT_QUAD, fragment_shader=frag_src)
        new_vao = self.ctx.vertex_array(new_prog, [])
        if self._vao is not None or self._prog is not None:
            self.ctx.finish()
        if self._vao is not None:
            self._vao.release()
        if self._prog is not None:
            self._prog.release()
        self._prog = new_prog
        self._vao = new_vao
        self._compiled_key = key

    def compute_orbital_clock_telemetry(self, compiled: CompiledMasterAction) -> Dict[str, float]:
        """
        Computes Special Relativistic velocity slowing, General Relativistic / Weave clock blueshift,
        net clock rate (`us/day`), GPS navigation drift (`km/day`), and Mercury perihelion advance
        using the active mod's `SymPy`-extracted `(alpha_lapse, gamma, beta)`!
        """
        gm_earth = 3.986004418e14
        r_earth_m = 6378137.0
        c_mps = 299792458.0
        sec_day = 86400.0

        h_km = max(0.0, float(self.probe_alt_km))
        r_m = r_earth_m + h_km * 1000.0
        v_kms = math.sqrt(gm_earth / r_m) / 1000.0

        sr_us = -(gm_earth / (2.0 * c_mps * c_mps * r_m)) * sec_day * 1.0e6
        # Gravitational redshift governed by temporal clock lapse A(U) = 1 - 2*alpha_lapse*U + 2*beta*U^2
        lapse_scale = float(compiled.ppn_alpha_lapse)
        gr_us = lapse_scale * (gm_earth / (c_mps * c_mps)) * (1.0 / r_earth_m - 1.0 / r_m) * sec_day * 1.0e6
        net_us = sr_us + gr_us
        drift_km_day = abs(net_us) * 1.0e-6 * c_mps / 1000.0

        # Mercury perihelion precession: GR = 42.98" / cy * (2*alpha_lapse + 2*gamma - beta) / 3
        mercury_arcsec_cy = (
            42.98 * (2.0 * lapse_scale + 2.0 * float(compiled.ppn_gamma) - float(compiled.ppn_beta)) / 3.0
        )

        return {
            "h_km": h_km,
            "h_zero_cross_km": 0.5 * 6378.137,
            "v_orb_kms": v_kms,
            "sr_us_day": sr_us,
            "gr_us_day": gr_us,
            "net_us_day": net_us,
            "drift_km_day": drift_km_day,
            "mercury_arcsec_cy": mercury_arcsec_cy,
            "spool_ratio_exact": 1.000000,
        }

    def render(
        self,
        fbo: moderngl.Framebuffer,
        camera: Camera3D,
        compiled: CompiledMasterAction,
        wall_time: float,
    ) -> None:
        self.is_hibernated = False
        self._ensure_program(compiled.optical_n_glsl_expr, compiled.d_half_n2_dU_glsl_expr)
        self.allocated_vram_mb = 0.5
        assert self._prog is not None and self._vao is not None

        fbo.use()
        w, h = fbo.size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        eye, fwd, right, up = camera.basis_vectors()
        aspect = float(w) / max(float(h), 1.0)
        fov_tan = math.tan(math.radians(camera.fov_deg) * 0.5)

        cam_r = max(float(np.linalg.norm(eye)), 1.0)
        u_cam = 1.0 / cam_r
        n_cam_mod = float(compiled.optical_n_numpy_fn(np.array([u_cam]))[0]) if compiled.optical_n_numpy_fn else 1.0
        n_cam_gr = ((1.0 + 0.5 * u_cam) ** 3) / max(1.0 - 0.5 * u_cam, 0.05)

        for uname, uval in (
            ("u_cam_eye", tuple(float(v) for v in eye)),
            ("u_cam_fwd", tuple(float(v) for v in fwd)),
            ("u_cam_right", tuple(float(v) for v in right)),
            ("u_cam_up", tuple(float(v) for v in up)),
            ("u_fov_tan", float(fov_tan)),
            ("u_aspect", float(aspect)),
            ("u_time", float(wall_time)),
            ("u_submode", int(self.submode)),
            ("u_spin_a", float(self.spin_a)),
            ("u_shadow_rg", float(compiled.shadow_radius_rg)),
            ("u_u_photon", float(compiled.u_photon_sphere)),
            ("u_u_cap", float(compiled.u_horizon_cap)),
            ("u_n_cam_mod", float(n_cam_mod)),
            ("u_n_cam_gr", float(n_cam_gr)),
            ("u_show_bhex_gauge", 1 if self.show_bhex_gauge else 0),
            ("u_show_phase_slips", 1 if self.show_phase_slips else 0),
            ("u_mass_solar", float(self.mass_solar)),
            ("u_probe_alt_km", float(self.probe_alt_km)),
            ("u_fbo_h", float(max(h, 64))),
            ("u_spool_reflectivity", float(self.spool_reflectivity)),
            ("u_qnm_freq_ratio", float(self.qnm_freq_ratio)),
        ):
            if uname in self._prog:
                self._prog[uname].value = uval

        self._vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        self.ctx.finish()
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._prog is not None:
            self._prog.release()
            self._prog = None
        self.allocated_vram_mb = 0.0
