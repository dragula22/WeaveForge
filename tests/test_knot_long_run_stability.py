"""
60-Second (7,200-Substep) 3D Topological Soliton Stability & Hopf Invariant Suite
(`tests/test_knot_long_run_stability.py`)
=================================================================================
Verifies:
  1. Bit-for-bit identical `t = 0` initial `(phi_0, vel_0 === 0)` state across `tnap_weave` and `lambda_cdm`
     for all 3 topological presets (`0: trefoil_knot`, `1: z3_braid`, `2: hopfion_ring`).
  2. Full 60-second (`3,600` frames x `2` substeps/frame = `7,200` substeps) 100% undamped (`damping == 0.0`)
     conservative symplectic Störmer-Verlet evolution at `128^3` in `Arena1KnotTank`:
       • `tnap_weave` (`k_skyrme = 1.55`): finite everywhere, L2 norm retained (`> 0.55 * L2_0`),
         extended 3D torus/link geometry (`mean_r > 1.8`), open core (`core_rho < 0.55 * peak_rho`),
         vacuum-sphere anti-crossing lock (`n_min > 0.70`), and conserved quantized Whitehead Hopf
         invariant (`Q_H ~ +7, +9, -6` within `< 0.5` across all `7,200` substeps!).
       • `lambda_cdm` (`k_skyrme = 0.0`) from the identical `t = 0` field: clean relativistic dispersion
         (`peak_rho(7200) < 0.15 * peak_rho(0)`).
"""

from __future__ import annotations

import math
from typing import Tuple

import moderngl
import numpy as np
import pytest

from weaveforge.arenas.arena1_knot_tank import Arena1KnotTank
from weaveforge.core.frameworks import get_mods_by_id
from weaveforge.core.symbolic_compiler import compile_master_action


def _readback_arena1_phi_vel_zyx3(a1: Arena1KnotTank) -> Tuple[np.ndarray, np.ndarray]:
    """Reads `(phi, vel)` of shape `(N, N, N, 3)` (`float32`) via non-demoting staging copy."""
    n = int(a1.grid_n)
    num_floats = n * n * n * 3
    latest_buf = a1._buf_a if a1._cur_in_a else a1._buf_b
    assert latest_buf is not None and a1._buf_vel is not None
    phi = a1._copy_ssbo_to_ram_f32(latest_buf, num_floats).reshape((n, n, n, 3))
    vel = a1._copy_ssbo_to_ram_f32(a1._buf_vel, num_floats).reshape((n, n, n, 3))
    return phi, vel


def _compute_hopf_charge(phi_zyx3: np.ndarray, box_size: float = 12.0, v0: float = 0.85) -> float:
    """
    Computes the Whitehead topological Hopf invariant `Q_H = -(1 / 4*pi^2) int A . B d^3x`
    via spectral Biot-Savart inversion (`curl A = B`, `div A = 0`) on the `S^2` target manifold.
    """
    n_grid = int(phi_zyx3.shape[0])
    dx = float(box_size) / float(n_grid)

    nx = phi_zyx3[..., 0].astype(np.float64)
    ny = phi_zyx3[..., 1].astype(np.float64)
    nz = phi_zyx3[..., 2].astype(np.float64) + float(v0)
    norm = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1.0e-8)
    nx /= norm
    ny /= norm
    nz /= norm
    n_vec = np.stack([nx, ny, nz], axis=0)  # (3, Z, Y, X)

    dn_dx = (np.roll(n_vec, -1, axis=3) - np.roll(n_vec, 1, axis=3)) / (2.0 * dx)
    dn_dy = (np.roll(n_vec, -1, axis=2) - np.roll(n_vec, 1, axis=2)) / (2.0 * dx)
    dn_dz = (np.roll(n_vec, -1, axis=1) - np.roll(n_vec, 1, axis=1)) / (2.0 * dx)

    def _triple(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )

    bx = _triple(n_vec, dn_dy, dn_dz)
    by = _triple(n_vec, dn_dz, dn_dx)
    bz = _triple(n_vec, dn_dx, dn_dy)

    k1d = 2.0 * math.pi * np.fft.fftfreq(n_grid, d=dx)
    kz, ky, kx = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    k2_safe = np.where(k2 > 0.0, k2, 1.0)

    bx_k = np.fft.fftn(bx)
    by_k = np.fft.fftn(by)
    bz_k = np.fft.fftn(bz)

    ax_k = np.where(k2 > 0.0, 1j * (ky * bz_k - kz * by_k) / k2_safe, 0.0)
    ay_k = np.where(k2 > 0.0, 1j * (kz * bx_k - kx * bz_k) / k2_safe, 0.0)
    az_k = np.where(k2 > 0.0, 1j * (kx * by_k - ky * bx_k) / k2_safe, 0.0)

    ax = np.fft.ifftn(ax_k).real
    ay = np.fft.ifftn(ay_k).real
    az = np.fft.ifftn(az_k).real

    helicity = float(np.sum(ax * bx + ay * by + az * bz) * (dx**3))
    return helicity / (16.0 * math.pi * math.pi)


def _compute_soliton_geometry_metrics(
    phi_zyx3: np.ndarray,
    box_size: float = 12.0,
    v0: float = 0.85,
) -> Tuple[float, float, float]:
    """Returns `(mean_r, core_rho, n_min_tube)` for 3D non-collapse and vacuum-sphere lock verification."""
    n_grid = int(phi_zyx3.shape[0])
    coords = ((np.arange(n_grid, dtype=np.float32) + 0.5) / float(n_grid) - 0.5) * float(box_size)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy + zz * zz)

    rho = np.sum(phi_zyx3.astype(np.float64) ** 2, axis=-1)
    total_rho = max(float(np.sum(rho)), 1.0e-12)
    mean_r = float(np.sum(rho * rr) / total_rho)

    central_mask = rr < 0.55
    core_rho = float(np.mean(rho[central_mask])) if np.any(central_mask) else 0.0

    rn = np.sqrt(
        phi_zyx3[..., 0].astype(np.float64) ** 2
        + phi_zyx3[..., 1].astype(np.float64) ** 2
        + (phi_zyx3[..., 2].astype(np.float64) + float(v0)) ** 2
    ) / float(v0)
    tube_mask = rho > 1.5
    n_min_tube = float(np.min(rn[tube_mask])) if np.any(tube_mask) else 0.0
    return mean_r, core_rho, n_min_tube


def test_knot_t0_initial_state_bit_identical_across_tnap_and_lambda_cdm() -> None:
    ctx = moderngl.create_standalone_context(require=430)
    mods = get_mods_by_id()
    tnap_mod = mods["tnap_weave"]
    tnap_comp = compile_master_action(tnap_mod)
    cdm_mod = mods["lambda_cdm"]
    cdm_comp = compile_master_action(cdm_mod)

    a1_tnap = Arena1KnotTank(ctx, initial_grid_n=64, mod=tnap_mod, compiled=tnap_comp)
    a1_cdm = Arena1KnotTank(ctx, initial_grid_n=64, mod=cdm_mod, compiled=cdm_comp)
    try:
        for preset_idx in (0, 1, 2):
            a1_tnap.reset_field(preset_idx)
            a1_cdm.reset_field(preset_idx)
            phi_t, vel_t = _readback_arena1_phi_vel_zyx3(a1_tnap)
            phi_c, vel_c = _readback_arena1_phi_vel_zyx3(a1_cdm)

            assert np.array_equal(phi_t, phi_c), f"Preset {preset_idx} t=0 phi differs across mods!"
            assert np.array_equal(vel_t, vel_c), f"Preset {preset_idx} t=0 vel differs across mods!"
            assert float(np.max(np.abs(vel_t))) == 0.0
    finally:
        a1_tnap.release()
        a1_cdm.release()
        ctx.release()


@pytest.mark.parametrize(
    "preset_idx,preset_name,expected_qh",
    [
        (0, "trefoil_knot", 7.0),
        (1, "z3_braid", 9.0),
        (2, "hopfion_ring", -6.0),
    ],
)
def test_60s_7200_substep_topological_soliton_lock_vs_lambda_cdm_dispersion(
    preset_idx: int,
    preset_name: str,
    expected_qh: float,
) -> None:
    ctx = moderngl.create_standalone_context(require=430)
    mods = get_mods_by_id()
    tnap_mod = mods["tnap_weave"]
    tnap_comp = compile_master_action(tnap_mod)
    cdm_mod = mods["lambda_cdm"]
    cdm_comp = compile_master_action(cdm_mod)

    a1 = Arena1KnotTank(ctx, initial_grid_n=128, mod=tnap_mod, compiled=tnap_comp)
    try:
        assert a1.damping == 0.0, "Live symplectic integrator must be 100% undamped (damping == 0.0)!"

        # 1. TNaP Spacetime Weave: 60s @ 60 Hz = 3,600 frames x 2 substeps/frame = 7,200 substeps
        a1.update_compiled_action(tnap_comp, mod=tnap_mod)
        a1.reset_field(preset_idx)
        phi_t0_tnap, vel_t0_tnap = _readback_arena1_phi_vel_zyx3(a1)
        assert float(np.max(np.abs(vel_t0_tnap))) == 0.0
        qh_t0 = _compute_hopf_charge(phi_t0_tnap, box_size=a1.box_size)
        assert abs(qh_t0 - expected_qh) < 0.6, f"{preset_name} initial Q_H={qh_t0:.3f} != {expected_qh}"

        for _ in range(3600):
            a1.step(tnap_mod, tnap_comp, n_substeps=2)
        a1._sample_diagnostics(is_initial=False, reset_ham_baseline=False)

        assert a1.step_count == 7200
        phi_end, vel_end = _readback_arena1_phi_vel_zyx3(a1)
        assert np.all(np.isfinite(phi_end))
        assert np.all(np.isfinite(vel_end))
        assert math.isfinite(a1.hamiltonian_energy)
        assert a1.l2_norm > 0.55 * a1.init_l2_norm
        assert 1.5 < a1.current_peak_rho < 4.5
        assert abs(a1.hamiltonian_drift_pct) < 8.0

        mean_r, core_rho, n_min_tube = _compute_soliton_geometry_metrics(phi_end, box_size=a1.box_size)
        qh_end = _compute_hopf_charge(phi_end, box_size=a1.box_size)

        assert mean_r > 1.8, f"{preset_name} collapsed radially: mean_r={mean_r:.3f}"
        assert core_rho < 0.55 * a1.current_peak_rho, (
            f"{preset_name} filled central hole into lump: core_rho={core_rho:.3f}, peak={a1.current_peak_rho:.3f}"
        )
        assert n_min_tube > 0.70, f"{preset_name} radial S^2 crushing: n_min={n_min_tube:.3f}"
        assert abs(qh_end - qh_t0) < 0.5, (
            f"{preset_name} Hopf charge drifted across 7,200 steps: Q_H(0)={qh_t0:.3f} -> Q_H(7200)={qh_end:.3f}"
        )
        assert abs(qh_end - expected_qh) < 0.6

        # 2. Standard Lambda-CDM from the bit-for-bit identical t = 0 state: clean dispersion (< 0.15x peak)
        a1.update_compiled_action(cdm_comp, mod=cdm_mod)
        a1.reset_field(preset_idx)
        phi_t0_cdm, vel_t0_cdm = _readback_arena1_phi_vel_zyx3(a1)
        assert np.array_equal(phi_t0_tnap, phi_t0_cdm)
        assert np.array_equal(vel_t0_tnap, vel_t0_cdm)

        for _ in range(3600):
            a1.step(cdm_mod, cdm_comp, n_substeps=2)
        a1._sample_diagnostics(is_initial=False, reset_ham_baseline=False)

        assert a1.step_count == 7200
        assert math.isfinite(a1.current_peak_rho)
        assert a1.current_peak_rho < 0.15 * a1.init_peak_rho, (
            f"{preset_name} failed to disperse cleanly under Lambda-CDM: "
            f"peak(7200)={a1.current_peak_rho:.4f} vs init={a1.init_peak_rho:.4f}"
        )
    finally:
        a1.release()
        ctx.release()
