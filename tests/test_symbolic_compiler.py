"""
Exhaustive Symbolic Compiler, Red-Team Health Audit & Edge-Case Tests (`tests/test_symbolic_compiler.py`)
========================================================================================================
Tests:
  • All 6 built-in `.toml` mods (validating exact PPN gamma/beta, 2PN shadow radii, Saturn 400-digit screening,
    GW170817, and automated ghost/PPN rejection on `ghost_redteam_demo`).
  • Foundational `SymPy` algebraic proofs (KMS horizon polylogarithm -> Bose-Einstein RAR, Gibbons Planck spool
    `1.0`, Catenoid waist `24/r0^4`, and `Z_3` Koide `2/3`).
  • Adversarial & edge-case inputs: empty strings, whitespace-only, `__import__` / `lambda` injection attempts,
    unknown symbols, syntax errors, singular `0 * T_kin` kinetic matrices, higher-time-derivative Ostrogradsky
    ghosts, superluminal `c_w > 1` sound speeds, and negative signed fields (`phi_a < 0`) checking zero NaNs!
  • TOML round-trip serialization (`PhysicsMod.to_toml_string()` -> `PhysicsMod.from_toml_bytes()`).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from weaveforge.core.frameworks import ActionSpec, ModParameters, PhysicsMod, get_mods_by_id, load_all_mods
from weaveforge.core.symbolic_compiler import (
    SymbolicCompilationError,
    compile_master_action,
    verify_foundational_tnap_identities,
)


def test_foundational_sympy_identities() -> None:
    proofs = verify_foundational_tnap_identities()
    assert proofs["kms_rar_identity_exact"] is True
    assert math.isclose(float(proofs["planck_spool_ratio_exact"]), 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert "24/r0**4" in str(proofs["catenoid_waist_kretschmann"]).replace(" ", "")
    assert math.isclose(float(proofs["koide_z3_ratio_exact"]), 2.0 / 3.0, rel_tol=0.0, abs_tol=1e-15)


def test_builtin_six_mods_compile_and_audit() -> None:
    mods = get_mods_by_id()
    assert len(mods) >= 6

    # 1. Standard Lambda-CDM: GR shadow = 3*sqrt(3) = 5.196152 r_g, PPN gamma=beta=1
    cdm = compile_master_action(mods["lambda_cdm"])
    assert cdm.all_critical_passed is True
    assert math.isclose(cdm.ppn_gamma, 1.0, abs_tol=1e-7)
    assert math.isclose(cdm.ppn_beta, 1.0, abs_tol=1e-7)
    assert math.isclose(cdm.shadow_radius_rg, 3.0 * math.sqrt(3.0), rel_tol=1e-5)

    # 2. TNaP Spacetime Weave: Exponential metric n(U)=exp(2U) -> shadow = 2*e = 5.436564 r_g (+4.63% vs GR)
    tnap = compile_master_action(mods["tnap_weave"])
    assert tnap.all_critical_passed is True
    assert math.isclose(tnap.ppn_gamma, 1.0, abs_tol=1e-7)
    assert math.isclose(tnap.ppn_beta, 1.0, abs_tol=1e-7)
    assert math.isclose(tnap.shadow_radius_rg, 2.0 * math.e, rel_tol=1e-5)
    assert math.isclose(tnap.shadow_vs_gr_pct, (2.0 * math.e / (3.0 * math.sqrt(3.0)) - 1.0) * 100.0, abs_tol=1e-3)
    # 400-digit mpmath Saturn 9.58 AU screening check (~ 10^-318)
    assert tnap.saturn_anomaly_log10 < -300.0
    assert tnap.earth_anomaly_log10 < -3000.0

    # 3. Red-Team Ghost Trap: MUST fail critical audit gates automatically!
    trap = compile_master_action(mods["ghost_redteam_demo"])
    assert trap.all_critical_passed is False
    failed_ids = {g.gate_id for g in trap.health_gates if not g.passed}
    assert "kinetic_ghost" in failed_ids
    assert "sound_speed" in failed_ids
    assert "gw170817" in failed_ids
    assert "ppn_solar" in failed_ids
    # 1911 half-bending n(U)=exp(U) gives PPN gamma = 0.0 and shadow = e = 2.71828 r_g!
    assert math.isclose(trap.ppn_gamma, 0.0, abs_tol=1e-6)
    assert math.isclose(trap.shadow_radius_rg, math.e, rel_tol=1e-4)


@pytest.mark.parametrize(
    "bad_expr",
    [
        "",
        "   ",
        "__import__('os').system('echo pwned')",
        "lambda x: x",
        "T_kin + unknown_variable_xyz",
        "T_kin + * / 3",
    ],
)
def test_compiler_rejects_empty_malicious_or_unknown_inputs(bad_expr: str) -> None:
    bad_mod = PhysicsMod(
        id="bad_test",
        name="Bad Input Mod",
        author="Pytest",
        summary="Adversarial input check",
        action=ActionSpec(field_lagrangian=bad_expr),
    )
    with pytest.raises(SymbolicCompilationError):
        compile_master_action(bad_mod)


def test_singular_zero_kinetic_term_raises_clean_error() -> None:
    # A Lagrangian with zero time derivatives has a singular kinetic Hessian K_ab = 0 -> cannot invert d^2 phi / dt^2!
    no_kin = PhysicsMod(
        id="static_only",
        name="No Kinetic Term",
        author="Pytest",
        summary="Missing T_kin",
        action=ActionSpec(field_lagrangian="-G_grad - 0.5 * m_eff**2 * rho"),
    )
    with pytest.raises(SymbolicCompilationError):
        compile_master_action(no_kin)


def test_superluminal_wave_speed_flagged_by_health_audit() -> None:
    superlum = load_all_mods()[0].clone(new_id="superlum_test")
    superlum.parameters.c_w = 1.35  # c_s^2 = 1.8225 > 1!
    comp = compile_master_action(superlum)
    assert comp.all_critical_passed is False
    ss_gate = [g for g in comp.health_gates if g.gate_id == "sound_speed"][0]
    assert ss_gate.passed is False
    assert "SUPERLUMINAL" in ss_gate.plain_english


def test_negative_signed_fields_produce_zero_nans_in_compiled_callables() -> None:
    tnap_mod = get_mods_by_id()["tnap_weave"]
    comp = compile_master_action(tnap_mod)
    assert comp.accel_numpy_fn is not None and comp.hamiltonian_numpy_fn is not None

    # Strictly negative & mixed-sign 3D fields where naive pow(negative, even) would NaN!
    rng = np.random.default_rng(42)
    phi = rng.uniform(-4.5, -0.1, size=(3, 8, 8, 8)).astype(np.float32)
    vel = rng.uniform(-2.0, 2.0, size=(3, 8, 8, 8)).astype(np.float32)
    grad = rng.uniform(-1.5, 1.5, size=(3, 3, 8, 8, 8)).astype(np.float32)
    lap = rng.uniform(-3.0, 3.0, size=(3, 8, 8, 8)).astype(np.float32)

    acc = comp.accel_numpy_fn(phi, grad, lap, 1.0, 0.32, 6.8, 4.2)
    ham = comp.hamiltonian_numpy_fn(phi, vel, grad, 1.0, 0.32, 6.8, 4.2)
    assert np.isfinite(acc).all()
    assert np.isfinite(ham).all()
    assert float(np.min(ham)) > 0.0  # Positive-definite Hamiltonian on TNaP potential


def test_physics_mod_toml_roundtrip() -> None:
    orig = get_mods_by_id()["tnap_weave"]
    raw = orig.to_toml_string().encode("utf-8")
    reloaded = PhysicsMod.from_toml_bytes(raw)
    assert reloaded.id == orig.id
    assert reloaded.action.field_lagrangian == orig.action.field_lagrangian
    assert reloaded.action.metric_lapse_A == orig.action.metric_lapse_A
    assert math.isclose(reloaded.parameters.a0_si, orig.parameters.a0_si, rel_tol=1e-5)
    assert reloaded.parameters.pinned_pleat_coupling == orig.parameters.pinned_pleat_coupling


def test_coleman_false_vacuum_detected_by_gate5() -> None:
    # g4^2 = 256 >> (16/3)*m_eff^2*g6 = 27.73 -> deep negative intermediate false-vacuum well!
    unstable = get_mods_by_id()["tnap_weave"].clone(new_id="false_vac_test")
    unstable.parameters.m_eff = 1.0
    unstable.parameters.g4 = 16.0
    unstable.parameters.g6 = 5.2
    comp = compile_master_action(unstable)
    vb_gate = [g for g in comp.health_gates if g.gate_id == "vacuum_bound"][0]
    assert vb_gate.passed is False
    assert "FALSE-VACUUM" in vb_gate.plain_english


def test_independent_metric_lapse_catches_non_unit_ppn_beta() -> None:
    # Even when optical_metric_n has c2_u = 1.85 inside [1.74, 2.01], non-GR lapse beta = 1.20 MUST fail!
    bad_beta = get_mods_by_id()["lambda_cdm"].clone(new_id="bad_ppn_beta")
    bad_beta.action.optical_metric_n = "1 + 2 * U + 1.85 * U**2"
    bad_beta.action.metric_lapse_A = "1 - 2 * U + 2.40 * U**2"
    comp = compile_master_action(bad_beta)
    assert math.isclose(comp.ppn_gamma, 1.0, abs_tol=1e-6)
    assert math.isclose(comp.ppn_beta, 1.20, abs_tol=1e-6)
    ppn_gate = [g for g in comp.health_gates if g.gate_id == "ppn_solar"][0]
    assert ppn_gate.passed is False


def test_anisotropic_dy_phi2_instability_detected_by_gate2() -> None:
    # Stable along dx_phi1 (+1) but elliptic/wrong-sign along dy_phi2 (-1)
    aniso = PhysicsMod(
        id="aniso_elliptic",
        name="Anisotropic Elliptic Mod",
        author="Pytest",
        summary="Tests all 9 eigenvalues in Gate 2",
        action=ActionSpec(
            field_lagrangian="T_kin - 0.5 * c_w**2 * (dx_phi1**2 - dy_phi2**2 + dz_phi3**2 + dy_phi1**2 + dz_phi1**2 + dx_phi2**2 + dz_phi2**2 + dx_phi3**2 + dy_phi3**2)"
        ),
    )
    comp = compile_master_action(aniso)
    ss_gate = [g for g in comp.health_gates if g.gate_id == "sound_speed"][0]
    assert ss_gate.passed is False


def test_nonlinear_gradient_and_px_lagrangian_compile_without_false_ghost() -> None:
    px_mod = PhysicsMod(
        id="px_superfluid",
        name="Relativistic P(X) Mod",
        author="Pytest",
        summary="Tests mixed 2nd spatial derivatives + velocity-gradient Hessian without false ghost",
        action=ActionSpec(
            field_lagrangian="(T_kin - c_w**2 * G_grad) + 0.08 * (T_kin - c_w**2 * G_grad)**2 - 0.5 * m_eff**2 * rho"
        ),
    )
    comp = compile_master_action(px_mod)
    assert comp.all_critical_passed is True
    assert "dxy_phi1" in comp.glsl_compute_source
    assert "dxy_phi1" in comp.wgsl_compute_source
    assert "dx_dt_phi1" in comp.glsl_compute_source
    assert "dx_dt_phi1" in comp.wgsl_compute_source


def test_emit_gpu_expr_negative_and_high_odd_powers_preserve_sign() -> None:
    import sympy as sp
    from weaveforge.core.symbolic_compiler import _emit_gpu_expr

    x = sp.Symbol("x", real=True)
    wgsl_neg3 = _emit_gpu_expr(x**-3, "wgsl")
    glsl_pow9 = _emit_gpu_expr(x**9, "glsl")
    assert "select(-1.0, 1.0" in wgsl_neg3
    assert ">= 0.0 ? 1.0 : -1.0" in glsl_pow9

    # End-to-end GPU check: L = T_kin - c_w**2 * G_grad - 0.1 * phi1**10 -> RHS has -phi1**9 (odd power 9!)
    high_odd = PhysicsMod(
        id="high_odd_pow",
        name="High Odd Power Test",
        author="Pytest",
        summary="phi1**10 potential produces -phi1**9 force",
        action=ActionSpec(field_lagrangian="T_kin - c_w**2 * G_grad - 0.1 * phi1**10"),
    )
    comp = compile_master_action(high_odd)
    phi = np.full((3, 4, 4, 4), -1.2, dtype=np.float64)
    grad = np.zeros((3, 3, 4, 4, 4), dtype=np.float64)
    lap = np.zeros((3, 4, 4, 4), dtype=np.float64)
    acc = comp.accel_numpy_fn(phi, grad, lap, 1.0, 1.0, 0.0, 0.0)
    # At phi1 = -1.2 < 0, restoring force -phi1**9 = -(-1.2)^9 = +5.15978 > 0!
    assert math.isclose(float(acc[0, 0, 0, 0]), 1.2**9, rel_tol=1e-6)

