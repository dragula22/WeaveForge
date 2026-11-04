"""
Observational Crucible & Dec 2, 2026 Gaia DR4 Pipeline Tests (`tests/test_crucible_and_gaia_dr4.py`)
====================================================================================================
Tests:
  • Ruthlessly honest pass/warn/fail behavior across all 6 competing physics frameworks:
      - `Classic MOND`: passes SPARC 175 with 0 halo knobs, FAILS Planck CMB Peak 3 (`R_32 < 0.5`),
        and WARNs on Saturn 9.58 AU ephemeris (`10^-6 > 2e-8`)!
      - `Lambda-CDM`: passes Planck CMB & Solar System, WARNs on SPARC (`350` free halo knobs),
        and predicts `0.0%` Gaia DR4 wide-binary boost!
      - `TNaP Spacetime Weave`: `Omega_m_effective == 0.30941`, waist-clamped `D_2 < 400 uK^2`,
        Saturn `< 10^-300`, and `+15%` Disk -> `+31%` Halo Gaia DR4 wide-binary boost!
  • Gaia DR4 CSV ingestion with strict `ruwe <= 1.2` unresolved-triple rejection, negative/zero
    separation rejection, empty sample handling, and mock survey significance!
"""

from __future__ import annotations

import math
from pathlib import Path

from weaveforge.arenas.arena4_cosmic_crucible import evaluate_observational_crucible
from weaveforge.core.frameworks import get_mods_by_id
from weaveforge.core.symbolic_compiler import compile_master_action
from weaveforge.data.gaia_dr4_pipeline import (
    analyze_gaia_wide_binary_sample,
    generate_mock_gaia_dr4_catalog,
    load_gaia_dr4_csv,
)


def test_crucible_scientific_honesty_across_competing_theories() -> None:
    mods = get_mods_by_id()

    # 1. Classic MOND must pass SPARC, FAIL CMB Peak 3, and WARN on Saturn ephemerides
    mond = mods["classic_mond"]
    mond_rep = evaluate_observational_crucible(mond, compile_master_action(mond))
    b_mond = {b.badge_id: b.status for b in mond_rep.badges}
    assert b_mond["sparc_175"] == "PASS"
    assert b_mond["planck_cmb"] == "FAIL"
    assert mond_rep.cmb_r32_ratio < 0.50
    assert b_mond["solar_gw"] == "WARN"

    # 2. Lambda-CDM must WARN on SPARC (350 halo knobs), PASS CMB, and predict 0% Gaia boost
    cdm = mods["lambda_cdm"]
    cdm_rep = evaluate_observational_crucible(cdm, compile_master_action(cdm))
    b_cdm = {b.badge_id: b.status for b in cdm_rep.badges}
    assert b_cdm["sparc_175"] == "WARN"
    assert b_cdm["planck_cmb"] == "PASS"
    assert abs(cdm_rep.wide_binary_disk_boost_pct) < 1e-6
    assert abs(cdm_rep.wide_binary_halo_boost_pct) < 1e-6

    # If user unticks CDM halos in Lambda-CDM without nu(y) boost, SPARC must FAIL!
    bary_gr = cdm.clone(new_id="pure_bary_gr")
    bary_gr.parameters.has_cdm_halo = False
    bary_rep = evaluate_observational_crucible(bary_gr, compile_master_action(bary_gr))
    b_bary = {b.badge_id: b.status for b in bary_rep.badges}
    assert b_bary["sparc_175"] == "FAIL"

    # 3. TNaP Spacetime Weave: exact Omega_m = 0.30941, clamped C_2, and Halo > Disk wide-binary boost
    tnap = mods["tnap_weave"]
    tnap_rep = evaluate_observational_crucible(tnap, compile_master_action(tnap))
    b_tnap = {b.badge_id: b.status for b in tnap_rep.badges}
    assert math.isclose(tnap_rep.omega_m_effective, 0.30941, abs_tol=5e-5)
    assert tnap_rep.cmb_d2_quadrupole_uK2 < 400.0
    assert tnap_rep.cross_scale_seam_open is True
    assert b_tnap["sparc_175"] == "WARN"
    assert b_tnap["solar_gw"] == "PASS"
    # Honest open flags on full-Boltzmann CMB & Hubble tension!
    assert b_tnap["planck_cmb"] == "WARN"
    assert b_tnap["desi_h0"] == "WARN"
    assert 12.0 <= tnap_rep.wide_binary_disk_boost_pct <= 20.0
    assert 26.0 <= tnap_rep.wide_binary_halo_boost_pct <= 36.0
    assert tnap_rep.wide_binary_halo_boost_pct > tnap_rep.wide_binary_disk_boost_pct + 10.0


def test_gaia_dr4_mock_and_csv_quality_cuts(tmp_path: Path) -> None:
    # Empty input edge case
    empty_stats = analyze_gaia_wide_binary_sample([])
    assert empty_stats["n_clean"] == 0
    assert empty_stats["halo_excess_sigma"] == 0.0

    # Mock Gaia DR4 TNaP vs Lambda-CDM statistical discrimination
    tnap_pairs = generate_mock_gaia_dr4_catalog(n_binaries=2400, truth_mode="tnap_weave", seed=7)
    cdm_pairs = generate_mock_gaia_dr4_catalog(n_binaries=2400, truth_mode="lambda_cdm", seed=7)

    s_tnap = analyze_gaia_wide_binary_sample(tnap_pairs)
    s_cdm = analyze_gaia_wide_binary_sample(cdm_pairs)
    assert s_tnap["halo_boost_pct"] > s_tnap["disk_boost_pct"] + 10.0
    assert s_tnap["halo_excess_sigma"] > 4.5
    assert abs(s_cdm["halo_boost_pct"]) < 5.0

    # CSV ingestor: must filter out ruwe > 1.2 triples and invalid <= 0 AU separations!
    csv_file = tmp_path / "gaia_dr4_sample.csv"
    csv_file.write_text(
        "pair_id,sep_au,z_gal_kpc,m_tot_msun,v_tilde_obs,ruwe_max\n"
        "CLEAN_INNER,1200.0,0.2,1.6,0.785,1.02\n"
        "CLEAN_DISK,14000.0,0.3,1.6,0.910,1.08\n"
        "CLEAN_HALO,15000.0,6.2,1.6,1.035,1.05\n"
        "BAD_TRIPLE,15000.0,0.2,1.6,2.450,1.85\n"
        "BAD_NEG_SEP,-50.0,0.2,1.6,0.800,1.01\n",
        encoding="utf-8",
    )
    loaded = load_gaia_dr4_csv(csv_file, max_ruwe=1.20)
    assert [p.pair_id for p in loaded] == ["CLEAN_INNER", "CLEAN_DISK", "CLEAN_HALO"]
    csv_stats = analyze_gaia_wide_binary_sample(loaded)
    assert csv_stats["halo_boost_pct"] > csv_stats["disk_boost_pct"] > 10.0


def test_bundled_published_sparc_175_mrt_catalog_and_rar_scatter() -> None:
    from weaveforge.data.sparc_catalog import load_sparc_175_catalog, score_full_sparc_175

    gals, src_tag = load_sparc_175_catalog()
    assert src_tag == "MRT_RAW_SPARC"
    assert len(gals) == 175
    total_pts = sum(int(g.r_kpc.size) for g in gals)
    assert total_pts == 3391

    tnap = get_mods_by_id()["tnap_weave"]
    comp = compile_master_action(tnap)
    assert comp.nu_numpy_fn is not None
    fit = score_full_sparc_175(
        gals,
        comp.nu_numpy_fn,
        tnap.parameters.a0_si,
        has_cdm_halo=False,
        optimize_ml_prior=True,
    )
    # Reproduces McGaugh, Lelli & Schombert (2016, PRL) ~0.13 dex RAR scatter & Li+ (2018) median chi2_red ~ 1.63
    assert 0.120 <= float(fit["rar_rms_dex"]) <= 0.145
    assert 1.20 <= float(fit["median_chi2_red"]) <= 2.10


def test_real_camb_boltzmann_and_astropy_desi_dr2_bao_integration() -> None:
    from weaveforge.arenas.arena4_cosmic_crucible import _HAS_CAMB

    assert _HAS_CAMB is True
    mods = get_mods_by_id()

    tnap_rep = evaluate_observational_crucible(mods["tnap_weave"], compile_master_action(mods["tnap_weave"]))
    cdm_rep = evaluate_observational_crucible(mods["lambda_cdm"], compile_master_action(mods["lambda_cdm"]))
    mond_rep = evaluate_observational_crucible(mods["classic_mond"], compile_master_action(mods["classic_mond"]))

    assert len(tnap_rep.cmb_ell) == 999
    assert tnap_rep.cmb_r32_ratio > 0.88
    assert cdm_rep.cmb_r32_ratio > 0.90
    assert mond_rep.cmb_r32_ratio < 0.50

    # Published 13-point DESI DR2 (2025) BAO chi^2/dof via Astropy w0waCDM
    assert tnap_rep.desi_bao_chi2_dof < 1.50
    assert cdm_rep.desi_bao_chi2_dof < 1.50
    assert mond_rep.desi_bao_chi2_dof > 50.0
    b_mond = {b.badge_id: b.status for b in mond_rep.badges}
    assert b_mond["desi_h0"] == "FAIL"

