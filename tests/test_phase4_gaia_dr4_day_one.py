"""
Phase 4 Milestone 4C: Live Gaia DR4 Day-One Raw Astrometric + RV Triple-Vetting
Ingestion & 3-Test Blind Falsification Engine (`tests/test_phase4_gaia_dr4_day_one.py`)
=======================================================================================
Verifies:
  1. Exact ICRS `(RA, Dec, parallax, pmRA, pmDec)` -> `(sep_au, dist_pc, r_gal_kpc, z_gal_kpc,
     v_tilde_obs, theta_gc_deg)` recovery & Sagittarius A* (`RA=266.4168, Dec=-29.0078`) compass geometry.
  2. Pre-registered Zenodo `10.5281/zenodo.23919217` sequential cut-flow (`min_poe >= 20`,
     parallax consistency, `nss == 0`, `ruwe <= 1.20`, and 3-sigma RV unbound-flyby / SB1-triple vetting).
  3. Blind Day-One rehearsal (`N = 4,500`, `inject_contaminants=True`) discriminating `TNaP`
     (`FAVORS_TNAP_WEAVE`, `delta_chi2 > 25`) vs `Lambda-CDM` (`FAVORS_LAMBDA_CDM`, `delta_chi2 < -25`).
  4. End-to-end CLI execution of `python -m weaveforge gaia-dr4 --export-raw-mock ... --day-one-verdict`.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from astropy import units as u
from astropy.coordinates import Galactocentric, SkyCoord
import pytest

from weaveforge.__main__ import main as weaveforge_main
from weaveforge.data.gaia_dr4_pipeline import (
    AU_M,
    G_SI,
    KMS_PER_MASYR_KPC,
    M_SUN_KG,
    SGR_A_STAR_DEC_DEG,
    SGR_A_STAR_RA_DEG,
    evaluate_gaia_dr4_day_one_verdict,
    export_mock_raw_gaia_dr4_astrometry_csv,
    ingest_gaia_dr4_raw_astrometry,
    load_gaia_dr4_csv,
)


def test_raw_astrometry_skycoord_galactocentric_and_compass_geometry(tmp_path: Path) -> None:
    """
    Analytical verification of two-star sky-level astrometry recovery on binary pairs
    placed due North of Sagittarius A* (`RA = 266.4168 deg, Dec = 0.0 deg`, `d = 200 pc`):
      - Pair 1 (`ALIGNED_GC`): North-South separation (`dRA = 0`, `dDec = 75 arcsec` -> `s = 15,000 AU`)
        pointing directly along the meridian toward Sgr A* (`theta_gc_deg == 0 deg`).
      - Pair 2 (`PERP_GC`): East-West separation (`dRA*cos(dec) = 75 arcsec`, `dDec = 0` -> `s = 15,000 AU`)
        perpendicular to Sgr A* (`theta_gc_deg == 90 deg`).
    """
    dist_pc = 200.0
    plx_mas = 1000.0 / dist_pc  # 5.0 mas
    sep_target_au = 15000.0
    dtheta_arcsec = sep_target_au / dist_pc  # 75.0 arcsec
    dtheta_deg = dtheta_arcsec / 3600.0

    m1_msun = 0.8
    m2_msun = 0.8
    m_tot = m1_msun + m2_msun
    v_circ_kms = 1.0e-3 * math.sqrt((G_SI * m_tot * M_SUN_KG) / (sep_target_au * AU_M))

    # Give ALIGNED_GC v_tilde = 1.1708 and PERP_GC v_tilde = 1.0000 analytically
    dpm_aligned_masyr = (1.1708 * v_circ_kms) / (KMS_PER_MASYR_KPC * (dist_pc / 1000.0))
    dpm_perp_masyr = (1.0000 * v_circ_kms) / (KMS_PER_MASYR_KPC * (dist_pc / 1000.0))

    ra0 = SGR_A_STAR_RA_DEG
    dec0 = 0.0

    csv_file = tmp_path / "analytical_compass.csv"
    fieldnames = [
        "pair_id",
        "ra1_deg",
        "dec1_deg",
        "parallax1",
        "parallax_error1",
        "pmra1",
        "pmdec1",
        "ruwe1",
        "radial_velocity1",
        "radial_velocity_error1",
        "non_single_star1",
        "mass1",
        "ra2_deg",
        "dec2_deg",
        "parallax2",
        "parallax_error2",
        "pmra2",
        "pmdec2",
        "ruwe2",
        "radial_velocity2",
        "radial_velocity_error2",
        "non_single_star2",
        "mass2",
    ]
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Pair 1: North-South along constant RA = 266.4168 deg -> aligned (0 deg) with Sgr A*
        writer.writerow(
            {
                "pair_id": "ALIGNED_GC",
                "ra1_deg": f"{ra0:.9f}",
                "dec1_deg": f"{dec0 - 0.5 * dtheta_deg:.9f}",
                "parallax1": f"{plx_mas:.6f}",
                "parallax_error1": "0.015",
                "pmra1": f"{10.0 - 0.5 * dpm_aligned_masyr:.8f}",
                "pmdec1": "-5.0",
                "ruwe1": "1.01",
                "radial_velocity1": "-12.10",
                "radial_velocity_error1": "0.25",
                "non_single_star1": 0,
                "mass1": f"{m1_msun}",
                "ra2_deg": f"{ra0:.9f}",
                "dec2_deg": f"{dec0 + 0.5 * dtheta_deg:.9f}",
                "parallax2": f"{plx_mas:.6f}",
                "parallax_error2": "0.015",
                "pmra2": f"{10.0 + 0.5 * dpm_aligned_masyr:.8f}",
                "pmdec2": "-5.0",
                "ruwe2": "1.03",
                "radial_velocity2": "-11.95",
                "radial_velocity_error2": "0.25",
                "non_single_star2": 0,
                "mass2": f"{m2_msun}",
            }
        )
        # Pair 2: East-West along equator Dec = 0 deg -> perpendicular (90 deg) to Sgr A* meridian
        writer.writerow(
            {
                "pair_id": "PERP_GC",
                "ra1_deg": f"{ra0 - 0.5 * dtheta_deg:.9f}",
                "dec1_deg": f"{dec0:.9f}",
                "parallax1": f"{plx_mas:.6f}",
                "parallax_error1": "0.015",
                "pmra1": "10.0",
                "pmdec1": f"{-5.0 - 0.5 * dpm_perp_masyr:.8f}",
                "ruwe1": "0.99",
                "radial_velocity1": "8.00",
                "radial_velocity_error1": "0.25",
                "non_single_star1": 0,
                "mass1": f"{m1_msun}",
                "ra2_deg": f"{ra0 + 0.5 * dtheta_deg:.9f}",
                "dec2_deg": f"{dec0:.9f}",
                "parallax2": f"{plx_mas:.6f}",
                "parallax_error2": "0.015",
                "pmra2": "10.0",
                "pmdec2": f"{-5.0 + 0.5 * dpm_perp_masyr:.8f}",
                "ruwe2": "1.04",
                "radial_velocity2": "8.10",
                "radial_velocity_error2": "0.25",
                "non_single_star2": 0,
                "mass2": f"{m2_msun}",
            }
        )

    pairs, cut_flow = ingest_gaia_dr4_raw_astrometry(csv_file)
    assert cut_flow.n_raw_rows == 2
    assert cut_flow.n_clean_final == 2
    assert len(pairs) == 2

    # Reference Astropy Galactocentric coordinates at (ra0, dec0, dist=200 pc)
    ref_gc = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg, distance=dist_pc * u.pc, frame="icrs").transform_to(
        Galactocentric(
            galcen_coord=SkyCoord(ra=SGR_A_STAR_RA_DEG * u.deg, dec=SGR_A_STAR_DEC_DEG * u.deg, frame="icrs"),
            galcen_distance=8.178 * u.kpc,
            z_sun=20.8 * u.pc,
        )
    )
    ref_r_gal = math.hypot(float(ref_gc.x.to_value(u.kpc)), float(ref_gc.y.to_value(u.kpc)))
    ref_z_gal = float(ref_gc.z.to_value(u.kpc))

    p_al, p_pe = pairs[0], pairs[1]
    assert p_al.pair_id == "ALIGNED_GC"
    assert p_al.dist_pc == pytest.approx(dist_pc, rel=1e-6)
    assert p_al.sep_au == pytest.approx(sep_target_au, rel=1e-5)
    assert p_al.r_gal_kpc == pytest.approx(ref_r_gal, abs=1e-4)
    assert p_al.z_gal_kpc == pytest.approx(ref_z_gal, abs=1e-4)
    assert p_al.v_tilde_obs == pytest.approx(1.1708, rel=1e-5)
    assert p_al.theta_gc_deg == pytest.approx(0.0, abs=0.05)
    assert p_al.delta_rv_kms == pytest.approx(0.15, abs=1e-6)

    assert p_pe.pair_id == "PERP_GC"
    assert p_pe.sep_au == pytest.approx(sep_target_au, rel=1e-5)
    assert p_pe.v_tilde_obs == pytest.approx(1.0000, rel=1e-5)
    assert p_pe.theta_gc_deg == pytest.approx(90.0, abs=0.05)

    # Verify load_gaia_dr4_csv seamless delegation on raw two-star headers
    delegated = load_gaia_dr4_csv(csv_file, max_ruwe=1.20)
    assert [p.pair_id for p in delegated] == ["ALIGNED_GC", "PERP_GC"]
    assert delegated[0].theta_gc_deg == pytest.approx(0.0, abs=0.05)
    assert delegated[1].theta_gc_deg == pytest.approx(90.0, abs=0.05)


def test_zenodo_preregistered_cutflow_rejects_ruwe_nss_and_rv_flybys(tmp_path: Path) -> None:
    """
    Verifies sequential Zenodo `10.5281/zenodo.23919217` cut-flow stages:
      1. `parallax / parallax_error >= 20.0`
      2. Consistent pair distance (`|plx1 - plx2|` within 3.5-sigma + orbital depth)
      3. `non_single_star == 0`
      4. `0 < ruwe <= 1.20`
      5. 3-sigma radial-velocity bound-orbit filter (`|RV1 - RV2|` rejects unbound flybys & SB1 triples,
         while keeping bound pairs and faint pairs lacking RVs).
    """
    base_row = {
        "pair_id": "CLEAN_BOUND",
        "ra1": "180.000000000",
        "dec1": "30.000000000",
        "parallax1": "5.000000",
        "parallax_error1": "0.020000",
        "pmra1": "-12.000000",
        "pmdec1": "8.000000",
        "pmra_error1": "0.004000",
        "pmdec_error1": "0.004000",
        "ruwe1": "1.02",
        "radial_velocity1": "15.00",
        "radial_velocity_error1": "0.30",
        "non_single_star1": "0",
        "mass1": "0.85",
        "ra2": "180.020000000",
        "dec2": "30.000000000",
        "parallax2": "5.010000",
        "parallax_error2": "0.020000",
        "pmra2": "-11.680000",
        "pmdec2": "8.000000",
        "pmra_error2": "0.004000",
        "pmdec_error2": "0.004000",
        "ruwe2": "1.05",
        "radial_velocity2": "15.25",
        "radial_velocity_error2": "0.30",
        "non_single_star2": "0",
        "mass2": "0.80",
    }

    rows = [
        dict(base_row, pair_id="CLEAN_WITH_RV"),
        dict(base_row, pair_id="CLEAN_MISSING_RV2", radial_velocity2=""),
        dict(base_row, pair_id="FAIL_LOW_POE", parallax_error1="0.400000"),  # PoE = 12.5 < 20
        dict(base_row, pair_id="FAIL_BG_PARALLAX", parallax2="6.200000"),  # |5.0 - 6.2| = 1.2 mas >> tol
        dict(base_row, pair_id="FAIL_NSS_ASTROM", non_single_star2="1"),
        dict(base_row, pair_id="FAIL_RUWE_TRIPLE", ruwe1="1.62"),
        dict(base_row, pair_id="FAIL_RV_FLYBY_28KMS", radial_velocity2="43.00"),  # |dRV| = 28 km/s unbound!
    ]

    csv_file = tmp_path / "cutflow_vetting.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(base_row.keys()))
        writer.writeheader()
        writer.writerows(rows)

    pairs, cf = ingest_gaia_dr4_raw_astrometry(csv_file, max_ruwe=1.20, min_poe=20.0, rv_sigma_clip=3.0)
    assert cf.n_raw_rows == 7
    assert cf.n_passed_snr_parallax == 6
    assert cf.n_passed_consistent_parallax == 5
    assert cf.n_passed_nss == 4
    assert cf.n_passed_ruwe == 3
    assert cf.n_passed_rv_bound == 2
    assert cf.n_clean_final == 2
    assert [p.pair_id for p in pairs] == ["CLEAN_WITH_RV", "CLEAN_MISSING_RV2"]
    assert pairs[0].delta_rv_kms == pytest.approx(0.25, abs=1e-6)
    assert pairs[1].delta_rv_kms is None


def test_blind_day_one_rehearsal_discriminates_tnap_vs_lambda_cdm(tmp_path: Path) -> None:
    """
    Full Day-One blind rehearsal (`N = 4,500`, `inject_contaminants=True` with ~12% triples, NSS, and 25 km/s RV flybys)
    from raw two-star ICRS sky astrometry + radial velocities under both `tnap_weave` and `lambda_cdm`.
    """
    tnap_csv = tmp_path / "mock_day_one_tnap.csv"
    cdm_csv = tmp_path / "mock_day_one_lcdm.csv"

    s_tnap_exp = export_mock_raw_gaia_dr4_astrometry_csv(
        tnap_csv, n_binaries=4500, truth_mode="tnap_weave", inject_contaminants=True, seed=20261202
    )
    s_cdm_exp = export_mock_raw_gaia_dr4_astrometry_csv(
        cdm_csv, n_binaries=4500, truth_mode="lambda_cdm", inject_contaminants=True, seed=20261202
    )
    assert s_tnap_exp["n_total_exported"] == 4500
    assert s_tnap_exp["n_contaminants_injected"] == pytest.approx(540, abs=5)

    tnap_pairs, cf_tnap = ingest_gaia_dr4_raw_astrometry(tnap_csv)
    cdm_pairs, cf_cdm = ingest_gaia_dr4_raw_astrometry(cdm_csv)

    # All ~540 injected RUWE/NSS/RV-flyby contaminants must be filtered out while retaining >98% of clean binaries
    assert cf_tnap.n_raw_rows == 4500
    assert 3850 <= cf_tnap.n_clean_final <= 4000
    assert 3850 <= cf_cdm.n_clean_final <= 4000

    v_tnap = evaluate_gaia_dr4_day_one_verdict(tnap_pairs, cut_flow=cf_tnap)
    v_cdm = evaluate_gaia_dr4_day_one_verdict(cdm_pairs, cut_flow=cf_cdm)

    # Blind discrimination: TNaP sky must strongly favor TNaP, Lambda-CDM sky must strongly favor Lambda-CDM!
    assert v_tnap["verdict"] == "FAVORS_TNAP_WEAVE"
    assert float(v_tnap["delta_chi2"]) > 25.0
    assert float(v_tnap["log10_bayes_factor"]) > 5.0
    assert 10.5 <= float(v_tnap["local_backyard_15k_boost_pct"]) <= 22.0
    assert float(v_tnap["env_step_halo_minus_inner_pct"]) > 3.0

    assert v_cdm["verdict"] == "FAVORS_LAMBDA_CDM"
    assert float(v_cdm["delta_chi2"]) < -25.0
    assert float(v_cdm["log10_bayes_factor"]) < -5.0
    assert abs(float(v_cdm["local_backyard_15k_boost_pct"])) < 5.5
    assert abs(float(v_cdm["env_step_halo_minus_inner_pct"])) < 5.5


def test_cli_gaia_dr4_day_one_verdict_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end CLI verification of `gaia-dr4 --export-raw-mock ... --day-one-verdict`."""
    raw_tnap = tmp_path / "cli_raw_tnap.csv"
    rc1 = weaveforge_main(
        [
            "gaia-dr4",
            "--export-raw-mock",
            str(raw_tnap),
            "--mock-count",
            "3200",
            "--truth",
            "tnap_weave",
            "--day-one-verdict",
        ]
    )
    out1 = capsys.readouterr().out
    assert rc1 == 0
    assert raw_tnap.exists()
    assert "Clean RUWE<=1.2 Pairs" in out1
    assert "10.5281/zenodo.23919217" in out1
    assert "Day-One Blind Verdict : FAVORS_TNAP_WEAVE" in out1

    raw_cdm = tmp_path / "cli_raw_lcdm.csv"
    rc2 = weaveforge_main(
        [
            "gaia-dr4",
            "--export-raw-mock",
            str(raw_cdm),
            "--mock-count",
            "3200",
            "--truth",
            "lambda_cdm",
            "--no-contaminants",
            "--day-one-verdict",
        ]
    )
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert raw_cdm.exists()
    assert "0 contaminants" in out2
    assert "Day-One Blind Verdict : FAVORS_LAMBDA_CDM" in out2

    # Direct `--csv` ingestion of exported raw sky CSV must also auto-run the Day-One scorecard
    rc3 = weaveforge_main(["gaia-dr4", "--csv", str(raw_tnap), "--day-one-verdict"])
    out3 = capsys.readouterr().out
    assert rc3 == 0
    assert "Day-One Blind Verdict : FAVORS_TNAP_WEAVE" in out3
