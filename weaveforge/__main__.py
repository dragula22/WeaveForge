"""
WeaveForge Unified CLI Entrypoint (`weaveforge/__main__.py`)
============================================================
Usage:
  python -m weaveforge                          # Launch 60-120+ FPS 3D Studio (Rig 1)
  python -m weaveforge studio --arena 3         # Launch directly into Arena 3 (1M-Star Collider)
  python -m weaveforge worker --port 8765       # Launch Rig 2 AMD RX 5700 XT Vulkan wgpu LAN Worker
  python -m weaveforge audit                    # Run SymPy Proofs + 5-Gate Audit + 5-Dataset Crucible
  python -m weaveforge gaia-dr4 --csv my.csv    # Run Dec 2, 2026 Gaia DR4 Wide-Binary |z_gal| Test
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _cmd_studio(args: argparse.Namespace) -> int:
    from weaveforge.studio.app import WeaveForgeStudioApp

    visible = not bool(args.hidden)
    app = WeaveForgeStudioApp(
        width=args.width,
        height=args.height,
        initial_arena=args.arena,
        initial_mod_id=args.mod,
        vsync=not bool(args.no_vsync),
        visible=visible,
        ui_scale=args.ui_scale,
        render_scale=args.render_scale,
        fps_cap=args.fps_cap,
    )
    snap = Path(args.screenshot_out) if args.screenshot_out else None
    app.run(max_frames=args.headless_frames, screenshot_out=snap)
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    import uvicorn
    from weaveforge.compute.lan_worker import create_worker_app

    print(f"[WeaveForge Rig 2 Worker] Starting Cross-Platform Vulkan/DX12 wgpu Server on {args.host}:{args.port}...")
    app = create_worker_app(prefer_discrete=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_audit(_args: argparse.Namespace) -> int:
    from weaveforge.arenas.arena4_cosmic_crucible import evaluate_observational_crucible
    from weaveforge.core.frameworks import load_all_mods
    from weaveforge.core.symbolic_compiler import (
        compile_master_action,
        verify_foundational_tnap_identities,
    )

    print("================================================================================")
    print(" WEAVEFORGE FOUNDATIONAL SYMPY ALGEBRAIC IDENTITIES AUDIT")
    print("================================================================================")
    proofs = verify_foundational_tnap_identities()
    for k, v in proofs.items():
        print(f"  * {k:34s}: {v}")

    print("\n================================================================================")
    print(" COMPETING PHYSICS MODS — 5-GATE HEALTH AUDIT & 5-DATASET OBSERVATIONAL CRUCIBLE")
    print("================================================================================")
    for mod in load_all_mods():
        comp = compile_master_action(mod)
        rep = evaluate_observational_crucible(mod, comp)
        if not comp.all_critical_passed:
            crit = "REJECTED BY RED-TEAM"
        elif comp.has_warnings or not comp.all_gates_passed:
            crit = "CAUTION (STABILITY WARNING)"
        else:
            crit = "ALL 5 GATES PASSED"
        print(f"\n[{mod.id}] {mod.name}")
        print(
            f"  Health: {crit} | PPN gamma={comp.ppn_gamma:.5f} | "
            f"2PN Shadow={comp.shadow_radius_rg:.4f} r_g ({comp.shadow_vs_gr_pct:+.2f}%) | "
            f"Omega_m_eff={rep.omega_m_effective:.5f}"
        )
        for b in rep.badges:
            print(f"    [{b.status:8s}] {b.dataset_name:32s} -> {b.headline_metric}")
    return 0


def _cmd_gaia(args: argparse.Namespace) -> int:
    import tempfile
    from weaveforge.data.gaia_dr4_pipeline import (
        GaiaCutFlowSummary,
        analyze_gaia_wide_binary_sample,
        evaluate_gaia_dr4_day_one_verdict,
        export_mock_raw_gaia_dr4_astrometry_csv,
        generate_mock_gaia_dr4_catalog,
        ingest_gaia_dr4_raw_astrometry,
        is_raw_gaia_astrometry_csv,
        load_gaia_dr4_csv,
    )

    inject_contam = not bool(args.no_contaminants)
    cut_flow: GaiaCutFlowSummary | None = None
    is_raw = False

    if args.export_raw_mock:
        out_path = Path(args.export_raw_mock)
        exp_stats = export_mock_raw_gaia_dr4_astrometry_csv(
            out_path,
            n_binaries=args.mock_count,
            truth_mode=args.truth,
            inject_contaminants=inject_contam,
        )
        print(
            f"[WeaveForge Gaia DR4 Raw Mock] Exported {exp_stats['n_total_exported']} rows "
            f"({exp_stats['n_clean_injected']} clean, {exp_stats['n_contaminants_injected']} contaminants) -> {out_path}"
        )

    if args.csv:
        p = Path(args.csv)
        if is_raw_gaia_astrometry_csv(p):
            pairs, cut_flow = ingest_gaia_dr4_raw_astrometry(p, max_ruwe=args.max_ruwe)
            is_raw = True
        else:
            pairs = load_gaia_dr4_csv(p, max_ruwe=args.max_ruwe)
        label = f"CSV ({p.name})"
    elif args.export_raw_mock:
        p = Path(args.export_raw_mock)
        pairs, cut_flow = ingest_gaia_dr4_raw_astrometry(p, max_ruwe=args.max_ruwe)
        is_raw = True
        label = f"Raw Mock CSV ({p.name}, {args.truth}, N={args.mock_count})"
    elif args.day_one_verdict:
        with tempfile.TemporaryDirectory() as tmp_d:
            tmp_csv = Path(tmp_d) / "gaia_dr4_day_one_rehearsal.csv"
            export_mock_raw_gaia_dr4_astrometry_csv(
                tmp_csv,
                n_binaries=args.mock_count,
                truth_mode=args.truth,
                inject_contaminants=inject_contam,
            )
            pairs, cut_flow = ingest_gaia_dr4_raw_astrometry(tmp_csv, max_ruwe=args.max_ruwe)
        is_raw = True
        label = f"Day-One 3D Sky Mock ({args.truth}, N={args.mock_count})"
    else:
        pairs = generate_mock_gaia_dr4_catalog(n_binaries=args.mock_count, truth_mode=args.truth)
        label = f"Synthetic Gaia DR4 Mock ({args.truth}, N={args.mock_count})"

    res = analyze_gaia_wide_binary_sample(pairs)
    print(f"[WeaveForge Gaia DR4 Pipeline] Source: {label}")
    print(f"  Clean RUWE<=1.2 Pairs : {res['n_clean']}")
    print(f"  Disk Boost (|z|<1.5k) : {res['disk_boost_pct']:+.2f}%")
    print(f"  Halo Boost (|z|>=4.5k): {res['halo_boost_pct']:+.2f}%")
    print(f"  Halo-vs-Disk Excess   : {res['halo_excess_sigma']:+.2f} sigma")

    if bool(args.day_one_verdict) or is_raw:
        verdict = evaluate_gaia_dr4_day_one_verdict(pairs, cut_flow=cut_flow)
        print(f"\n[Zenodo {verdict['zenodo_doi']} — 3-Test Blind Falsification Scorecard]")
        if cut_flow is not None:
            print(
                f"  Cut-Flow              : raw={cut_flow.n_raw_rows} -> poe={cut_flow.n_passed_snr_parallax} "
                f"-> plx={cut_flow.n_passed_consistent_parallax} -> nss={cut_flow.n_passed_nss} "
                f"-> ruwe={cut_flow.n_passed_ruwe} -> rv={cut_flow.n_passed_rv_bound} -> clean={cut_flow.n_clean_final}"
            )
        print(
            f"  [Test A] Local <=300pc: {verdict['local_backyard_15k_boost_pct']:+.2f}% +/- "
            f"{verdict['local_backyard_15k_err_pct']:.2f}% (TNaP {verdict['tnap_pred_local_15k_boost_pct']:+.2f}% vs LCDM +0.00%)"
        )
        print(
            f"  [Test B] GC Compass   : {verdict['compass_split_pct']:+.2f}% +/- "
            f"{verdict['compass_split_err_pct']:.2f}% (TNaP {verdict['tnap_pred_compass_split_pct']:+.2f}% vs LCDM +0.00%)"
        )
        print(
            f"  [Test C] Env Step     : {verdict['env_step_halo_minus_inner_pct']:+.2f}% +/- "
            f"{verdict['env_step_err_pct']:.2f}% (TNaP {verdict['tnap_pred_env_step_pct']:+.2f}% vs LCDM +0.00%)"
        )
        print(
            f"  Combined Chi^2        : chi2_tnap={verdict['chi2_tnap']:.2f} | "
            f"chi2_lambda_cdm={verdict['chi2_lambda_cdm']:.2f} | "
            f"delta_chi2={verdict['delta_chi2']:+.2f} | log10_BF={verdict['log10_bayes_factor']:+.2f}"
        )
        print(f"  Day-One Blind Verdict : {verdict['verdict']}")
    return 0


def _cmd_fetch_sparc(_args: argparse.Namespace) -> int:
    from weaveforge.data.sparc_catalog import fetch_official_sparc_tables, load_sparc_175_catalog

    files = fetch_official_sparc_tables()
    gals, mode = load_sparc_175_catalog()
    n_pts = sum(len(g.r_kpc) for g in gals)
    print(f"[WeaveForge SPARC] Downloaded {[f.name for f in files]} -> {len(gals)} galaxies, {n_pts} radial points ({mode})")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or (raw[0].startswith("-") and raw[0] not in ("-h", "--help")):
        raw = ["studio", *raw]

    parser = argparse.ArgumentParser(prog="weaveforge", description="WeaveForge Multi-Scale 3D Physics Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_studio = sub.add_parser("studio", help="Launch Interactive 60-120+ FPS 3D Studio")
    p_studio.add_argument("--arena", type=int, default=1, choices=[1, 2, 3, 4], help="Initial 3D Arena (1-4)")
    p_studio.add_argument("--mod", type=str, default="tnap_weave", help="Initial PhysicsMod id")
    p_studio.add_argument("--width", type=int, default=1680)
    p_studio.add_argument("--height", type=int, default=960)
    p_studio.add_argument("--ui-scale", type=float, default=None, help="UI DPI scale factor (default: Auto 4K/DPI)")
    p_studio.add_argument(
        "--render-scale", type=float, default=None, help="3D internal FBO render scale 0.5-1.0 (default: Auto)"
    )
    p_studio.add_argument(
        "--fps-cap", type=int, default=None, help="Max FPS cap, e.g. 60, 120, 240, or 0 for uncapped (default: Auto)"
    )
    p_studio.add_argument("--no-vsync", action="store_true")
    p_studio.add_argument("--hidden", action="store_true", help="Create hidden GLFW window for automated QA")
    p_studio.add_argument("--headless-frames", type=int, default=None, help="Run N frames then exit cleanly")
    p_studio.add_argument("--screenshot-out", type=str, default=None, help="Save full Studio PNG before exit")
    p_studio.set_defaults(func=_cmd_studio)

    p_worker = sub.add_parser("worker", help="Launch Rig 2 Cross-Platform Vulkan/DX12 wgpu LAN Worker")
    p_worker.add_argument("--host", type=str, default="0.0.0.0")
    p_worker.add_argument("--port", type=int, default=8765)
    p_worker.set_defaults(func=_cmd_worker)

    p_audit = sub.add_parser("audit", help="Run CLI Symbolic Proofs + 5-Gate Health Audit + Crucible")
    p_audit.set_defaults(func=_cmd_audit)

    p_gaia = sub.add_parser("gaia-dr4", help="Run Dec 2, 2026 Gaia DR4 Wide-Binary |z_gal| Pipeline")
    p_gaia.add_argument("--csv", type=str, default=None, help="Path to Gaia DR4 wide-binary CSV")
    p_gaia.add_argument("--max-ruwe", type=float, default=1.20)
    p_gaia.add_argument("--mock-count", type=int, default=2400)
    p_gaia.add_argument("--truth", type=str, default="tnap_weave", choices=["tnap_weave", "lambda_cdm"])
    p_gaia.add_argument(
        "--day-one-verdict",
        action="store_true",
        help="Evaluate Zenodo 10.5281/zenodo.23919217 3-test blind falsification scorecard & verdict",
    )
    p_gaia.add_argument(
        "--export-raw-mock",
        type=str,
        default=None,
        help="Export realistic 3D Keplerian sky-level Gaia DR4 two-star astrometry CSV to path",
    )
    p_gaia.add_argument(
        "--no-contaminants",
        action="store_true",
        help="Disable ~12%% RUWE-triple / NSS / RV-flyby contaminant injection in raw mock export",
    )
    p_gaia.set_defaults(func=_cmd_gaia)

    p_sparc = sub.add_parser("fetch-sparc", help="Download/refresh official Lelli+ 2016 SPARC .mrt tables")
    p_sparc.set_defaults(func=_cmd_fetch_sparc)

    parsed = parser.parse_args(raw)
    return int(parsed.func(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
