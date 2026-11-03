"""
Rig 2 Distributed LAN Compute Worker & Async Client (`weaveforge/compute/lan_worker.py`)
========================================================================================
Designed specifically for Dual-PC Hardware Arsenals:
  • Rig 1 (`Ryzen 9 9900X + NVIDIA RTX 5080`): Runs 60–120+ FPS 3D Studio without stutters.
  • Rig 2 (`Ryzen 7 3700X 16-Thread + AMD Radeon RX 5700 XT 8GB`): Runs this zero-CUDA
    Vulkan/`wgpu` + multi-core worker over home Ethernet (`http://<RIG2_IP>:8765`), chewing
    through all 175 SPARC galaxy IMF-prior fits, 2D `(g4, g6)` 3D Trefoil knot stability maps,
    and Gaia DR4 wide-binary catalog sweeps!

Includes seamless local-thread fallback when Rig 2 is offline so every Studio button works
on a single machine out of the box!
"""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import queue
import socket
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI
import httpx
from pydantic import BaseModel, Field

from weaveforge.compute.wgpu_engine import WgpuComputeEngine
from weaveforge.core.frameworks import PhysicsMod
from weaveforge.core.symbolic_compiler import compile_master_action
from weaveforge.data.gaia_dr4_pipeline import (
    analyze_gaia_wide_binary_sample,
    generate_mock_gaia_dr4_catalog,
)
from weaveforge.data.sparc_catalog import load_sparc_175_catalog, score_full_sparc_175


class WorkerSweepRequest(BaseModel):
    toml_source: str
    g4_grid: List[float] = Field(default_factory=lambda: [0.0, 4.0, 7.5, 9.8, 12.0])
    g6_grid: List[float] = Field(default_factory=lambda: [1.0, 2.5, 3.5, 5.5])
    knot_grid_n: int = 48
    knot_steps: int = 75
    optimize_sparc_ml: bool = True
    gaia_mock_count: int = 2400
    gaia_mock_pairs: Optional[int] = None


def _is_tcp_endpoint_reachable(url: str, timeout_s: float = 0.15) -> bool:
    """Fast `< 160 ms` raw TCP reachability gate so powered-off LAN nodes never stall HTTP for `2+ s`."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not (1 <= int(port) <= 65535):
            return False
        with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
            return True
    except (OSError, ValueError):
        return False


def execute_crunch_job_sync(
    req: WorkerSweepRequest,
    engine: Optional[WgpuComputeEngine] = None,
    prefer_igpu_when_shared: bool = False,
) -> Dict[str, Any]:
    """Executes the full crunch bundle (wgpu 3D knot + phase sweep + 175 SPARC M/L fit + Gaia DR4)."""
    t0 = time.perf_counter()
    if engine is None:
        engine = WgpuComputeEngine(prefer_igpu_when_shared=prefer_igpu_when_shared)

    mod = PhysicsMod.from_toml_bytes(req.toml_source.encode("utf-8"))
    compiled = compile_master_action(mod)
    galaxies, sparc_mode = load_sparc_175_catalog()
    n_gaia = int(req.gaia_mock_pairs if req.gaia_mock_pairs is not None else req.gaia_mock_count)
    safe_grid_n = min(int(req.knot_grid_n), 256) if engine.is_shared_igpu_mode else int(req.knot_grid_n)

    # Run CPU-intensive 175-galaxy SPARC M/L optimization + Gaia DR4 Monte Carlo concurrently with wgpu sweep!
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 8, 16)) as pool:
        fut_sparc = pool.submit(
            score_full_sparc_175,
            galaxies,
            compiled.nu_numpy_fn,
            mod.parameters.a0_si,
            mod.parameters.has_cdm_halo,
            mod.parameters.cdm_halo_scale,
            req.optimize_sparc_ml,
        )
        fut_gaia = pool.submit(
            lambda: analyze_gaia_wide_binary_sample(
                generate_mock_gaia_dr4_catalog(
                    n_binaries=n_gaia,
                    truth_mode=mod.id,
                    a0_si=mod.parameters.a0_si,
                    nu_fn=compiled.nu_numpy_fn,
                )
            )
        )
        knot_single = engine.run_3d_knot_evolution(
            compiled,
            mod,
            grid_n=safe_grid_n,
            n_steps=int(req.knot_steps),
            preset="trefoil_knot",
        )
        knot_sweep = engine.run_knot_stability_phase_sweep(
            mod,
            g4_values=req.g4_grid,
            g6_values=req.g6_grid,
            grid_n=safe_grid_n,
            n_steps=int(req.knot_steps),
        )
        sparc_res = fut_sparc.result()
        gaia_res = fut_gaia.result()

    return {
        "status": "ok",
        "mod_id": mod.id,
        "mod_name": mod.name,
        "gpu_adapter": engine.adapter_summary,
        "worker_gpu": engine.adapter_summary,
        "cpu_threads": os.cpu_count() or 1,
        "sparc_source_mode": sparc_mode,
        "sparc_175": {
            "n_galaxies": sparc_res["n_galaxies"],
            "n_radial_points": sparc_res["n_radial_points"],
            "median_chi2_red": sparc_res["median_chi2_red"],
            "mean_chi2_red": sparc_res["mean_chi2_red"],
            "rar_rms_dex": sparc_res["rar_rms_dex"],
            "passed": sparc_res["passed"],
            "optimized_ml": bool(req.optimize_sparc_ml),
        },
        "knot_3d": {
            "grid_n": knot_single["grid_n"],
            "n_steps": knot_single["n_steps"],
            "retention_ratio": float(knot_single["retention_ratio"]),
            "init_l2_norm": float(knot_single["init_l2_norm"]),
            "final_l2_norm": float(knot_single["final_l2_norm"]),
            "elapsed_ms": float(knot_single["elapsed_ms"]),
        },
        "gaia_dr4_mock": gaia_res,
        "knot_phase_sweep": knot_sweep,
        "total_elapsed_ms": (time.perf_counter() - t0) * 1000.0,
    }


def _subprocess_crunch_entry(req_payload: Dict[str, Any], prefer_igpu: bool, out_q: Any) -> None:
    """Isolated `spawn` child-process worker: zero Python GIL contention and zero RTX 5080 VRAM impact!"""
    try:
        req = WorkerSweepRequest(**req_payload)
        res = execute_crunch_job_sync(req, engine=None, prefer_igpu_when_shared=bool(prefer_igpu))
        out_q.put(("ok", res))
    except Exception as exc:
        out_q.put(("err", f"{type(exc).__name__}: {exc}"))


def create_worker_app(
    prefer_discrete: bool = True,
    prefer_vendor: Optional[str] = None,
) -> FastAPI:
    """Builds the FastAPI LAN Worker service for Rig 2 (`AMD Ryzen 7 3700X + RX 5700 XT`)."""
    app = FastAPI(title="WeaveForge Rig 2 LAN Compute Worker", version="0.1.0")
    state: Dict[str, Any] = {"engine": None, "jobs_completed": 0}

    def _eng() -> WgpuComputeEngine:
        if state["engine"] is None:
            vendor = prefer_vendor
            if vendor is None and not prefer_discrete:
                vendor = "integrated"
            state["engine"] = WgpuComputeEngine(
                prefer_vendor=vendor,
                prefer_igpu_when_shared=(not prefer_discrete and vendor is None),
            )
        return state["engine"]

    @app.get("/health")
    def health() -> Dict[str, Any]:
        eng = _eng()
        return {
            "status": "online",
            "gpu_adapter": eng.adapter_summary,
            "cpu_threads": os.cpu_count() or 1,
            "jobs_completed": state["jobs_completed"],
        }

    @app.post("/crunch")
    def crunch(req: WorkerSweepRequest) -> Dict[str, Any]:
        res = execute_crunch_job_sync(req, engine=_eng())
        state["jobs_completed"] += 1
        return res

    return app


create_worker_fastapi_app = create_worker_app


def _normalize_worker_url(raw_url: str) -> str:
    """Normalizes user-entered LAN endpoints (accepting both `http://host:port` and bare `host:port`)."""
    u = str(raw_url or "").strip().rstrip("/")
    if not u:
        return ""
    if "://" not in u:
        u = f"http://{u}"
    return u


_VALID_EXECUTION_MODES = ("auto", "local", "local_igpu", "lan", "lan_only")


class AsyncLANWorkerDispatcher:
    """
    Non-blocking job dispatcher used inside Rig 1's Studio UI:
      • `execution_mode = "auto"`: `0.15s` raw TCP socket probe to `worker_url`; if Rig 2 is powered off,
        fails over in `< 160 ms` to an isolated `multiprocessing` (`spawn`) worker running on the
        Ryzen 9 9900X multi-core CPU + `AMD Radeon(TM) Graphics` Vulkan iGPU + DDR5 System RAM (`0 MiB` RTX VRAM)!
      • `execution_mode = "local"` / `"local_igpu"`: skips LAN probes entirely (`0 ms`) and runs in an
        isolated local `spawn` process targeting the integrated/secondary Vulkan adapter + DDR5 RAM.
      • `execution_mode = "lan"` / `"lan_only"`: probes remote LAN node only (`< 200 ms`) and reports
        clear offline status if unreachable.
    """

    def __init__(
        self,
        worker_url: str = "http://127.0.0.1:8765",
        execution_mode: str = "auto",
        prefer_igpu_on_local: bool = True,
        probe_timeout_s: float = 0.15,
    ) -> None:
        self.worker_url = _normalize_worker_url(worker_url)
        mode_norm = str(execution_mode).strip().lower()
        self.execution_mode: str = mode_norm if mode_norm in _VALID_EXECUTION_MODES else "auto"
        self.prefer_igpu_on_local: bool = True if self.execution_mode == "local_igpu" else bool(prefer_igpu_on_local)
        self.probe_timeout_s: float = float(max(min(probe_timeout_s, 1.0), 0.02))
        self.is_running: bool = False
        self.last_status: str = "Idle (Ready for Multi-Core / Vulkan iGPU or LAN Crunch)"
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_probe_ms: float = 0.0
        self._local_engine: Optional[WgpuComputeEngine] = None
        self._active_proc: Optional[Any] = None
        atexit.register(self.shutdown)

    def probe_lan_reachable(self, timeout_s: Optional[float] = None) -> bool:
        """Probes `self.worker_url` via raw TCP socket within `timeout_s` (`0.15s` default, `< 200 ms`)."""
        url_norm = _normalize_worker_url(self.worker_url)
        if not url_norm.startswith(("http://", "https://")):
            self.last_probe_ms = 0.0
            return False
        t0 = time.perf_counter()
        ok = _is_tcp_endpoint_reachable(
            url_norm,
            timeout_s=float(self.probe_timeout_s if timeout_s is None else timeout_s),
        )
        self.last_probe_ms = (time.perf_counter() - t0) * 1000.0
        return ok

    def shutdown(self) -> None:
        proc = self._active_proc
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)
            except Exception:
                pass
            self._active_proc = None

    def _run_local_isolated(self, req: WorkerSweepRequest, force_igpu: bool = False) -> Dict[str, Any]:
        """Runs local crunch in a `spawn` child process (zero GIL / zero RTX VRAM), with thread fallback."""
        igpu_flag = True if force_igpu else bool(self.prefer_igpu_on_local)
        target_lbl = "Local Multi-Core + Vulkan iGPU (DDR5)" if igpu_flag else "Local Multi-Core Vulkan Worker"
        self.last_status = f"Crunching on {target_lbl}..."

        out_q: Optional[Any] = None
        proc: Optional[Any] = None
        spawn_started = False
        msg: Optional[Any] = None
        child_pid: Optional[int] = None
        try:
            ctx_mp = mp.get_context("spawn")
            out_q = ctx_mp.Queue()
            proc = ctx_mp.Process(
                target=_subprocess_crunch_entry,
                args=(req.model_dump(), igpu_flag, out_q),
                daemon=True,
            )
            self._active_proc = proc
            proc.start()
            spawn_started = True
            child_pid = proc.pid

            deadline = time.perf_counter() + 120.0
            while time.perf_counter() < deadline:
                try:
                    msg = out_q.get(timeout=0.08)
                    break
                except queue.Empty:
                    if not proc.is_alive():
                        try:
                            msg = out_q.get_nowait()
                        except queue.Empty:
                            pass
                        break

            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=0.5)
        except Exception:
            if spawn_started:
                raise
        finally:
            self.shutdown()
            if out_q is not None:
                try:
                    out_q.close()
                    out_q.join_thread()
                except Exception:
                    pass

        if spawn_started:
            if msg is not None:
                kind, payload = msg
                if kind == "ok" and isinstance(payload, dict):
                    payload["execution_backend"] = "local_subprocess"
                    payload["worker_pid"] = child_pid
                    payload["prefer_igpu"] = bool(igpu_flag)
                    return payload
                raise RuntimeError(str(payload))
            exit_code = proc.exitcode if proc is not None else None
            raise RuntimeError(
                f"Isolated local worker process (pid={child_pid}) exited without returning a result (exitcode={exit_code})."
            )

        # Seamless fallback to background daemon thread ONLY if `spawn` startup is unavailable in constrained harnesses
        if self._local_engine is None or self._local_engine.is_shared_igpu_mode != igpu_flag:
            self._local_engine = WgpuComputeEngine(prefer_igpu_when_shared=igpu_flag)
        self.last_status = f"Crunching on Background Vulkan Worker ({self._local_engine.adapter_summary})..."
        res = execute_crunch_job_sync(req, engine=self._local_engine, prefer_igpu_when_shared=igpu_flag)
        res["execution_backend"] = "local_thread"
        res["worker_pid"] = os.getpid()
        res["prefer_igpu"] = bool(igpu_flag)
        return res

    def dispatch_async(
        self,
        mod: PhysicsMod,
        use_lan_if_online: Optional[bool] = None,
        execution_mode: Optional[str] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.is_running:
            return
        if execution_mode is not None:
            mode_in = str(execution_mode).strip().lower()
            self.execution_mode = mode_in if mode_in in _VALID_EXECUTION_MODES else "auto"
            if self.execution_mode == "local_igpu":
                self.prefer_igpu_on_local = True
        mode = str(self.execution_mode).strip().lower()
        if mode not in _VALID_EXECUTION_MODES:
            mode = "auto"
        if use_lan_if_online is False and mode == "auto":
            eff_mode = "local"
        else:
            eff_mode = mode

        url_norm = _normalize_worker_url(self.worker_url)
        if url_norm:
            self.worker_url = url_norm

        req_kwargs: Dict[str, Any] = {"toml_source": mod.to_toml_string()}
        if request_overrides:
            req_kwargs.update(request_overrides)
        req = WorkerSweepRequest(**req_kwargs)

        self.is_running = True
        self.last_status = f"Dispatching 175-SPARC + 3D Knot Phase Sweep for '{mod.name}' ({eff_mode.upper()})..."

        def _worker_thread() -> None:
            try:
                is_lan_only = eff_mode in ("lan", "lan_only")
                if eff_mode == "auto" or is_lan_only:
                    if url_norm.startswith(("http://", "https://")):
                        t_probe = time.perf_counter()
                        reachable = _is_tcp_endpoint_reachable(url_norm, timeout_s=self.probe_timeout_s)
                        self.last_probe_ms = (time.perf_counter() - t_probe) * 1000.0
                        if reachable:
                            try:
                                with httpx.Client(timeout=1.5) as probe:
                                    h = probe.get(f"{url_norm}/health")
                                    h.raise_for_status()
                                    gpu_lbl = h.json().get("gpu_adapter", "Remote LAN GPU")
                                self.last_status = f"Crunching on Remote LAN Node ({gpu_lbl})..."
                                with httpx.Client(timeout=120.0) as client:
                                    r = client.post(f"{url_norm}/crunch", json=req.model_dump())
                                    r.raise_for_status()
                                    res_json = r.json()
                                    res_json["execution_backend"] = "remote_lan"
                                    res_json["lan_probe_ms"] = self.last_probe_ms
                                    self.last_result = res_json
                                    self.last_status = (
                                        f"LAN Complete ({self.last_result['total_elapsed_ms']:.0f} ms on {gpu_lbl})"
                                    )
                                    return
                            except Exception as lan_exc:
                                if is_lan_only:
                                    self.last_status = f"LAN Node Offline ({url_norm}): {lan_exc}"
                                    return
                        elif is_lan_only:
                            self.last_status = (
                                f"LAN Node Offline ({url_norm}) -- switch mode to Auto or Local iGPU+DDR5"
                            )
                            return
                    elif is_lan_only:
                        self.last_probe_ms = 0.0
                        self.last_status = (
                            f"LAN Node Offline (invalid endpoint '{self.worker_url}') -- switch mode to Auto or Local iGPU+DDR5"
                        )
                        return
                else:
                    self.last_probe_ms = 0.0

                res = self._run_local_isolated(req, force_igpu=(eff_mode == "local_igpu"))
                res["lan_probe_ms"] = self.last_probe_ms
                self.last_result = res
                gpu_used = res.get("gpu_adapter", "Local Vulkan")
                self.last_status = f"Complete ({res['total_elapsed_ms']:.0f} ms on {gpu_used})"
            except Exception as exc:
                self.last_status = f"Worker Error: {exc}"
            finally:
                self.is_running = False

        try:
            threading.Thread(target=_worker_thread, daemon=True).start()
        except Exception:
            self.is_running = False
            raise


LanComputeDispatcher = AsyncLANWorkerDispatcher
