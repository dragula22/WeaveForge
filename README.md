# WeaveForge v1.0.0 — The Un-Gameable Physics Wind Tunnel

**Open-Source, Real-Time 60–120+ FPS 3D Moddable Action-to-GPU Physics Sandbox & 7-Dataset Observational Referee Engine**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23919217.svg)](https://doi.org/10.5281/zenodo.23919217)
[![Code License: AGPL v3](https://img.shields.io/badge/Code_License-GNU_AGPLv3-blue.svg)](LICENSE)
[![Docs & Visuals: CC BY 4.0](https://img.shields.io/badge/Docs_%26_Visuals-CC_BY_4.0-lightgrey.svg)](LICENSE-DOCS.md)

> *"Don't trust a physics theory because of a verbal story. Type its Master Cost Sheet (Lagrangian) into the compiler, let `SymPy` derive its exact 3D equations of motion, let the Red-Team Audit check it for ghosts and Solar-System violations, compile it straight into 3D GPU compute shaders, and watch it fight `Lambda-CDM`, `MOND`, `Superfluid Dark Matter`, `AeST`, and `TNaP Spacetime Weave` live against 175 galaxies, the CMB, black-hole shadows, and Gaia DR4 wide binaries."*

![WeaveForge Studio — Arena 1: 256³ Quantum & Topological Knot Tank](docs/screenshots/studio_arena1_knot_tank.png)

---

## Plain-English Overview: What Is `WeaveForge`?

Most physics simulations hardcode one set of rules and render pretty animations. **`WeaveForge`** works like a **game engine for the laws of the universe** paired with a **ruthless automated referee**:

1. **Moddable Laws of Physics (`mods/*.toml`)**: Switch with one click between **6 built-in competing frameworks** (`Standard Lambda-CDM`, `Classic MOND / QUMOND`, `Superfluid Dark Matter`, `AeST Relativistic MOND`, `TNaP Spacetime Weave`, and a `Red-Team Stress Trap` mod) — or write your own `.toml` physics mod in seconds.
2. **"One Master Cost Sheet" Symbolic-to-GPU Compiler (`SymPy` $\to$ `GLSL 430` + `WGSL`)**: Instead of hand-coding forces, you type a symbolic 3D field Lagrangian $\mathcal{L}$, an optical refractive metric $n(U)$, an independent gravitational lapse $A(U)$, an acceleration law $\nu(y)$, and a tensor wave speed $c_T/c$. `WeaveForge` symbolically differentiates the full 3D Euler–Lagrange PDEs (including mixed spatial Hessians and relativistic $P(X)$ velocity couplings) & Hamiltonian $H$, runs a **5-Gate Red-Team Health Audit** (Ostrogradsky/kinetic ghosts, 9-eigenvalue 3D hyperbolicity, `GW170817`, independent 1PN PPN $\gamma,\beta$, and Coleman false-vacuum / Planck-core stability), and compiles **both** native OpenGL 4.6 (`GLSL 430`) and cross-platform Vulkan/DX12/Metal (`WGSL`) 3D GPU compute kernels.
3. **4 Connected Multi-Scale 3D GPU Arenas (`WASD + Mouse` at `60–1,200+ FPS`)**:
   - **Arena `[1]` Quantum & Topological Knot Tank (`128³` / `256³` / `512³`)**: Volumetric 3D relativistic Skyrme–Faddeev triad + causal weave-tension wave tank (`Box U = -kappa rho_H` AND 5-channel spin-2 transverse-traceless `h_+ / h_x` tensor gravitational waves across OpenGL 4.6 + Vulkan `WGSL`), live virtual LIGO `(h_+, h_x)` strain oscilloscope, 3D GPU parallel Hamiltonian energy $H$ reduction, two-knot `1/r^2` gravity & binary trefoil inspiral, and Finkelstein–Rubinstein spin-$1/2$ fermionic Hopfion Pauli core repulsion (`Preset 7`).
   - **Arena `[2]` Strong-Gravity & Black-Hole Forge**: Real-time 3D Fermat optical-Hamiltonian (`SymPy` $\to$ `GLSL`) curved-geodesic raytracer comparing spinning Kerr vs. Weave black holes (`a/M` up to `0.998`), the `+4.63%` `BHEX` photon-ring shadow gauge (`5.437 r_g` vs `5.196 r_g`), 3D superfluid quantum phase-slip threads, the `1.000000` Gibbons Planck-tension fishing-reel spool, the `3,189.1 km` (`0.00 µs/day`) Earth orbital clock zero-crossing explorer, and **`Submode 3` (`Strong-Field BSSN Funnel & Spinning Kerr-Weave / LVK O4 Ringdown`)** pairing a 3D frame-dragged funnel & quadrupolar spiral ringdown crests with live `(2,2,0) / (2,2,1) / (3,3,0)` QNM spectroscopy across `a/M in [0, 0.95]`, `GW150914` free-`(M_f, a_f)` + IMR + hierarchical LVK O3b/O4 `TGR` audits, Gibbons Planck-spool echoes, and 20-field 3+1D BSSNok + Lichnerowicz strong-field telemetry.
   - **Arena `[3]` 3D Galaxy & Cluster Collider (`1,000,000`–`2,000,000` GPU Stars + 5-Level `64³` 3D Geometric Multigrid Poisson Solver)**: Simulates grand-design spirals (`NGC 3198`, `Milky Way`), low-acceleration dwarfs (`DDO 154`), any of the **175 real `SPARC` galaxies** via 1-click 3D spawning, the **`4,574 km/s` Bullet Cluster pass-through** (collisional crimson X-ray ICM bow-shock vs. moving dark-matter halos / ballistic pleats + $-\nabla\cdot\mathbf{g}_{\text{eff}}$ weak-lensing convergence peaks), and the **`14 km/s` Slow Dwarf Merger sideways wave-pleat splash**!
   - **Arena `[4]` Cosmic Hourglass & Un-Gameable 7-Row Observational Crucible**: Interactive 3D Two-Wing `CPT` Catenoid Hourglass ($R(\tau)=\sqrt{r_0^2+\tau^2}$, finite $K_{\max}=24/r_0^4 < \infty$, neutral `Observer Channel A` & `Observer Channel B` slices, Landauer $\Delta Q \ge k_B T \ln 2$ memory creases, and 3D Kibble–Zurek quench + Weyl chiral baryogenesis telemetry) paired with an **AST Free-Knob Auditor**, **Single-`H_0` Cross-Scale Continuity Gate (`NGC 3198` double-counting check)**, and live `ImPlot` scorecards across **all 7 empirical benchmarks**: (1) **SPARC-175 (`Lelli+ 2016` MRT)**, (2) **Planck 2018 CMB (`CAMB` `TT / TE / EE` $\ell=2\dots2500$ Knox cosmic-variance + detector-noise $\chi^2/\nu$)**, (3) **Primordial BBN (`D/H` & `Y_p`)**, (4) **3D Linear Growth & Weak-Lensing `S_8` (`DES Y3 / KiDS-1000`)**, (5) **Cassini/Saturn & GW170817**, (6) **13-point Anchored DESI DR2 (2025) BAO + `SH0ES` `H_0`**, and (7) **Dec 2, 2026 `Gaia DR4` Wide-Binary Day-One Raw Astrometry & RV Vetting**.

| Arena `[2]` Strong-Gravity & Black-Hole Forge | Arena `[3]` 1,000,000-Star 3D Galaxy Collider |
| :---: | :---: |
| ![Arena 2](docs/screenshots/studio_arena2_black_hole.png) | ![Arena 3](docs/screenshots/studio_arena3_galaxy_1m.png) |

---

## Ruthless Scientific Honesty (Zero Hardcoded Favoritism)

Every theory — including `Lambda-CDM` and `TNaP Spacetime Weave` — goes through the **exact same** symbolic differentiation and empirical scoring pipeline:

![Red-Team Ghost & PPN Rejection Demo](docs/screenshots/studio_arena4_redteam_audit.png)

- **`Classic MOND`** (`5` global + `0` halo knobs) passes 175 real SPARC galaxies (`median chi2_red = 1.63`, `RAR = 0.133 dex`), **automatically FAILS** `CAMB` Planck CMB (`Knox chi2/dof = 52.00`, `R_32 = 0.45`), Weak-Lensing `S_8` (`17.96` unscreened blowout), and `DESI DR2` BAO (`chi2/13 = 319.3`), and shows a `5.6x` smaller Bullet Cluster weak-lensing offset (`22 kpc` vs `123 kpc`).
- **`Lambda-CDM`** (`3` global + `350` per-galaxy NFW halo knobs) passes `CAMB` Planck 2018 CMB (`Knox chi2/dof = 1.000`, `R_32 = 0.99`), Primordial BBN (`1.04σ`), Solar-System & spinning Kerr `GW150914`/LVK O4 ringdown (`0.27σ` IMR, `0.33σ` O4 `TGR`), **earns `WARN` on SPARC** (`350` per-galaxy halo knobs), **`WARN` on `S_8`** (`0.830`, `3.16σ` vs `DES Y3 / KiDS-1000`), and **`WARN` on `DESI DR2 / H_0`** (`67.36` vs `SH0ES 73.04`), while predicting **`0.0%` wide-binary boost** at all `|z_gal|`.
- **`TNaP Spacetime Weave`** (`19` AST-audited global knobs + `0` per-galaxy halo knobs, with `a_0` locked to $c H_0 / (\sqrt{3}\pi)$ at single unified `H_0 = 67.36 km/s/Mpc`) screens Saturn to `< 10^-318`, clamps the CMB low-$\ell$ quadrupole (`D_2 = 325 uK^2`), stakes a cosmology-independent **`+15.4%` Disk $\to$ `+30.7%` Halo** wide-binary forecast on **ESA `Gaia DR4` (December 2, 2026)**, and **candidly surfaces every quantitative Red-Team seam live in the Wind Tunnel**:
  1. **CMB-to-Galaxy Cross-Scale Continuity (`NGC 3198` Seam)**: Populating $\Omega_{\text{pleat}} \approx 0.260$ in the CMB without an explicit halo screening scale (`pleat_screening_mpc = 0.0`) double-counts pleats + QUMOND in galaxies (`NGC 3198` outer rotation `256.1 km/s` vs `150 km/s`, `25.4σ` $\to$ `SPARC = WARN`); turning on `pleat_screening_mpc = 0.25 Mpc` closes the seam (`161.3 km/s` $\to$ `PASS`) at the cost of `+1` explicit AST-counted knob.
  2. **Single-`H_0` Planck Knox $\chi^2/\nu$ & Primordial BBN**: Pre-BBN pleats (`delta_neff_pleat = 0.259`, $N_{\text{eff}} = 3.303$) yield `Knox chi2/dof = 1.594` (`TE r = 0.9900`, `WARN`) and BBN `D/H = 2.601e-5, Y_p = 0.2506` (`2.87σ WARN`); switching to post-BBN pleats (`delta_neff_pleat = 0.0`) drops `Knox chi2/dof -> 1.027` (`TE r = 0.9995`), `BBN -> 1.04σ PASS`, and anchored `DESI DR2 BAO chi2/13 -> 0.997`.
  3. **3D Linear Growth & Weak-Lensing `S_8`**: Unscreened cosmic-web QUMOND blows up linear clustering to `S_8 = 22.46` (`FAIL`), whereas explicit Mpc screening (`pleat_screening_mpc = 0.25`) restores `S_8 = 0.825` (`2.87σ WARN`).
  4. **Spinning Kerr-Weave Ringdown vs. `GW150914` & LVK O4 (`Track 6B`)**: Because the unmodified classical `exp(2U)` catenoid throat stalls above $R_{\min} = e\,M = 2.718\,M$, frame dragging *deepens* the `(2,2,0)` shift from `-7.72%` (`a/M = 0`) to **`-16.46%`** at `GW150914` spin `a/M = 0.67` and caps the mass-independent quality factor at $Q_{220}^{\max} = 2.231$ vs `GW150914` $Q_{\text{obs}} = 3.154 \pm 0.257$ — rejecting the bare classical throat at **`3.72σ` even under 100% free `(M_f, a_f)`**, **`4.76σ` (`FAIL`)** under IMR consistency, and **`5.92σ` (`FAIL`)** across hierarchical LVK O3b/O4 `TGR` (proving a compact Gibbons Planck-spool / horizon phase transition inside $r < r_{\text{ph}}$ is required!).
- **`Red-Team Stress Trap`** (`-T_kin`, `n(U) = exp(U)` 1911 half-bending with `A(U) = exp(-2U)`, `c_T = 0.998 c`) is automatically flagged and rejected across Ostrogradsky ghosts, Hadamard blowup, `GW170817`, `Cassini` PPN $\gamma=0$, SPARC (`median chi2_red = 17.35`), `CAMB` CMB Peak 3, and `DESI DR2` BAO!

---

## Quick Start (1-Click or CLI)

### Option A: 1-Click Windows Launchers
- **Launch 3D Studio (Single-PC or Rig 1)**: Double-click **`Launch_WeaveForge_Studio.bat`**
- **Launch Rig 2 Vulkan LAN Worker (AMD / NVIDIA / Intel — Zero CUDA Needed)**: Double-click **`Launch_Rig2_LAN_Worker.bat`**
- **Run Full Symbolic Proofs + Pytest Suite**: Double-click **`Run_Test_And_QA_Suite.bat`**

### Option B: Command Line
```powershell
# Launch interactive 3D Studio (Auto 4K DPI, Auto 3D Render Scale, Auto 120Hz Pacing on >=165Hz displays)
.\.venv\Scripts\python.exe -m weaveforge studio

# Explicit 4K UI zoom, internal 3D viewport render scale, and FPS pacing override
.\.venv\Scripts\python.exe -m weaveforge studio --ui-scale 1.75 --render-scale 0.75 --fps-cap 120

# Launch directly into Arena 3 (1,000,000-Star Galaxy & Bullet Cluster Collider)
.\.venv\Scripts\python.exe -m weaveforge studio --arena 3

# Run CLI Symbolic Proofs + 5-Gate Health Audit + 6-Theory Crucible Table
.\.venv\Scripts\python.exe -m weaveforge audit

# Verify or refresh bundled published Lelli+ (2016) 175-galaxy SPARC MRT tables
.\.venv\Scripts\python.exe -m weaveforge fetch-sparc

# Run Dec 2, 2026 Gaia DR4 Day-One Raw Astrometry + RV Triple-Vetting & 3-Test Blind Verdict
.\.venv\Scripts\python.exe -m weaveforge gaia-dr4 --mock-count 4500 --truth tnap_weave --day-one-verdict
.\.venv\Scripts\python.exe -m weaveforge gaia-dr4 --csv path\to\gaia_dr4_raw_astrometry.csv --day-one-verdict
```

---

## Studio UX, 60/120 FPS NVENC Video Capture, 4K High-DPI & `32 GB DDR5` Architecture

`WeaveForge` is engineered for butter-smooth single-PC operation on 4K high-refresh displays while optionally scaling out over LAN:

- **Zero-Stall `60 / 120 FPS` `.mp4` Video Recorder & Screenshot Capture (`captures/`)**:
  - **Triple-PBO Async GPU Readback (`weaveforge/studio/video_recorder.py`)**: Click **`Rec 60fps`** in the Top Bar (or press **`F9`**) to record smooth H.264 `.mp4` video directly into `captures/` without stalling the 3D viewport. Uses a 3-slot OpenGL `GL_PIXEL_PACK_BUFFER` (`GL_STREAM_READ`) + `glFenceSync` DMA ring into a pre-allocated `DDR5` buffer pool and a background `ffmpeg` encoder thread (`h264_nvenc` hardware encoding on RTX GPUs with automatic `libx264` CPU fallback, plus optional GPU `glBlitFramebuffer` `1080p` downscaling and automatic even-dimension alignment).
  - **Instant PNG Snapshots (`F12` / `Snap PNG`)**: Captures full-resolution window screenshots (`captures/snap_*.png`) seamlessly — even mid-recording.
- **Intuitive Left-Sidebar "Reset Defaults" & Comprehensive Right-Click Context Menu**:
  - **1-Click & Per-Slider Reset**: Tab 2 (`Arena & Couplings`) features a live **`Reset Modified Sliders to Defaults (N changed)`** button, an inline header **`Reset`** button, and per-slider **`[R]`** micro-buttons (with default-value hover tooltips) that appear whenever any coupling slider or checkbox diverges from the active `.toml` mod's pristine defaults. Tab 1 (`Master Action`) provides **`Restore Default Mod Equations`** whenever symbolic equations are edited.
  - **Right-Click Quick-Actions Menu (`RMB` Click vs `RMB` Drag)**: A clean stationary right-click anywhere in the Studio opens **`WeaveForge Studio — Quick Actions`** (Pause/Resume, Step `+1` Frame, Reset Arena, Exit DDR5 Replay, Save PNG / Start-Stop `.mp4` Video & `60/120 FPS` + `Native/1080p` Settings, Open `captures/`, Reset Sliders / All Sidebar Settings, Switch Mod/Arena/Presets, Reset Camera, UI/3D Scale, FPS Cap, and DDR5 Hibernation) while preserving `Right-Mouse-Drag` 3D camera panning.
  - **Viewport Hotkeys** *(auto-disabled while typing equations)*: **`Space`** (Pause/Resume), **`R`** (Reset Active Arena), **`Home`** (Reset 3D Camera), **`F9`** (Start/Stop `.mp4` Video), **`F12`** (Save PNG Screenshot), **`1`–`4`** (Switch Arena), **`Ctrl + +` / `-` / `0`** (UI Zoom).
- **Crisp 4K High-DPI UI & Precision Frame Pacing**:
  - **`Auto` Per-Monitor V2 DPI Scaling (`>= 175%` on 4K)**: Automatically detects per-monitor content scale and resolution, with live **`100%–250%` UI zoom** (`Ctrl + +` / `Ctrl + -` / `Ctrl + 0` or Top Bar `UI` selector) that scales fonts, plots, and controls cleanly without compounding styles.
  - **Decoupled `3D Render Scale` (`50% Perf` / `75% Balanced` / `100% Native`)**: Renders heavy 3D ray-marched volumes and `1M–2M` star fields into an internal offscreen FBO (default `75%` on 4K) while keeping text, plots, and badges razor-sharp at native 4K resolution.
  - **Honest Post-Swap Telemetry & `CreateWaitableTimerExW` Pacing (`60 / 120 / 240 FPS / Uncapped`)**: Sub-millisecond Win32 high-resolution waitable timer pacing prevents needless 240Hz GPU saturation (`swap_interval(0)` on sub-refresh caps to avoid DWM VBlank-1 double-blocking), paired with live post-swap FPS / `ms`, `GDDR7 VRAM`, and `DDR5 RAM` telemetry.
- **`32 GB DDR5` System RAM Architecture (Zero VRAM Thrashing)**:
  - **Outgoing-First Inactive-Arena DDR5 Hibernation**: Switching tabs automatically offloads outgoing GPU arenas (`Arena 1` `256³`/`512³` fields & `Arena 3` `1M–2M` star SSBOs + Multigrid pyramid) into `DDR5` System RAM via non-demoting `glCopyBufferSubData` PCIe staging *before* waking the incoming arena (`0 MiB` inactive VRAM residency).
  - **`DDR5 Time-Machine` Phase-Space Replay Scrubber**: Streams GPU-packed `float16` phase-space keyframes (`(\phi, \dot{\phi})` in Arena 1, `(\mathbf{x}, \mathbf{v})` in Arena 3) into a bounded `DDR5` ring buffer for instant timeline scrubbing and lossless `float32` live restoration (`Resume Live`).
  - **DDR5-Cached `SymPy` & `CAMB` LUTs**: Memoizes symbolic variational templates, 400-digit Saturn screening, and asynchronous `CAMB` `CubicSpline` Planck CMB TT grids in `DDR5` RAM for `< 1 ms` live coupling slider updates.
- **Single-PC Zen 5 RDNA2 iGPU Ready + Dual-PC LAN Scale-Out**:
  - **Rig 1 (`Ryzen 9 9900X + 32 GB DDR5 + NVIDIA RTX 5080 16 GB`)**: Runs the zero-copy OpenGL 4.6 interactive 3D Studio. When **Rig 2 is powered off**, `Auto` (`< 160 ms` TCP failover) and `Single-PC Only` modes execute background `SPARC-175 + 3D Knot + Gaia DR4` sweeps in an isolated `spawn` process on the **Zen 5 multi-core CPU + integrated RDNA2 `AMD Radeon(TM) Graphics` Vulkan iGPU** backed by `DDR5` RAM — **`0 MiB` RTX 5080 VRAM impact** and zero viewport stutter!
  - **Rig 2 (`Ryzen 7 3700X + AMD Radeon RX 5700 XT 8 GB`, Optional)**: Runs `Launch_Rig2_LAN_Worker.bat` via **`wgpu` native Vulkan/DX12 (`WGSL`)** + `FastAPI`, exchanging compact `< 2 KB` `.toml` specs and `< 8 KB` sweep tables over Ethernet.

---

## How to Write & Share a Physics Mod (`mods/*.toml`)

Drop any `.toml` file into [`mods/`](mods/) (or edit live inside the Left Studio Panel and click **`Save Custom .toml`**):

```toml
id = "my_custom_gravity"
name = "My Custom Gravity Mod"
author = "@YourHandle"
summary = "Testing a true-vacuum Coleman Q-ball soliton & exponential optical metric"

[action]
field_lagrangian = "T_kin - c_w**2 * G_grad - 0.5 * m_eff**2 * rho + 0.25 * g4 * rho**2 - (1.0/6.0) * g6 * rho**3"
optical_metric_n = "exp(2 * U)"
metric_lapse_A   = "exp(-2 * U)"
interpolation_nu = "1.0 / (1.0 - exp(-sqrt(y)))"
gw_speed_ratio   = "1"

[parameters]
c_w = 1.0
m_eff = 2.50
g4 = 9.80
g6 = 3.10
a0_si = 1.2031e-10
has_cdm_halo = false
cdm_halo_scale = 0.0
waist_clamping_ell = 3.21
pinned_pleat_coupling = 1.0
w0_de = -0.91
wa_de = -0.28
```

See [`ROADMAP.md`](ROADMAP.md) for the full system architecture diagram and multi-phase open-source roadmap.

---

## Open-Science Dual Licensing (`AGPL-3.0` + `CC BY 4.0`)

Copyright &copy; 2026 **David Stoler** (*WeaveForge* / *Theoretically Not A Physicist* — [`stoler.systems`](https://stoler.systems) | [`@TNaPhysicist`](https://www.youtube.com/@TNaPhysicist)).

- **Source Code, Shaders & Simulation Engines ([`LICENSE`](LICENSE)):** Licensed under the **GNU Affero General Public License v3.0 (`AGPL-3.0-only`)**. Anyone is free to inspect, run, test, and modify the code—provided all modified or network-hosted versions **remain 100% open-source under `AGPL-3.0`** and preserve attribution to David Stoler ([`NOTICE`](NOTICE)). Closed-source or proprietary relicensing is strictly prohibited.
- **Documentation, Audit Reports, Screenshots & Visual Assets ([`LICENSE-DOCS.md`](LICENSE-DOCS.md)):** Licensed under **Creative Commons Attribution 4.0 International (`CC BY 4.0`)**—free to share, cite, and build upon with clear attribution to David Stoler (`https://stoler.systems` / [`https://github.com/dragula22/WeaveForge`](https://github.com/dragula22/WeaveForge) / DOI [`10.5281/zenodo.23919217`](https://doi.org/10.5281/zenodo.23919217)).
