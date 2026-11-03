"""
Moddable Physics Framework Schema & TOML Loader/Saver (`weaveforge/core/frameworks.py`)
=======================================================================================
Every physics theory in WeaveForge — whether consensus `Lambda-CDM`, `Classic MOND`,
`Superfluid Dark Matter`, `AeST`, `TNaP Spacetime Weave`, or a user's custom mod —
lives on an equal footing as a `PhysicsMod` ("One Master Cost Sheet").

No theory gets secret hardcoded `PASS` badges: every equation string inside `[action]`
is symbolically differentiated by `SymPy`, audited for ghosts/instabilities/PPN/GW170817,
compiled into native 3D GPU compute kernels (`GLSL 430` + `WGSL`), and scored live
against real empirical datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODS_DIR = REPO_ROOT / "mods"


def _infer_default_lapse(optical_metric_n: str) -> str:
    """Infers canonical isotropic temporal lapse `A(U) = -g_00` when omitted in legacy/minimal `.toml` files."""
    compact = optical_metric_n.replace(" ", "")
    if "1-U/2" in compact or "1-0.5*U" in compact:
        return "((1 - U/2) / (1 + U/2))**2"
    return "exp(-2 * U)"


@dataclass
class ActionSpec:
    """Symbolic expressions defining the Master Cost Sheet across scales."""

    # 3D relativistic triad/condensate Lagrangian density L(T_kin, G_grad, rho, ...)
    field_lagrangian: str = (
        "T_kin - c_w**2 * G_grad - 0.5 * m_eff**2 * rho "
        "+ 0.25 * g4 * rho**2 - (1.0/6.0) * g6 * rho**3"
    )
    # Effective optical metric refractive index n(U) = sqrt(B(U)/A(U)), where U = GM / (r * c^2)
    optical_metric_n: str = "exp(2 * U)"
    # Temporal clock lapse A(U) = -g_00(U) = 1 - 2*alpha*U + 2*beta*U^2 + O(U^3)
    metric_lapse_A: str = "exp(-2 * U)"
    # Low-acceleration gravitational interpolation function nu(y), where y = |grad Phi_N| / a0
    interpolation_nu: str = "1.0 / (1.0 - exp(-sqrt(y)))"
    # Tensor gravitational-wave speed ratio c_T / c (audited against GW170817 |c_T/c - 1| <= 3e-15)
    gw_speed_ratio: str = "1"


@dataclass
class ModParameters:
    """Numerical coupling constants & cosmological parameters for the active mod."""

    c_w: float = 1.0
    m_eff: float = 2.50
    g4: float = 9.80
    g6: float = 3.10
    a0_si: float = 1.2031e-10
    has_cdm_halo: bool = False
    cdm_halo_scale: float = 0.0
    waist_clamping_ell: float = 3.21
    pinned_pleat_coupling: float = 1.0
    w0_de: float = -0.91
    wa_de: float = -0.28


@dataclass
class PhysicsMod:
    """Complete, self-contained Moddable Physics Framework specification."""

    id: str
    name: str
    author: str
    summary: str
    action: ActionSpec = field(default_factory=ActionSpec)
    parameters: ModParameters = field(default_factory=ModParameters)
    source_path: Optional[Path] = None

    def clone(self, new_id: Optional[str] = None, new_name: Optional[str] = None) -> "PhysicsMod":
        return PhysicsMod(
            id=new_id or self.id,
            name=new_name or self.name,
            author=self.author,
            summary=self.summary,
            action=ActionSpec(**asdict(self.action)),
            parameters=ModParameters(**asdict(self.parameters)),
            source_path=self.source_path,
        )

    def to_toml_string(self) -> str:
        """Serializes this PhysicsMod cleanly into a human-editable `.toml` string."""
        def _esc(s: str) -> str:
            return (
                s.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\r", "\\r")
                .replace("\n", "\\n")
                .replace("\t", "\\t")
            )

        clean_hdr_name = " ".join(self.name.splitlines()).strip() or "Custom Mod"
        p = self.parameters
        a = self.action
        lines = [
            "# ==============================================================================",
            f"# WeaveForge Physics Mod: {clean_hdr_name}",
            "# ==============================================================================",
            f'id = "{_esc(self.id)}"',
            f'name = "{_esc(self.name)}"',
            f'author = "{_esc(self.author)}"',
            f'summary = "{_esc(self.summary)}"',
            "",
            "[action]",
            f'field_lagrangian = "{_esc(a.field_lagrangian)}"',
            f'optical_metric_n = "{_esc(a.optical_metric_n)}"',
            f'metric_lapse_A = "{_esc(a.metric_lapse_A)}"',
            f'interpolation_nu = "{_esc(a.interpolation_nu)}"',
            f'gw_speed_ratio = "{_esc(a.gw_speed_ratio)}"',
            "",
            "[parameters]",
            f"c_w = {float(p.c_w):.6g}",
            f"m_eff = {float(p.m_eff):.6g}",
            f"g4 = {float(p.g4):.6g}",
            f"g6 = {float(p.g6):.6g}",
            f"a0_si = {float(p.a0_si):.6e}",
            f"has_cdm_halo = {'true' if p.has_cdm_halo else 'false'}",
            f"cdm_halo_scale = {float(p.cdm_halo_scale):.6g}",
            f"waist_clamping_ell = {float(p.waist_clamping_ell):.6g}",
            f"pinned_pleat_coupling = {float(p.pinned_pleat_coupling):.6g}",
            f"w0_de = {float(p.w0_de):.6g}",
            f"wa_de = {float(p.wa_de):.6g}",
            "",
        ]
        return "\n".join(lines)

    def save_toml(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_toml_string(), encoding="utf-8")
        self.source_path = path
        return path

    @classmethod
    def from_toml_bytes(cls, raw_bytes: bytes, source_path: Optional[Path] = None) -> "PhysicsMod":
        data = tomllib.loads(raw_bytes.decode("utf-8"))
        act_d = data.get("action", {})
        par_d = data.get("parameters", {})
        opt_n = str(act_d.get("optical_metric_n", ActionSpec.optical_metric_n))
        lapse_a = str(act_d.get("metric_lapse_A", _infer_default_lapse(opt_n)))
        action = ActionSpec(
            field_lagrangian=str(act_d.get("field_lagrangian", ActionSpec.field_lagrangian)),
            optical_metric_n=opt_n,
            metric_lapse_A=lapse_a,
            interpolation_nu=str(act_d.get("interpolation_nu", ActionSpec.interpolation_nu)),
            gw_speed_ratio=str(act_d.get("gw_speed_ratio", ActionSpec.gw_speed_ratio)),
        )
        params = ModParameters(
            c_w=float(par_d.get("c_w", 1.0)),
            m_eff=float(par_d.get("m_eff", 2.50)),
            g4=float(par_d.get("g4", 0.0)),
            g6=float(par_d.get("g6", 0.0)),
            a0_si=float(par_d.get("a0_si", 1.2031e-10)),
            has_cdm_halo=bool(par_d.get("has_cdm_halo", False)),
            cdm_halo_scale=float(par_d.get("cdm_halo_scale", 0.0)),
            waist_clamping_ell=float(par_d.get("waist_clamping_ell", 0.0)),
            pinned_pleat_coupling=float(par_d.get("pinned_pleat_coupling", 0.0)),
            w0_de=float(par_d.get("w0_de", -1.0)),
            wa_de=float(par_d.get("wa_de", 0.0)),
        )
        return cls(
            id=str(data.get("id", "custom_mod")),
            name=str(data.get("name", "Custom Modded Physics")),
            author=str(data.get("author", "Community Modder")),
            summary=str(data.get("summary", "Custom symbolic Lagrangian action.")),
            action=action,
            parameters=params,
            source_path=source_path,
        )

    @classmethod
    def from_toml_file(cls, path: Path) -> "PhysicsMod":
        return cls.from_toml_bytes(path.read_bytes(), source_path=path)


def load_all_mods(mods_dir: Path = DEFAULT_MODS_DIR) -> List[PhysicsMod]:
    """Loads all `.toml` physics mods sorted by filename, with safe fallback presets."""
    mods: List[PhysicsMod] = []
    if mods_dir.exists():
        for p in sorted(mods_dir.glob("*.toml")):
            try:
                mods.append(PhysicsMod.from_toml_file(p))
            except Exception as exc:
                print(f"[WeaveForge] Warning: skipped malformed mod {p.name}: {exc}")
    if not mods:
        mods.append(PhysicsMod(
            id="tnap_weave",
            name="TNaP Spacetime Weave (Default Fallback)",
            author="WeaveForge Built-in",
            summary="Default zero-knob spacetime weave model.",
        ))
    return mods


def get_mods_by_id(mods_dir: Path = DEFAULT_MODS_DIR) -> Dict[str, PhysicsMod]:
    return {m.id: m for m in load_all_mods(mods_dir)}
