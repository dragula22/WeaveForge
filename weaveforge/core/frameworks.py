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

import ast
from dataclasses import asdict, dataclass, field
import math
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
    k_skyrme: float = 0.0
    g_knot: float = 0.0
    retarded_weave_u: bool = True
    a0_si: float = 1.202712e-10
    has_cdm_halo: bool = False
    cdm_halo_scale: float = 0.0
    waist_clamping_ell: float = 3.21
    pinned_pleat_coupling: float = 1.0
    w0_de: float = -0.91
    wa_de: float = -0.28
    h0_kms_mpc: float = 67.36
    delta_neff_pleat: float = 0.0
    pleat_screening_mpc: float = 0.0


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
            f"k_skyrme = {float(p.k_skyrme):.6g}",
            f"g_knot = {float(p.g_knot):.6g}",
            f"retarded_weave_u = {'true' if p.retarded_weave_u else 'false'}",
            f"a0_si = {float(p.a0_si):.6e}",
            f"has_cdm_halo = {'true' if p.has_cdm_halo else 'false'}",
            f"cdm_halo_scale = {float(p.cdm_halo_scale):.6g}",
            f"waist_clamping_ell = {float(p.waist_clamping_ell):.6g}",
            f"pinned_pleat_coupling = {float(p.pinned_pleat_coupling):.6g}",
            f"w0_de = {float(p.w0_de):.6g}",
            f"wa_de = {float(p.wa_de):.6g}",
            f"h0_kms_mpc = {float(p.h0_kms_mpc):.6g}",
            f"delta_neff_pleat = {float(p.delta_neff_pleat):.6g}",
            f"pleat_screening_mpc = {float(p.pleat_screening_mpc):.6g}",
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
            k_skyrme=float(par_d.get("k_skyrme", 0.0)),
            g_knot=float(par_d.get("g_knot", 0.0)),
            retarded_weave_u=bool(par_d.get("retarded_weave_u", True)),
            a0_si=float(par_d.get("a0_si", 1.202712e-10)),
            has_cdm_halo=bool(par_d.get("has_cdm_halo", False)),
            cdm_halo_scale=float(par_d.get("cdm_halo_scale", 0.0)),
            waist_clamping_ell=float(par_d.get("waist_clamping_ell", 0.0)),
            pinned_pleat_coupling=float(par_d.get("pinned_pleat_coupling", 0.0)),
            w0_de=float(par_d.get("w0_de", -1.0)),
            wa_de=float(par_d.get("wa_de", 0.0)),
            h0_kms_mpc=float(par_d.get("h0_kms_mpc", 67.36)),
            delta_neff_pleat=float(par_d.get("delta_neff_pleat", 0.0)),
            pleat_screening_mpc=float(par_d.get("pleat_screening_mpc", 0.0)),
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


_CANONICAL_ACTION_NUMBERS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
_C_M_S_KNOB: float = 299792458.0
_MPC_M_KNOB: float = 3.0856775814913673e22
_SAFE_NU_MATH: Dict[str, object] = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "abs": abs,
    "pi": math.pi,
    "e": math.e,
}


def _extract_expr_numeric_literals(expr_str: str) -> List[float]:
    """Extracts signed numeric `ast.Constant` literals from a symbolic expression via `ast.parse`."""
    s = str(expr_str or "").strip()
    if not s:
        return []
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return []

    found: List[float] = []

    def _visit(node: ast.AST) -> None:
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
        ):
            raw = float(node.operand.value)
            found.append(-raw if isinstance(node.op, ast.USub) else raw)
            return
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            found.append(float(node.value))
            return
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree.body)
    return found


def _is_interpolation_nu_active(nu_expr: str) -> bool:
    """Returns `True` when `interpolation_nu` depends non-trivially on `y` (`nu(y) != 1`)."""
    s = str(nu_expr or "").strip()
    if not s:
        return False
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return False
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if "y" not in names:
        return False
    try:
        code = compile(tree, "<interpolation_nu>", "eval")
        for y_sample in (0.05, 0.50, 2.50):
            env = dict(_SAFE_NU_MATH)
            env["y"] = y_sample
            val = float(eval(code, {"__builtins__": {}}, env))  # noqa: S307
            if math.isfinite(val) and abs(val - 1.0) > 1.0e-9:
                return True
        return False
    except Exception:
        return True


def count_mod_free_knobs(mod: PhysicsMod) -> Dict[str, object]:
    """
    Transparently audits active free parameters and non-canonical AST tuning constants
    across `[action]` and `[parameters]` for any `PhysicsMod` without hardcoded ID tables.
    """
    act = mod.action
    p = mod.parameters

    action_tuning_literals: List[float] = []
    seen_literals: set[float] = set()
    for field_name in (
        "field_lagrangian",
        "optical_metric_n",
        "metric_lapse_A",
        "interpolation_nu",
        "gw_speed_ratio",
    ):
        expr_val = getattr(act, field_name, "")
        for raw_num in _extract_expr_numeric_literals(expr_val):
            mag = abs(float(raw_num))
            if not math.isfinite(mag):
                continue
            if any(math.isclose(mag, c_num, rel_tol=0.0, abs_tol=1.0e-12) for c_num in _CANONICAL_ACTION_NUMBERS):
                continue
            key = round(mag, 12)
            if key not in seen_literals:
                seen_literals.add(key)
                action_tuning_literals.append(mag)

    pde_coupling_knobs: List[str] = []
    if abs(float(p.c_w) - 1.0) > 1.0e-9:
        pde_coupling_knobs.append("c_w")
    if abs(float(p.m_eff)) > 1.0e-12:
        pde_coupling_knobs.append("m_eff")
    if abs(float(p.g4)) > 1.0e-12:
        pde_coupling_knobs.append("g4")
    if abs(float(p.g6)) > 1.0e-12:
        pde_coupling_knobs.append("g6")
    if abs(float(p.k_skyrme)) > 1.0e-12:
        pde_coupling_knobs.append("k_skyrme")
    if abs(float(p.g_knot)) > 1.0e-12:
        pde_coupling_knobs.append("g_knot")

    cosmo_knobs: List[str] = []
    if float(p.h0_kms_mpc) > 0.0:
        cosmo_knobs.append("h0_kms_mpc")

    if bool(p.has_cdm_halo):
        cosmo_knobs.append("cdm_halo_scale")
        per_galaxy_halo_knobs = 350
    else:
        per_galaxy_halo_knobs = 0

    nu_active = _is_interpolation_nu_active(act.interpolation_nu)
    a0_unruh_si = (
        _C_M_S_KNOB
        * (max(float(p.h0_kms_mpc), 0.0) * 1000.0 / _MPC_M_KNOB)
        / (math.sqrt(3.0) * math.pi)
    )
    a0_val = float(p.a0_si)
    a0_locked_to_h0 = bool(
        nu_active
        and a0_val > 0.0
        and (abs(a0_val - a0_unruh_si) / max(abs(a0_val), 1.0e-30) <= 0.002)
    )
    if nu_active and not a0_locked_to_h0:
        cosmo_knobs.append("a0_si")

    ppc = float(p.pinned_pleat_coupling)
    if abs(ppc) > 1.0e-9 and abs(ppc - 1.0) > 1.0e-9:
        cosmo_knobs.append("pinned_pleat_coupling")
    if float(p.waist_clamping_ell) > 1.0e-9:
        cosmo_knobs.append("waist_clamping_ell")
    if abs(float(p.delta_neff_pleat)) > 1.0e-9:
        cosmo_knobs.append("delta_neff_pleat")
    if float(p.pleat_screening_mpc) > 1.0e-9:
        cosmo_knobs.append("pleat_screening_mpc")
    if abs(float(p.w0_de) - (-1.0)) > 1.0e-9:
        cosmo_knobs.append("w0_de")
    if abs(float(p.wa_de)) > 1.0e-9:
        cosmo_knobs.append("wa_de")

    act_cnt = len(action_tuning_literals)
    pde_cnt = len(pde_coupling_knobs)
    cos_cnt = len(cosmo_knobs)
    global_total = act_cnt + pde_cnt + cos_cnt

    lock_tag = ", a0=cH0/(sqrt(3)pi)" if a0_locked_to_h0 else ""
    summary_label = (
        f"{global_total} global knobs ({act_cnt} action, {pde_cnt} PDE, {cos_cnt} cosmo{lock_tag}) "
        f"+ {per_galaxy_halo_knobs} galaxy-halo knobs"
    )

    return {
        "action_tuning_literals": action_tuning_literals,
        "action_literal_count": act_cnt,
        "pde_coupling_knobs": pde_coupling_knobs,
        "pde_knob_count": pde_cnt,
        "cosmology_and_halo_knobs": cosmo_knobs,
        "cosmo_knob_count": cos_cnt,
        "global_knob_total": global_total,
        "per_galaxy_halo_knobs": per_galaxy_halo_knobs,
        "a0_locked_to_h0": a0_locked_to_h0,
        "summary_label": summary_label,
    }
