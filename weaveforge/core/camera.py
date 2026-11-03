"""
60+ FPS Interactive 3D Camera (`weaveforge/core/camera.py`)
===========================================================
Supports both Game-Engine Free-Fly (`WASD + Q/E + Shift + Mouse Look`) and
Turntable Orbit (`Mouse Drag + Scroll Zoom + Pan`) across all 4 3D GPU Arenas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Tuple

import numpy as np


@dataclass
class Camera3D:
    """Interactive 3D camera generating exact view, projection, and ray-marching basis vectors."""

    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    yaw: float = 0.65          # Horizontal angle [rad]
    pitch: float = 0.42        # Elevation angle [rad] in [-1.52, +1.52]
    distance: float = 4.6      # Orbit distance from target
    fov_deg: float = 46.0
    near: float = 0.05
    far: float = 500.0
    free_fly: bool = False
    move_speed: float = 3.5

    def reset(
        self,
        yaw: float = 0.65,
        pitch: float = 0.42,
        distance: float = 4.6,
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        fov_deg: float = 46.0,
    ) -> None:
        self.yaw = float(yaw)
        self.pitch = float(np.clip(pitch, -1.52, 1.52))
        self.distance = max(0.25, float(distance))
        self.target = np.array(target, dtype=np.float32)
        self.fov_deg = float(fov_deg)

    def basis_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns `(eye, forward, right, up)` unit vectors in world space (`Z`-up or `Y`-up consistent)."""
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        # Spherical offset around target (Y-up studio convention)
        offset = np.array([cp * sy, sp, cp * cy], dtype=np.float32)
        eye = self.target + self.distance * offset
        forward = -offset
        norm_f = float(np.linalg.norm(forward))
        if norm_f > 1e-8:
            forward /= norm_f

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        norm_r = float(np.linalg.norm(right))
        if norm_r < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= norm_r

        up = np.cross(right, forward)
        up /= max(float(np.linalg.norm(up)), 1e-8)
        return eye, forward, right, up

    def orbit_drag(
        self,
        dx_px: float,
        dy_px: float,
        sensitivity: float = 0.0055,
        dpi_scale: float = 1.0,
    ) -> None:
        dx = float(dx_px)
        dy = float(dy_px)
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return
        inv_dpi = 1.0 / max(float(dpi_scale) if math.isfinite(float(dpi_scale)) else 1.0, 0.25)
        eff_sens = float(sensitivity) * inv_dpi
        self.yaw -= dx * eff_sens
        self.pitch = float(np.clip(self.pitch + dy * eff_sens, -1.52, 1.52))

    def pan_drag(
        self,
        dx_px: float,
        dy_px: float,
        sensitivity: float = 0.0018,
        dpi_scale: float = 1.0,
    ) -> None:
        dx = float(dx_px)
        dy = float(dy_px)
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return
        inv_dpi = 1.0 / max(float(dpi_scale) if math.isfinite(float(dpi_scale)) else 1.0, 0.25)
        _, _, right, up = self.basis_vectors()
        scale = float(sensitivity) * self.distance * inv_dpi
        self.target = self.target - (dx * scale) * right + (dy * scale) * up

    def zoom_scroll(self, scroll_y: float) -> None:
        sy = float(scroll_y)
        if not math.isfinite(sy):
            return
        sy_clamped = float(np.clip(sy, -12.0, 12.0))
        factor = math.exp(-0.11 * sy_clamped)
        self.distance = float(np.clip(self.distance * factor, 0.30, 180.0))

    def process_wasd(
        self,
        dt: float,
        w: bool = False,
        s: bool = False,
        a: bool = False,
        d: bool = False,
        q: bool = False,
        e: bool = False,
        sprint: bool = False,
    ) -> None:
        if not (w or s or a or d or q or e):
            return
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f <= 0.0:
            return
        dt_safe = min(dt_f, 0.20)
        _, forward, right, _ = self.basis_vectors()
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        if w:
            vel += forward
        if s:
            vel -= forward
        if d:
            vel += right
        if a:
            vel -= right
        if e:
            vel += world_up
        if q:
            vel -= world_up
        n = float(np.linalg.norm(vel))
        if n > 1e-6:
            vel /= n
            speed = self.move_speed * (0.35 * self.distance + 0.5) * (2.8 if sprint else 1.0)
            self.target = self.target + vel * float(speed * dt_safe)

    def view_matrix(self) -> np.ndarray:
        """Returns 4x4 row-major float32 LookAt view matrix (transpose for GLSL column-major upload)."""
        eye, forward, right, up = self.basis_vectors()
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = right
        m[0, 3] = -float(np.dot(right, eye))
        m[1, :3] = up
        m[1, 3] = -float(np.dot(up, eye))
        m[2, :3] = -forward
        m[2, 3] = float(np.dot(forward, eye))
        return m

    def projection_matrix(self, aspect: float) -> np.ndarray:
        """Returns 4x4 row-major float32 OpenGL perspective matrix."""
        aspect = max(float(aspect), 1e-3)
        f = 1.0 / math.tan(math.radians(self.fov_deg) * 0.5)
        nf = 1.0 / (self.near - self.far)
        p = np.zeros((4, 4), dtype=np.float32)
        p[0, 0] = f / aspect
        p[1, 1] = f
        p[2, 2] = (self.far + self.near) * nf
        p[2, 3] = (2.0 * self.far * self.near) * nf
        p[3, 2] = -1.0
        return p

    def view_proj_gl_bytes(self, aspect: float) -> bytes:
        """Returns column-major float32 `P @ V` bytes ready for `moderngl` uniform `mat4`."""
        vp = self.projection_matrix(aspect) @ self.view_matrix()
        return np.ascontiguousarray(vp.T, dtype=np.float32).tobytes()
