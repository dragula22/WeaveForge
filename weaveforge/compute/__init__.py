"""Cross-platform wgpu (Vulkan/DX12/Metal) compute engine & Rig 2 LAN worker."""

from weaveforge.compute.lan_worker import AsyncLANWorkerDispatcher, LanComputeDispatcher
from weaveforge.compute.wgpu_engine import WgpuComputeEngine

__all__ = [
    "AsyncLANWorkerDispatcher",
    "LanComputeDispatcher",
    "WgpuComputeEngine",
]
