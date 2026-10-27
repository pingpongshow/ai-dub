"""GPU availability: wait while another process is using the card."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class GpuState:
    present: bool
    name: str = ""
    total_mb: int = 0
    used_mb: int = 0
    processes: list[str] | None = None

    @property
    def free_mb(self) -> int:
        return max(0, self.total_mb - self.used_mb)

    def describe(self) -> str:
        if not self.present:
            return "no GPU visible"
        users = ", ".join(self.processes or []) or "other processes"
        return f"{self.used_mb} of {self.total_mb} MB in use by {users}"


def query() -> GpuState:
    if not shutil.which("nvidia-smi"):
        return GpuState(present=False)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        name, total, used = [x.strip() for x in out[0].split(",")]
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        procs = []
        for a in apps:
            parts = [x.strip() for x in a.split(",")]
            if len(parts) == 2:
                procs.append(f"{parts[0].rsplit('/', 1)[-1] or 'process'} ({parts[1]} MB)")
        return GpuState(True, name, int(float(total)), int(float(used)), procs)
    except Exception:
        return GpuState(present=False)


def wait_for(min_free_mb: int, *, poll_s: int, on_wait: Callable[[str], None],
             cancelled: Callable[[], bool], enabled: bool = True) -> None:
    """Block until `min_free_mb` of VRAM is free. Without a GPU (development), return at once."""
    if not enabled:
        return
    notified = False
    while True:
        st = query()
        if not st.present or st.free_mb >= min_free_mb:
            return
        if cancelled():
            return
        on_wait(f"waiting for GPU ({st.describe()}; need {min_free_mb} MB free)")
        notified = True
        time.sleep(poll_s)
        if notified and cancelled():
            return
