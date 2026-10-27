"""Runtime shared by every model worker.

A worker is a long-lived process started by Ai Dub for one pipeline stage. It
reads one JSON request per line on stdin and writes one JSON reply per line on
stdout. Anything else it prints goes to stderr (the job log).

    request:  {"id": 3, "op": "run", "args": {...}}
    reply:    {"id": 3, "ok": true, "result": {...}}
              {"id": 3, "ok": false, "error": "..."}
    progress: {"id": 3, "progress": 0.4, "message": "..."}

Every worker implements `load(args)` (called once) and any number of ops.
Audio and other large data move as file paths, never inside the JSON.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Callable

# The protocol gets a private copy of stdout; file descriptor 1 itself is pointed at
# stderr, so stray prints from libraries (and child processes they spawn) can't
# corrupt the protocol.
_out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr

_current_id: int | None = None


def progress(fraction: float, message: str = "") -> None:
    _send({"id": _current_id, "progress": round(float(fraction), 4), "message": message})


def log(*parts: Any) -> None:
    print(*parts, file=sys.stderr, flush=True)


def _send(obj: dict) -> None:
    _out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _out.flush()


def device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def model_dir() -> str:
    return os.environ.get("AIDUB_MODELS", "/models")


def serve(ops: dict[str, Callable[[dict], Any]]) -> None:
    """Main loop. `ops` maps an op name to a function taking the args dict."""
    global _current_id
    _send({"id": None, "ok": True, "result": {"ready": True, "pid": os.getpid()}})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _send({"id": None, "ok": False, "error": f"bad request: {exc}"})
            continue
        _current_id = req.get("id")
        op = req.get("op")
        if op == "exit":
            _send({"id": _current_id, "ok": True, "result": {}})
            break
        if op == "ping":
            _send({"id": _current_id, "ok": True, "result": {"pong": True}})
            continue
        fn = ops.get(op)
        if fn is None:
            _send({"id": _current_id, "ok": False, "error": f"unknown op {op!r}"})
            continue
        try:
            result = fn(req.get("args") or {})
            _send({"id": _current_id, "ok": True, "result": result if result is not None else {}})
        except Exception as exc:  # report, keep serving
            traceback.print_exc(file=sys.stderr)
            _send({"id": _current_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})


def read_mono(path: str, sr: int = 16000):
    """Load audio as mono float32 at `sr` (numpy array)."""
    import numpy as np
    import soundfile as sf
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if rate != sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=rate, target_sr=sr)
        except ImportError:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(rate, sr)
            data = resample_poly(data, sr // g, rate // g)
    return np.ascontiguousarray(data, dtype=np.float32)


def torch_info() -> dict:
    """Facts for `aidub doctor`: library versions and whether the GPU works."""
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            x = torch.randn(256, 256, device="cuda")
            info["matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
    except Exception as exc:  # report, don't crash doctor
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info
