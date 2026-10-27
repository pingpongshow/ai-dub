"""Bandwidth extension to 48 kHz with FlashSR (one-step diffusion).

The model works on fixed 5.12 s windows (245,760 samples at 48 kHz); longer
lines are processed in overlapping windows with a raised-cosine crossfade.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from aidub_worker import device, log, model_dir, read_mono, serve, torch_info

WIN = 245760
OVERLAP = 24000
state = {}


def _weights() -> Path:
    from huggingface_hub import snapshot_download
    d = Path(model_dir()) / "flashsr"
    if not (d / "student_ldm.pth").exists():
        snapshot_download("jakeoneijk/FlashSR_weights", repo_type="dataset", local_dir=str(d))
    return d


def load(args):
    from FlashSR.FlashSR import FlashSR
    d = _weights()
    state["m"] = FlashSR(str(d / "student_ldm.pth"), str(d / "sr_vocoder.pth"), str(d / "vae.pth")).to(device()).eval()
    return {"model": "FlashSR (jakeoneijk/FlashSR_weights)"}


def _run(x: np.ndarray) -> np.ndarray:
    import torch
    n = len(x)
    if n <= WIN:
        t = torch.from_numpy(np.pad(x, (0, WIN - n))).unsqueeze(0).to(device())
        with torch.no_grad():
            return state["m"](t, lowpass_input=True)[0, :n].float().cpu().numpy()
    hop = WIN - OVERLAP
    out = np.zeros(n + WIN, dtype=np.float32)
    weight = np.zeros(n + WIN, dtype=np.float32)
    fade = np.ones(WIN, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, OVERLAP, dtype=np.float32))
    fade[:OVERLAP], fade[-OVERLAP:] = ramp, ramp[::-1]
    for start in range(0, n, hop):
        seg = x[start:start + WIN]
        y = _run(seg) if len(seg) < WIN else _run_exact(seg)
        out[start:start + len(y)] += y * fade[:len(y)]
        weight[start:start + len(y)] += fade[:len(y)]
    return out[:n] / np.maximum(weight[:n], 1e-6)


def _run_exact(seg: np.ndarray) -> np.ndarray:
    import torch
    t = torch.from_numpy(seg).unsqueeze(0).to(device())
    with torch.no_grad():
        return state["m"](t, lowpass_input=True)[0].float().cpu().numpy()


def upsample(args):
    done = []
    for it in args["items"]:
        x = read_mono(it["input"], 48000)
        peak_in = float(np.max(np.abs(x))) or 1.0
        y = _run(x)
        rms_in = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
        rms_out = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
        y = y * (rms_in / rms_out)
        y = np.clip(y, -max(peak_in, 0.99), max(peak_in, 0.99))
        sf.write(it["output"], y.astype(np.float32), 48000, subtype="FLOAT")
        done.append(it["output"])
    log(f"upsampled {len(done)} lines")
    return {"done": done}


def prefetch(args):
    return {"path": str(_weights())}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "upsample": upsample, "prefetch": prefetch, "doctor": doctor})
