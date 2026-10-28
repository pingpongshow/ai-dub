"""Cinematic separation with Bandit v2 (speech / music / effects).

Dialogue = the speech stem; music & effects = music + effects stems. Long audio
is processed in slices with a crossfade (the handler keeps a slice on the GPU).
"""

import glob
import os
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from aidub_worker import device, log, model_dir, serve, torch_info

FS = 48000
STEMS = ["speech", "music", "sfx"]
URL = "https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt"
SLICE_S, OVERLAP_S = 180.0, 4.0
state = {}


def _src():
    for p in glob.glob(str(Path(sys.prefix) / "src" / "bandit-v2")):
        return p
    raise RuntimeError("bandit-v2 source not found in the environment")


def _ckpt() -> Path:
    d = Path(model_dir()) / "bandit"
    sd = d / "bandit_multi_state_dict.pt"
    if sd.exists():
        return sd
    d.mkdir(parents=True, exist_ok=True)
    raw = d / "checkpoint-multi.ckpt"
    if not raw.exists():
        import httpx
        log("downloading Bandit v2 weights")
        with httpx.stream("GET", URL, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            with open(str(raw) + ".part", "wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        os.replace(str(raw) + ".part", raw)
    import torch
    ck = torch.load(str(raw), map_location="cpu", weights_only=False)  # Lightning checkpoint from the authors
    torch.save({k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, sd)
    raw.unlink()
    return sd


def load(args):
    import torch
    from torch import nn
    sys.path.insert(0, _src())
    sys.modules.setdefault("pytorch_lightning", types.SimpleNamespace(LightningModule=nn.Module))
    try:  # torchaudio.io was removed in 2.9; stub it only when it's really missing
        import torchaudio.io  # noqa: F401
    except ImportError:
        tio = types.ModuleType("torchaudio.io")
        tio.StreamReader = None
        sys.modules["torchaudio.io"] = tio
    from src.models.bandit.bandit import Bandit
    from src.system.inference_handler import StandardTensorChunkedInferenceHandler
    model = Bandit(in_channels=1, stems=STEMS, fs=FS, band_type="musical", n_bands=64,
                   normalize_channel_independently=False, treat_channel_as_feature=True, n_sqm_modules=8,
                   emb_dim=128, rnn_dim=256, bidirectional=True, rnn_type="GRU", mlp_dim=512,
                   hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True,
                   use_freq_weights=True, n_fft=2048, win_length=2048, hop_length=512, window_fn="hann_window",
                   wkwargs=None, power=None, center=True, normalized=True, pad_mode="reflect", onesided=True)
    res = model.load_state_dict(torch.load(str(_ckpt()), map_location="cpu", weights_only=True), strict=False)
    if res.missing_keys:
        raise RuntimeError(f"Bandit weights missing keys: {res.missing_keys[:5]}")
    state["model"] = model.to(device()).eval()
    state["handler"] = StandardTensorChunkedInferenceHandler(chunk_size_seconds=8.0, hop_size_seconds=1.0,
                                                             inference_batch_size=8, fs=FS).to(device())
    return {"model": "Bandit v2 multi"}


def _run(x: np.ndarray) -> dict:
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x.T))[None].to(device())
    with torch.no_grad():
        out = state["handler"](t, state["model"])
    return {s: out["estimates"][s]["audio"][0].float().cpu().numpy().T[: len(x)] for s in STEMS}


def separate(args):
    import librosa
    x, sr = sf.read(args["input"], dtype="float32", always_2d=True)
    if sr != FS:
        x = librosa.resample(x.T, orig_sr=sr, target_sr=FS).T
    n, sl, ov = len(x), int(SLICE_S * FS), int(OVERLAP_S * FS)
    acc = {s: np.zeros_like(x) for s in STEMS}
    weight = np.zeros((n, 1), dtype=np.float32)
    start = 0
    while start < n:
        end = min(n, start + sl)
        part = _run(x[start:end])
        w = np.ones((end - start, 1), dtype=np.float32)
        if start > 0:
            w[:ov, 0] = np.linspace(0, 1, min(ov, end - start))
        if end < n:
            w[-ov:, 0] = np.linspace(1, 0, ov)
        for s in STEMS:
            acc[s][start:end] += part[s][: end - start] * w
        weight[start:end] += w
        if end == n:
            break
        start = end - ov
    out = Path(args["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    weight = np.maximum(weight, 1e-6)
    speech = acc["speech"] / weight
    mne = (acc["music"] + acc["sfx"]) / weight
    sf.write(out / "dialogue.wav", speech, FS, subtype="FLOAT")
    sf.write(out / "mne.wav", mne, FS, subtype="FLOAT")
    return {"vocals": str(out / "dialogue.wav"), "instrumental": str(out / "mne.wav")}


def prefetch(args):
    return {"path": str(_ckpt())}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "separate": separate, "prefetch": prefetch, "doctor": doctor})
