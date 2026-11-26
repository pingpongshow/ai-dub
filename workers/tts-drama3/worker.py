"""House voices spoken by Fish Audio Drama 3, cloned from each character's designed voice.

Each line can carry a direction: the original line's emotion as an inline tag ([angry]).
Drama 3 (36B, mixture of experts) needs 42 GB with its experts in FP8; on a smaller card the
routed experts of the last `offload_layers` layers stay in pinned RAM and only the 8 experts
each step picks are copied over (32 GB card, 22 of 40 layers: 26.8 GB at peak, a quarter of
real time). No duration control. Output: mono WAV at the codec's sample rate.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("FISH_DRAMA3_CUDA_GRAPH", "0")  # experts fetched per step can't be graph-captured
os.environ.setdefault("TQDM_DISABLE", "1")             # a progress bar per step would flood the job's log

from aidub_worker import log, model_dir, serve, torch_info

REPO = "fishaudio/drama-3"
TAGS = {"ANGRY": "[angry]", "HAPPY": "[excited]", "SAD": "[sad]", "FEARFUL": "[scared]",
        "SURPRISED": "[surprised]", "DISGUSTED": "[disgusted]"}
PEAK = 0.95
state = {"prompts": {}}


def _src():
    src = Path(sys.prefix) / "src" / "fish-speech"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _weights(repo: str) -> Path:
    from huggingface_hub import snapshot_download
    target = Path(model_dir()) / "fish" / repo.split("/")[-1].lower()
    if not (target / "config.json").exists() or not (target / "codec.pth").exists():
        snapshot_download(repo, local_dir=str(target))
    return target


def _offload(layers: int) -> None:
    """Keep the routed experts of the last `layers` layers in RAM, fetching the ones each step uses."""
    import torch
    from fish_speech.models.text2semantic.fish_omni import backbone as B, moe as M
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward
    if getattr(B.FishOmniBackbone, "_aidub", False):
        return
    B.FishOmniBackbone._aidub = True
    cuda = torch.device("cuda:0")

    def materialize(self, devices):
        n = len(self.model.layers)
        off = set(range(max(0, n - layers), n))
        for m in (self.model.embed_tokens, self.model.norm, self.lm_head):
            m.to_empty(device=cuda)
        for i, layer in enumerate(self.model.layers):
            if i not in off:
                layer.to_empty(device=cuda)
                continue
            ex = layer.mlp._modules.pop("experts")
            layer.to_empty(device=cuda)
            ex.to_empty(device="cpu")
            ex.aidub_offloaded = True
            layer.mlp._modules["experts"] = ex
        self.inverse_frequencies.clear()
        if off:
            log(f"experts of layers {min(off)}-{max(off)} kept in RAM")

    keys = ("w13_weight", "w13_scale_inv", "w2_weight", "w2_scale_inv")
    first = M.MoEExperts.forward

    def forward(self, x, topk_ids, topk_weights):
        if not getattr(self, "aidub_offloaded", False):
            return first(self, x, topk_ids, topk_weights)
        if not getattr(self, "aidub_pinned", False):
            for k in keys:
                setattr(self, k, getattr(self, k).pin_memory())
            self.aidub_pinned = True
        ids = torch.unique(topk_ids)
        on_cpu = ids.cpu()
        pick = lambda w: M.select_fp8(w, on_cpu) if w.dtype == M.FP8_DTYPE else w[on_cpu]
        w13, s13, w2, s2 = (pick(getattr(self, k)).to(x.device, non_blocking=True) for k in keys)
        return fp8_moe_forward(x, torch.searchsorted(ids, topk_ids), topk_weights, w13, s13, w2, s2)

    B.FishOmniBackbone.materialize = materialize
    M.MoEExperts.forward = forward


def load(args):
    import torch
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")  # fish-speech logs every chunk it generates
    _src()
    from fish_speech.models.text2semantic import inference as I
    from fish_speech.models.text2semantic.fish_omni import model as FM
    d = _weights(args.get("model") or REPO)
    layers = int(args.get("offload_layers", 22))
    card_gb = torch.cuda.mem_get_info()[1] / 2 ** 30
    if card_gb >= 46:  # the whole model fits
        layers = 0
    _offload(layers)
    state["m"] = FM.FishOmniModel.from_pretrained(d, devices=["cuda:0"], tower_device="cuda:0", dtype=torch.bfloat16)
    state["codec"] = I.load_codec_model(I.resolve_codec_path(d, None), "cuda:0", state["m"].dtype)
    state["I"], state["prompts"] = I, {}
    return {"model": args.get("model") or REPO, "duration_control": False, "emotion_ref": False,
            "emotion_vector": False, "emotion_tags": True, "sample_rate": int(state["codec"].sample_rate),
            "offload_layers": layers}


def _prompt(ref: str):
    key = (ref, os.path.getmtime(ref))
    if key not in state["prompts"]:
        state["prompts"][key] = state["I"].encode_audio(ref, state["codec"], "cuda:0").cpu()
    return state["prompts"][key]


def synthesize(args):
    """Items: text, speaker_ref (a designed voice), speaker_text (its words), emotion (a line's
    emotion tag, optional), seed, out."""
    import numpy as np
    import soundfile as sf
    import torch
    I, done = state["I"], []
    for it in args["items"]:
        out = Path(it["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if it.get("seed") is not None:
                torch.manual_seed(int(it["seed"]))
            text = "<|speaker:0|>" + TAGS.get(str(it.get("emotion", "")).upper(), "") + it["text"]
            codes = [r.codes for r in I.generate_long(
                model=state["m"], device="cuda:0", decode_one_token=None, text=text,
                prompt_text=["<|speaker:0|>" + it.get("speaker_text", "")], prompt_tokens=[_prompt(it["speaker_ref"])])
                if r.action == "sample"]
            if not codes:
                raise RuntimeError("no speech generated")
            wav = I.decode_to_audio(torch.cat(codes, dim=1).to("cuda:0"), state["codec"]).cpu().float().numpy()
            peak = float(np.abs(wav).max()) if wav.size else 0.0
            sf.write(str(out), wav * (PEAK / peak) if peak > PEAK else wav, int(state["codec"].sample_rate))
            done.append(str(out))
        except Exception as exc:
            log(f"synthesis failed for {out.name}: {exc}")
    return {"done": done}


def prefetch(args):
    return {"path": str(_weights(args.get("model") or REPO))}


def doctor(args):
    _src()
    import fish_speech  # noqa: F401
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "synthesize": synthesize, "prefetch": prefetch, "doctor": doctor})
