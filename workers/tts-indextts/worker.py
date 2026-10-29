"""Voice-cloned English speech with IndexTTS-2.5.

Speaker timbre from a reference clip, emotion from the original line (or an
8-value emotion vector), and a duration factor (>1 slower/longer, 0.5-2.0).
Speed settings per item: num_beams (candidates weighed while writing the speech
codes, model default 3) and diffusion_steps (refinement steps, model default 25).
Output: 22.05 kHz mono WAV.
"""

from pathlib import Path

from aidub_worker import device, log, model_dir, serve, torch_info

REPO = "IndexTeam/IndexTTS-2.5"
state = {}


def _weights(repo: str) -> Path:
    from huggingface_hub import snapshot_download
    target = Path(model_dir()) / repo.split("/")[-1].lower()
    if not (target / "config.yaml").exists():
        snapshot_download(repo, local_dir=str(target))
    return target


def load(args):
    from indextts.infer_v2_5 import IndexTTS2
    repo = args.get("model") or REPO
    d = _weights(repo)
    state["tts"] = IndexTTS2(cfg_path=str(d / "config.yaml"), model_dir=str(d), use_bf16=device() == "cuda",
                             use_cuda_kernel=False, use_qwen_emo=False)
    return {"model": repo, "duration_control": True, "emotion_ref": True, "emotion_vector": True,
            "sample_rate": 22050}


def _refinement_steps(tts) -> None:
    """The model fixes its refinement steps at 25 inside infer(); its refiner is
    called with the step count as the sixth argument, so a wrapper can set it."""
    cfm = tts.s2mel.models["cfm"]
    if getattr(cfm, "_aidub_inference", None):
        return
    cfm._aidub_inference = cfm.inference

    def inference(*a, **k):
        a = list(a)
        if state.get("steps") and len(a) > 5:
            a[5] = state["steps"]
        return cfm._aidub_inference(*a, **k)

    cfm.inference = inference


def synthesize(args):
    import torch
    tts = state["tts"]
    _refinement_steps(tts)
    done = []
    for it in args["items"]:
        out = Path(it["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        if it.get("seed") is not None:
            torch.manual_seed(int(it["seed"]))
        kw = dict(spk_audio_prompt=it["speaker_ref"], text=it["text"], output_path=str(out), lang="EN",
                  duration_factor=float(it.get("duration_factor", 1.0)), verbose=False)
        if it.get("num_beams"):
            kw["num_beams"] = int(it["num_beams"])
        state["steps"] = int(it.get("diffusion_steps") or 0)
        if it.get("emo_vector"):
            kw["emo_vector"] = list(it["emo_vector"])
            kw["emo_alpha"] = float(it.get("emo_alpha", 0.65))
        elif it.get("emotion_ref"):
            kw["emo_audio_prompt"] = it["emotion_ref"]
            kw["emo_alpha"] = float(it.get("emo_alpha", 0.65))
        try:
            tts.infer(**kw)
            done.append(str(out))
        except Exception as exc:
            log(f"synthesis failed for {out.name}: {exc}")
    return {"done": done}


def prefetch(args):
    return {"path": str(_weights(args.get("model") or REPO))}


def doctor(args):
    import indextts
    return {**torch_info(), "indextts": getattr(indextts, "__version__", "2.5")}


if __name__ == "__main__":
    serve({"load": load, "synthesize": synthesize, "prefetch": prefetch, "doctor": doctor})
