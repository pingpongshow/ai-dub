"""Speech recognition with word timings: Qwen3-ASR + Qwen3-ForcedAligner.

Also used for the English back-check of dubbed takes and for force-aligning
subtitle text to the speech.
"""

from pathlib import Path

from aidub_worker import device, log, model_dir, read_mono, serve, torch_info

NAMES = {"ja": "Japanese", "ko": "Korean", "zh": "Chinese", "yue": "Cantonese", "en": "English", "es": "Spanish",
         "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
         "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "hi": "Hindi"}
CODES = {v.lower(): k for k, v in NAMES.items()}
state = {}


def _name(code):
    if not code:
        return None
    return NAMES.get(code.lower(), code[:1].upper() + code[1:].lower())


def _code(name):
    return CODES.get((name or "").lower(), (name or "").lower()[:2])


def load(args):
    import torch
    dtype = torch.bfloat16 if device() == "cuda" else torch.float32
    dmap = "cuda:0" if device() == "cuda" else "cpu"
    if args.get("aligner_only"):
        from qwen_asr import Qwen3ForcedAligner
        state["aligner"] = Qwen3ForcedAligner.from_pretrained(args.get("aligner") or "Qwen/Qwen3-ForcedAligner-0.6B",
                                                              dtype=dtype, device_map=dmap)
        return {"model": args.get("aligner")}
    from qwen_asr import Qwen3ASRModel
    aligner = args.get("aligner") or "Qwen/Qwen3-ForcedAligner-0.6B"
    state["asr"] = Qwen3ASRModel.from_pretrained(
        args.get("model") or "Qwen/Qwen3-ASR-1.7B", dtype=dtype, device_map=dmap,
        max_inference_batch_size=16, max_new_tokens=1024,
        forced_aligner=aligner, forced_aligner_kwargs=dict(dtype=dtype, device_map=dmap))
    state["model"] = args.get("model")
    return {"model": f"{args.get('model')} + {aligner}"}


def _words(stamps):
    out = []
    for it in stamps or []:
        t = str(getattr(it, "text", "")).strip()
        if t:
            out.append({"text": t, "start": round(float(it.start_time), 3), "end": round(float(it.end_time), 3)})
    return out


def transcribe(args):
    model = state["asr"]
    items = args["items"]
    audio = [(read_mono(it["path"], 16000), 16000) for it in items]
    ts = bool(args.get("timestamps", True))
    res = model.transcribe(audio=audio, language=_name(args.get("language")), return_time_stamps=ts)
    out = []
    for r in res:
        out.append({"text": r.text, "language": _code(r.language), "words": _words(r.time_stamps) if ts else []})
    return {"results": out}


def align(args):
    al = state.get("aligner")
    if al is None:
        al = state["asr"].forced_aligner if hasattr(state.get("asr"), "forced_aligner") else None
    if al is None:
        raise RuntimeError("aligner not loaded")
    lang = _name(args.get("language"))
    out = []
    for it in args["items"]:
        try:
            res = al.align(audio=(read_mono(it["path"], 16000), 16000), text=it["text"], language=lang)
            out.append({"words": _words(res[0])})
        except Exception as exc:
            log(f"align failed: {exc}")
            out.append({"words": []})
    return {"results": out}


LEXICON_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/"  # CMU Pronouncing Dictionary, BSD


def prefetch(args):
    from huggingface_hub import snapshot_download
    for repo in (args.get("model") or "Qwen/Qwen3-ASR-1.7B", args.get("aligner") or "Qwen/Qwen3-ForcedAligner-0.6B"):
        snapshot_download(repo)
    _lexicon()
    return {}


def _lexicon():
    """The pronouncing dictionary the back-check uses to tell a different spelling
    of the same sound from a mispronunciation (models/lexicon/)."""
    import urllib.request
    d = Path(model_dir()) / "lexicon"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("cmudict.dict", "LICENSE"):
        if not (d / name).exists():
            tmp = d / (name + ".part")
            urllib.request.urlretrieve(LEXICON_URL + name, tmp)
            tmp.replace(d / name)
            log(f"downloaded {name} (CMU Pronouncing Dictionary)")


def doctor(args):
    import qwen_asr
    return {**torch_info(), "qwen_asr": getattr(qwen_asr, "__version__", "?")}


if __name__ == "__main__":
    serve({"load": load, "transcribe": transcribe, "align": align, "prefetch": prefetch, "doctor": doctor})
