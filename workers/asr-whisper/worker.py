"""Whisper-family recognition (Whisper large-v3, anime-whisper, kotoba-whisper) for comparisons.

Same ops as the main recognizer. Distilled two-layer models ship alignment heads
copied from large-v3; they are corrected so word timings can work. When a model
gives no usable timings, results carry text only.
"""

from aidub_worker import device, log, read_mono, serve, torch_info

LANG = {"ja": "japanese", "ko": "korean", "zh": "chinese", "en": "english", "yue": "cantonese"}
state = {}


def load(args):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    mid = args.get("model") or "openai/whisper-large-v3"
    dtype = torch.float16 if device() == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(mid, torch_dtype=dtype, attn_implementation="eager")
    proc = AutoProcessor.from_pretrained(mid)
    if model.config.decoder_layers == 2:
        model.generation_config.alignment_heads = [[1, h] for h in range(model.config.decoder_attention_heads)]
    state["pipe"] = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                             feature_extractor=proc.feature_extractor, device=0 if device() == "cuda" else -1,
                             torch_dtype=dtype)
    state["model"] = mid
    return {"model": mid}


def transcribe(args):
    pipe = state["pipe"]
    lang = LANG.get(args.get("language") or "", None)
    if lang:
        pipe.tokenizer.set_prefix_tokens(language=lang, task="transcribe")
    gen = {"task": "transcribe", "num_beams": 1, "condition_on_prev_tokens": False}
    if lang:
        gen["language"] = lang
    out = []
    for it in args["items"]:
        audio = {"raw": read_mono(it["path"], 16000), "sampling_rate": 16000}
        words = []
        try:
            if args.get("timestamps", True):
                r = pipe(audio, return_timestamps="word", generate_kwargs=gen)
                for c in r.get("chunks", []):
                    a, b = c.get("timestamp") or (None, None)
                    if a is not None and c.get("text", "").strip():
                        words.append({"text": c["text"].strip(), "start": round(float(a), 3),
                                      "end": round(float(b if b is not None else a + 0.2), 3)})
            else:
                r = pipe(audio, generate_kwargs=gen)
        except Exception as exc:  # e.g. a model without working timestamps: text only
            log(f"timestamps failed ({exc}); text only")
            r = pipe(audio, generate_kwargs=gen)
            words = []
        out.append({"text": r.get("text", "").strip(), "language": args.get("language") or "", "words": words})
    return {"results": out}


def prefetch(args):
    from huggingface_hub import snapshot_download
    for m in ("openai/whisper-large-v3", "litagin/anime-whisper"):
        snapshot_download(m)
    return {}


def doctor(args):
    return torch_info()


if __name__ == "__main__":
    serve({"load": load, "transcribe": transcribe, "prefetch": prefetch, "doctor": doctor})
