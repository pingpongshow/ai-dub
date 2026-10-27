"""Local translation LLM with vLLM (offline engine, no server, no network).

Default model: a 4-bit build of Qwen3.8-27B that fits one 32 GB card with
CUDA graphs off (--enforce-eager), text-only, thinking switched off.
"""

from aidub_worker import log, serve

state = {}


def load(args):
    from vllm import LLM
    name = args.get("model") or "RedHatAI/Qwen3.8-27B-MXFP4"
    kw = dict(model=name, gpu_memory_utilization=float(args.get("gpu_memory_utilization", 0.80)),
              max_model_len=int(args.get("max_model_len", 16384)), max_num_seqs=int(args.get("max_num_seqs", 8)),
              enforce_eager=True, kv_cache_dtype="fp8", trust_remote_code=False)
    try:
        state["llm"] = LLM(language_model_only=True, **kw)
    except TypeError:
        state["llm"] = LLM(**kw)  # model or vLLM without the vision/text split
    state["name"] = name
    log(f"loaded {name}")
    return {"model": name}


def chat(args):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=float(args.get("temperature", 0.3)), top_p=0.8, top_k=20,
                        max_tokens=int(args.get("max_tokens", 4096)), presence_penalty=0.5)
    outs = state["llm"].chat(args["conversations"], sampling_params=sp, use_tqdm=False,
                             chat_template_kwargs={"enable_thinking": False})
    return {"outputs": [o.outputs[0].text for o in outs]}


def prefetch(args):
    from huggingface_hub import snapshot_download
    return {"path": snapshot_download(args.get("model") or "RedHatAI/Qwen3.8-27B-MXFP4")}


def doctor(args):
    import torch
    import vllm
    return {"vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}


if __name__ == "__main__":
    serve({"load": load, "chat": chat, "prefetch": prefetch, "doctor": doctor})
