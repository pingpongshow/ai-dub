"""Local translation LLM with vLLM (offline engine, no server, no network).

Default model: a 4-bit build of Qwen3.8-27B that fits one 32 GB card with
CUDA graphs off (--enforce-eager), text-only, thinking switched off.
"""

from aidub_worker import log, serve

state = {}


def load(args):
    from vllm import LLM
    name = args.get("model") or "RedHatAI/Qwen3.8-27B-MXFP4"
    # On the RTX 5090 (sm_120) vLLM 0.32's FlashInfer FP4 GEMM and XQA attention
    # paths aren't usable (unsupported arch / need nvcc), so default to Marlin for
    # the 4-bit layers and Triton for attention. Both are configurable.
    kw = dict(model=name, gpu_memory_utilization=float(args.get("gpu_memory_utilization", 0.80)),
              max_model_len=int(args.get("max_model_len", 16384)), max_num_seqs=int(args.get("max_num_seqs", 8)),
              enforce_eager=True, kv_cache_dtype="fp8", trust_remote_code=False,
              kernel_config={"linear_backend": args.get("linear_backend", "marlin")},
              attention_config={"backend": args.get("attention_backend", "TRITON_ATTN")})
    if args.get("language_model_only", True):
        try:
            state["llm"] = LLM(language_model_only=True, **kw)
        except (TypeError, ValueError) as exc:  # a text-only model, or a vLLM without the split
            log(f"language_model_only not applicable ({exc}); loading normally")
            state["llm"] = LLM(**kw)
    else:
        state["llm"] = LLM(**kw)
    state["name"] = name
    log(f"loaded {name}")
    return {"model": name}


def chat(args):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=float(args.get("temperature", 0.3)), top_p=float(args.get("top_p", 0.8)),
                        top_k=int(args.get("top_k", 20)), max_tokens=int(args.get("max_tokens", 4096)),
                        presence_penalty=float(args.get("presence_penalty", 0.5)),
                        repetition_penalty=float(args.get("repetition_penalty", 1.0)))
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
