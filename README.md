# Ai Dub

Local, voice-cloned dubbing of foreign-language video into English.

Ai Dub takes a film or episode (Japanese, Korean, Mandarin, Cantonese, …) and
adds an **"English (AI dub)"** audio track to the file:

- every speaker keeps their own voice, cloned from their own lines;
- each line is timed to the original performance;
- the original music, effects, laughs, gasps, songs and lines already in
  English are kept as they were;
- a good subtitle is used as the source of the text when there is one, with
  speech recognition filling the gaps;
- subtitle tracks are added for languages the file has none in.

Everything runs locally on your own GPU. No audio or text is sent to any
cloud service. The only outside contact is subtitle lookups through your own
Bazarr (optional).

> For personal use with media you own. The dub clones real actors' voices:
> don't distribute dubbed files. Added tracks are labelled "AI dub" and carry
> a version stamp.

## How it works

```
ingest → separate dialogue / music & effects → who speaks when → speech recognition
→ sound events → subtitles (find, sync, score) → align lines → show's voice bank
→ English script (local LLM: translate, review, rewrite to fit) → voice-cloned takes
→ back-check and timing fit → restore full bandwidth → mix → new file
```

| Step | Default model | Licence |
|---|---|---|
| Dialogue separation | MelBand-RoFormer via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | MIT tool; weights per their authors |
| Diarization | [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) | OpenMDW-1.1 |
| Voice embeddings | [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Apache-2.0 |
| Gender / age range | [audeering wav2vec2 age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) | CC-BY-NC-SA-4.0 (non-commercial) |
| Speech recognition | [Qwen3-ASR 1.7B + ForcedAligner](https://github.com/QwenLM/Qwen3-ASR) | Apache-2.0 |
| Sound events / emotion | [SenseVoice-Small](https://github.com/FunAudioLLM/SenseVoice) | FunASR model licence |
| Picture subtitles (OCR) | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 |
| Translation | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), 4-bit, on [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 |
| Voice cloning | [IndexTTS-2.5](https://github.com/index-tts/index-tts) | bilibili Model Use License |
| Bandwidth extension | [FlashSR](https://github.com/jakeoneijk/FlashSR_Inference) | no licence declared by its authors |
| Generic voices | [Qwen3-TTS VoiceDesign](https://github.com/QwenLM/Qwen3-TTS) | Apache-2.0 |

**No model is included in this repository or the Docker image.** Each model
is downloaded from its official source when you install it from the
**Models** page, and runs in its own Python environment inside the container.
Check each model's licence before you install it. Models can be swapped in
`config.yaml`; a model that is chosen but not installed stops the job with a
clear error rather than silently using another.

## Your files are safe

- The original is never re-encoded, changed or deleted by a job. The new file
  is written next to it under a temporary name, verified (every track,
  chapter and attachment present, durations equal, decodes at start and
  end), and only then does it take the original's name. The original is set
  aside as `<name>.<ext>.aidub-original`.
- Set-aside originals are listed on **Originals pending deletion**, where you
  can restore them or delete them yourself. Nothing deletes them
  automatically.
- Files that already have English audio are skipped.
- Existing subtitle tracks are never replaced.

## Requirements

- An NVIDIA GPU with recent drivers. Built for a 32 GB card (RTX 5090);
  the translation model alone uses about 26 GB. Stages load one model at a
  time.
- Docker with the NVIDIA container runtime.
- About 60 GB of disk for models, plus room for one extra copy of each file
  you dub until you delete its set-aside original.

## Install (Docker)

```bash
docker run -d --name ai-dub --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e PUID=99 -e PGID=100 \
  -p 8790:8790 \
  -v /path/to/appdata:/config \
  -v /path/to/models:/models \
  -v /path/to/work:/work \
  -v /path/to/movies:/media/movies \
  -v /path/to/shows:/media/shows \
  -v /path/to/anime:/media/anime \
  ghcr.io/pingpongshow/ai-dub:latest
```

An Unraid template is in [`docker/unraid/ai-dub.xml`](docker/unraid/ai-dub.xml).

Then open `http://<server>:8790`:

1. **Models**: install everything (tens of GB; one at a time, with a log),
   then **Make generic voices** once. Or start the container with
   `AIDUB_INSTALL_ON_START=true`.
2. **Settings**: optionally switch on Bazarr (URL, API key, and how Bazarr's
   paths map to Ai Dub's). The diarization and recognition defaults need no
   Hugging Face token.
3. **Queue**: browse your media, tick files, add them. One job runs at a
   time; if another program is using the GPU, the job waits and says so.

## Configuration

`/config/config.yaml` holds your changes; everything else comes from
[`aidub/default_config.yaml`](aidub/default_config.yaml). Keys and tokens are
entered on the Settings page and stored in `/config/secrets.yaml` (never shown
back). Per-show settings (honorifics, style), voices and glossary live in each
show's profile.

## Development

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" pillow
.venv/bin/pytest -q            # needs ffmpeg; mkvmerge and rubberband optional
```

The tests run the whole pipeline on generated media with stand-in workers
(`tests/fakes`), so they need no GPU or models. `aidub run FILE --preview 5m
--output out.mkv` dubs the first minutes of a file into a separate file
without touching the source. See [`dev/`](dev/) for model comparison tools.

Layout: `aidub/` (pipeline, queue, API), `workers/` (one folder per model,
each with its own requirements), `web/` (UI), `tests/`, `dev/`, `docker/`.

## Credits

Ideas borrowed from [OpenDub](https://github.com/estebanstifli/OpenDub)
(per-line audio, resumable steps, no silent fallbacks) and
[VideoLingo](https://github.com/Huanshere/VideoLingo) (translate → reflect →
rewrite, glossary extraction). Neither is included or run.

## Licence

MIT for this code. Each model has its own licence (see the table above).
