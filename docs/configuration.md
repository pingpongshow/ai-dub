# Configuration

Ai Dub works with no configuration beyond the folder mounts. This page is for
when you want to change how it behaves. To install it, see [Setup](setup.md).

## Where settings come from

Ai Dub reads its settings from four places. Each one overrides those above
it:

1. **Built-in defaults:** [`aidub/default_config.yaml`](../aidub/default_config.yaml).
   This file lists every setting, with a comment on what it does, so it is
   the reference.
2. **Your changes:** `/config/config.yaml`. Only what you put here
   overrides the defaults. A section you leave out keeps all its defaults.
   Within a section, set only the keys you want to change. A list, such as
   `media`, replaces the default list as a whole.
3. **Environment variables:** `AIDUB_APPDATA`, `AIDUB_MODELS` and
   `AIDUB_WORK` override the three `paths`.
4. **`${NAME}` placeholders:** any string value can contain `${NAME}`. It is
   filled in from `/config/secrets.yaml`, or from an environment variable of
   that name. Keys are kept out of `config.yaml` this way.

On first start, the container copies
[`docker/config.example.yaml`](../docker/config.example.yaml) to
`/config/config.yaml`. That copy lists the media folders and the Bazarr
section, ready to edit.

**Restart the container after any change**, whether you edited the file or
used the Settings page. Settings are read when Ai Dub starts.

If `config.yaml` can't be used, Ai Dub refuses to start, and the container
log names the problem. This happens when the file isn't valid YAML, or when a
section that should be a mapping isn't (see
[Troubleshooting](troubleshooting.md#the-container-wont-start)).

> The Settings page saves by rewriting `config.yaml`, so comments you added
> by hand are lost when you save there. If you keep notes in the file,
> make your changes by hand too.

## Environment variables

| Variable | Default | What it does |
|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | — | Which GPUs the container sees (`all`, or an index or UUID). Required. |
| `PUID` / `PGID` | `99` / `100` | The user and group Ai Dub runs as, and that its new files belong to. |
| `TZ` | UTC | Time zone for the times in the logs. |
| `AIDUB_INSTALL_ON_START` | `false` | `true` installs every missing model when the container starts. |
| `AIDUB_CONFIG` | `/config/config.yaml` | Another settings file. |
| `AIDUB_APPDATA`, `AIDUB_MODELS`, `AIDUB_WORK` | `/config`, `/models`, `/work` | Move a data folder inside the container. To use another drive, change the mount instead: see [Putting the work folder on another drive](setup.md#putting-the-work-folder-on-another-drive). |
| `HF_TOKEN` | — | A Hugging Face token, for a gated model you switched to. The defaults don't need one. Also settable as a key. |

## Keys

Keys are entered under **Settings → Keys** and stored in
`/config/secrets.yaml`, which is readable only by Ai Dub's user. They are
never shown back on the page; you can replace or clear them.

| Key | Used for |
|---|---|
| `BAZARR_API_KEY` | Asking Bazarr for subtitles (Bazarr → Settings → General → Security → API Key) |
| `HF_TOKEN` | Hugging Face downloads of gated models; the defaults don't need it |

## Media folders

Only folders listed under `media` appear in the queue's browser. Ai Dub can
read and change files only in those folders. The list can't be changed from
the Settings page, on purpose.

```yaml
media:
  - { name: Movies, path: /media/movies }
  - { name: Shows,  path: /media/shows }
  - { name: Anime,  path: /media/anime }
  - { name: Korean drama, path: /media/kdrama }   # also mount it: -v /srv/kdrama:/media/kdrama
```

Ai Dub guesses the show from Sonarr- and Radarr-style paths:

- `Show/Season 01/Show - S01E07.mkv` → *Show*
- `Movie (Year)/Movie (Year).mkv` → *Movie (Year)*
- a file directly in a media folder → the file name, without the episode
  part

You can type another show when adding a file.

## Bazarr

Ai Dub looks for English subtitles in this order:

1. inside the file;
2. next to the file (`Episode.en.srt`, `Episode.eng.ass`, …);
3. from Bazarr, if it is switched on.

Picture subtitles (PGS) are read with OCR.

On the **Settings** page:

1. Tick **Use Bazarr**, and enter its address as Ai Dub's container can
   reach it, for example `http://10.1.1.10:6767`, or `http://bazarr:6767` on
   a shared Docker network.
2. Add `BAZARR_API_KEY` under **Keys**.
3. Fill in the **Folder map**. Bazarr names files by its own container paths,
   so Ai Dub has to translate between the two:

   ```
   /media/movies = /movies
   /media/shows = /tv
   /media/anime = /anime
   ```

   On the left is Ai Dub's path; on the right, the path Bazarr uses for the
   same folder. Look them up in Bazarr under Settings → Sonarr/Radarr → Path
   Mappings, or in Bazarr's container mounts.
4. **Save Bazarr settings** and restart Ai Dub, then **Test**. The test uses
   the saved key, which Ai Dub reads only at start.

The same settings in `config.yaml`:

```yaml
subtitles:
  providers:
    bazarr:
      enabled: true
      url: http://10.1.1.10:6767
      api_key: "${BAZARR_API_KEY}"
      search_missing: true        # ask Bazarr to search when it has nothing yet
      path_map:
        - { aidub: /media/movies, bazarr: /movies }
        - { aidub: /media/shows,  bazarr: /tv }
        - { aidub: /media/anime,  bazarr: /anime }
```

## Per-show settings

Each show has a profile in `/config/profiles/<show>/`. It holds:

- `profile.json`: settings, characters, glossary and rules;
- `refs/`: the reference clips for each character's voice;
- `fingerprints/`: the openings and endings it has learnt.

Edit these from the **Shows** page rather than by hand.

| Setting | Where | Choices |
|---|---|---|
| Honorifics | Shows → Settings | Keep (-san, -kun…) or drop. Applies to lines translated from speech; lines dubbed from English subtitles keep the subtitle's wording. |
| Style | Shows → Settings | Natural English, or close to the original |
| Review before dubbing | Shows → Settings → Show rules | The default, Off, When unsure or Always |
| Subtitle habits | Shows → Settings → Show rules | Romaji lines are lyrics · italics in quotation marks are lyrics · "~ text ~" is a sign, not spoken · the narrator's opening is dubbed. All on by default; switch off what doesn't fit the show. |
| Recurring regions | Shows → Settings → Show rules | Openings, endings and songs marked on a review page "for every episode". Remove them here. |
| Characters | Shows → Voices | Rename, merge, split off wrongly matched clips, delete, and hear the references; merge suggestions for voices that sound alike |
| Glossary | Shows → Glossary | Names and terms with their fixed English; click the English to change it |

The review mode is chosen in this order: the episode's choice when you added
it, then the show's rule, then `review.mode`.

## Settings worth knowing

Everything else is in [`default_config.yaml`](../aidub/default_config.yaml). The
ones you're most likely to want are below.

| Setting | Default | What it does |
|---|---|---|
| `review.mode` | `unsure` | Stop before translating: `off`, `unsure` (only when something needs a look) or `always`. Also on the Settings page. |
| `verify.enabled` | `true` | Keep a preview of each finished dub to check before its original is deleted. With `false`, a finished dub's working files are deleted at once, and the original stays set aside on the Originals page. |
| `translate.lead` | `subtitle_verbatim` | Where the English comes from. `subtitle_verbatim` dubs the English subtitles word for word wherever they cover a line. `subtitle_only` uses the subtitles as the script for every line they cover, but not word for word. `subtitle` uses a subtitle only where it means what was said. `audio` translates the speech and treats subtitles as hints. |
| `translate.style`, `translate.honorifics` | `natural`, `keep` | The defaults for shows that don't set their own |
| `ingest.skip_english_audio` | `true` | Skip files that already have English audio, including untagged audio that turns out to be English |
| `ingest.song_chapters` | opening, op, ending, ed, credits, … | Chapter names whose audio is always kept as it is |
| `subtitles.min_score` | `0.65` | How well a subtitle track must match the speech to be trusted. A track for another episode scores under about 0.5. |
| `songs.detect_by_pitch` | `true` | Find sung passages by held notes over music |
| `songs.match_episodes` | `true` | Find the opening and ending shared with neighbouring episodes in the folder |
| `speakers.link_threshold` | `0.62` | How alike two voices must sound (cosine similarity) to be treated as one speaker. Raise it if different people get merged; lower it if one person is split in two. |
| `speak.takes` | `2` | Takes voiced per line; the best one is kept |
| `speak.redo_takes` | `4` | Takes for a line you send back from the check screen, with new seeds each time |
| `speak.processes` | `2` | Voice-model processes sharing the GPU |
| `fit.timing` | `auto` | `window` fits each line to the time the character's mouth moves. `start` starts with the original line at its natural length. `auto` uses `start` for word-for-word subtitle dubs and `window` otherwise. |
| `fit.free_early_ms`, `fit.free_max_late_ms`, `fit.free_gentle_tempo` | `400`, `350`, `1.12` | With `start` timing: how early a line may start when that leaves the next on time (never back over a shot change or a line kept in the original), how late one may run, and how far the lines running into it are sped up together before any goes up to `fit.force_stretch_max` |
| `condense.enabled` | `true` | Write shorter English for lines that still don't fit |
| `readback.enabled`, `readback.min_clear` | `true`, `0.5` | Listen to the finished track before delivery; stop the job if it is silent while the original plays, or if less than this share of its lines can be heard |
| `mix.min_level_db` | `-12` | No dubbed line quieter than this under the film's usual dialogue level, however quiet the original voice was |
| `stages.faces.enabled` | `true` | Use the picture, once **Who is talking on screen** is installed on the Models page: a line the voice check doubts stays with the character whose face is talking, or moves to them |
| `deliver.surround`, `deliver.surround_bitrate` | `true`, `384k` | A 5.1 original gets a 5.1 English track, the dub in its front and centre channels and its music, effects and surrounds as they were. Other layouts get stereo. |
| `stages.bwe.enabled` | `true` | Restore full 48 kHz brightness to the cloned voice (`mode: auto` does it only when the original is brighter) |
| `deliver.track_title` | `English (AI dub)` | The name of the added audio track |
| `deliver.add_subtitle_tracks` | `true` | Add the subtitle tracks Ai Dub found, or made, for languages the file has none in |
| `deliver.owner` | `{uid: 99, gid: 100}` | Owner given to the new file. Match it to `PUID`/`PGID`. |
| `cleanup.keep_failed_work` | `true` | Keep a failed job's working folder, so that **Retry** carries on where it stopped |
| `gpu.wait`, `gpu.min_free_mb` | `true`, per step | Wait (and say so) while another program uses the GPU, until this much memory is free |

Example `config.yaml`:

```yaml
media:
  - { name: Anime, path: /media/anime }

review:
  mode: always

translate:
  lead: subtitle            # dub the subtitle only where it means what was said

deliver:
  owner: { uid: 1000, gid: 1000 }

subtitles:
  providers:
    bazarr:
      enabled: true
      url: http://bazarr:6767
      api_key: "${BAZARR_API_KEY}"
      path_map:
        - { aidub: /media/anime, bazarr: /anime }
```

## Models

Each step's model is set under `stages`:

```yaml
stages:
  translate: { model: RedHatAI/Qwen3.8-27B-MXFP4, gpu_memory_utilization: 0.80, max_model_len: 16384 }
```

The worker for a step decides which models it can run. A model of the same
family (another size of Qwen3-ASR, say) usually works. Anything else needs
changes to the worker in `workers/<name>/`.

After changing a model, use **Models → Reinstall** to download it. A job never
falls back to another model: a missing one stops it with an error that says
which.

### Translation on another machine

If your GPU is too small for the translation model, or you already run a
language-model server, you can send translation there instead. It works with
any OpenAI-compatible API, such as llama.cpp's server, Ollama, vLLM or LM
Studio:

```yaml
llm_endpoint:
  enabled: true
  url: http://10.1.1.20:11434/v1
  model: qwen3:32b
  api_key: ""
```

Every translation request then goes to that server; the built-in model isn't
loaded. Use a strong multilingual model: the quality of the English script
depends on it. Ai Dub was tuned with Qwen3.8-27B. The script, the episode's
text and the show's glossary go to this server, so keep it on your own
network.
