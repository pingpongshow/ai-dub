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
   reach it, for example `http://192.168.1.10:6767`, or `http://bazarr:6767` on
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
      url: http://192.168.1.10:6767
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
- `house/`: each character's designed house voice, when the show uses house voices;
- `fingerprints/`: the openings and endings it has learnt.

Edit these from the **Shows** page rather than by hand.

| Setting | Where | Choices |
|---|---|---|
| Dub voices | Shows → Settings → Voices | The actors' own voices, cloned from the film (`clone`), or house voices designed to suit each character (`house`). Default: `voice.mode`. A title can also be given its own choice when you add it (Add → Dub voices). |
| Honorifics | Shows → Settings | Keep (-san, -kun…) or drop. Applies to lines translated from speech; lines dubbed from English subtitles keep the subtitle's wording. |
| Style | Shows → Settings | Natural English, or close to the original |
| Review before dubbing | Shows → Settings → Show rules | The default, Off, When unsure or Always |
| Subtitle habits | Shows → Settings → Show rules | Romaji lines are lyrics · italics in quotation marks are lyrics · "~ text ~" is a sign, not spoken · the narrator's opening is dubbed. All on by default; switch off what doesn't fit the show. |
| Recurring regions | Shows → Settings → Show rules | Openings, endings and songs marked on a review page "for every episode". Remove them here. |
| Characters | Shows → Voices | Rename, merge, split off wrongly matched clips, delete, and hear the references; merge suggestions for voices that sound alike. With house voices: hear each character's house voice, read the description it was designed from, and **Recast** it for the next dub |
| Glossary | Shows → Glossary | Names and terms with their fixed English; click the English to change it |

The review mode is chosen in this order: the episode's choice when you added
it, then the show's rule, then `review.mode`.

## Settings worth knowing

Everything else is in [`default_config.yaml`](../aidub/default_config.yaml). The
ones you're most likely to want are below.

| Setting | Default | What it does |
|---|---|---|
| `voice.mode` | `clone` | `clone`: each actor's own voice, cloned from the film. `house`: a voice of Ai Dub's own for each character, designed from what the original says about them (gender, age range, pitch, pace, loudness, liveliness, usual mood) and kept for every episode and spoken by the same model that designed it (see `voice.house.speaker`). Needs the house-voice model (Models page). Per show on its Settings tab. |
| `voice.house.speaker` | `drama3` | Which model speaks house voices. `drama3`: Fish Audio Drama 3, with each line's emotion as a direction ([angry], [excited], [sad]); chosen by ear, but on a 32 GB card about half its experts stay in RAM (`stages.house_drama3.offload_layers`) and it runs at about a quarter of real time, hours a film. `qwen`: Qwen3-TTS, which designed them; clean, one steady voice per character, about 60 times real time, no emotion. `indextts`: the voice-cloning model, which carries the original line's emotion (`voice.house.emo_alpha`) and fits each line's length, but sounded more robotic. Only indextts has duration control: long lines are sped up afterwards. If the chosen model isn't installed, the next one speaks |
| `stages.recognize_ja` | Haas-ja-v1.0 | Japanese: the words come from this model (on Woman in the Dunes they matched the English subtitles' meaning more often than Qwen3-ASR's: closer on 33 % of lines, further on 12 %), timed by Qwen3's aligner. `enabled: false`, or the model not installed: Qwen3-ASR's words |
| `voice.house.design_min_s` | `10` | Characters with less speech than this get one of the generic voices of their gender and age range instead of one designed for them |
| `speakers.face_links`, `speakers.absorb_s` | `true`, `20` | Telling one actor from another: speakers whose lines show the same face talking are joined, and a speaker with less than this many seconds joins the main speaker of the same gender it sounds most like |
| `review.mode` | `unsure` | Stop before translating: `off`, `unsure` (only when something needs a look) or `always`. Also on the Settings page. |
| `verify.enabled` | `true` | Keep a preview of each finished dub to check before its original is deleted. With `false`, a finished dub's working files are deleted at once, and the original stays set aside on the Originals page. |
| `translate.lead` | `subtitle_verbatim` | Where the English comes from. `subtitle_verbatim` dubs the English subtitles word for word wherever they cover a line. `subtitle_only` uses the subtitles as the script for every line they cover, but not word for word. `subtitle` uses a subtitle only where it means what was said. `audio` translates the speech and treats subtitles as hints. |
| `translate.style`, `translate.honorifics` | `natural`, `keep` | The defaults for shows that don't set their own |
| `ingest.skip_english_audio` | `true` | Skip files that already have English audio, including untagged audio that turns out to be English |
| `ingest.song_chapters` | opening, op, ending, ed, credits, … | Chapter names whose audio is always kept as it is |
| `subtitles.require` | `true` | A file with no usable English subtitle waits under **Waiting for you** until you choose another subtitle, search Bazarr, or allow a dub from the speech. With none in the file or next to it, it waits before any work starts. `false` dubs such files from the speech without asking. |
| `subtitles.prefer` | `guided` | With both a person's English subtitle and one in the spoken language: `guided` translates the spoken-language one with the English as a guide (names spelled as the English spells them); `english` dubs the English word for word. In blind listening, guided was preferred in 16 of 36 scenes and English in 13: it keeps more of what was said, while English subtitles are often shortened. |
| `translate.repair_speech` | `true` | A dub from the speech alone: first correct what recognition heard, in context, before translating |
| `speak.emo_alpha` | `0.6` | How much of the original line's emotion the English takes (0 to 1). 0.6 was preferred in 9 of 14 scenes, 0.35 in 3. Lower it if delivery sounds overdone. |
| `speak.rescue_wer` | `0.35` | When no take passes the back-check, a take this close (at most one key word unclear) is used and flagged on the Check tab instead of leaving the line in the original language |
| `mix.level_spread` | `0` | How much of the film's volume differences between lines the dub keeps. `0` brings them all to the scene's usual level; `1` matches each original line. Raise it (to `0.5`, say) if shouts and whispers sound too alike. |
| `mix.even_within` | `4` | Evens out loud and quiet moments inside each dubbed line. `1` is off; higher is stronger. With `level_spread: 0` this was preferred over the film's own levels in a blind test. |
| `mix.room_tone` | `true` | Fill the scene's own background noise back in under each dubbed line. In a noisy scene much of the noise is in the dialogue stem, so without it the noise dips as each English voice starts and the voice sounds spliced in. |
| `stages.separate.surrounds` | `true` | On a 5.1 film, also separate the surround channels when they carry the speech (reverb, say), so the original voice doesn't stay behind the dub as an echo. Costs one more separation pass on those films. |
| `mix.polish` | `off` | Smooths each voice before mixing: harsh and hissy moments are turned down, the rough top end smoothed, the level evened and a little room added. `off`, `gentle` or `strong`. Off by default: in a blind test none of the three was preferred. |
| `subtitles.min_score` | `0.65` | How well a subtitle track must match the speech to be trusted. A track for another episode scores under about 0.5. |
| `songs.detect_by_pitch` | `true` | Find sung passages by held notes over music |
| `songs.match_episodes` | `true` | Find the opening and ending shared with neighbouring episodes in the folder |
| `speakers.link_threshold` | `0.62` | How alike two voices must sound (cosine similarity) to be treated as one speaker. Raise it if different people get merged; lower it if one person is split in two. |
| `speakers.same_person_ratio` | `0.86` | Two speakers the speaker detection told apart become one character only if they also sound alike line by line: their lines must be this alike to each other, compared with how alike each one's own lines are. Two different men can match on average and still fail this. A character that still ends up with two voices is flagged on the review page. |
| `speakers.prefer_own_clip` | `true` | Voice each line from its own clip, not the character's usual voice. In a blind listening test the own clip was acceptable on 36 of 48 lines, the usual voice on 16. On the review page, **Try the other voice** swaps one line. |
| `speak.max_wer` | `0.35` | How much of a take may be heard wrong by the back-check and still pass (word error rate) |
| `speak.takes` | `1` | Takes voiced per line; the best one that passes the check is kept |
| `speak.retakes` | `3` | New takes (new seeds) for each line whose takes all fail the check, voiced and checked again automatically |
| `speak.retake_rounds` | `2` | How many times the check voices failed lines again; a line that still fails is left in the original and listed (0: never) |
| `speak.redo_takes` | `4` | Takes for a line you send back with **↻ Redo**, with new seeds each time |
| `speak.processes` | `2` | Voice-model processes sharing the GPU |
| `fit.timing` | `auto` | `window` fits each line to the time the character's mouth moves. `start` starts with the original line at its natural length. `sync` is `start`, then each line follows the speaker's mouth: its pieces start with the original's phrases, pause where the original pauses (gaps of `fit.sync_pause_ms`) and are slowed or sped up a little (`fit.sync_slowest`, `fit.sync_fastest`) to end with it; most useful with voices that have no duration control (house voices). `auto` uses `start` for subtitle dubs (word for word or guided) and for dubs from the speech alone, and `window` for `translate.lead: subtitle` or `subtitle_only`. Fitting the speech-only translations into the mouth movement lost words; natural length kept 39–57 % more at no timing cost. |
| `fit.free_early_ms`, `fit.free_max_late_ms`, `fit.free_gentle_tempo` | `400`, `350`, `1.12` | With `start` timing: how early a line may start when that leaves the next on time (never back over a shot change or a line kept in the original), how late one may run, and how far the lines running into it are sped up together before any goes up to `fit.force_stretch_max` |
| `condense.enabled` | `true` | Write shorter English for lines that still don't fit |
| `readback.enabled`, `readback.min_clear` | `true`, `0.5` | Listen to the finished track before delivery; stop the job if it is silent while the original plays, or if less than this share of its lines can be heard |
| `mix.min_level_db` | `-12` | No dubbed line quieter than this under the film's usual dialogue level, however quiet the original voice was |
| `stages.faces.enabled` | `true` | Use the picture, once **Who is talking on screen** is installed on the Models page: a line the voice check doubts is given to the character whose face is talking, and one person split into two characters is pointed out. About 7 minutes a film. |
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
  url: http://192.168.1.20:11434/v1
  model: qwen3:32b
  api_key: ""
```

Every translation request then goes to that server; the built-in model isn't
loaded. Use a strong multilingual model: the quality of the English script
depends on it. Ai Dub was tuned with Qwen3.8-27B. The script, the episode's
text and the show's glossary go to this server, so keep it on your own
network.
