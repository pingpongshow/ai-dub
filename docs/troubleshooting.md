# Troubleshooting

Start with the logs, which usually name the problem.

| Log | Where |
|---|---|
| A job's log | **Log** on the job card (Queue or History), or `/work/job-<id>/job.log` |
| A model install | **Models → Log** (kept until the container restarts) |
| Ai Dub itself | `docker logs ai-dub` |
| Model health | `docker exec -it ai-dub aidub doctor`: each model should say `ok` and name your GPU |

## The container won't start

**`config.yaml: '…' must be a mapping`, or a YAML error, in `docker logs`.**
Your `/config/config.yaml` has a typo. A common one is a section written as
a value (`review: always` instead of `review: { mode: always }`). Fix the
file and start the container again. If you're stuck, rename the file: the
container writes a fresh copy of the example on its next start.

**`unknown or invalid runtime name: nvidia`, or `could not select device driver`.**
The NVIDIA Container Toolkit isn't installed, or Docker hasn't been restarted
since it was installed. On Unraid, install the **Nvidia Driver** plugin and
reboot. To test the GPU outside Ai Dub:

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ubuntu nvidia-smi
```

**It starts, but the page doesn't load.** Check the port mapping
(`-p 8790:8790`) and that nothing else uses port 8790 on the host. Ai Dub
listens on all addresses inside the container.

## Installing models

**An install says Failed.** Open its **Log**. Common causes:

| In the log | What to do |
|---|---|
| `No space left on device` | `/models` needs about 95 GB in all. Free space and install again. |
| Connection errors or timeouts | Hugging Face, ModelScope, GitHub or PyPI couldn't be reached. Install again; finished downloads are kept. |
| `401` / `403` / gated repository | You switched to a gated model. Accept its terms on Hugging Face and add `HF_TOKEN` under Settings → Keys, then restart. |
| A build error in a package | Run `docker exec -it ai-dub aidub setup <name> --force` to rebuild that model's environment from scratch, and look at the full output. |

**`Weights missing`.** The environment installed, but the download didn't
finish. Press **Install** again.

**`aidub doctor` says `FAILED … CUDA` or no GPU.** The model sees no GPU, or
the driver is too old for CUDA 12.8. Update the driver to 570 or later, and
check `NVIDIA_VISIBLE_DEVICES`.

**A job fails with `model '…' is not installed`.** That model hasn't
finished installing; see the Models page. Ai Dub never switches to another
model on its own.

## A job waits

**`waiting for GPU (…; need N MB free)`.** Another program is using video
memory: a media server transcoding, another AI app, a game, a desktop session.
The job starts by itself once enough memory is free.

To find out what is using the GPU, run `nvidia-smi` on the host. If you'd
rather share the card, lower that step's figure under `gpu.min_free_mb`. To
fail at once instead of waiting, set `gpu.wait: false`. A step that starts
without enough memory fails with a CUDA out-of-memory error.

**`Needs subtitles` under Waiting for you.** Dubs follow English subtitles, so a file without
a usable English one waits for you while other files carry on.

- With no English subtitle in the file, next to it or from Bazarr, it waits as soon as it
  is added, and no work has started on it.
- An English subtitle that turns out not to match the speech can only be told after the
  speech is analysed, so that file stops then; its analysis is kept.

On the card you can **Use this subtitle** (one in another language, translated into English),
**Search Bazarr** for a language, add an `.en.srt` next to the file and **Look for English
again**, or **Dub from the speech anyway…**, the least accurate choice. To dub without asking,
set `subtitles.require: false`.

**The queue doesn't move.** Check whether the queue is paused: the Queue page
then shows a banner and **▶ Resume the queue**. To dub one title without resuming
the rest, press **Start now** on it. Also look under **Waiting for you**: an episode paused for
review waits there until you open it and press **Start dubbing**. Other
episodes carry on in the meantime.

**There's no time estimate, or a strange one.** The estimates come from
the dubs you've already finished. They improve after the first few, and are
per show once a show has its own history.

## A job is skipped

| Message | Why | What to do |
|---|---|---|
| `skipped: English audio exists` | The file already has an English audio track | Nothing, or set `ingest.skip_english_audio: false` |
| `skipped: the audio is already English (its track has no language tag)` | The untagged track turned out to be English | If the file has another, foreign-language track, queue it again and pick that **Audio track** |
| `skipped: already dubbed by Ai Dub` | The file has an Ai Dub track already | Queue it again with **Redo** to rebuild it |
| `skipped: no foreign-language audio track` | No usable audio track | Pick the **Audio track** and **Language** by hand when adding it |

## A job fails

The message names the step, for example ``failed at 'Speak': …``. Read the
job **Log** first. Then press **Retry**: the job carries on from the failed
step, because by default a failed job's working folder is kept
(`cleanup.keep_failed_work`).

| In the log | Likely cause | What to do |
|---|---|---|
| `CUDA out of memory` | Something else took video memory mid-step, or settings were raised | Close the other program and **Retry**. For voicing, set `speak.processes: 1`. For translation, lower `stages.translate.gpu_memory_utilization` a little. |
| `worker '…' exited (code -9)` | The host ran out of RAM and the kernel killed the model | Free RAM (other containers) and **Retry** |
| The job stops at **Mix** and Ai Dub restarts | The mix holds a whole film's audio in memory, up to about 60 GB for a two-hour 5.1 film, and the host ran out | `dmesg` on the host shows `Out of memory` if this was the cause. Free RAM; after the restart the job carries on from its first unfinished step. |
| `worker '…' failed to start` | The model's environment is broken | **Models → Reinstall** that model |
| `No space left on device` | `/work` or the media disk is full | See [Disk space](#disk-space) |
| `Permission denied` on the media file | Ai Dub can't write next to the file | See [Permissions](#permissions) |
| `the English track is silent from …`, or `only N% of the English lines can be heard` | The **Listen back** step found the finished track broken | Nothing was delivered and the original is untouched. Open an issue with the log. |
| `… is over 4 GiB and would read back cut short` | A very long audio file was written in a form that loses its end | Nothing was delivered. Open an issue with the log. |
| A delivery check fails (`stream … missing`, durations differ) | The new file didn't pass the safety check | Nothing was changed; the original is untouched. Open an issue with the log. |

A container restart in the middle of a job is not a failure: the job resumes
by itself from the step it was on.

## The dub doesn't sound right

Everything here can be fixed on the review page, where a finished dub is
checked. For one or two lines, select the line (a moment on the **Check** tab, a
subtitle, or the timeline), change its English, speaker or voice in **Details**,
and press **↻ Redo**: only those lines are voiced again, with fresh takes. For
bigger changes, press **Redub with these changes**. The analysis is kept, so only
translation and voicing run again.

| Problem | Fix |
|---|---|
| Foreign speech left in | The **Check** tab lists the lines that stayed in the original language. Usually every take was heard back wrong, often a sound word or name recognition can't spell ("peep-oo"). **↻ Redo this line**, with new English if it needs it, or set the stretch to **Dub** (key D). |
| Voices sound spliced in over a noisy scene | Check that `mix.room_tone` is on: it keeps the scene's background noise steady under the English lines. If one speaker stands out as much louder, lower `mix.level_spread` (0 is the default). |
| The original voice can be heard behind the dub, like an echo | On a 5.1 film this is speech in the surround channels. Check that `stages.separate.surrounds` is on; a dub made before it was needs to be queued again with **Redo** to separate them. On stereo, the separation left some voice in the music: there is no setting for that. |
| Shouts and whispers sound too alike, or flat | Raise `mix.level_spread` (to `0.5`, say) and lower `mix.even_within` (to `2`) |
| Delivery sounds overdone, or flat | Lower or raise `speak.emo_alpha` (default `0.6`) |
| Word endings sound cut off | The trim keeps a final consonant that is still sounding; if a word still sounds short, **↻ Redo this line** gives it fresh takes |
| Loud lines crackle or sound clipped | The voice model's output is turned down before it would be cut at full scale. A take that still sounds rough: **↻ Redo this line**. |
| A song was dubbed | Mark it as a song: **♪ Mark a song**, or select it and press S. Mark it "for every episode" if it's the opening or ending. |
| Speech was kept as if it were a song | Select the region and press D (dub) |
| A line has the wrong voice | Select the line and press the speaker's name: hear each speaker's clip and give it to the right one. If the speaker is right but their voice is wrong, find a line of theirs that sounds right in the original and press **Make this line their voice** under it; their other lines go on the **↻ Redo** list with it. If two characters keep getting mixed up, merge or split them on **Shows → Voices**. |
| A line is in a voice of the wrong gender, or someone else's | Speaker detection is least sure of voices with only a few seconds of speech: those lines borrow another character's voice, picked by how the speaker sounds. The review page flags them as "Dubbed with a borrowed voice" with the voices they sound most like; give the lines to the right person (on a finished dub this puts them on the **↻ Redo** list). |
| Two people share one character (one voice for both, or the wrong one) | The review page flags "may be more than one person" with the lines in question. Give them to the right speaker or a new one; on a finished dub this puts them on the **↻ Redo** list. On **Shows → Voices**, split the other person's clips off into a character of their own. |
| One person is split into several voices | Merge them on **Shows → Voices**, or try the "sound like the same person?" suggestions. You can also lower `speakers.link_threshold`. |
| Names are wrong or mispronounced | Fix them on **Shows → Glossary**. Glossary names are also accepted by the back-check. With guided subtitles, names follow the English subtitle; a name it never says is read from the original's writing and can come out wrong. |
| The English came from the wrong subtitle | A subtitle for another episode or cut scores low and is dropped; raise `subtitles.min_score` if a bad one got through. Check the sidecar file's name and language tag (`.en.srt`). |
| Reviews keep correcting the same thing | `aidub corrections` lists what each review changed; the files in `appdata/corrections` (no audio) survive **Keep the dub**. Worth including in an issue, with the film's name removed. |
| You want to know a finished dub is all there | `docker exec ai-dub aidub audit --all --output /work/audit.json` hears the English track of every delivered dub against its subtitles: stretches with no English, and subtitles never heard. It changes nothing. |
| Lines are rushed or cut off | Raise `condense.rounds` to give lines more tries at shorter English. With `fit.timing: start`, lines can run on past the mouth movement. |
| The voice sounds muffled | Check that the bandwidth-extension model is installed and `stages.bwe.enabled` is on |

If a dub is beyond fixing, choose **Put the original back** on the check
screen.

## Subtitles and Bazarr

**Test says it can't connect.** The address must work from inside Ai Dub's
container. `localhost` there is Ai Dub itself. Use the host's IP
(`http://192.168.1.10:6767`), or Bazarr's container name if both containers
share a Docker network.

**Test says `no Bazarr API key is set` or `Bazarr refused the API key`.**
Copy the key again from Bazarr (Settings → General → Security → API Key),
save it under **Keys**, and restart Ai Dub before testing again. Ai Dub reads
keys only when it starts.

**Bazarr connects, but finds nothing.** The **Folder map** is probably wrong.
Bazarr knows each file by its own path (`/tv/Show/…`), and Ai Dub must be able
to turn `/media/shows/Show/…` into that path. Every line in the map should
pair the same folder as seen by both containers. Bazarr also has to know the
series or movie (through Sonarr or Radarr) and have English as a wanted
language.

**Picture subtitles (PGS) give poor text.** They are read with OCR, which is
right most of the time. A text subtitle (SRT/ASS) next to the file, or from
Bazarr, is preferred when there is one.

## Playback

**I can't hear the dub.** The dub is added as an extra audio track named
**English (AI dub)**. It is not made the default, so your original audio
stays the default. Choose the track in your player. In Plex or Jellyfin,
refresh the item's metadata if the new track doesn't show up yet.

**The file was replaced but my media server shows the old one.** Run a
library scan or refresh. The file has the same name, but its contents
changed.

## Disk space

- **`/work` grows.** Failed jobs keep their working folders so **Retry** can
  resume them, which can be several GB each. Removing a failed job from
  History deletes its folder. So does setting
  `cleanup.keep_failed_work: false`. To give it a drive of its own, see
  [Putting the work folder on another drive](setup.md#putting-the-work-folder-on-another-drive).

  Dubs waiting to be checked keep their check files until you choose
  **Keep the dub** or **Put the original back**. An episode paused at review
  keeps its whole working folder until it finishes.
- **The media disk fills up.** Every dub leaves the original set aside until
  you keep the dub. The **Originals** page lists them with their sizes and the
  free space. Checking dubs and choosing **Keep the dub** frees the space.
- **`/models` is huge.** ~95 GB is normal. `/models/uv-cache` holds package
  downloads; it can be deleted while nothing is installing, but reinstalls
  then download them again.

## Permissions

**`Permission denied` writing next to a media file.** Ai Dub runs as
`PUID:PGID` (99:100 by default) and adds the new track by writing a new file
in the same folder. Make sure that:

- the media mounts are read-write (not `:ro`);
- `PUID`/`PGID` can write to the media folders;
- `deliver.owner` matches, so that your media server can read the new file.

**New files have the wrong owner.** Set `deliver.owner` (and `PUID`/`PGID`)
to the user and group your media files belong to, then restart.

## The pages

**A page looks out of date after an update.** Open pages reload themselves
when Ai Dub's version changes. If one doesn't, reload it in the browser
(Shift-reload clears the cache).

**A setting I changed hasn't taken effect.** Restart the container: settings
and keys are read at start.

**I chose Keep the dub by mistake.** The original has already been deleted
by then. Delete the dubbed file, restore the original from your backup, and
queue it again.

## Getting help

Open an issue at https://github.com/pingpongshow/ai-dub/issues. Include:

- the job **Log**, or `docker logs ai-dub`;
- the output of `aidub doctor`;
- your GPU and driver version;
- what you expected to happen.

Remove keys and personal paths first.
