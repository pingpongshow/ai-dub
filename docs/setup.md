# Setup

This guide covers what Ai Dub needs, how to install it, the first run, and
keeping it up to date. For settings, see [Configuration](configuration.md);
when something goes wrong, see [Troubleshooting](troubleshooting.md).

## What you need

**GPU.** An NVIDIA card with a driver new enough for CUDA 12.8 (driver 570 or
later). Ai Dub was built and tested on an RTX 5090 (32 GB) with driver 580.

Ai Dub loads one model at a time. The free video memory each step waits for
(`gpu.min_free_mb`) is:

| Step | Needs free |
|---|---|
| Translation (Qwen3.8-27B, 4-bit, vLLM) | ~26 GB |
| Voice cloning | ~10 GB |
| Speech recognition | ~8 GB |
| Dialogue separation | ~6 GB |
| Everything else | 2–4 GB |

On a card with less than about 28 GB, the built-in translation model won't
fit. Point translation at an OpenAI-compatible server you run elsewhere
instead (`llm_endpoint`, see [Configuration](configuration.md#translation-on-another-machine)).
By the numbers above, every other step needs 10 GB or less, but only the
32 GB setup has been tested.

**Docker** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
so that `--runtime=nvidia` works. On Unraid, install the **Nvidia Driver**
plugin.

**Memory.** Ai Dub has only been run on a server with 128 GB of RAM. The mix
holds the whole film's audio at once: about 60 GB at its peak for a two-hour 5.1
film, less for stereo and shorter episodes. Allow for that on top of whatever
else the server runs. How much less RAM it can manage with hasn't been measured.

**Disk:**

| Folder | Size |
|---|---|
| Models (`/models`) | ~95 GB once everything is installed: ~50 GB of model downloads plus the Python environments and caches. |
| Work (`/work`) | Several GB per episode while it runs, ~13 GB for a two-hour film. When a dub finishes, most of it is deleted. What's needed to check the dub and redub it is kept until you check it. |
| Media | Room for one extra copy of each dubbed file until you keep the dub, because the original is set aside rather than deleted. |

Put `/models` and `/work` on fast local storage (SSD or NVMe) if you can.
Reading model weights from a slow array makes every step start slowly.

**Network.** The model downloads come from Hugging Face, ModelScope, GitHub
and PyPI. After installation, Ai Dub needs no internet; Bazarr is the only
service it talks to, and only if you switch it on.

## Folders and mounts

| Container path | What it holds | Notes |
|---|---|---|
| `/config` | Settings (`config.yaml`), keys (`secrets.yaml`), the job database (`aidub.db`), show profiles, generic voices, the queue pause flag | Small; **back this up** |
| `/models` | Model environments (`envs/`), weights (`hf/`, `modelscope/`, …), the uv cache | Large; can be downloaded again |
| `/work` | One `job-<id>` folder per job | Temporary; can have its own drive (see [below](#putting-the-work-folder-on-another-drive)) |
| `/media/movies`, `/media/shows`, `/media/anime` | Your media | Must be **read-write**: the new track is added to the file in place |

The three media paths are the defaults in `config.yaml`. To use other folder
names, or more folders, mount them and list them under `media:` (see
[Configuration](configuration.md#media-folders)). Ai Dub can only see and change
files under the folders listed there.

Files are written as `PUID:PGID` (99:100 by default, Unraid's `nobody:users`).
Set these to the user that owns your media, or the new files may end up with
an owner your media server can't read.

## Install

### Docker

```bash
docker run -d --name ai-dub --runtime=nvidia --restart unless-stopped \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PUID=99 -e PGID=100 \
  -e TZ=America/Los_Angeles \
  -p 8790:8790 \
  -v /path/to/appdata:/config \
  -v /path/to/models:/models \
  -v /path/to/work:/work \
  -v /path/to/movies:/media/movies \
  -v /path/to/shows:/media/shows \
  -v /path/to/anime:/media/anime \
  --log-opt max-size=50m --log-opt max-file=1 \
  ghcr.io/pingpongshow/ai-dub:latest
```

Leave out any media mount you don't have. `TZ` only affects the times shown
in logs. The two `--log-opt` flags cap the container log's size.

### Docker Compose

```yaml
services:
  ai-dub:
    image: ghcr.io/pingpongshow/ai-dub:latest
    container_name: ai-dub
    runtime: nvidia
    restart: unless-stopped
    ports:
      - "8790:8790"
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      PUID: "1000"
      PGID: "1000"
      TZ: Europe/London
      AIDUB_INSTALL_ON_START: "false"
    volumes:
      - ./appdata:/config
      - /fast/disk/ai-dub/models:/models
      - /fast/disk/ai-dub/work:/work
      - /srv/media/movies:/media/movies
      - /srv/media/tv:/media/shows
      - /srv/media/anime:/media/anime
    logging:
      options:
        max-size: "50m"
        max-file: "1"
```

### Unraid

1. Install the **Nvidia Driver** plugin and reboot, if you haven't already.
2. Copy [`docker/unraid/ai-dub.xml`](../docker/unraid/ai-dub.xml) to
   `/boot/config/plugins/dockerMan/templates-user/my-ai-dub.xml`.
3. **Docker → Add Container**, choose **ai-dub** from the template list, and
   set the paths.

   The template's defaults put models and work under
   `/mnt/user/appdata/ai-dub/`. Point them at a cache pool or other fast disk
   if you can. To change this later, see
   [Putting the work folder on another drive](#putting-the-work-folder-on-another-drive).
4. Leave **PUID 99 / PGID 100**, which match Unraid's usual media owner.

### The image

`ghcr.io/pingpongshow/ai-dub:latest` is built from `main` by GitHub Actions
after the tests pass. Each build is also tagged with its short commit hash
(`sha-2b6d8b5`). To pin a known-good build, use that tag instead of
`latest`.

The image contains the app, ffmpeg, mkvtoolnix, rubberband, sox and
[uv](https://github.com/astral-sh/uv). It has no models, and no model
environments.

## First run

Open `http://<server>:8790`.

1. **Models → Install all missing.** Each model installs one at a time, into
   its own Python environment under `/models/envs`, and then downloads its
   weights. Expect about 50 GB of downloads; how long that takes depends on your
   connection. **Log** shows each model's progress. A model that fails can
   be installed again on its own.

   To install everything when the container starts, set
   `AIDUB_INSTALL_ON_START=true`. Do this for one start only: it is quick when
   nothing is missing, but it's easy to forget it is on.
2. **Models → Make generic voices** (once, after the voice-design model is
   installed). This builds a small set of stock voices, used for speakers
   with too little clean audio to clone.
3. **Check the install**:

   ```bash
   docker exec -it ai-dub aidub doctor
   ```

   Each model should report `ok` and name your GPU. `not installed` means its
   install hasn't finished, or failed.
4. **Settings**: choose the review mode. If you use Bazarr, turn it on,
   enter its address, map its folders and add `BAZARR_API_KEY` under Keys.
   **Restart the container** after changing settings or keys, then press
   **Test** next to Bazarr's address.
5. **Try one short episode.** Queue it with **Review before dubbing: Always**,
   so that you see what Ai Dub found before it spends time voicing. Check the
   result on the review page's Check tab, then keep the dub or put the original back.

A no-risk test from the command line dubs the first five minutes into a
separate file and never touches the source:

```bash
docker exec -it ai-dub aidub run "/media/anime/Show/Season 1/Show S01E01.mkv" \
  --preview 5m --output /work/preview-test.mkv
```

The preview lands in your work folder.

## What a typical episode costs

On the RTX 5090, a 24-minute episode takes 12–14 minutes; a film takes
proportionally longer. Translation and voice cloning take most of that time.
Once a few dubs have finished, the queue shows its own estimate for each
job.

## Updating

1. Wait until nothing is running. A job paused at **review**, or a dub
   waiting to be checked, is fine to leave.

   A restart in the middle of a job is safe too: the job resumes from its
   first unfinished step. But the step that was interrupted starts again,
   which can waste up to half an hour on translation or voicing.
2. Pull the new image and recreate the container with the same settings:

   ```bash
   docker pull ghcr.io/pingpongshow/ai-dub:latest
   docker stop ai-dub && docker rm ai-dub
   # the same docker run command as before
   ```

   With Compose: `docker compose pull && docker compose up -d`. On Unraid:
   **Docker → Check for updates → Apply update**.
3. Open browser tabs reload themselves when they notice the new version. A
   review page with unsaved changes asks you to save first.

Models live in `/models` and survive updates. A model needs reinstalling
only if the release notes say so (**Models → Reinstall**).

Old images build up. Remove the previous one by its ID (`docker images`,
then `docker rmi <id>`).

## Putting the work folder on another drive

Every job's working files go in `/work`. It's the most heavily written folder:
allow about 15 GB for each title that is running or waiting to be checked. You
can put it on any local drive, such as a dedicated SSD, by changing which
folder on the host is mapped to `/work`. The web page has no setting for this,
because the container can only see folders that are mapped into it.

**Choosing the drive.** Use a local SSD or NVMe, not a network share. It must
be mounted before Docker starts. If it isn't, Docker creates an empty folder
in its place and the work goes to the wrong disk (on Unraid, into RAM). On
Unraid, add the drive as its own pool under **Main**, and use its direct path
(`/mnt/scratch/…`) rather than `/mnt/user/…`.

**Moving it:**

1. Pause the queue and let the running job finish. Dubs waiting for review or
   a check can stay as they are; step 3 takes their files along.
2. Stop the container.
3. Copy the folder's contents to the new place. Keep the trailing slashes:

   ```bash
   rsync -a /old/path/work/ /mnt/scratch/ai-dub-work/
   ```

4. Change the host side of the mapping, and leave the container side as `/work`:
   - **Unraid:** Docker → **ai-dub** → **Edit**, set **Work** to
     `/mnt/scratch/ai-dub-work`, then **Apply**.
   - **Compose or `docker run`:** change the left side of the volume, as in
     `/mnt/scratch/ai-dub-work:/work`, and create the container again.
5. Check that it took. This should name the new drive:

   ```bash
   docker exec ai-dub df -h /work
   ```

   Then open a dub that's waiting to be checked and play it.
6. Resume the queue, and delete the old folder once you're happy.

Don't change the container side or `paths.work`. Saved jobs store full paths
under `/work`, so they would stop working. If you skip step 3, jobs in progress
or waiting to be checked lose their working files.

**What stays where it is:**

- **The finished file.** It's written next to the original on your media drive,
  so that it can be swapped in safely. Delivery needs room there, whichever
  drive `/work` is on.
- **Model weights.** `/models` is a separate mapping. Move it the same way if
  you want it on the new drive too.

## Backups

Back up `/config`. It holds everything you've taught Ai Dub:

- each show's voices and characters;
- its glossary and rules;
- the job history;
- your settings and keys.

Copy it while the container is stopped, or at least while no job is running.
`aidub.db` is SQLite, so copying it mid-write can give a torn copy.

`/models` can be downloaded again. `/work` is temporary, except for dubs
waiting to be checked: their check previews live there. It's large and changes
constantly, so leave it out of scheduled backups.

## Removing Ai Dub

Before you remove it, open **Originals** and deal with every set-aside file:
**Restore** puts the original back, **Delete** keeps the dub. Ai Dub never
deletes these files itself. Without it, they sit next to your media as
`*.aidub-original` until you clean them up by hand.

Then remove the container, the image, and the `/models` and `/work` folders.
