[![Discord](https://img.shields.io/discord/1048892118732656731?logo=discord)](https://discord.gg/rYz9wsxYYK)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakuri255/UltraSinger/blob/master/colab/UltraSinger.ipynb)
![Status](https://img.shields.io/badge/status-development-yellow)

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rakuri255/UltraSinger/main.yml)
[![GitHub](https://img.shields.io/github/license/rakuri255/UltraSinger)](https://github.com/rakuri255/UltraSinger/blob/main/LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/rakuri255/ultrasinger/badge)](https://www.codefactor.io/repository/github/rakuri255/ultrasinger)
[![Check Requirements](https://github.com/rakuri255/UltraSinger/actions/workflows/main.yml/badge.svg)](https://github.com/rakuri255/UltraSinger/actions/workflows/main.yml)
[![Pytest](https://github.com/rakuri255/UltraSinger/actions/workflows/pytest.yml/badge.svg)](https://github.com/rakuri255/UltraSinger/actions/workflows/pytest.yml)
[![docker](https://github.com/rakuri255/UltraSinger/actions/workflows/docker.yml/badge.svg)](https://hub.docker.com/r/rakuri255/ultrasinger)

<p align="center" dir="auto">
<img src="https://repository-images.githubusercontent.com/594208922/4befe3da-a448-4cbc-b6ef-93899119071b" style="height: 300px;width: auto;" alt="UltraSinger Logo">
</p>

# UltraSinger

> ⚠️ _This project is still under development!_

UltraSinger is a tool to automatically create UltraStar.txt, midi and notes from music.
It automatically pitches UltraStar files, adding text and tapping to UltraStar files and creates separate UltraStar karaoke files.
It also can re-pitch current UltraStar files and calculates the possible in-game score.

Multiple AI models are used to extract text from the voice and to determine the pitch.

Please mention UltraSinger in your UltraStar.txt file if you use it. It helps others find this tool, and it helps this tool get improved and maintained.
You should only use it on Creative Commons licensed songs.

## ❤️ Support
There are many ways to support this project. Starring ⭐️ the repo is just one 🙏

You can also support this work on <a href="https://github.com/sponsors/rakuri255">GitHub sponsors</a> or <a href="https://patreon.com/Rakuri">Patreon</a> or <a href="https://www.buymeacoffee.com/rakuri255" target="_blank">Buy Me a Coffee</a>.

This will help me a lot to keep this project alive and improve it.

<a href="https://www.buymeacoffee.com/rakuri255" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="height: 60px !important;width: 217px !important;" ></a>
<a href="https://patreon.com/Rakuri"><img src="https://raw.githubusercontent.com/rakuri255/UltraSinger/main/assets/patreon.png" alt="Become a Patron" style="height: 60px !important;width: 217px !important;"/> </a>
<a href="https://github.com/sponsors/rakuri255"><img src="https://raw.githubusercontent.com/rakuri255/UltraSinger/main/assets/mona-heart-featured.webp" alt="GitHub Sponsor" style="height: 60px !important;width: auto;"/> </a>

## Table of Contents

- [UltraSinger](#ultrasinger)
  - [❤️ Support](#️-support)
  - [Table of Contents](#table-of-contents)
  - [💻 How to use this source code](#-how-to-use-this-source-code)
    - [Installation](#installation)
    - [Run](#run)
  - [📖 How to use the App](#-how-to-use-the-app)
    - [Presets and stages](#presets-and-stages)
    - [📟 Use GPU](#-use-gpu)
      - [Considerations for Windows users](#considerations-for-windows-users)
      - [Crashes due to low VRAM](#crashes-due-to-low-vram)
    - [📦 Containerized](#containerized-docker-or-podman)

## 💻 How to use this source code

### Installation

* Install Python 3.12 **(older/newer versions may have compatibility issues)**. [Download](https://www.python.org/downloads/)
* Also download or install ffmpeg with PATH. [Download](https://www.ffmpeg.org/download.html)
* Go to folder `install` and run install script for your OS:
  * Choose `GPU` if you have an NVIDIA CUDA GPU.
  * Choose `CPU` if you don't have an NVIDIA GPU or want CPU-only processing.

### Run

* In root folder just run `run_on_windows.bat`, `run_on_linux.sh` or `run_on_mac.command` to start the app.
* Now you can use the UltraSinger source code with `py UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]`. See [How to use](#-how-to-use-the-app) for more information.

## 📖 How to use the App

```commandline
python src/UltraSinger.py -i "Artist - Title.mp3"          # audio or video file
python src/UltraSinger.py -i https://www.youtube.com/...   # YouTube
python src/UltraSinger.py -i chart.txt                     # re-pitch an existing UltraStar chart
```

Options:

```commandline
  -i, --input        audio/video file, YouTube URL, or UltraStar txt (re-pitch)
  -o, --output       output folder (default: 'output' next to the input)
  --preset NAME      processing preset from presets/ (default: best; repitch for a txt input)
  --set OVERRIDE     change the preset (repeatable, applied in order):
                       step.param=value        a parameter
                       step.after=other        move a step (also step.before=other)
                       step.use=other_stage    swap a stage
                       -step                   drop a step (write it as --set=-step)
                       +stage[#id][@after=step]  insert a step
  --language CODE    skip language detection, e.g. en, de
  --midi             also write a MIDI file
  --no-stems         don't write the vocal/instrumental audio
  --no-lookup        no MusicBrainz lookup: artist and title from the file name
  --cpu              never use the GPU
  --cache DIR        persistent cache (default: inside the song folder, removed at the end)
  --keep-cache       keep the song's cache folder
  --fresh STEP       recompute STEP, ignoring its cache
  --cookiefile FILE  cookies for YouTube
  --ffmpeg PATH      ffmpeg location if it is not on PATH
  -q, --quiet        console: warnings and errors only
  --debug            console: everything live (library output, progress bars)
  --list             list all stages and their parameters
  --check            validate the preset (with your --set changes) and exit
  --where STEP       show every position where STEP could run, and exit
  --estimate         estimated runtime for the input, from earlier runs on this machine, and exit
```

### Presets and stages

Every processing decision lives in a preset (`presets/*.yaml`): an ordered list of stages with their
parameters. `--list` shows every stage, what it reads and writes, and its parameters.

| preset | what it is |
|---|---|
| `best` | best known configuration: htdemucs_ft, Whisper large-v3-turbo, LRCLIB lyrics correction, RMVPE pitch, GAME note segmentation, edge snapping, timing calibration, planned line breaks |
| `cpu_fullsong` | like best with lighter models (htdemucs, Whisper medium) for CPU-only machines |
| `repitch` | keep a chart's syllables and timing, detect only the pitches |
| `bench`, `oracle_lyrics` | evaluation presets (see `eval/`) |

Examples:

```commandline
python src/UltraSinger.py -i song.mp3 --set transcribe.whisper.model=medium --set separate.model=htdemucs
python src/UltraSinger.py -i song.mp3 --set=-lyrics.lrclib --set=-words.correct_lyrics
python src/UltraSinger.py --where words.strip_punctuation          # where could this step go?
python src/UltraSinger.py -i song.mp3 --set words.strip_punctuation.after=syllables.hyphenate
```

Text stages work on whichever text level is current: moved after hyphenation, punctuation
cleanup, silence splitting or a segmenter operate on syllables instead of words.

Each output folder contains the chart, `lineage.json` (time, code version, every step and
parameter that produced it) and `run_log.jsonl` (per-stage timings, cache hits, memory, and every
message including library warnings and output, which the console shows only once and in short).
Every computed stage also records its runtime (with audio length, device, parameters) in
`~/.cache/ultrasinger/runtimes.jsonl`; runs log an estimate up front and `--estimate` prints one.

Models with their own licenses: GAME note segmentation (CC BY-NC-SA 4.0, set
`segment.game.model_dir`; `segment.contour` is a license-free alternative), the punctuation model
for line breaks (MIT, downloaded on first use).

### 📟 Use GPU

With a GPU you can speed up the process. Also the quality of the transcription and pitching is better.

You need an NVIDIA CUDA device for this to work. Sorry, there is no CUDA device for macOS.

For GPU support on Windows and Linux, the installation script automatically installs PyTorch with CUDA support.

It is optional (but recommended) to install the CUDA driver for your GPU: see [CUDA driver](https://developer.nvidia.com/cuda-downloads).
Also check your GPU CUDA support. See [CUDA support](https://gist.github.com/standaloneSA/99788f30466516dbcc00338b36ad5acf)

For manual installation, you can use:
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

#### Crashes due to low VRAM

If something crashes because of low VRAM, use a smaller Whisper model or batch size
(`--set transcribe.whisper.model=medium`, `--set transcribe.whisper.batch_size=2`) or the
`cpu_fullsong` preset. Whisper needs more than 8GB VRAM in the `large` models.

You can also force CPU usage with `--cpu`.

### 📦 Containerized (Docker or Podman)

See [container/README.md](container/README.md)
