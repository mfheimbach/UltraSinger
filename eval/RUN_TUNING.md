# Running the staged tuning on your machine

Everything is checked up front by `eval/preflight.py`; if it passes, the run needs no attention.
A crashed or stopped run resumes where it stopped.

## 1. Setup (once)

```bash
git fetch && git checkout pipeline-refactor && git pull
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env   # uv, if missing
uv sync
uv pip install -r eval/requirements.txt   # mir_eval, jiwer, optuna
# GPU: GAME and RMVPE use onnxruntime; for CUDA there:  uv pip install onnxruntime-gpu
```

- **GAME model** (the zip holds the folder `GAME-1.0.3-small-onnx/`):
  ```bash
  mkdir -p ~/.cache/game && cd ~/.cache/game
  curl -LO https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-small-onnx.zip
  unzip GAME-1.0.3-small-onnx.zip && rm GAME-1.0.3-small-onnx.zip && cd -
  ```
- **Data** (private, never in git): the collection stays where it is; a data folder next to the repo
  holds a link to it and the generated windows, and the repo links to that folder:
  ```bash
  mkdir -p ../data/eval_singstar
  ln -s /home/mo/songs ../data/eval_singstar/raw               # <edition>/<song>/ as Syncer writes it
  ln -s "$(realpath ../data/eval_singstar)" eval_singstar      # gitignored (folder or link)
  uv run python eval/splits/make_tuning_sets.py --check        # every train song found?
  uv run python eval/splits/make_tuning_sets.py                # ~200 windows: tune_train + tune_holdout (~1 min)
  ```
  (The quick bench and selection windows from `ultrasinger_eval_windows.tar` are only needed for
  `tune/staged_singstar.yaml`, the small 12-song spec.)

## 2. Preflight

```bash
uv run python eval/preflight.py tune/staged_singstar_full.yaml
```

- It checks packages, ffmpeg, GPU, the spec (every search value builds a valid pipeline), the data, the GAME model and
  disk space.
- It then runs one song through the whole pipeline. The first time, this downloads the models (~5 GB).
- Every line should say OK/INFO and it ends with "all checks passed".

## 3. Run

```bash
nohup uv run python eval/tune.py tune/staged_singstar_full.yaml > tune_run.log 2>&1 &     # Linux/macOS
# Windows: run  uv run python eval/tune.py tune/staged_singstar_full.yaml > tune_run.log 2>&1  in a terminal you leave open
```

- **Device:** picked automatically (CUDA if available); the first log line says which.
- **Warm-up:** without runtime records on this machine, the tuner first runs the base pipeline on every tuning song. The
  timings decide which parameters are cheap to search.
- **Groups:** 19, early stages first. Each writes `tune/out/staged_singstar_full/report.md`, `preset.yaml` and `trials.csv`
  as it finishes.
- **Holdout:** at the end, `tune_holdout` (~50 windows, other artists) is scored once with the start and the tuned configuration.

**If it stops**, look at the last `=== N. <group>` line in the log and resume there (report and trials are appended):

```bash
nohup uv run python eval/tune.py tune/staged_singstar_full.yaml --from-group N >> tune_run.log 2>&1 &
```

## 4. Send back

`tune/out/staged_singstar_full/report.md`, `trials.csv`, `preset.yaml` and `tune_run.log`.

Notes:
- The noise thresholds in `tune/noise.yaml` were measured on CPU. On GPU (Whisper in float16) the chaos is
  similar but not identical; the GPU noise run (10+ seeds) will replace them.
- Lines starting `NO EFFECT` (a parameter changes no chart, e.g. punctuation windows on 30 s windows) and `FALLBACK`
  are information, not errors.
