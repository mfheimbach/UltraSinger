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
- **Data** (private, never in git): unpack `ultrasinger_eval_windows.tar` outside the repo and link it in, so the
  paths in the specs (`eval_singstar/bench`, `eval_singstar/selection`) work unchanged:
  ```bash
  mkdir -p ../data && tar -xf ultrasinger_eval_windows.tar -C ../data
  ln -s "$(realpath ../data/eval_singstar)" eval_singstar     # gitignored (folder or link)
  ```

## 2. Preflight

```bash
uv run python eval/preflight.py tune/staged_singstar.yaml
```

- It checks packages, ffmpeg, GPU, the spec (every search value builds a valid pipeline), the data, the GAME model and
  disk space.
- It then runs one song through the whole pipeline. The first time, this downloads the models (~5 GB).
- Every line should say OK/INFO and it ends with "all checks passed".

## 3. Run

```bash
nohup uv run python eval/tune.py tune/staged_singstar.yaml > tune_run.log 2>&1 &     # Linux/macOS
# Windows: run  python eval/tune.py tune/staged_singstar.yaml > tune_run.log 2>&1  in a terminal you leave open
```

- **Device:** picked automatically (CUDA if available); the first log line says which.
- **Warm-up:** without runtime records on this machine, the tuner first runs the base pipeline on the 12 songs. The
  timings decide which parameters are cheap to search.
- **Groups:** 19, early stages first. Each writes `tune/out/staged_singstar/report.md`, `preset.yaml` and `trials.csv`
  as it finishes.
- **Holdout:** at the end, the selection set (36 windows) is scored once with the start and the tuned configuration.

**If it stops**, look at the last `=== N. <group>` line in the log and resume there (report and trials are appended):

```bash
nohup uv run python eval/tune.py tune/staged_singstar.yaml --from-group N >> tune_run.log 2>&1 &
```

## 4. Send back

`tune/out/staged_singstar/report.md`, `trials.csv`, `preset.yaml` and `tune_run.log`.

Notes:
- The noise thresholds in `tune/noise.yaml` were measured on CPU. On GPU (Whisper in float16) the chaos is
  similar but not identical; the GPU noise run (10+ seeds) will replace them.
- Lines starting `NO EFFECT` (a parameter changes no chart, e.g. punctuation windows on 30 s windows) and `FALLBACK`
  are information, not errors.
