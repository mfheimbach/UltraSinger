# Eval workspace map

## Data (`eval_singstar/`, not in git: private, not redistributable)
- `raw/`: USDB Syncer download of 11 SingStar editions (.opus, untouched YouTube stream)
- `train/` (125), `test/` (59), `reserve/` (13), `hard/` (8): symlinks into `raw/`, frozen split from
  `eval/splits/singstar_split.csv` (grouped by artist, stratified by language x style, one song per
  artist in test). Test is scored at milestones only.
- `bench/`: quick bench, 12 x 30 s windows from 12 train artists (`eval/splits/make_quick_bench.py`)
- `bench_full/`: the same 12 songs at full length (confirm tier)

## Work dirs (not in git)
- `eval_work_ng/`: current work dir. `cache/<song>/` holds the pipeline stage caches
  (`stage_<name>_<key>.pkl` + `.lineage.json`), `runs/<name>/` the results.
- `eval/archive/all_runs_metrics_*.csv.gz`: metrics of every run before the SingStar switch.

## Running
- One config, one process per song: `python eval/run_eval.py --songs eval_singstar/bench --work eval_work_ng
  --name x --preset bench --gt-language [--set step.param=value] [--mode repitch] [--oracle-lyrics]`
- Many configs in one process: `python eval/batch_eval.py --songs eval_singstar/bench --work eval_work_ng
  --prefix x --preset bench --gt-language --config a: --config b:segment.game.boundary_threshold=0.3`
- Re-score without running: `run_eval.py ... --name x --score-only`
- `--gt-language` takes the language from the chart (30 s windows break Whisper's language detection).
- BPM jitter (`--jitter`) is no longer needed: with GAME segmentation it moves the bench mean by 0.002.
- Strict contracts: `ULTRASINGER_STRICT=1` turns contract warnings into errors.

## Baseline (quick bench, preset `bench`, 2026-11-25)
usdx_medium 0.609, usdx_hard 0.444, WER 0.358, line F1 0.463. Noise floor: demucs seeds alone move
the bench mean by ~0.02 (0.609 / 0.622 / 0.648); compare configs on the same seed, decide on
bigger sets.

## Tuning (`eval/tune.py`, specs in `tune/`)
- A spec names base preset, data (train / optional holdout / folds), objective (weighted metrics) and
  the search space: parameter ranges, discrete values, module alternatives (`step.use`), step order
  (`step.after`). Cheap dimensions (from the runtime records) run as in-memory tails (~0.5 s per trial
  on the bench); expensive ones (separation, Whisper) are discrete choices computed on demand.
- Reports artist-grouped CV gain with CI, the in-sample tuned config, fANOVA importance, holdout once.
  Output: `tune/out/<name>/report.md`, `preset.yaml` (base + overrides), `trials.csv`.
- Staged runs (`stages:`): each group starts from the last best; resume with
  `--start tune/out/<name>/preset.yaml --from-group N`. A group's search stops at whichever comes first:
  `trials`, no gain > min gain for `stop.patience` trials, `stop.max_seconds`. A group is adopted only if
  its CV gain > min gain = max(`min_gain`, `noise_sigma` x noise SD of the first chaotic module at or
  after the group's earliest step).
- Noise levels per engine live in `tune/noise.yaml` (per song set; re-measure when an engine changes,
  other engines borrow with a warning). **TODO (GPU run):** measure them with 10+ seeds / placebos
  per chaotic module (separate, Whisper, GAME); current n=3..6 is rough.
- `tune/notes_demo.yaml`: note parameters on the bench, CV gain +0.005 (CI -0.007..+0.015): the
  current values are near optimal for usdx_medium alone; end_extension wants to grow (metric gaming,
  add voicing precision to the objective).

## Tools
- `splits/`: SingStar split and quick bench (frozen; don't re-roll for nicer scores)
- `check_charts.py`, `error_analysis.py`, `compare.py`, `make_clips.py` (chart cutting helpers)
- `experiments/`: one-off studies; `experiments/archived/` and `experiments/pre_pipeline/`: superseded tools
