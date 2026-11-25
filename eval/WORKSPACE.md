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
usdx_medium 0.609, usdx_hard 0.444, WER 0.358, line F1 0.463.

## Tools
- `splits/`: SingStar split and quick bench (frozen; don't re-roll for nicer scores)
- `fit_line_breaks.py`: refit the line-break model
- `check_charts.py`, `error_analysis.py`, `compare.py`, `make_clips.py` (chart cutting helpers)
- `experiments/`: one-off studies; `experiments/pre_pipeline/`: tools of the old monolith (archived)
