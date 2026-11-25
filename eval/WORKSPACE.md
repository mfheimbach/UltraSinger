# Eval workspace map

## Data (`eval_data/`, not in git)
- `usdb/` 9 USDB full songs (5 EN, 3 DE, 1 JA = Butter-fly, outlier), `dev/` 7 CC songs (Systemabsturz excluded)
- `usdb_clips/`, `dev_clips/`: 45 s "most sung" clips (old workbench, biased to loud/dense parts)
- `bench/`: new workbench, 16 windows of ~20 s (8 diverse + 8 weak spots), built by `make_bench.py`

## Work dirs (not in git)
- `eval_work/`: full songs; `eval_work_clips/`: old clips; `eval_work_bench/`: new bench
- caches per song: `<work>/cache/<song>/` (+ `demucs_<model>[+second pass]/` per separator)
- `eval_work/all_runs_metrics.csv.gz`: metrics of all 345 runs up to 2026-11-25 (columns: work, run, song, metrics, config)
- reference runs kept: `fullsong_baseline` (upstream), `fullsong2_game_all`, `fsnap_s50e06`, `plf_*` (current best full songs), `plfdev_planned`, `devfs_best`, `al_*` (clip baselines)

## Configs (`eval/configs/`)
- `best_clips.txt`: best config for clips/bench (htdemucs_ft, large-v3-turbo)
- `best_fullsong_cpu.txt`: CPU proxy for full songs (htdemucs, Whisper medium) incl. LRCLIB

Usage: `B=$(cat eval/configs/best_clips.txt); python eval/batch_eval.py --songs eval_data/bench --work eval_work_bench --prefix x --base $B --jitter 0.99 1 1.01 --config name:key=value`
- `/`-separated lists inside a value (commas split overrides), e.g. `pitch_shift_passes=-1/0/1`
- fresh transcriptions: pre-generate with `run_eval.py` (one process per song) to avoid OOM

## Tools
- `bench_profile.py` + `make_bench.py`: window features/scores of full songs -> workbench
- `fit_line_breaks.py`: refit the line-break model
- `experiments/`: one-off studies (paths inside point to /tmp outputs): transcript voting, pitch-variant voting, bench PCA, line-break LOSO
