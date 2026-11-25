# UltraSinger evaluation harness

Measures how close UltraSinger's output gets to hand-made UltraStar charts,
so changes can be judged by numbers instead of by ear.

## Quick start

```bash
uv pip install -r eval/requirements.txt          # mir_eval, jiwer (on top of uv sync)
python eval/run_eval.py --songs eval_singstar/bench --work eval_work_ng --name baseline --preset bench --gt-language
python eval/run_eval.py --songs eval_singstar/bench --work eval_work_ng --name repitch --mode repitch
python eval/compare.py eval_work_ng/runs/baseline eval_work_ng/runs/my_change
```

Runs use pipeline presets (`presets/*.yaml`); any stage parameter can be changed per run, and
steps can be dropped:

```bash
python eval/run_eval.py ... --name game03 --set segment.game.boundary_threshold=0.3
python eval/run_eval.py ... --name nosnap --set=-notes.snap_edges
python eval/batch_eval.py --songs eval_singstar/bench --work eval_work_ng --prefix gt --preset bench \
    --gt-language --config base: --config t03:segment.game.boundary_threshold=0.3
```

Current data, split and baseline: see `WORKSPACE.md`.

## Test set layout

One folder per song, each with an UltraStar txt and the audio it references
(`#MP3`/`#AUDIO`). This is what USDB Syncer produces, so a real test set is:
pick songs in USDB Syncer, download, point `--songs` at the folder.
Audio never goes into git (`eval_data/` and `eval_work/` are ignored).

## Modes

| mode | input | what it isolates |
|---|---|---|
| `full` | audio | the whole pipeline: separation, transcription, segmentation, pitch |
| `repitch` | ground-truth txt + audio | pitch detection only (timings and lyrics come from the chart) |

## Clips, oracle lyrics, sweeps

- `splits/make_quick_bench.py` builds the quick bench (random 30 s windows with at least 40 %
  singing; the old "densest stretch" clips of `make_clips.py` were biased to loud parts).
- `run_eval.py ... --oracle-lyrics` aligns the chart's own lyrics (plain text, no
  timing) instead of transcribing (preset `oracle_lyrics`): the upper bound for
  "download lyrics, then align".
- Alternative transcriber: `--set transcribe.whisper.use=transcribe.qwen3` swaps the stage
  (needs `pip install qwen-asr`; if it bumps `opentelemetry-api`, pin it back to the version in
  uv.lock or WhisperX's VAD breaks).

## Metrics

Frame level, 10 ms grid (robust to different note splits):

- `usdx_medium`: share of chart points a singer would get on the real chart by
  singing the generated one perfectly (USDX medium: within ±1 semitone,
  octave ignored, golden notes count double). The single headline number.
- `usdx_hard`: the same with exact pitch class required (USDX hard). Key
  quantization trades one against the other, so watch both.
- `pitch_mae`: mean octave-folded pitch distance in semitones (0–6) on frames where
  both chart and output have a note. Continuous, so it separates "1 off" from "5 off".
- `chroma_acc` / `chroma_acc_pm1`: pitch correct (±0 / ±1 semitone, octave
  ignored) over ground-truth pitched frames; unvoiced predictions count as wrong.
- `chroma_acc_cond`: same, only where something was predicted. Separates
  "wrong pitch" from "missed the note".
- `voicing_recall` / `voicing_precision` / `voicing_f1`: singing detected when it should be.
- `transpose` (repitch mode): dominant semitone shift vs. the chart. Nonzero means
  the chart is transposed relative to this audio (wrong version) — fix or drop the
  song (`--exclude`). Full mode reports the same as `pitch_shift`, informational
  only, since there UltraSinger itself can cause a shift.
- `best_offset_ms`: how late the prediction is overall. Large values mean a
  GAP/offset problem, not note problems.

Note level: `onset_f1_50ms`, `onset_f1_100ms`, `onset_err_median_ms`,
`note_count_ratio` (>1 = too many notes), `note_f1_onset_chroma`.

Lyrics: `wer`, `cer` (case and punctuation ignored).

Repitch mode: `note_chroma_acc`, `note_chroma_acc_pm1`,
`note_chroma_acc_dur` (duration weighted), `octave_errors`.

## Caching

Each song gets `<work>/cache/<song>/`, shared by all runs. Every stage result is cached under a key
made of: stage name, version and parameters, the content of its inputs, the upstream chain of steps
(order matters; `--cache-by content` ignores it), the stage's source code (and the `modules.*`
files it imports, except for the expensive model stages), and the content of model files given as
parameters. Each cache file has a `.lineage.json` with every step order that produced it.

## Notes

- `--only` / `--exclude` filter songs by folder-name substring.
- `check_charts.py --songs DIR` checks chart timing against the separated vocals
  (independent of UltraSinger); combine with a repitch run's `transpose` to vet a new test set.
- `run_eval.py` runs each song in a subprocess (bounded memory); `batch_eval.py` runs all in one.
- No MusicBrainz lookup: metadata comes from the chart.
- Whisper large-v3 needs more than 8 GB RAM on CPU; use turbo or medium on small machines.
- Reproducibility: demucs' random shift and GAME's sampling noise are seeded, timings are rounded
  to 1 ms; the same input gives byte-identical charts.
