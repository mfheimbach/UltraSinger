# UltraSinger evaluation harness

Measures how close UltraSinger's output gets to hand-made UltraStar charts,
so changes can be judged by numbers instead of by ear.

## Quick start

```bash
uv pip install -r eval/requirements.txt          # mir_eval, jiwer (on top of uv sync)
bash eval/fetch_dev_set.sh                        # 7 Creative Commons songs -> eval_data/dev
python eval/run_eval.py --songs eval_data/dev --name baseline
python eval/run_eval.py --songs eval_data/dev --name repitch --mode repitch
python eval/compare.py eval_work/runs/baseline eval_work/runs/my_change
```

Any `Settings` field can be overridden per run:

```bash
python eval/run_eval.py --songs eval_data/dev --name medium --set whisper_model=medium --set whisper_batch_size=4
```

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

## Metrics

Frame level, 10 ms grid (robust to different note splits):

- `usdx_medium`: share of chart points a singer would get on the real chart by
  singing the generated one perfectly (USDX medium: within ±1 semitone,
  octave ignored, golden notes count double). The single headline number.
- `chroma_acc` / `chroma_acc_pm1`: pitch correct (±0 / ±1 semitone, octave
  ignored) over ground-truth pitched frames; unvoiced predictions count as wrong.
- `chroma_acc_cond`: same, only where something was predicted. Separates
  "wrong pitch" from "missed the note".
- `voicing_recall` / `voicing_precision` / `voicing_f1`: singing detected when it should be.
- `transpose`: dominant semitone shift vs. the chart. Nonzero means the chart
  is transposed relative to this audio (wrong version) — fix or drop the song.
- `best_offset_ms`: how late the prediction is overall. Large values mean a
  GAP/offset problem, not note problems.

Note level: `onset_f1_50ms`, `onset_f1_100ms`, `onset_err_median_ms`,
`note_count_ratio` (>1 = too many notes), `note_f1_onset_chroma`.

Lyrics: `wer`, `cer` (case and punctuation ignored).

Repitch mode: `note_chroma_acc`, `note_chroma_acc_pm1`,
`note_chroma_acc_dur` (duration weighted), `octave_errors`.

## Caching

Each song gets `eval_work/cache/<song>/`, shared by all runs: Demucs stems,
transcription (keyed by Whisper config) and pitch.

Caveat: UltraSinger's pitch cache key is only `swiftf0_<mode>`. It doesn't
include SwiftF0 parameters or whether separation was on. When changing
anything that affects pitch, add `--set skip_cache_pitch_detection=true`.

## Notes

- `--only` / `--exclude` filter songs by folder-name substring.
- Runs each song in a subprocess (UltraSinger keeps settings in module globals).
- MusicBrainz lookup and sheet-music export are stubbed out; metadata comes from the chart.
- Whisper large-v2 needs more than 8 GB RAM on CPU; use `medium` on small machines.
