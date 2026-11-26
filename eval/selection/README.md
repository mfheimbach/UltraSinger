# Picking between parallel options without the chart

Wherever a step has parallel options (separation seeds or models, transcribers, GAME seeds, ...),
run each as a candidate, describe every candidate with reference-free features, and test which
features (or models over them) pick the better candidate.

```bash
# candidates = batch configs that differ in one option (first = default)
python eval/batch_eval.py --songs eval_singstar/bench --work eval_work_ng --prefix sel --preset bench \
    --gt-language --config sep0: --config sep1:separate.seed=1 --config sep2:separate.seed=2
python eval/selection/analyze.py --work eval_work_ng --group sep=sel_sep0,sel_sep1,sel_sep2 --out sel.csv
```

- `features.py`: separation (`sep_`), pitch track (`pit_`), transcript (`txt_`, incl. LM plausibility),
  notes (`not_`), agreement with the other candidates (`agr_`). Needs `store.pkl` in each run (written
  by `run_song.py`).
- `analyze.py`: default / oracle (best candidate per window, the upper bound) / random / single
  features (direction learned leave-one-song-out) / pairwise logistic ranker; gains with bootstrap CIs.
  `--out` writes the long table (one row per candidate) for R.
- Discovery on the bench, confirmation on `eval_singstar/selection` (36 windows, other train artists).

## First results (bench, 12 windows x 3 candidates, 2026-11-26)
- Demucs seeds: bench mean 0.609 / 0.622 / 0.648; oracle +0.057 over seed 0. Best single features:
  share of voiced frames covered by notes +0.049, unique-word share +0.047, word count +0.043,
  transcript agreement +0.041. Unconfirmed (multiple comparisons on 12 windows).
- GAME seeds: oracle only +0.008; not worth voting over.
