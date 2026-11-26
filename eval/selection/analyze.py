"""Can we pick the better candidate without the chart?

    python eval/selection/analyze.py --work eval_work_ng --group sep=sel_sep0,sel_sep1,sel_sep2 \
        --group game=sel_sep0,sel_game1,sel_game2 [--target usdx_medium] [--out sel.csv]

Each group lists runs that differ in one parallel option; the first run is the default. For every
window/song, candidates get reference-free features (features.py) and their true score (from the
run's metrics.csv). Selectors are compared by the score of the candidate they pick:
  default      the first run
  oracle       the best candidate (upper bound)
  random       mean over candidates
  feature      argmax or argmin of one feature; the direction is learned on the other songs
  ranker       pairwise logistic regression on within-window feature differences
All learned selectors are validated leave-one-song-out (artist-level once songs share artists).
Bootstrap CIs over songs. The long table (one row per candidate) is written for R/pandas.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from features import agreement, features, load_candidate  # noqa: E402


def collect(work: str, group: str, runs: list[str], target: str) -> pd.DataFrame:
    metrics = {r: pd.read_csv(os.path.join(work, "runs", r, "metrics.csv")).set_index("song") for r in runs}
    songs = sorted(set.intersection(*(set(m.index) for m in metrics.values())))
    rows = []
    for song in songs:
        stores = {}
        for r in runs:
            d = os.path.join(work, "runs", r, _slug(song))
            if os.path.exists(os.path.join(d, "store.pkl")):
                stores[r] = load_candidate(d)
        if len(stores) < 2:
            continue
        agr = agreement(stores)
        for k, r in enumerate(runs):
            if r not in stores:
                continue
            row = {"group": group, "song": song, "candidate": r, "is_default": k == 0}
            for t in ("usdx_medium", "usdx_hard", "wer", "voicing_f1", "line_f1"):
                if t in metrics[r].columns:
                    row[f"y_{t}"] = metrics[r].loc[song, t]
            row.update(features(stores[r]))
            row.update(agr[r])
            rows.append(row)
    return pd.DataFrame(rows)


def _slug(name: str) -> str:
    import run_eval
    return run_eval.slug(name)


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.split("_", 1)[0] in ("sep", "pit", "txt", "not", "agr")
            and pd.api.types.is_numeric_dtype(df[c])]
    # only features that vary between candidates somewhere
    return [c for c in cols if df.groupby("song")[c].nunique().max() > 1]


def _pick(df_song: pd.DataFrame, col: str, sign: int) -> float:
    vals = df_song[col].to_numpy(float) * sign
    if np.all(np.isnan(vals)):
        return float(df_song.loc[df_song.is_default, "y"].iloc[0])
    return float(df_song["y"].iloc[int(np.nanargmax(vals))])


def evaluate(df: pd.DataFrame, target: str, higher_is_better: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["y"] = df[f"y_{target}"] * (1 if higher_is_better else -1)
    songs = df.song.unique()
    by = {s: g.reset_index(drop=True) for s, g in df.groupby("song")}
    cols = feature_cols(df)
    picks = {"default": [], "oracle": [], "random": []}
    for s in songs:
        g = by[s]
        picks["default"].append(float(g.loc[g.is_default, "y"].iloc[0]))
        picks["oracle"].append(float(g.y.max()))
        picks["random"].append(float(g.y.mean()))
    for c in cols:
        sel = []
        for s in songs:  # leave-one-song-out: learn the direction on the other songs
            others = df[df.song != s]
            dm = others.groupby("song")[[c, "y"]].transform(lambda x: x - x.mean())
            corr = np.nansum(dm[c] * dm["y"])
            sign = 1 if corr >= 0 else -1
            sel.append(_pick(by[s], c, sign))
        picks[f"feature:{c}"] = sel
    picks["ranker"] = _ranker_loso(df, cols, by, songs)
    base = np.array(picks["default"])
    oracle_gain = np.array(picks["oracle"]) - base
    rng = np.random.default_rng(0)
    out = []
    for name, vals in picks.items():
        gain = np.array(vals) - base
        boots = [gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(2000)]
        out.append({"selector": name, "mean": np.mean(vals) * (1 if higher_is_better else -1),
                    "gain": gain.mean() * (1 if higher_is_better else -1),
                    "ci_low": np.percentile(boots, 2.5) * (1 if higher_is_better else -1),
                    "ci_high": np.percentile(boots, 97.5) * (1 if higher_is_better else -1),
                    "share_of_oracle": gain.sum() / oracle_gain.sum() if oracle_gain.sum() > 0 else np.nan,
                    "wins": int((gain > 1e-9).sum()), "losses": int((gain < -1e-9).sum())})
    res = pd.DataFrame(out)
    if not higher_is_better:
        res[["ci_low", "ci_high"]] = res[["ci_high", "ci_low"]].values
    return res.sort_values("gain", ascending=not higher_is_better).reset_index(drop=True)


def _ranker_loso(df, cols, by, songs) -> list[float]:
    """Pairwise logistic regression on within-window feature differences (standardised)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return [float(by[s].loc[by[s].is_default, "y"].iloc[0]) for s in songs]
    X = df[cols].to_numpy(float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    X = np.nan_to_num(X)
    df = df.assign(_i=np.arange(len(df)))
    sel = []
    for s in songs:
        train = df[df.song != s]
        rows, ys = [], []
        for _, g in train.groupby("song"):
            idx = g._i.to_numpy()
            for a in range(len(idx)):
                for b in range(len(idx)):
                    if a != b and g.y.iloc[a] != g.y.iloc[b]:
                        rows.append(X[idx[a]] - X[idx[b]])
                        ys.append(int(g.y.iloc[a] > g.y.iloc[b]))
        g = df[df.song == s]
        if len(set(ys)) < 2:
            sel.append(float(g.loc[g.is_default, "y"].iloc[0]))
            continue
        D = np.array(rows)
        sd = D.std(axis=0) + 1e-9
        model = LogisticRegression(C=0.3, max_iter=2000).fit(D / sd, ys)
        scores = (X[g._i.to_numpy()] / sd) @ model.coef_[0]
        sel.append(float(g.y.iloc[int(np.argmax(scores))]))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--group", action="append", required=True, metavar="NAME=run0,run1,...")
    ap.add_argument("--target", default="usdx_medium")
    ap.add_argument("--out", help="long table (CSV) of candidates x features x scores")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    tables = []
    for spec in args.group:
        name, runs = spec.split("=", 1)
        df = collect(args.work, name, runs.split(","), args.target)
        tables.append(df)
        lower = args.target in ("wer", "cer")
        res = evaluate(df, args.target, higher_is_better=not lower)
        print(f"\n== {name}: {df.song.nunique()} songs x {df.candidate.nunique()} candidates, target {args.target}")
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(res.head(args.top).round(4).to_string(index=False))
    if args.out:
        pd.concat(tables).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
