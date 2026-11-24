"""Mixture-of-experts study for note pitch: can a selector beat GAME alone?
Needs the export_note_candidates patch from branch moe-experiment and runs with export_note_candidates=true (prefix moe_usdb_export / moe_dev_export).
Leave-one-song-out CV; prints GAME/RMVPE/SwiftF0 alone, 2-of-3 vote, oracle and learned selectors."""
import sys, json, glob, os, numpy as np
sys.path.insert(0,'eval')
import usdx, run_eval
from sklearn.linear_model import LogisticRegression
SHIFT = -0.07
data = []  # per segment: song, dur, label, candidates dict
for set_ in ['usdb','dev']:
    for name, txt in run_eval.discover(f'eval_data/{set_}_clips', None, ['Systemabsturz']):
        fs = glob.glob(f'eval_work_clips/runs/moe_{set_}_export/{run_eval.slug(name)}/*/note_candidates.json')
        if not fs: continue
        rows = json.load(open(fs[0])); gt = usdx.read(txt)
        gn = [n for n in gt.notes if n.pitched]
        for r in rows:
            s, e = r['start'] + SHIFT, r['end'] + SHIFT
            if e - s <= 0.02: continue
            ov = {}
            for n in gn:
                o = min(e, n.end) - max(s, n.start)
                if o > 0: ov[n.pitch] = ov.get(n.pitch, 0) + o
            if not ov: continue
            p, o = max(ov.items(), key=lambda kv: kv[1])
            if o < 0.5 * (e - s): continue
            data.append((name, e - s, p, r))
print(len(data), 'labelled segments')
C = ['game', 'rmvpe', 'swiftf0']
def ok(cand, lab):
    return cand is not None and (int(round(cand)) - lab) % 12 == 0
def feats(r, c):
    x = r.get(c)
    others = [r.get(o) for o in C if o != c]
    agree = sum(1 for o in others if o is not None and x is not None and (int(round(o)) - int(round(x))) % 12 == 0)
    return [C.index(c) == 0, C.index(c) == 1, C.index(c) == 2,
            abs(x - round(x)) if x is not None else 0.5,
            r.get(c + '_conf', 0) or 0, r.get(c + '_voiced', 0) or 0, min(r.get(c + '_spread', 2) or 2, 3),
            agree, np.log(r['end'] - r['start'])]
songs = sorted(set(d[0] for d in data))
w = np.array([d[1] for d in data])
res = {k: [] for k in ['game', 'rmvpe', 'swiftf0', 'vote', 'oracle', 'selector']}
for test in songs:
    tr = [d for d in data if d[0] != test]; te = [d for d in data if d[0] == test]
    X, y = [], []
    for _, dur, lab, r in tr:
        for c in C:
            if r.get(c) is None: continue
            X.append(feats(r, c)); y.append(ok(r[c], lab))
    clf = LogisticRegression(max_iter=1000).fit(np.array(X, float), np.array(y))
    for _, dur, lab, r in te:
        res['game'].append((dur, ok(r['game'], lab)))
        res['rmvpe'].append((dur, ok(r['rmvpe'], lab)))
        res['swiftf0'].append((dur, ok(r['swiftf0'], lab)))
        cands = [c for c in C if r.get(c) is not None]
        votes = {}
        for c in cands: votes.setdefault(int(round(r[c])) % 12, []).append(c)
        best = max(votes.values(), key=len) if votes else []
        pick = 'game' if (r.get('game') is not None and (len(best) < 2 or 'game' in best)) else (best[0] if best else None)
        res['vote'].append((dur, ok(r.get(pick), lab) if pick else False))
        res['oracle'].append((dur, any(ok(r.get(c), lab) for c in cands)))
        if cands:
            probs = clf.predict_proba(np.array([feats(r, c) for c in cands], float))[:, 1]
            res['selector'].append((dur, ok(r[cands[int(np.argmax(probs))]], lab)))
        else:
            res['selector'].append((dur, False))
for k, v in res.items():
    d = np.array(v, float)
    print(f"{k:9s} accuracy {d[:,1].mean():.3f}  (duration-weighted {np.average(d[:,1], weights=d[:,0]):.3f})")

# --- non-linear selector with richer features
from sklearn.ensemble import HistGradientBoostingClassifier
def feats2(r, c):
    base = feats(r, c)
    x = r.get(c)
    diffs = []
    for o in C:
        v = r.get(o)
        if v is None or x is None: diffs.append(6.0); continue
        d = abs(x - v) % 12; diffs.append(min(d, 12 - d))
    return base + diffs + [x % 12 if x is not None else -1, x if x is not None else 0]
out = []
for test in songs:
    tr = [d for d in data if d[0] != test]; te = [d for d in data if d[0] == test]
    X, y = [], []
    for _, dur, lab, r in tr:
        for c in C:
            if r.get(c) is None: continue
            X.append(feats2(r, c)); y.append(ok(r[c], lab))
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05).fit(np.array(X, float), np.array(y))
    for _, dur, lab, r in te:
        cands = [c for c in C if r.get(c) is not None]
        if not cands: out.append((dur, False)); continue
        probs = clf.predict_proba(np.array([feats2(r, c) for c in cands], float))[:, 1]
        out.append((dur, ok(r[cands[int(np.argmax(probs))]], lab)))
d = np.array(out, float)
print(f"selector2 accuracy {d[:,1].mean():.3f}  (duration-weighted {np.average(d[:,1], weights=d[:,0]):.3f})")
