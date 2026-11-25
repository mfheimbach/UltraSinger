"""Quick bench: 12 x 30 s windows from 12 different TRAIN artists (SingStar split).

- songs: stratified by bucket, proportional to train (en_pop 4, en_rock 3, de_party 2,
  de_poprock 2, en_party 1); one song per artist group; fixed seed
- window: random line start whose 30 s window is at least 40% sung (no "most singing" pick:
  the old clips were biased loud/dense); same seed -> same windows
- confirm tier: the same 12 songs at full length (eval_singstar/bench_full, symlinks)
Frozen: re-running with the same seed reproduces it; don't re-roll for nicer scores.

  python eval/splits/make_quick_bench.py
"""
import csv, json, os, random, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "eval"))
import usdx  # noqa: E402
from make_clips import write_chart  # noqa: E402

SEED, SECONDS, MIN_SUNG = 20261125, 30.0, 0.40
QUOTA = {"en_pop": 4, "en_rock": 3, "de_party": 2, "de_poprock": 2, "en_party": 1}
LEAD_IN = 0.5


def main():
    rng = random.Random(SEED)
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "singstar_split.csv"), encoding="utf-8"))
            if r["split"] == "train"]
    picked = []
    for b, n in QUOTA.items():
        cand = sorted(r for r in ((r["group"], r["song"]) for r in rows if r["bucket"] == b))
        rng.shuffle(cand)
        used = set()
        for g, s in cand:
            if g in used:
                continue
            used.add(g)
            picked.append((b, s))
            if sum(x[0] == b for x in picked) == n:
                break
    out = os.path.join(ROOT, "eval_singstar", "bench")
    full = os.path.join(ROOT, "eval_singstar", "bench_full")
    os.makedirs(out, exist_ok=True)
    os.makedirs(full, exist_ok=True)
    for b, s in picked:
        src = os.path.join(ROOT, "eval_singstar", "train", s)
        song = usdx.read(usdx.find_song_txt(src))
        notes = song.notes
        options = []
        for i, n in enumerate(notes):
            if not n.line_start:
                continue
            start = max(0.0, n.start - LEAD_IN)
            kept = [j for j in range(i, len(notes)) if notes[j].end <= start + SECONDS]
            sung = sum(notes[j].end - notes[j].start for j in kept)
            if kept and sung / SECONDS >= MIN_SUNG:
                options.append((start, kept, sung / SECONDS))
        start, kept, share = rng.choice(options)
        end = min(start + SECONDS, notes[kept[-1]].end + 1.0)
        name = f"{s} @{int(start)}s"
        dst = os.path.join(out, name)
        os.makedirs(dst, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                        "-i", song.audio_path(), "-vn", "-ac", "2", "-ar", "44100", os.path.join(dst, "audio.wav")],
                       check=True)
        write_chart(song, os.path.join(dst, "song.txt"), "audio.wav", start, kept)
        json.dump({"song": s, "bucket": b, "start": round(start, 3), "end": round(end, 3),
                   "sung_share": round(share, 2)}, open(os.path.join(dst, "window.json"), "w"))
        link = os.path.join(full, s)
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(os.path.realpath(src), full), link)
        print(f"{b:10s} {name:60s} {end - start:4.1f}s sung {share:.0%}")


if __name__ == "__main__":
    main()
