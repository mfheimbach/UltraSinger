"""Train/test split of the SingStar collections (Mo's Syncer download, .opus, no fixes).

Rules
- One song per artist in test (diversity); its other songs go to 'reserve', never train.
- Grouped by artist (incl. feat./&/+/vs. partners): one voice never sits in both sets.
- Stratified by language x style bucket so test mirrors train.
- Excluded now: Bollywood (Hindi, outlier like Butter-fly), duplicate cuts,
  and 'hard' candidates (live, stadium/crowd chants, spoken novelty) -> saved for the hard set.
- Fixed seed; output is frozen in singstar_split.csv. Do not re-run to "get a better split".

Extension (2026-11-26, `python eval/splits/make_singstar_split.py extend`): Mo's collection grew to
328 songs (singstar_files_v2.txt: ten more editions, Bollywood gone). Every existing row keeps its
split; new songs join their artist group's side (a group with a test song sends new songs to
reserve), new groups get the same 30% test rule per bucket (seed + 2). Clear contaminants go to
excluded/hard: Hindi (Jai Ho, now also removed from train), second cuts, live/TV versions, rap
and spoken novelty.
"""
import csv, os, random, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, TEST_SHARE = 20261125, 0.30

HARD = {
    "Wolfgang Petry - Wahnsinn (Live)": "live",
    "Bruder Leichtfuß - Deutsche Nationalhymne (Stadion Version)": "stadium crowd",
    "Bayern-Fans United - Stern des Südens": "fan choir",
    "Die Original Deutschmacher - Lebenslang Grünweiß (Werder Bremen 2004)": "fan choir",
    "Franz Beckenbauer - Gute Freunde kann niemand trennen": "non-singer, 1966 recording",
    "Mike der Bademeister - Eskimo (Der BRRR-Song)": "novelty / shouted",
    "Höhner - Wenn nicht jetzt, wann dann (Fußball Version)": "alt. crowd version",
    "Höhner - Wenn nicht jetzt, wann dann (Halbfinal Version)": "alt. crowd version",
}
DUPLICATE = {"LeAnn Rimes - Can’t Fight The Moonlight (Album Version)": "second cut of the same song"}
ENGLISH_IN_GERMAN_EDITIONS = {
    "Gerry & The Pacemakers - You’ll Never Walk Alone", "Chubby Checker - Let’s Twist Again",
    "Wanda Jackson - Let’s Have A Party", "Hermes House Band - I Will Survive (La La La)",
    "Medina - You And I", "Agnes - Release Me", "Alexandra Burke feat. Pitbull - All Night Long",
    "Daniela Katzenberger - Nothing’s Gonna Stop Me Now", "Lady Gaga - Bad Romance",
    "Yoomiii - A Kiss Is All I Miss", "Hermes House Band - Country Roads", "Right Said Fred - You’re My Mate",
    "Bellamy Brothers - Let Your Love Flow", "Banaroo - Dubi Dam Dam",
}
GERMAN_IN_AMPED_DE = {"Die Toten Hosen - Warum werde ich nicht satt", "In Extremo - Küss mich"}

# extension 2026-11-26
HARD_V2 = {
    "Geier Sturzflug - Bruttosozialprodukt (ZDF Hitparade)": "live TV version",
    "Sir Mix-A-Lot - Baby Got Back": "rap",
    "MC Hammer - U Can’t Touch This": "rap",
    "Arrested Development - Tennessee": "rap",
    "Meneer Kaktus - Het grote meneer Kaktus lied": "children's TV novelty",
}
DUPLICATE_V2 = {
    "Stone Temple Pilots - Plush (Acoustic)": "second cut of Plush",
    "Seal - Kiss From A Rose (Album Version)": "second cut of the same song",
    "Ostrowska - Szklana Pogoda": "same song and singer as Lombard - Szklana pogoda",
}
HINDI = {"A.R. Rahman & The Pussycat Dolls feat. Nicole Scherzinger - Jai Ho! (You Are My Destiny)"}
ENGLISH_IN_80S = {
    "Modern Talking - Cheri, Cheri Lady", "Marietta - Fire And Ice", "Cyndi Lauper - She Bop",
    "Limahl - The Never Ending Story", "Dolly Dots - Do Wah Diddy Diddy", "F. R. David - Words",
    "Imagination - Just An Illusion", "INXS - Need You Tonight",
}


def bucket_v2(edition, song):
    """Buckets for the editions added in the extension (language x era)."""
    if song in HINDI:
        return None
    if edition.startswith("SingStar ’80s"):
        if song in ENGLISH_IN_80S:
            return "en_retro"
        return {"[DE]": "de_retro", "[NL]": "nl", "[IT]": "it", "[PL]": "pl", "[FR]": "fr"}[edition.split()[-1]]
    if edition.startswith("SingStar ’90s"):
        return "en_retro"
    if edition == "SingStar zingt met Disney":
        return "nl"
    if edition == "SingStar top.it":
        return "it"
    return bucket(edition, song)


def bucket(edition, song):
    if edition == "SingStar Bollywood":
        return None
    if edition == "SingStar Amped [DE]":
        return "de_rock" if song in GERMAN_IN_AMPED_DE else "en_rock"
    if edition.startswith("SingStar Amped"):
        return "en_rock"
    if edition == "SingStar Deutsch Rock-Pop":
        return "de_poprock"
    if edition.startswith("SingStar Après-Ski") or edition == "SingStar Fußballhits":
        return "en_party" if song in ENGLISH_IN_GERMAN_EDITIONS else "de_party"
    return "en_pop"  # Boy/Girl Bands, Chart Hits, Anthems


def voices(song):
    artist = song.split(" - ", 1)[0]
    return {p.strip() for p in re.split(r" feat\. | & | \+ | vs\. |, ", artist) if p.strip()}


def main():
    rows = []
    for line in open(os.path.join(HERE, "singstar_files.txt"), encoding="utf-8"):
        line = line.strip()
        if not line.endswith(".opus"):
            continue
        _, edition, song, _ = line.split("/", 3)
        rows.append({"edition": edition, "song": song, "bucket": bucket(edition, song)})

    # union-find over shared voices -> artist groups
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in rows:
        vs = sorted(voices(r["song"]))
        for v in vs[1:]:
            parent[find(v)] = find(vs[0])
        r["group"] = find(vs[0])
    for r in rows:
        r["group"] = find(r["group"])

    for r in rows:
        if r["bucket"] is None:
            r["split"], r["note"] = "excluded", "Hindi (outlier)"
        elif r["song"] in DUPLICATE:
            r["split"], r["note"] = "excluded", DUPLICATE[r["song"]]
        elif r["song"] in HARD:
            r["split"], r["note"] = "hard", HARD[r["song"]]
        else:
            r["split"], r["note"] = None, ""

    # a group goes where its songs mostly are by bucket; assign groups bucket by bucket
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for r in rows:
        if r["split"] is None:
            groups[r["group"]].append(r)
    by_bucket = defaultdict(list)
    for g, members in groups.items():
        main_bucket = max({m["bucket"] for m in members}, key=lambda b: sum(m["bucket"] == b for m in members))
        by_bucket[main_bucket].append(g)
    for b in sorted(by_bucket):
        gs = sorted(by_bucket[b])
        rng.shuffle(gs)
        total = sum(len(groups[g]) for g in gs)
        target, n_test = round(total * TEST_SHARE), 0
        for g in gs:
            take = n_test < target
            members = sorted(groups[g], key=lambda m: m["song"])
            keep = rng.choice(members) if take else None
            for m in members:
                if not take:
                    m["split"] = "train"
                elif m is keep:
                    m["split"] = "test"
                else:  # one song per artist in test; the rest can't go to train (same voice)
                    m["split"], m["note"] = "reserve", "same artist as a test song"
            n_test += 1 if take else 0

    out = os.path.join(HERE, "singstar_split.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["split", "bucket", "edition", "song", "group", "note"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["split"], r["bucket"] or "", r["song"])):
            w.writerow({k: r[k] for k in w.fieldnames})
    print("wrote", out)


def extend():
    """Add singstar_files_v2.txt songs to the frozen split (see module docstring)."""
    path = os.path.join(HERE, "singstar_split.csv")
    old = list(csv.DictReader(open(path, encoding="utf-8")))
    known = {r["song"] for r in old}
    for r in old:   # contaminant found later: out of train/test
        if r["song"] in HINDI and r["split"] != "excluded":
            r["split"], r["note"], r["bucket"] = "excluded", "Hindi (outlier)", ""
    new = []
    for line in open(os.path.join(HERE, "singstar_files_v2.txt"), encoding="utf-8"):
        line = line.strip()
        if not line.endswith(".opus"):
            continue
        _, edition, song, _ = line.split("/", 3)
        if song in known:
            continue
        new.append({"edition": edition, "song": song, "bucket": bucket_v2(edition, song), "note": ""})

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in old + new:
        vs = sorted(voices(r["song"]))
        for v in vs[1:]:
            parent[find(v)] = find(vs[0])
    old_sides = defaultdict(set)
    for r in old:
        old_sides[find(sorted(voices(r["song"]))[0])].add(r["split"])
    for r in new:
        r["group"] = find(sorted(voices(r["song"]))[0])
        if r["bucket"] is None:
            r["split"], r["note"] = "excluded", "Hindi (outlier)"
        elif r["song"] in DUPLICATE_V2:
            r["split"], r["note"] = "excluded", DUPLICATE_V2[r["song"]]
        elif r["song"] in HARD_V2:
            r["split"], r["note"] = "hard", HARD_V2[r["song"]]
        elif "test" in old_sides[r["group"]] or "reserve" in old_sides[r["group"]]:
            r["split"], r["note"] = "reserve", "same artist as a test song"
        elif "train" in old_sides[r["group"]]:
            r["split"], r["note"] = "train", "artist already in train"
        else:
            r["split"] = None

    rng = random.Random(SEED + 2)
    groups = defaultdict(list)
    for r in new:
        if r["split"] is None:
            groups[r["group"]].append(r)
    by_bucket = defaultdict(list)
    for g, members in groups.items():
        main_bucket = max({m["bucket"] for m in members}, key=lambda b: sum(m["bucket"] == b for m in members))
        by_bucket[main_bucket].append(g)
    for b in sorted(by_bucket):
        gs = sorted(by_bucket[b])
        rng.shuffle(gs)
        total = sum(len(groups[g]) for g in gs)
        target, n_test = round(total * TEST_SHARE), 0
        for g in gs:
            take = n_test < target
            members = sorted(groups[g], key=lambda m: m["song"])
            keep = rng.choice(members) if take else None
            for m in members:
                if not take:
                    m["split"] = "train"
                elif m is keep:
                    m["split"] = "test"
                else:
                    m["split"], m["note"] = "reserve", "same artist as a test song"
            n_test += 1 if take else 0
    rows = old + [{k: r[k] for k in ("split", "bucket", "edition", "song", "group", "note")} for r in new]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["split", "bucket", "edition", "song", "group", "note"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["split"], r["bucket"] or "", r["song"])):
            w.writerow({k: r[k] or "" for k in w.fieldnames})
    print("wrote", path, f"({len(new)} new songs)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extend":
        extend()
    else:
        main()
