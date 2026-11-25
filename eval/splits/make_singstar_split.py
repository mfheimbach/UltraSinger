"""Train/test split of the SingStar collections (Mo's Syncer download, .opus, no fixes).

Rules
- One song per artist in test (diversity); its other songs go to 'reserve', never train.
- Grouped by artist (incl. feat./&/+/vs. partners): one voice never sits in both sets.
- Stratified by language x style bucket so test mirrors train.
- Excluded now: Bollywood (Hindi, outlier like Butter-fly), duplicate cuts,
  and 'hard' candidates (live, stadium/crowd chants, spoken novelty) -> saved for the hard set.
- Fixed seed; output is frozen in singstar_split.csv. Do not re-run to "get a better split".
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


if __name__ == "__main__":
    main()
