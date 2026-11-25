"""Lyrics correction via LRCLIB (https://lrclib.net).

The transcription keeps the timing; LRCLIB provides the correct words. The reference lyrics
are matched to the transcribed words (sequence alignment on normalized words); matched and
substituted words take the reference spelling, keeping the transcribed timing. If too few
words match, the lyrics are assumed to belong to another song/version and are not used.
"""


import logging
import copy
import difflib
import json
import re
import unicodedata
import urllib.parse
import urllib.request

from modules.console_colors import red_highlighted

log = logging.getLogger("ultrasinger.lyrics_lookup")

API = "https://lrclib.net/api"
TIMESTAMP_RE = re.compile(r"\[\d+:\d+(?:\.\d+)?\]")


def _query(params: dict) -> list:
    url = f"{API}/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "UltraSinger (https://github.com/rakuri255/UltraSinger)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def _clean(text: str) -> str:
    """Drop parenthesised/bracketed additions like '(Studio Footage)' or '(feat. X)'."""
    return re.sub(r"\s*[\(\[].*?[\)\]]", "", text or "").strip()


def fetch_lyrics(artist: str, title: str, duration: float, max_duration_diff: float = 15.0) -> list[dict]:
    """Candidate lyrics (closest duration first) as {"plain": text, "synced": LRC text or None}."""
    for a, t in ((artist, title), (_clean(artist), _clean(title))):
        if not t:
            continue
        try:
            results = _query({"track_name": t, "artist_name": a})
        except Exception as e:  # network problems: behave like "not found"
            log.warning(f"{red_highlighted('LRCLIB lookup failed')}: {e}")
            return []
        results = [r for r in results if (r.get("plainLyrics") or r.get("syncedLyrics"))
                   and abs((r.get("duration") or 0) - duration) <= max_duration_diff]
        if results:
            results.sort(key=lambda r: abs((r.get("duration") or 0) - duration))
            return [{"plain": r.get("plainLyrics") or TIMESTAMP_RE.sub("", r.get("syncedLyrics") or ""),
                     "synced": r.get("syncedLyrics")} for r in results[:3]]
    return []


def _norm(word: str) -> str:
    w = unicodedata.normalize("NFC", word.lower())
    return re.sub(r"[^\w']", "", w).strip("'")


def correct_words(transcribed, lyrics_text: str, min_match: float = 0.4):
    """Replace mis-heard words with the reference lyrics, keeping the transcribed timing.
    Returns (corrected list, match share) or (None, share) if the lyrics don't fit."""
    ref = [w for w in re.split(r"\s+", lyrics_text) if _norm(w)]
    hyp = [t for t in transcribed if _norm(str(t.word))]
    if not ref or not hyp:
        return None, 0.0
    a = [_norm(str(t.word)) for t in hyp]
    b = [_norm(w) for w in ref]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    blocks = sm.get_opcodes()
    matched = sum(i2 - i1 for tag, i1, i2, j1, j2 in blocks if tag == "equal")
    share = matched / len(hyp)
    if share < min_match:
        return None, share
    out = []
    for tag, i1, i2, j1, j2 in blocks:
        if tag == "equal":
            for k in range(i2 - i1):
                t = copy.deepcopy(hyp[i1 + k])
                t.word = ref[j1 + k] + " "
                out.append(t)
        elif tag == "replace" and (j2 - j1 > 2 * (i2 - i1) + 1 or i2 - i1 > 2 * (j2 - j1) + 1):
            # very different lengths: usually backing vocals/repeats this version doesn't have
            out.extend(copy.deepcopy(hyp[i1:i2]))
        elif tag == "replace":
            # reference words take over the time span of the mis-heard words
            start, end = hyp[i1].start, hyp[i2 - 1].end
            words = ref[j1:j2]
            weights = [max(1, len(_norm(w))) for w in words]
            total = sum(weights)
            cursor = start
            for w, wt in zip(words, weights):
                t = copy.deepcopy(hyp[i1])
                t.start = cursor
                t.end = cursor + (end - start) * wt / total
                t.word = w + " "
                cursor = t.end
                out.append(t)
        elif tag == "delete":
            out.extend(copy.deepcopy(hyp[i1:i2]))  # sung but not in the lyrics (ad-libs): keep
        # "insert": in the lyrics but not heard (e.g. a repeat this version skips): drop
    return out, share
