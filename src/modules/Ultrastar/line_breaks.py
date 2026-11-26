"""Line break planning.

Facts from 15 charts (USDB + dev, 843 lines): 95% of lines have at most 38 characters
(none more than 45) and last at most 4.8 s; the median line has 23 characters and lasts 1.9 s.

Each pause between two words gets a break score from a small logistic model (fitted on our
own output against chart line starts, leave-one-song-out line F1 0.71 vs 0.60 for the
original pause rule):
  - the pause length
  - how likely a multilingual punctuation-restoration model puts a sentence/clause mark
    after the word (text only, lower-cased, no casing or punctuation cues)
A dynamic program then picks the breaks that maximise these scores plus a prior on line
length (characters and duration) learned from the charts.
"""

import logging

import math
import re
from dataclasses import dataclass

import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.console_colors import blue_highlighted

log = logging.getLogger("ultrasinger.line_breaks")

@dataclass
class LineBreakModel:
    """log-odds(break) = intercept + w_pause * log(pause + 0.02) + w_punct * logit(p_punct), plus a
    prior on line length (characters: normal; duration: log-normal) from the charts. Values come
    from the preset (fitted with eval/fit_line_breaks.py)."""
    intercept: float
    w_pause: float
    w_punct: float
    chars_mean: float
    chars_sd: float
    chars_max: int
    dur_median: float
    dur_log_sd: float
    prior_weight: float
    max_words: int


def _words(midi_segments: list[MidiSegment]) -> list[tuple[int, int, float, float, str]]:
    """(first segment, last segment, start, end, text) per word."""
    words, first = [], 0
    for k, seg in enumerate(midi_segments):
        if seg.word.endswith(" ") or k == len(midi_segments) - 1:
            text = "".join(s.word.replace("~", "").strip() for s in midi_segments[first:k + 1])
            words.append((first, k, midi_segments[first].start, seg.end, text))
            first = k + 1
    return words


_PUNCT_MEMO: dict = {}


def punctuation_probabilities(words: list[str], model_name: str, device: str = "cpu",
                              chunk: int = 150, context: int = 30) -> np.ndarray:
    """P(punctuation after each word) from a token-classification punctuation model.
    Memoized per word sequence within a process (tuning runs plan the same text many times)."""
    key = (tuple(words), model_name)
    if key not in _PUNCT_MEMO:
        if len(_PUNCT_MEMO) > 2000:
            _PUNCT_MEMO.clear()
        _PUNCT_MEMO[key] = _punctuation_probabilities(words, model_name, device, chunk, context)
    return _PUNCT_MEMO[key].copy()


def _punctuation_probabilities(words, model_name, device, chunk, context) -> np.ndarray:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device).eval()
    no_punct = [k for k, v in model.config.id2label.items() if v == "0"][0]
    cleaned = [re.sub(r"[^\w']", "", w.lower()) or "x" for w in words]
    out = np.zeros(len(cleaned))
    i = 0
    while i < len(cleaned):
        a, b = max(0, i - context), min(len(cleaned), i + chunk)
        enc = tokenizer(cleaned[a:b + context], is_split_into_words=True, return_tensors="pt",
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            probs = model(**enc).logits[0].softmax(-1).cpu().numpy()
        last_token = {}
        for t, w in enumerate(enc.word_ids()):
            if w is not None:
                last_token[w] = t
        for w, t in last_token.items():
            if i <= a + w < b:
                out[a + w] = 1.0 - probs[t, no_punct]
        i = b
    del model
    return out


def plan_line_breaks(midi_segments: list[MidiSegment], model_name: str, device: str, m: LineBreakModel,
                     bonus_starts: set[int] | None = None, bonus: float = 0.0) -> None:
    """Set line_start on the first note of each planned line. bonus_starts: segment indices that
    get an extra break score (e.g. LRCLIB line starts)."""
    words = _words(midi_segments)
    n = len(words)
    if n < 2:
        return
    log.info(f"Planning line breaks with {blue_highlighted(model_name)}")
    p = np.clip(punctuation_probabilities([w[4] for w in words], model_name, device), 1e-3, 1 - 1e-3)
    score = [0.0] * n
    for i in range(1, n):
        pause = max(0.0, words[i][2] - words[i - 1][3])
        score[i] = m.intercept + m.w_pause * math.log(pause + 0.02) + m.w_punct * math.log(p[i - 1] / (1 - p[i - 1]))
        if bonus_starts and words[i][0] in bonus_starts:
            score[i] += bonus
    chars = [0]
    for w in words:
        chars.append(chars[-1] + len(w[4]))
    best, back = [-math.inf] * (n + 1), [0] * (n + 1)
    best[0] = 0.0
    for b in range(1, n + 1):  # a line covers words a..b-1
        for a in range(max(0, b - m.max_words), b):
            c = chars[b] - chars[a] + (b - 1 - a)
            if c > m.chars_max and b - 1 > a:
                continue
            dur = max(0.05, words[b - 1][3] - words[a][2])
            prior = -0.5 * ((c - m.chars_mean) / m.chars_sd) ** 2 - 0.5 * ((math.log(dur) - math.log(m.dur_median)) / m.dur_log_sd) ** 2
            s = best[a] + m.prior_weight * prior + (score[a] if a > 0 else 0.0)
            if s > best[b]:
                best[b], back[b] = s, a
    for seg in midi_segments:
        seg.line_start = False
    b = n
    while b > 0:
        a = back[b]
        if a > 0:
            midi_segments[words[a][0]].line_start = True
        b = a
