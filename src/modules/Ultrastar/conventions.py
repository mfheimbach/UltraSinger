"""Community conventions for UltraStar charts, applied to a written txt.

What USDB Syncer and YASS normalise in every chart players download, so our output looks the
same. Own implementation of the rules (no code taken from those GPL projects):

- notes never touch: a note that ends on or after the next note's start is shortened so one
  beat stays free (if it can't be shortened, the next note starts one beat later)
- line-break timing: the '- N' beat where the old line disappears follows the YASS table
  (gap >= 4 s: 2 s after the line; >= 2 s: 1 s after; tiny gaps: at the line end; otherwise a
  few beats before the next line, depending on the gap in beats)
- typographic apostrophes (’) instead of ' ´ `
- the first letter of each line is capitalised
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("ultrasinger.conventions")

NOTE = re.compile(r"^([:*FRG]) (-?\d+) (\d+) (-?\d+) ?(.*)$")
BREAK = re.compile(r"^- ?(-?\d+)(?: (-?\d+))?\s*$")
APOSTROPHE = re.compile(r"(?<=\w)['´`](?=\w)|(?<=\w)'(?=\s|$)")


def _parse(lines):
    """Split into header lines and body tokens: ('note', [type, start, dur, pitch, text]) / ('break', n) / ('other', line)."""
    header, body = [], []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not body and line.startswith("#"):
            header.append(line)
            continue
        m = NOTE.match(line)
        if m:
            body.append(["note", [m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)]])
            continue
        m = BREAK.match(line)
        if m:
            body.append(["break", int(m.group(1))])
            continue
        body.append(["other", line])
    return header, body


def _bpm(header) -> float:
    for h in header:
        if h.upper().startswith("#BPM:"):
            return float(h.split(":", 1)[1].replace(",", "."))
    raise ValueError("chart has no #BPM")


def fix_touching_notes(body) -> int:
    notes = [t[1] for t in body if t[0] == "note"]
    fixed = 0
    for cur, nxt in zip(notes, notes[1:]):
        gap = nxt[1] - (cur[1] + cur[2])
        if gap >= 1:
            continue
        if cur[2] - (1 - gap) >= 1:
            cur[2] -= 1 - gap
        else:
            cur[2] = max(1, cur[2])
            nxt[1] = cur[1] + cur[2] + 1
        fixed += 1
    return fixed


def fix_line_break_timing(body, bpm: float, rules: dict) -> int:
    """YASS-style position of '- N' between the previous line's end and the next line's start.
    rules (from the preset): long_gap_s / long_offset_s and mid_gap_s / mid_offset_s (seconds after the
    line end for long pauses), touch_beats (gaps up to this break at the line end), beat_rules
    [[max_gap, beats_before_next], ...] for short gaps, far_offset_beats after the end otherwise."""
    beat_s = 60.0 / (bpm * 4)
    fixed = 0
    for i, tok in enumerate(body):
        if tok[0] != "break":
            continue
        prev = next((t[1] for t in reversed(body[:i]) if t[0] == "note"), None)
        nxt = next((t[1] for t in body[i + 1:] if t[0] == "note"), None)
        if prev is None or nxt is None:
            continue
        end, start = prev[1] + prev[2], nxt[1]
        gap = start - end
        gap_s = gap * beat_s
        rule = next((before for max_gap, before in rules["beat_rules"] if gap <= max_gap), None)
        if gap_s >= rules["long_gap_s"]:
            pos = end + round(rules["long_offset_s"] / beat_s)
        elif gap_s >= rules["mid_gap_s"]:
            pos = end + round(rules["mid_offset_s"] / beat_s)
        elif gap <= rules["touch_beats"]:
            pos = end
        elif rule is not None:
            pos = start - rule
        else:
            pos = end + rules["far_offset_beats"]
        pos = min(max(pos, end), start)
        if pos != tok[1]:
            tok[1] = pos
            fixed += 1
    return fixed


def fix_apostrophes(body) -> int:
    fixed = 0
    for tok in body:
        if tok[0] == "note":
            new = APOSTROPHE.sub("’", tok[1][4])
            if new != tok[1][4]:
                tok[1][4], fixed = new, fixed + 1
    return fixed


def fix_first_word_capitalisation(body) -> int:
    fixed, line_start = 0, True
    for tok in body:
        if tok[0] == "break":
            line_start = True
        elif tok[0] == "note" and line_start:
            text = tok[1][4]
            for k, ch in enumerate(text):
                if ch.isalpha():
                    if ch.islower():
                        tok[1][4] = text[:k] + ch.upper() + text[k + 1:]
                        fixed += 1
                    break
            line_start = False
    return fixed


def apply_conventions(path: str, touching: bool, line_breaks: bool, apostrophes: bool, capitalise: bool,
                      break_rules: dict) -> dict:
    """Rewrite the chart at path in place. Returns how many items each rule changed."""
    from modules.Ultrastar.ultrastar_txt import FILE_ENCODING
    with open(path, encoding=FILE_ENCODING) as f:
        header, body = _parse(f.readlines())
    counts = {}
    if touching:
        counts["touching_notes"] = fix_touching_notes(body)
    if line_breaks:
        counts["line_breaks"] = fix_line_break_timing(body, _bpm(header), break_rules)
    if apostrophes:
        counts["apostrophes"] = fix_apostrophes(body)
    if capitalise:
        counts["capitalised_lines"] = fix_first_word_capitalisation(body)
    out = header[:]
    for kind, val in body:
        if kind == "note":
            t, s, d, p, text = val
            out.append(f"{t} {s} {d} {p} {text}")
        elif kind == "break":
            out.append(f"- {val}")
        else:
            out.append(val)
    with open(path, "w", encoding=FILE_ENCODING) as f:
        f.write("\n".join(out) + "\n")
    log.info("conventions: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return counts
