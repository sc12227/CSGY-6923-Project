import os
import json
import csv
import argparse
from typing import Dict, Any, Tuple, Optional

from music21 import converter, note, chord, stream


def find_abc_files(root: str):
    out = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".abc"):
                out.append(os.path.join(r, fn))
    return sorted(out)


def basic_header_checks(text: str) -> Dict[str, bool]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    flags = {
        "has_X": any(ln.startswith("X:") for ln in lines),
        "has_T": any(ln.startswith("T:") for ln in lines),
        "has_M": any(ln.startswith("M:") for ln in lines),
        "has_L": any(ln.startswith("L:") for ln in lines),
        "has_K": any(ln.startswith("K:") for ln in lines),
    }
    return flags


def count_events(s: stream.Stream) -> Tuple[int, int, int]:
    num_notes = 0
    num_chords = 0
    num_rests = 0

    flat = s.flatten()
    for el in flat:
        if isinstance(el, note.Note):
            num_notes += 1
        elif isinstance(el, chord.Chord):
            num_chords += 1
        elif isinstance(el, note.Rest):
            num_rests += 1

    return num_notes, num_chords, num_rests


def parse_abc_with_music21(path: str) -> Tuple[Optional[stream.Stream], Optional[str]]:
    try:
        s = converter.parse(path, format="abc")
        return s, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def write_midi(s: stream.Stream, midi_path: str) -> Optional[str]:
    try:
        os.makedirs(os.path.dirname(midi_path), exist_ok=True)
        s.write("midi", fp=midi_path)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abc_root", default="./samples/conditional_abc")
    ap.add_argument("--midi_out", default="./midi_out_cond")
    ap.add_argument("--report_csv", default="./sample_report.csv")
    ap.add_argument("--summary_json", default="./sample_summary.json")
    args = ap.parse_args()

    abc_files = find_abc_files(args.abc_root)
    if not abc_files:
        raise RuntimeError(f"No .abc files found under: {args.abc_root}")

    rows = []
    total = 0
    parse_ok = 0
    midi_ok = 0
    nontrivial_ok = 0 

    for p in abc_files:
        total += 1
        rel = os.path.relpath(p, args.abc_root)

        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            rows.append({
                "abc_path": rel,
                "syntactic_valid": 0,
                "midi_success": 0,
                "parse_error": f"ReadError: {e}",
                "midi_error": "",
                "num_notes": 0,
                "num_chords": 0,
                "num_rests": 0,
                "nontrivial_music": 0,
                **basic_header_checks(""),
            })
            continue

        header_flags = basic_header_checks(text)

        s, perr = parse_abc_with_music21(p)
        if s is None:
            rows.append({
                "abc_path": rel,
                "syntactic_valid": 0,
                "midi_success": 0,
                "parse_error": perr or "UnknownParseError",
                "midi_error": "",
                "num_notes": 0,
                "num_chords": 0,
                "num_rests": 0,
                "nontrivial_music": 0,
                **header_flags,
            })
            continue

        parse_ok += 1
        n_notes, n_chords, n_rests = count_events(s)
        nontrivial = 1 if (n_notes + n_chords) > 0 else 0
        if nontrivial:
            nontrivial_ok += 1

        midi_path = os.path.join(args.midi_out, os.path.splitext(rel)[0] + ".mid")
        merr = write_midi(s, midi_path)
        if merr is None:
            midi_ok += 1

        rows.append({
            "abc_path": rel,
            "syntactic_valid": 1,
            "midi_success": 1 if merr is None else 0,
            "parse_error": "",
            "midi_error": "" if merr is None else merr,
            "num_notes": n_notes,
            "num_chords": n_chords,
            "num_rests": n_rests,
            "nontrivial_music": nontrivial,
            **header_flags,
        })

    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.report_csv), exist_ok=True)
    with open(args.report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "total_samples": total,
        "syntactic_valid_count": parse_ok,
        "midi_success_count": midi_ok,
        "nontrivial_music_count": nontrivial_ok,
        "syntactic_valid_pct": (parse_ok / total * 100.0) if total else 0.0,
        "midi_success_pct": (midi_ok / total * 100.0) if total else 0.0,
        "nontrivial_music_pct": (nontrivial_ok / total * 100.0) if total else 0.0,
        "notes_about_metric": {
            "syntactic_valid": "music21 can parse the ABC file without exception",
            "midi_success": "parsed stream can be written to MIDI without exception",
            "nontrivial_music": "contains at least one Note or Chord (not only rests)",
        }
    }

    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("===== DONE =====")
    print(f"Total samples           : {total}")
    print(f"Syntactic valid (parse) : {parse_ok} ({summary['syntactic_valid_pct']:.2f}%)")
    print(f"MIDI success            : {midi_ok} ({summary['midi_success_pct']:.2f}%)")
    print(f"Nontrivial music        : {nontrivial_ok} ({summary['nontrivial_music_pct']:.2f}%)")
    print(f"CSV report saved        : {args.report_csv}")
    print(f"Summary JSON saved      : {args.summary_json}")
    print(f"MIDI output folder      : {args.midi_out}")


if __name__ == "__main__":
    main()
