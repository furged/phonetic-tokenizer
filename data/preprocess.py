"""
Phase 1 preprocessing: turn a raw parallel corpus into a clean train.txt
of Hinglish sentences for tokenizer prototyping.

Supports two input formats:
  1. IIIT-H corpus (Dhar et al., 2018): two aligned plain-text files,
     one sentence per line -- data/raw/s-enhi.txt (code-mixed) and
     data/raw/t-en.txt (English translation, not used here).
  2. PHINC (fallback): a single CSV at data/raw/phinc.csv.

The script auto-detects which is present. If both exist, IIIT-H is
preferred (it's the corpus specified in the original project guidance).

Output: data/processed/train.txt  (~1000 cleaned, deduped Hinglish sentences)

Run from project root:
    python data/preprocess.py
"""

import csv
import re
import sys
from pathlib import Path

IIITH_PATH = Path("data/raw/s-enhi.txt")
PHINC_PATH = Path("data/raw/phinc.csv")
OUT_PATH = Path("data/processed/train.txt")
TARGET_SENTENCES = 1000

CANDIDATE_COLUMNS = [
    "English-Hindi code-mixed",
    "english_hindi_code_mixed",
    "code_mixed",
    "codemixed",
    "hinglish",
]


def find_text_column(fieldnames):
    lowered = {f.lower(): f for f in fieldnames}
    for candidate in CANDIDATE_COLUMNS:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for f in fieldnames:
        fl = f.lower()
        if fl == "english" or "translation" in fl or "id" in fl or "index" in fl or fl == "":
            continue
        return f
    raise ValueError(f"Could not find a Hinglish text column. Columns found: {fieldnames}")


def clean_sentence(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid(text: str) -> bool:
    if not text:
        return False
    if len(text.split()) < 3:
        return False
    if len(text) > 500:
        return False
    return True


def load_iiith():
    print(f"Using IIIT-H corpus: {IIITH_PATH}")
    with open(IIITH_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


def load_phinc():
    print(f"Using PHINC corpus: {PHINC_PATH}")
    with open(PHINC_PATH, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("ERROR: could not read CSV header.")
            sys.exit(1)
        text_col = find_text_column(reader.fieldnames)
        print(f"Using column: '{text_col}'")
        for row in reader:
            yield row.get(text_col, "")


def main():
    if IIITH_PATH.exists():
        source = load_iiith()
    elif PHINC_PATH.exists():
        source = load_phinc()
    else:
        print(f"ERROR: no corpus found. Expected {IIITH_PATH} or {PHINC_PATH}.")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    cleaned = []

    for raw_text in source:
        text = clean_sentence(raw_text)
        if not is_valid(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)

    print(f"Total valid, deduped sentences found: {len(cleaned)}")

    if len(cleaned) < TARGET_SENTENCES:
        print(
            f"WARNING: only found {len(cleaned)} sentences, fewer than the "
            f"target of {TARGET_SENTENCES}. Writing all of them."
        )
        subset = cleaned
    else:
        subset = cleaned[:TARGET_SENTENCES]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for line in subset:
            f.write(line + "\n")

    print(f"Wrote {len(subset)} sentences to {OUT_PATH}")


if __name__ == "__main__":
    main()