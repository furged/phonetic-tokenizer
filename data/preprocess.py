"""
Phase 1 preprocessing: turn the raw PHINC CSV into a clean train.txt
of Hinglish sentences for tokenizer prototyping.

Input:  data/raw/phinc.csv        (columns: English-Hindi code-mixed, English)
Output: data/processed/train.txt  (~1000 cleaned, deduped Hinglish sentences)

Run from project root:
    python data/preprocess.py
"""

import csv
import re
import sys
from pathlib import Path

RAW_PATH = Path("data/raw/phinc.csv")
OUT_PATH = Path("data/processed/train.txt")
TARGET_SENTENCES = 1000

# PHINC's code-mixed column name can vary slightly depending on export;
# we check a few likely candidates instead of hardcoding one.
CANDIDATE_COLUMNS = [
    "English-Hindi code-mixed",
    "english_hindi_code_mixed",
    "code_mixed",
    "codemixed",
    "hinglish",
]


def find_text_column(fieldnames):
    """Find the column holding the Hinglish sentence, case-insensitively."""
    lowered = {f.lower(): f for f in fieldnames}
    for candidate in CANDIDATE_COLUMNS:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    # Fallback: first column that isn't an obvious translation/id/index column
    for f in fieldnames:
        fl = f.lower()
        if "english" == fl or "translation" in fl or "id" in fl or "index" in fl or fl == "":
            continue
        return f
    raise ValueError(
        f"Could not find a Hinglish text column. Columns found: {fieldnames}"
    )


def clean_sentence(text: str) -> str:
    """Basic cleanup: strip whitespace, collapse repeated spaces,
    drop stray quote characters left over from CSV export."""
    text = text.strip().strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid(text: str) -> bool:
    if not text:
        return False
    if len(text.split()) < 3:  # drop very short/degenerate rows
        return False
    if len(text) > 500:  # drop obvious outliers/garbage rows
        return False
    return True


def main():
    if not RAW_PATH.exists():
        print(f"ERROR: {RAW_PATH} not found. Did you download phinc.csv into data/raw/?")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    cleaned = []

    with open(RAW_PATH, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("ERROR: could not read CSV header. Check the file isn't corrupted.")
            sys.exit(1)

        text_col = find_text_column(reader.fieldnames)
        print(f"Using column: '{text_col}'")

        for row in reader:
            raw_text = row.get(text_col, "")
            text = clean_sentence(raw_text)

            if not is_valid(text):
                continue

            key = text.lower()  # dedupe case-insensitively
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