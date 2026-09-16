"""
Quick heuristic check: what fraction of train.txt looks like it has
zero Hindi content (i.e. is actually English-only, not code-mixed)?

This is a rough heuristic, not a real language identifier: it checks
for the presence of any common Hindi function words written in Roman
script. A sentence with none of these words is flagged as "likely
English-only." Real work should use a proper LID tool later (e.g.
fasttext lid or a HingBERT-LID model) -- this is just a fast sanity
check before you commit to using this data.

Run from project root:
    python data/check_language_mix.py
"""

from pathlib import Path

TRAIN_PATH = Path("data/processed/train.txt")

# Common Hindi function words / markers in Roman script. Not exhaustive,
# but these are frequent enough that a genuinely code-mixed sentence will
# almost always contain at least one.
HINDI_MARKERS = {
    "hai", "hain", "ho", "hoga", "hogi", "tha", "thi", "the",
    "ka", "ki", "ke", "ko", "se", "mein", "me", "par",
    "aur", "ya", "toh", "to", "bhi", "nahi", "nahin", "haan",
    "kya", "kyun", "kaise", "kab", "kaun", "kahan",
    "hum", "hamara", "tum", "tumhara", "aap", "unka", "uska", "iska",
    "yeh", "ye", "woh", "wo", "is", "us",
    "kar", "kiya", "karo", "karta", "karti", "karte", "raha", "rahe", "rahi",
    "wala", "wale", "wali", "bahot", "bahut", "kuch", "sab", "log",
    "acha", "accha", "theek", "thik",
}


def looks_english_only(sentence: str) -> bool:
    words = sentence.lower().split()
    words = [w.strip(".,!?#@:;\"'()") for w in words]
    return not any(w in HINDI_MARKERS for w in words)


def main():
    if not TRAIN_PATH.exists():
        print(f"ERROR: {TRAIN_PATH} not found.")
        return

    with open(TRAIN_PATH, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    total = len(lines)
    flagged = [l for l in lines if looks_english_only(l)]

    print(f"Total sentences: {total}")
    print(f"Flagged as likely English-only: {len(flagged)} ({100 * len(flagged) / total:.1f}%)")
    print()
    print("Sample flagged sentences:")
    for s in flagged[:10]:
        print(f"  - {s}")


if __name__ == "__main__":
    main()