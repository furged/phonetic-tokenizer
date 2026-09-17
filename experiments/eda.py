"""
Exploratory Data Analysis on the Hinglish corpus (data/processed/train.txt).

Produces:
  - Console summary stats (sentence/word counts, vocab size, code-mixing rate)
  - results/plots/sentence_length_hist.png
  - results/plots/word_frequency_top20.png
  - results/tables/eda_summary.csv

Usage:
    python experiments/eda.py
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt

TRAIN_PATH = Path("data/processed/train.txt")
PLOTS_DIR = Path("results/plots")
TABLES_DIR = Path("results/tables")

# Same heuristic marker list as check_language_mix.py -- rough code-mixing
# indicator, not a real language identifier. See that script's docstring
# for the caveat: it will under-detect Hindi content that uses words
# outside this list or unusual spellings (which is itself relevant to
# this project's whole premise about spelling variation).
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


def tokenize_words(sentence: str):
    return re.findall(r"[a-zA-Z']+", sentence.lower())


def has_hindi_marker(words):
    return any(w in HINDI_MARKERS for w in words)


def has_devanagari(sentence: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in sentence)


def main():
    if not TRAIN_PATH.exists():
        print(f"ERROR: {TRAIN_PATH} not found. Run data/preprocess.py first.")
        sys.exit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_PATH, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    n = len(sentences)
    sentence_word_lengths = []
    sentence_char_lengths = []
    vocab = Counter()
    devanagari_count = 0
    likely_code_mixed = 0

    for s in sentences:
        words = tokenize_words(s)
        sentence_word_lengths.append(len(words))
        sentence_char_lengths.append(len(s))
        vocab.update(words)

        if has_devanagari(s):
            devanagari_count += 1
        if has_hindi_marker(words):
            likely_code_mixed += 1

    avg_words = sum(sentence_word_lengths) / n
    avg_chars = sum(sentence_char_lengths) / n
    vocab_size = len(vocab)
    top20 = vocab.most_common(20)

    print("=" * 55)
    print(f"EDA SUMMARY -- {n} sentences")
    print("=" * 55)
    print(f"Avg words/sentence:      {avg_words:.2f}")
    print(f"Min / Max words:         {min(sentence_word_lengths)} / {max(sentence_word_lengths)}")
    print(f"Avg chars/sentence:      {avg_chars:.2f}")
    print(f"Unique word vocab size:  {vocab_size}")
    print(f"Sentences with Devanagari script: {devanagari_count} ({100*devanagari_count/n:.1f}%)")
    print(f"Sentences flagged as likely code-mixed (heuristic): "
          f"{likely_code_mixed} ({100*likely_code_mixed/n:.1f}%)")
    print()
    print("Top 20 most frequent words:")
    for word, count in top20:
        print(f"  {word:<15} {count}")
    print("=" * 55)

    # Save summary table
    summary_path = TABLES_DIR / "eda_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["sentence_count", n])
        writer.writerow(["avg_words_per_sentence", round(avg_words, 2)])
        writer.writerow(["min_words_per_sentence", min(sentence_word_lengths)])
        writer.writerow(["max_words_per_sentence", max(sentence_word_lengths)])
        writer.writerow(["avg_chars_per_sentence", round(avg_chars, 2)])
        writer.writerow(["unique_word_vocab_size", vocab_size])
        writer.writerow(["pct_sentences_with_devanagari", round(100*devanagari_count/n, 1)])
        writer.writerow(["pct_sentences_likely_code_mixed_heuristic", round(100*likely_code_mixed/n, 1)])
    print(f"Saved summary table to {summary_path}")

    # Plot 1: sentence length distribution
    plt.figure(figsize=(8, 5))
    plt.hist(sentence_word_lengths, bins=30, color="#4C72B0", edgecolor="black")
    plt.title("Sentence Length Distribution (words/sentence)")
    plt.xlabel("Words per sentence")
    plt.ylabel("Number of sentences")
    plt.tight_layout()
    hist_path = PLOTS_DIR / "sentence_length_hist.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved plot to {hist_path}")

    # Plot 2: top-20 word frequency
    words_, counts_ = zip(*top20)
    plt.figure(figsize=(9, 6))
    plt.barh(words_[::-1], counts_[::-1], color="#DD8452")
    plt.title("Top 20 Most Frequent Words")
    plt.xlabel("Frequency")
    plt.tight_layout()
    freq_path = PLOTS_DIR / "word_frequency_top20.png"
    plt.savefig(freq_path, dpi=150)
    plt.close()
    print(f"Saved plot to {freq_path}")


if __name__ == "__main__":
    main()