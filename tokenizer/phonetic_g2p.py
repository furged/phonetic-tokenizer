"""
Phonetic tokenizer for romanized Hinglish text.

DESIGN NOTE (important, document this in your paper's Method/Limitations):
Standard G2P tools (Epitran's hin-Deva, Festvox Indic) expect Devanagari
script input and convert it to phonemes. Our corpus (PHINC) is romanized
Hinglish -- Latin script, code-mixed, noisy social-media text. Those tools
can't be applied directly without a transliteration step first.

Instead, this module implements a lightweight RULE-BASED phonetic
normalizer that operates directly on the Latin spelling: it maps common
digraphs/trigraphs used in Hindi romanization (aa, ee, oo, bh, ch, dh, gh,
jh, kh, ph, sh, th, ng) to single phoneme symbols, then falls back to
single-character mapping for anything else. This is NOT linguistically
rigorous G2P -- it's an approximation chosen to match the actual text
(romanized, code-mixed, noisy) rather than force a Devanagari-based tool
onto data it can't read.

Known limitation: this does not solve schwa deletion (e.g. "kamal" should
phonetically drop the final inherent vowel in real Hindi pronunciation --
this rule set does not model that). Note this explicitly as a limitation,
per your professor's guidance -- pick one approach, document what it
doesn't handle, don't over-engineer in Phase 1.

Usage:
    python tokenizer/phonetic_g2p.py train                    # build vocab, save
    python tokenizer/phonetic_g2p.py encode "some sentence"   # quick test
"""

import json
import sys
from pathlib import Path

TRAIN_PATH = Path("data/processed/train.txt")
MODEL_DIR = Path("tokenizer/phonetic_model")
VOCAB_PATH = MODEL_DIR / "vocab.json"

# Longest-match-first digraph/trigraph -> phoneme symbol.
# Symbols are plain ASCII tags (not IPA) to keep files simple to inspect/diff.
MULTI_CHAR_RULES = [
    # trigraphs first
    ("chh", "CHH"),
    # digraphs (aspirated consonants -- common in Hindi romanization)
    ("bh", "BH"), ("ch", "CH"), ("dh", "DH"), ("gh", "GH"),
    ("jh", "JH"), ("kh", "KH"), ("ph", "PH"), ("sh", "SH"),
    ("th", "TH"), ("ng", "NG"), ("ny", "NY"),
    # long vowels
    ("aa", "AA"), ("ee", "EE"), ("ii", "EE"), ("oo", "OO"),
    ("ai", "AI"), ("au", "AU"),
]

# Single-character fallback for anything not caught by a digraph rule.
SINGLE_CHAR_RULES = {
    "a": "A", "e": "E", "i": "I", "o": "O", "u": "U",
    "b": "B", "c": "K", "d": "D", "f": "F", "g": "G",
    "h": "H", "j": "J", "k": "K", "l": "L", "m": "M",
    "n": "N", "p": "P", "q": "K", "r": "R", "s": "S",
    "t": "T", "v": "V", "w": "W", "x": "KS", "y": "Y", "z": "Z",
}

WORD_BOUNDARY = "<WB>"


def word_to_phonemes(word: str) -> list:
    """Convert a single lowercase word into a list of phoneme-symbol tokens."""
    word = word.lower()
    phonemes = []
    i = 0
    while i < len(word):
        matched = False
        for pattern, symbol in MULTI_CHAR_RULES:
            if word.startswith(pattern, i):
                phonemes.append(symbol)
                i += len(pattern)
                matched = True
                break
        if not matched:
            ch = word[i]
            if ch in SINGLE_CHAR_RULES:
                phonemes.append(SINGLE_CHAR_RULES[ch])
            # else: skip punctuation/digits/symbols/non-Latin chars silently
            i += 1
    return phonemes


def sentence_to_phonemes(sentence: str) -> list:
    """Convert a sentence into a flat phoneme-token list with word boundaries."""
    words = sentence.strip().split()
    tokens = []
    for idx, w in enumerate(words):
        tokens.extend(word_to_phonemes(w))
        if idx != len(words) - 1:
            tokens.append(WORD_BOUNDARY)
    return tokens


def build_vocab():
    if not TRAIN_PATH.exists():
        print(f"ERROR: {TRAIN_PATH} not found. Run data/preprocess.py first.")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    vocab = set()
    total_tokens = 0
    total_sentences = 0

    with open(TRAIN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = sentence_to_phonemes(line)
            vocab.update(tokens)
            total_tokens += len(tokens)
            total_sentences += 1

    vocab.add(WORD_BOUNDARY)
    sorted_vocab = sorted(vocab)
    id_map = {sym: i for i, sym in enumerate(sorted_vocab)}

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    print(f"Sentences processed: {total_sentences}")
    print(f"Total phoneme tokens: {total_tokens}")
    print(f"Avg tokens/sentence: {total_tokens / total_sentences:.2f}")
    print(f"Vocab size: {len(id_map)}")
    print(f"Saved vocab to {VOCAB_PATH}")


def encode(sentence: str):
    if not VOCAB_PATH.exists():
        print("ERROR: no vocab found. Run 'python tokenizer/phonetic_g2p.py train' first.")
        sys.exit(1)

    with open(VOCAB_PATH, encoding="utf-8") as f:
        id_map = json.load(f)

    tokens = sentence_to_phonemes(sentence)
    ids = [id_map.get(t, -1) for t in tokens]  # -1 = unseen phoneme (OOV)

    print(f"Input:  {sentence}")
    print(f"Tokens: {tokens}")
    print(f"IDs:    {ids}")
    print(f"Token count: {len(tokens)}")
    if -1 in ids:
        print("NOTE: -1 marks phoneme symbols not seen during vocab training.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "encode"):
        print("Usage:")
        print("  python tokenizer/phonetic_g2p.py train")
        print("  python tokenizer/phonetic_g2p.py encode \"some sentence\"")
        sys.exit(1)

    if sys.argv[1] == "train":
        build_vocab()
    elif sys.argv[1] == "encode":
        if len(sys.argv) < 3:
            print("Usage: python tokenizer/phonetic_g2p.py encode \"some sentence\"")
            sys.exit(1)
        encode(sys.argv[2])