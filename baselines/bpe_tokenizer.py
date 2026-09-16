"""
BPE baseline tokenizer for the phonetic-vs-BPE comparison.

Trains a byte-level BPE tokenizer (Hugging Face `tokenizers` library) on
data/processed/train.txt and saves it to baselines/bpe_tokenizer.json.

Usage:
    python baselines/bpe_tokenizer.py train                 # train + save
    python baselines/bpe_tokenizer.py encode "some sentence"  # quick test
"""

import sys
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

TRAIN_PATH = Path("data/processed/train.txt")
MODEL_DIR = Path("baselines/bpe_model")
VOCAB_SIZE = 2000  # placeholder -- match this to phonetic vocab size later,
                    # per professor's guidance ("match BPE & phonetic vocab sizes")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]


def train():
    if not TRAIN_PATH.exists():
        print(f"ERROR: {TRAIN_PATH} not found. Run data/preprocess.py first.")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(TRAIN_PATH)],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(str(MODEL_DIR))
    print(f"Trained BPE tokenizer, vocab_size={VOCAB_SIZE}")
    print(f"Saved to {MODEL_DIR}/vocab.json and {MODEL_DIR}/merges.txt")

    # Actual resulting vocab size (BPE training doesn't always hit the exact
    # target if the corpus is small -- worth checking for the vocab-matching
    # requirement).
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Actual vocab size after training: {actual_vocab_size}")


def encode(sentence: str):
    vocab_file = MODEL_DIR / "vocab.json"
    merges_file = MODEL_DIR / "merges.txt"
    if not vocab_file.exists() or not merges_file.exists():
        print("ERROR: no trained model found. Run 'python baselines/bpe_tokenizer.py train' first.")
        sys.exit(1)

    tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
    output = tokenizer.encode(sentence)
    print(f"Input:  {sentence}")
    print(f"Tokens: {output.tokens}")
    print(f"IDs:    {output.ids}")
    print(f"Token count: {len(output.ids)}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "encode"):
        print("Usage:")
        print("  python baselines/bpe_tokenizer.py train")
        print("  python baselines/bpe_tokenizer.py encode \"some sentence\"")
        sys.exit(1)

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "encode":
        if len(sys.argv) < 3:
            print("Usage: python baselines/bpe_tokenizer.py encode \"some sentence\"")
            sys.exit(1)
        encode(sys.argv[2])