"""Training script for nanoGPT with phonetic or BPE tokenizers. Reports bits per character."""

import sys
sys.path.insert(0, '.')

import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from model.nanogpt import GPT, GPTConfig
from tokenizer.phonetic_g2p import build_phonetic_bpe
from baselines.bpe_tokenizer import BPETokenizer

SEED = 42
TEST_FRAC = 0.1
FULL_PATH = 'data/processed/train.txt'
TRAIN_PATH = 'data/processed/split_train.txt'
TEST_PATH = 'data/processed/split_test.txt'
MAX_SAMPLES = None  # set to 100 only for quick tests


def make_split(corpus_path, train_path, test_path, test_frac=TEST_FRAC, seed=SEED):
    """Fixed, seeded split. Both tokenizers use the same files."""
    with open(corpus_path, encoding='utf-8') as f:
        sentences = [l.strip() for l in f if l.strip()]
    idx = list(range(len(sentences)))
    random.Random(seed).shuffle(idx)
    n_test = int(len(sentences) * test_frac)
    test_idx = set(idx[:n_test])
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(s for i, s in enumerate(sentences) if i not in test_idx) + '\n')
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(s for i, s in enumerate(sentences) if i in test_idx) + '\n')
    print(f"Split: {len(sentences) - n_test} train, {n_test} test (seed={seed})")


def encode_ids(tokenizer, sent):
    enc = tokenizer.encode(sent)
    return enc if isinstance(enc, list) else enc.ids


class HinglishDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, max_seq_length=256, max_samples=None):
        self.max_seq_length = max_seq_length
        with open(corpus_path, encoding='utf-8') as f:
            sentences = [l.strip() for l in f if l.strip()]
        if max_samples:
            sentences = sentences[:max_samples]
        self.token_ids = []
        for sent in sentences:
            self.token_ids.extend(encode_ids(tokenizer, sent))
        self.sequences = []
        for i in range(len(self.token_ids) - max_seq_length):
            seq = self.token_ids[i:i + max_seq_length]
            target = self.token_ids[i + 1:i + max_seq_length + 1]
            self.sequences.append((seq, target))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


@torch.no_grad()
def compute_bpc(model, tokenizer, test_path, config):
    """Total test loss in bits, divided by raw characters of the test text.
    Non-overlapping windows, so each target token is scored exactly once."""
    model.eval()
    device = config.device
    with open(test_path, encoding='utf-8') as f:
        sentences = [l.strip() for l in f if l.strip()]
    total_chars = sum(len(s) for s in sentences)

    ids = []
    for s in sentences:
        ids.extend(encode_ids(tokenizer, s))

    L = config.max_seq_length
    total_nats, n_scored = 0.0, 0
    for start in range(0, len(ids) - 1, L):
        chunk = ids[start:start + L + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        loss_sum = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
        total_nats += loss_sum.item()
        n_scored += y.numel()

    bits = total_nats / math.log(2)
    bpc = bits / total_chars
    model.train()
    return bpc, n_scored, total_chars


def train_model(tokenizer_name):
    print(f"\n{'=' * 60}\nTraining with {tokenizer_name.upper()} tokenizer\n{'=' * 60}\n")
    config = GPTConfig()
    device = config.device
    print(f"Device: {device}\n")

    if tokenizer_name == 'phonetic':
        print("Building phonetic tokenizer...")
        tokenizer = build_phonetic_bpe(TRAIN_PATH, vocab_size=2000)
        config.vocab_size = tokenizer.get_vocab_size()
    else:
        print("Building BPE tokenizer...")
        tokenizer = BPETokenizer(vocab_size=2000)
        tokenizer.train(TRAIN_PATH)
        config.vocab_size = tokenizer.vocab_size()
    print(f"Vocab size: {config.vocab_size}\n")

    dataset = HinglishDataset(TRAIN_PATH, tokenizer, config.max_seq_length, MAX_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print(f"Train sequences: {len(dataset)}\n")

    model = GPT(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config.num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {epoch_loss/(batch_idx+1):.4f}")
        avg = epoch_loss / max(len(dataloader), 1)
        print(f"\nEpoch {epoch+1} - train loss (nats/token): {avg:.4f}\n")

    torch.save(model.state_dict(), f'results/model_{tokenizer_name}_final.pt')

    bpc, n_scored, total_chars = compute_bpc(model, tokenizer, TEST_PATH, config)
    print(f"Test: {n_scored} tokens scored over {total_chars} characters")
    print(f"BITS PER CHARACTER ({tokenizer_name}): {bpc:.4f}\n")
    return bpc


if __name__ == '__main__':
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    make_split(FULL_PATH, TRAIN_PATH, TEST_PATH)

    results = {}
    for name in ['bpe', 'phonetic']:
        results[name] = train_model(name)

    print("=" * 60)
    print("BPC SUMMARY (lower is better)")
    print("=" * 60)
    for name, bpc in results.items():
        print(f"{name.upper()}: {bpc:.4f} bits per character")

    with open('results/tables/bpc.csv', 'w', newline='', encoding='utf-8') as f:
        f.write('tokenizer,bits_per_char\n')
        for name, bpc in results.items():
            f.write(f'{name},{bpc:.4f}\n')