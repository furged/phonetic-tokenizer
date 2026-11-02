"""Training script for nanoGPT with phonetic or BPE tokenizers.
Reports bits per character. Resumable (epoch and mid-epoch checkpoints).
Usage:  RUN_TAG=100k python experiments/train.py bpe
        RUN_TAG=100k python experiments/train.py phonetic
"""

import sys
sys.path.insert(0, '.')

import os
import math
import random
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from model.nanogpt import GPT, GPTConfig
from tokenizer.phonetic_g2p import build_phonetic_bpe, PhoneticBPE, PhoneticTokenizer
from baselines.bpe_tokenizer import BPETokenizer
from tokenizers import Tokenizer

SEED = 42
TEST_FRAC = 0.1
RUN_TAG = os.environ.get("RUN_TAG", "50k")
FULL_PATH = 'data/processed/train.txt' if RUN_TAG == "50k" else f'data/processed/train_{RUN_TAG}.txt'
TRAIN_PATH = f'data/processed/split_train_{RUN_TAG}.txt'
TEST_PATH = f'data/processed/split_test_{RUN_TAG}.txt'
MAX_SAMPLES = None  # set to 100 only for quick tests
MID_CKPT_EVERY = 2000  # batches

Path('results/logs').mkdir(parents=True, exist_ok=True)
Path('results/tables').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(f'results/logs/train_{RUN_TAG}.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger('train')


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
    log.info(f"Split: {len(sentences) - n_test} train, {n_test} test (seed={seed}, tag={RUN_TAG})")


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


def save_checkpoint(path, model, optimizer, epoch, extra):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'extra': extra,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch']


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
    log.info(f"Training {tokenizer_name.upper()} tokenizer ({RUN_TAG})")
    config = GPTConfig()
    device = config.device
    log.info(f"Device: {device}")

    tok_path = Path(f'results/tokenizer_{tokenizer_name}_{RUN_TAG}.json')
    if tokenizer_name == 'phonetic':
        if tok_path.exists():
            tokenizer = PhoneticBPE(Tokenizer.from_file(str(tok_path)), PhoneticTokenizer())
            log.info("Loaded saved phonetic tokenizer")
        else:
            log.info("Building phonetic tokenizer...")
            tokenizer = build_phonetic_bpe(TRAIN_PATH, vocab_size=2000)
            tokenizer.tok.save(str(tok_path))
        config.vocab_size = tokenizer.get_vocab_size()
    else:
        if tok_path.exists():
            tokenizer = BPETokenizer(vocab_size=2000)
            tokenizer.tokenizer = Tokenizer.from_file(str(tok_path))
            log.info("Loaded saved BPE tokenizer")
        else:
            log.info("Building BPE tokenizer...")
            tokenizer = BPETokenizer(vocab_size=2000)
            tokenizer.train(TRAIN_PATH)
            tokenizer.tokenizer.save(str(tok_path))
        config.vocab_size = tokenizer.vocab_size()
    log.info(f"Vocab size: {config.vocab_size}")

    dataset = HinglishDataset(TRAIN_PATH, tokenizer, config.max_seq_length, MAX_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    log.info(f"Train sequences: {len(dataset)}")

    model = GPT(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    log.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_path = Path(f'results/ckpt_{tokenizer_name}_{RUN_TAG}.pt')
    start_epoch = 0
    if ckpt_path.exists():
        start_epoch = load_checkpoint(ckpt_path, model, optimizer, device) + 1
        log.info(f"Resuming from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, config.num_epochs):
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
                log.info(f"Epoch {epoch+1}/{config.num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                         f"Loss: {epoch_loss/(batch_idx+1):.4f}")
            if (batch_idx + 1) % MID_CKPT_EVERY == 0:
                save_checkpoint(ckpt_path, model, optimizer, epoch - 1, {'mid_batch': batch_idx + 1})
        avg = epoch_loss / max(len(dataloader), 1)
        log.info(f"Epoch {epoch+1} - train loss (nats/token): {avg:.4f}")
        save_checkpoint(ckpt_path, model, optimizer, epoch, {'loss': avg})
        log.info(f"Checkpoint saved after epoch {epoch+1}")

    final_path = f'results/model_{tokenizer_name}_{RUN_TAG}_final.pt'
    torch.save(model.state_dict(), final_path)

    bpc, n_scored, total_chars = compute_bpc(model, tokenizer, TEST_PATH, config)
    log.info(f"Test: {n_scored} tokens scored over {total_chars} characters")
    log.info(f"BITS PER CHARACTER ({tokenizer_name}, {RUN_TAG}): {bpc:.4f}")
    return bpc


if __name__ == '__main__':
    try:
        make_split(FULL_PATH, TRAIN_PATH, TEST_PATH)
        names = sys.argv[1:] or ['bpe', 'phonetic']
        for name in names:
            log.info(f"Starting {name} ({RUN_TAG})")
            bpc = train_model(name)
            log.info(f"Finished {name}: BPC {bpc:.4f}")
            with open(f'results/tables/bpc_{RUN_TAG}.csv', 'a', encoding='utf-8') as f:
                f.write(f'{name},{bpc:.4f}\n')
    except Exception:
        log.exception("Training crashed")
        raise