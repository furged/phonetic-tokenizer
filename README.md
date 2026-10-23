# Sound Over Spelling

Phonetic vs. Subword Tokenization for Hindi/Hinglish Language Modeling

Does turning Hinglish into sounds make a language model more robust to spelling variation than standard subword tokenization?



![Python](https://img.shields.io/badge/Python-3.10%2B-blue)




![PyTorch](https://img.shields.io/badge/PyTorch-nanoGPT-orange)




![Status](https://img.shields.io/badge/Status-Training%20in%20progress-yellow)




![Data](https://img.shields.io/badge/Data-L3Cube--HingCorpus-green)



---

## The Idea

Hinglish has no fixed spelling. The same word is written many ways:

    acha   accha   achha   achcha   ->  "good"
    kuch   kuchh                    ->  "something"

BPE splits text by spelling, so each variant becomes a different token sequence. The model has to learn each spelling separately.

Hypothesis: tokenizing by sound (eSpeak phonemes) should make spelling variants collapse into the same or similar sequences, so the model should degrade less under misspelled input, even though sound sequences are longer.

    spelling -> BPE        -> [acha] [accha] [achha]     three different sequences
    sound    -> phonetic   -> a t S a / a t S h a        near-identical

---

## Results So Far

| Measure | BPE | Phonetic-BPE | Notes |
|---|---|---|---|
| Vocab size | 2000 | 2000 | Matched target |
| Avg tokens per sentence | 33.97 | 100.36 | Preliminary, 1000 sentences |
| Collision rate (spelling groups) | 0 / 114 | 19 / 114 | Preliminary, groups not yet verified |
| Bits per character (smoke test) | 3.70 | 3.19 | 1 epoch, 100 sentences. Pipeline check only, not a result |

The full bits-per-character comparison runs after the full training. The smoke-test numbers only confirm that the pipeline works.

---

## Pipeline

    L3Cube-HingCorpus
        -> 50k seeded sample (seed 42)
        -> 90/10 train/test split (seed 42)
        -> BPE tokenizer        (spelling)
        -> Phonetic tokenizer   (text -> eSpeak IPA -> BPE)
        -> nanoGPT, identical config for both
        -> bits per character on the same held-out text

Bits per character is used because perplexity is measured per token. The two tokenizers cut text into different-sized pieces, so their perplexities cannot be compared. BPC divides total loss by raw characters of the same test text.

---

## Setup

    git clone https://github.com/furged/phonetic-tokenizer.git
    cd phonetic-tokenizer
    pip install -r requirements.txt datasets

eSpeak NG is required for the phonetic path. Install it, then set these as user environment variables (Windows):

    PHONEMIZER_ESPEAK_LIBRARY = C:\Program Files\eSpeak NG\libespeak-ng.dll
    PHONEMIZER_ESPEAK_PATH    = C:\Program Files\eSpeak NG\espeak-ng.exe

Set a Hugging Face token so the 2.49 GB archive downloads quickly:

    export HF_TOKEN=your_token_here

---

## Reproduce

Run from the repo root, in this order.

| Step | Command | Output |
|---|---|---|
| 1. Build subset | `python data/prepare_subset.py` | `data/processed/train.txt` (50,000 lines) |
| 2. Tokenizer stats | `python experiments/tokenizer_stats.py` | `results/tables/tokenizer_stats.csv` |
| 3. Collision rate | `python experiments/collision_rate.py` | `results/tables/collision_rate.csv` |
| 4. Train BPE | `python experiments/train.py bpe` | checkpoints and a BPC row |
| 5. Train phonetic | `python experiments/train.py phonetic` | checkpoints and a BPC row |

Pausing: press Ctrl+C. Progress is saved at the end of each epoch, so rerunning the same command resumes from the last completed epoch. Copy `results/` to move between machines.

---

## Repository Map

    phonetic-tokenizer/
    ├── data/
    │   ├── prepare_subset.py    seeded 50k subset from the archive
    │   ├── raw/                 inputs (git-ignored)
    │   └── processed/           derived splits (git-ignored)
    ├── tokenizer/
    │   └── phonetic_g2p.py      eSpeak G2P and phonetic BPE wrapper
    ├── baselines/
    │   └── bpe_tokenizer.py     BPE baseline
    ├── model/
    │   └── nanogpt.py           small GPT, about 1.3M parameters
    ├── experiments/
    │   ├── tokenizer_stats.py   vocab size and tokens per sentence
    │   ├── collision_rate.py    do spelling variants collide?
    │   ├── train.py             training, BPC, resumable checkpoints
    │   └── eda.py               exploratory analysis
    └── results/                 tables, checkpoints, saved tokenizers

---

## Model

| Setting | Value |
|---|---|
| Architecture | Decoder-only Transformer (nanoGPT style) |
| Layers / heads / embedding | 4 / 4 / 128 |
| Context length | 256 tokens |
| Parameters | about 1.34M |
| Optimizer | Adam, learning rate 3e-4 |
| Epochs | 5 |
| Vocabulary | 2000 for both tokenizers |

Only the tokenizer changes between runs. Architecture, optimizer, data, and seed are identical.

---

## Data

| Use | Source | Status |
|---|---|---|
| Language modeling | L3Cube-HingCorpus (roman), 50k sample, seed 42 | Ready |
| Held-out evaluation | 10% of the 50k sample | Ready |
| Spelling groups | Transliteration-based grouping from teammate | Pending verification |
| Sentiment (planned) | SemEval-2020 Task 9 Hinglish | Pending cleaning and label confirmation |

---

## Limitations

- **Small scale.** About 1.3M parameters on a 50k-sentence sample. Results may not hold for larger models.
- **Imperfect G2P.** eSpeak reads spellings like `accha` differently from `acha`, so some variants never collapse.
- **Approximate fairness.** The phonetic model predicts sound sequences, so its BPC is only approximately comparable to BPE's.
- **Rare merges.** The 2000-token phonetic vocabulary is partly built from rare pairs.
- **Unverified groups.** Spelling groups include function words and inflections, and their source rows are not yet checked.
- **Sensitive content.** Social-media sentiment data contains abusive and communal language. Handling is pending mentor review.

---

## Roadmap

- [x] Phonetic tokenizer on eSpeak G2P
- [x] Bits-per-character metric and resumable training
- [x] 50k L3Cube subset
- [ ] Full training run, both tokenizers
- [ ] Verified spelling groups
- [ ] Robustness test on clean vs perturbed sentiment data
- [ ] Sentiment fine-tuning

---

## References

1. Goriely et al. (2024). Babble to Words.
2. Tokenization Limits Phonological Knowledge (2026).
3. L3Cube-HingCorpus, L3Cube Pune.
4. Hinglish-Hindi Transliteration Dataset, Hugging Face.
5. SemEval-2020 Task 9: sentiment analysis of code-mixed tweets.
6. eSpeak NG and Phonemizer.

---

## License

Code: to be decided. Data is governed by the licences of its original sources (L3Cube, transliteration dataset, SemEval-2020 Task 9). Check each before redistributing.