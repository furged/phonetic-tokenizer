```markdown
# Sound Over Spelling: Phonetic vs. Subword Tokenization for Hindi/Hinglish Language Modeling

Comparative study testing whether phonetic (sound-based) tokenization is more robust to spelling variation than standard BPE (spelling-based) tokenization, using Hinglish as a stress test for inconsistent spelling.

**Research question:** Does phonetic tokenization make language models more robust to spelling variation than BPE, when trained at matched scale?

## Hypothesis

Phonetic tokenization maps spelling variants of the same word (e.g. `acha`, `accha`, `achha`) to the same or similar token sequences more often than BPE does, so models trained on it should degrade less under spelling noise, even though phonetic sequences are longer.

## Status

| Phase | Component | Status |
|---|---|---|
| 0 | Environment, repo, dependencies | Done |
| 1 | Phonetic tokenizer (eSpeak G2P via Phonemizer) + BPE baseline | Done |
| 1 | Tokenizer stats (vocab size, tokens/sentence) | Done (preliminary, on 1000 sentences) |
| 1 | Collision rate script | Done, preliminary result (BPE 0/114, phonetic 19/114) |
| 2 | Training data: L3Cube-HingCorpus subset (50k, seed 42) | Done |
| 2 | nanoGPT training with BPC metric, resumable | Smoke test done; full run pending |
| 3 | Spelling groups (transliteration-based) | Pending: regrouped file from teammate |
| 3 | Robustness test (clean vs perturbed) | Not started |
| 3 | Sentiment fine-tuning (cleaned sentiment data) | Not started, data cleaning in progress |
| 4 | Paper | Drafting |

## Repository Layout

```

baselines/     BPE tokenizer baseline (Hugging Face tokenizers)
data/
  prepare_subset.py   Reservoir-samples a fixed 50k subset from the L3Cube archive
  preprocess.py       Original cleaning script (older, 1000-sentence path)
  raw/                Raw inputs (ignored by git): archive, spelling groups, sentiment files
  processed/          Derived splits (ignored by git)
experiments/
  tokenizer_stats.py  Vocab size and tokens/sentence for both tokenizers
  collision_rate.py   Do spelling variants of the same word get the same tokens?
  train.py            nanoGPT training with BPC, checkpoint resume, per-tokenizer runs
  eda.py              Exploratory data analysis
model/
  nanogpt.py          Small GPT (4 layers, 4 heads, 128 dim, ~1.3M parameters)
tokenizer/
  phonetic_g2p.py     eSpeak G2P (via Phonemizer), phonetic BPE wrapper
results/              Tables, checkpoints, saved tokenizers (ignored by git: *.pt, tokenizer files)

```

## Setup

```bash
git clone https://github.com/furged/phonetic-tokenizer.git
cd phonetic-tokenizer
pip install -r requirements.txt datasets
```

**eSpeak NG is required** for the phonetic tokenizer. Install it from the eSpeak NG GitHub releases, then set:

```bash
export PHONEMIZER_ESPEAK_LIBRARY="C:/Program Files/eSpeak NG/libespeak-ng.dll"
export PHONEMIZER_ESPEAK_PATH="C:/Program Files/eSpeak NG/espeak-ng.exe"
```

(On Windows, set these as user environment variables so they persist.)

**Hugging Face token** (needed to download the L3Cube archive quickly):

```bash
export HF_TOKEN=your_token_here
```

## Reproducing the Pipeline

Run from the repo root, in this order.

1. **Build the training subset** (downloads the 2.49 GB archive once, about 40 minutes on a typical connection):
   ```bash
   python data/prepare_subset.py
   ```
   Writes 50,000 sentences to `data/processed/train.txt`, with seed 42.

2. **Tokenizer stats** (Phase 1 numbers):
   ```bash
   python experiments/tokenizer_stats.py
   ```

3. **Collision rate** (needs `data/raw/spelling_variants.csv` in the target format: `hindi, variant_1 ... variant_n, source`):
   ```bash
   python experiments/collision_rate.py
   ```

4. **Train the language models** (resumable, run one at a time):
   ```bash
   python experiments/train.py bpe
   python experiments/train.py phonetic
   ```
   Pausing with Ctrl+C loses only the current epoch. Rerunning resumes from the last completed epoch. Results go to `results/tables/bpc.csv`.

## Metrics

- **Bits per character (BPC):** total test loss in bits, divided by the number of raw characters in the test text. Both tokenizers are scored on the same characters, so BPC is comparable across tokenizers. Replaces perplexity, which is not comparable when token sizes differ.
- **Collision rate:** share of spelling-variant groups whose variants all produce identical token sequences.

## Data

| Use | Source | Notes |
|---|---|---|
| Language model training | L3Cube-HingCorpus (`hinglish-orthobench/HingCorpus`, roman archive) | 50k-sentence seeded sample of a much larger corpus |
| Language model evaluation | 10% held-out split of the 50k sample | Same split for both tokenizers |
| Spelling groups | Transliteration dataset, via teammate's grouped file | Source row IDs not yet verified; function words still included |
| Sentiment (planned) | SemEval-2020 Task 9 Hinglish, via Person C | Labels: 0 = neutral, 1 = positive, 2 = negative. Needs cleaning |

The original HuggMachas corpus (movie-heavy) was used only for early pipeline tests and is no longer part of the main experiment.

## Known Limitations

- **Scale:** the model is small (~1.3M parameters) and trained on a 50k-sentence sample, so results may not transfer to larger models.
- **Phonetic G2P:** eSpeak via Phonemizer, not Epitran (no Latin-script Hindi map) or Festvox (Linux-oriented). Pronunciation rules are imperfect for romanized spelling, so some variants still map to different sounds (e.g. `accha` vs `acha`).
- **Metric fairness:** the phonetic model predicts sound sequences, not exact text, so its BPC is only approximately comparable to BPE's.
- **Phonetic vocabulary:** the eSpeak symbol set has 51 base symbols, and the 2000-token phonetic vocab is partly built from rare merges.
- **Spelling groups:** the teammate's grouping is frequency-based in parts, includes function words and some inflection variants, and its source IDs are unverified. The collision result must be reported with this caveat.
- **Sentiment data:** source contains abusive and communal language and emoji names. Cleaning rules and the emoji treatment are pending mentor approval.

## Team

- Project lead: pipeline, tokenizers, training, metrics
- Teammate (Person C): spelling groups, sentiment data preparation
- Teammate (Person B): sentiment data and paper sections

## References

- Goriely et al. (2024). *Babble to Words*.
- *Tokenization Limits Phonological Knowledge* (2026).
- L3Cube-HingCorpus (L3Cube Pune).
- Hinglish-Hindi Transliteration Dataset (Hugging Face).
- SemEval-2020 Task 9: Overview of sentiment analysis of code-mixed tweets.
- eSpeak NG; Phonemizer.

## Licence

TBD. Data is subject to the licences of its original sources (L3Cube, transliteration dataset, SemEval-2020 Task 9). Check each before redistributing.
```