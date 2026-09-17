# Sound Over Spelling: Phonetic vs. BPE Tokenization for Hindi/Hinglish

Comparative study testing whether phonetic (sound-based) tokenization is more
compact and more robust to spelling variation than standard BPE (spelling-based)
tokenization, using Hindi/Hinglish code-mixed text as a natural stress test for
spelling inconsistency.

**Research question:** Does phonetic tokenization make language models more
robust to spelling errors and orthographic noise than standard subword
tokenization, when trained at the same scale?

## Status (as of last session)

**Phase 0 (environment) — done.**
- Project set up at `C:\dev\phonetic-tokenizer\`, git initialized, pushed to GitHub
- Dependencies installed (`requirements.txt`)

**Phase 1 (data + tokenizers) — in progress.**
- Dataset: IIIT-H English-Hindi code-mixed corpus (Dhar et al., 2018),
  downloaded from https://github.com/mrinaldhar/en-hi-codemixed-corpus
  (source file `s-enhi.txt`). ~5700 valid sentences found; 1000 used for
  Phase 1 prototyping. (Note: an earlier session used PHINC as a substitute
  before the real IIIT-H corpus link was found — PHINC is no longer used.)
- Preprocessing (`data/preprocess.py`): cleans, dedupes, writes
  `data/processed/train.txt`.
- EDA (`experiments/eda.py`): sentence length distribution, vocab size,
  word frequency, rough code-mixing rate. Outputs to `results/plots/` and
  `results/tables/eda_summary.csv`.
- BPE baseline tokenizer (`baselines/bpe_tokenizer.py`): Hugging Face
  byte-level BPE, vocab_size=2000 (placeholder, see open question below).
- Phonetic tokenizer (`tokenizer/phonetic_g2p.py`): **rule-based**, not
  Epitran/Festvox. Epitran's Hindi module expects Devanagari script, but
  this corpus is romanized (Latin script) Hinglish, so a lightweight
  digraph-based phonetic normalizer was built instead, operating directly
  on the Latin spelling. Does not model schwa deletion. Documented as a
  limitation, not a bug — see script docstring for full reasoning.
- Comparison script (`experiments/tokenizer_stats.py`): runs both
  tokenizers over `train.txt` and reports vocab size + tokens/sentence
  side by side. Saves to `results/tables/tokenizer_stats.csv`.
- Model identification (`results/model_identification.md`): justifies
  nanoGPT as the model choice, written up in advance of Phase 2 (no
  GPU/HPC access used yet — no training has happened).

## Open questions (need professor input)

1. **Vocab-size mismatch:** the phonetic tokenizer's raw vocab is capped
   at ~40 symbols (fixed phoneme inventory), while BPE's is a tunable
   hyperparameter (currently 2000). These can't be matched without an
   extra step — possibly a BPE-style merge pass applied on top of the
   phoneme stream to reach a target vocab size (2^8/2^12/2^16), which may
   be what "8/12/16-bit" in the original project notes meant. Not yet
   resolved — pending professor confirmation.
2. **Sequence length:** the phonetic tokenizer produces far more tokens
   per sentence than BPE (~50-62 vs. ~7-10 in current testing), which
   affects context-length and compute decisions for Phase 2 model
   training.

## Project structure
```
data/           raw/ (downloaded corpora) and processed/ (cleaned splits)
tokenizer/      phonetic G2P tokenizer + phoneme-to-ID vocabulary
baselines/      BPE tokenizer baseline
model/          nanoGPT-style model with tokenizer hooks (not yet built)
experiments/    tokenizer stats, EDA, training scripts, evaluation
results/        output tables, plots, and writeups
```

## Setup
```bash
pip install -r requirements.txt
```

## Pipeline
1. Preprocess Hinglish corpus (dedupe, clean, language ID)
2. Build BPE tokenizer (baseline) and phonetic tokenizer (G2P + phoneme→ID)
3. Tokenizer-level eval: vocab size, tokens/sentence, collision rate on
   spelling variants (collision-rate testing still pending — requires the
   Hinglish-Hindi Transliteration Dataset, not yet downloaded)
4. Train nanoGPT with each tokenizer (identical config) — Phase 2, not started
5. Evaluate: perplexity, fine-tune on Hinglish sentiment task, accuracy on
   clean vs. spelling-perturbed test sets (headline result) — Phase 3, not started

See `PHONETIC_TOKENIZER_PROJECT_HANDOFF.md` for full project context, locked-in
design decisions, and phase-by-phase plan.

## Datasets
- IIIT-H code-mixed corpus (6k sentences) — pipeline prototyping (**in use**)
- L3Cube-HingCorpus — scale-up (not yet downloaded)
- Hinglish-Hindi Transliteration Dataset — spelling-variant / robustness test set (not yet downloaded)
- Hinglish-24k sentiment corpus — downstream evaluation task (not yet downloaded)

## Note on corpus content
The IIIT-H corpus is unfiltered, real-world social media commentary and
contains occasional hostile/communal/offensive language, as is common for
social-media-scraped NLP datasets. Worth a one-line disclosure in the
paper's data/ethics section.

## License
TBD