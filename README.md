# Sound Over Spelling: Phonetic vs. BPE Tokenization for Hindi/Hinglish

Comparative study testing whether phonetic (sound-based) tokenization is more
compact and more robust to spelling variation than standard BPE (spelling-based)
tokenization, using Hindi/Hinglish code-mixed text as a natural stress test for
spelling inconsistency.

**Research question:** Does phonetic tokenization make language models more
robust to spelling errors and orthographic noise than standard subword
tokenization, when trained at the same scale?

## Status
Phase 0 (environment setup) — in progress.

## Project structure
```
data/           raw/ (downloaded corpora) and processed/ (cleaned splits)
tokenizer/      phonetic G2P tokenizer + phoneme-to-ID vocabulary
baselines/      BPE tokenizer baseline
model/          nanoGPT-style model with tokenizer hooks
experiments/    tokenizer stats, training scripts, evaluation
results/        output tables and plots
```

## Setup
```bash
pip install -r requirements.txt
```

## Pipeline
1. Preprocess Hinglish corpus (dedupe, clean, language ID)
2. Build BPE tokenizer (baseline) and phonetic tokenizer (G2P + phoneme→ID)
3. Tokenizer-level eval: vocab size, tokens/sentence, collision rate on
   spelling variants
4. Train nanoGPT with each tokenizer (identical config)
5. Evaluate: perplexity, fine-tune on Hinglish sentiment task, accuracy on
   clean vs. spelling-perturbed test sets (headline result)

See `PHONETIC_TOKENIZER_PROJECT_HANDOFF.md` for full project context, locked-in
design decisions, and phase-by-phase plan.

## Datasets
- IIIT-H code-mixed corpus (6k sentences) — pipeline prototyping
- L3Cube-HingCorpus — scale-up
- Hinglish-Hindi Transliteration Dataset — spelling-variant / robustness test set
- Hinglish-24k sentiment corpus — downstream evaluation task

## License
TBD
