# Model Identification

## Chosen model: nanoGPT

**What it is:** nanoGPT (Karpathy) is a minimal, readable GPT-2-style
decoder-only Transformer implementation — small enough to train on modest
compute, but architecturally identical in kind (multi-head self-attention,
learned positional embeddings, standard Transformer blocks) to the models
used in the papers this project builds on (Goriely et al., 2024 use a
GPT-2-style model for their phoneme-vs-orthography comparison).

## Why nanoGPT specifically (not a larger pretrained model)

1. **Controlled comparison requires training from scratch.** The research
   question is about the *tokenizer*, not about which pretrained model
   performs best. Fine-tuning a large pretrained model (which was itself
   pretrained on orthographic text with its own tokenizer) would confound
   the comparison — any difference in results could come from the
   pretraining, not the tokenizer. Training two small models from scratch,
   identical in every way except tokenizer, isolates the variable this
   project is testing.
2. **Compute constraints.** Available compute is a weak local CPU plus
   HPC access for actual training runs (per project constraints). nanoGPT
   is designed to be trainable on limited hardware, unlike larger
   from-scratch alternatives.
3. **Precedent in the literature.** Goriely et al. (2024) use the same
   class of model (GPT-2-style, trained from scratch) for their
   phoneme-vs-orthography comparison, which makes this project's results
   more directly comparable to the existing literature.

## Planned configuration (Phase 2, pending HPC access)

*Not yet run — this is the intended setup, to be adjusted once actual
training begins.*

| Setting | Value | Notes |
|---|---|---|
| Architecture | GPT-2-style decoder-only Transformer (nanoGPT) | Identical config for both runs |
| Layers | TBD (small, e.g. 4-6) | Sized to dataset scale (1k-6k sentences at prototyping stage) |
| Context length | TBD | Must accommodate phonetic tokenizer's longer sequences (~50-60 tokens/sentence vs. BPE's ~7-10) |
| Vocab size (input embedding) | BPE: 2000 (placeholder). Phonetic: 42 raw, or larger if a merge step is added | **Open question — see vocab-matching decision below** |
| Training data | data/processed/train.txt (currently 1000 sentences, IIIT-H corpus) | Will scale to L3Cube-HingCorpus in Phase 4 |
| Two runs, identical config | Run A: BPE tokenizer. Run B: phonetic tokenizer | Only the tokenizer differs between runs — this is the core controlled comparison |

## Known open decision (flagged for professor)

The phonetic tokenizer's context length requirement is much longer than
BPE's for the same sentence (measured: ~50-62 phonetic tokens vs. ~7-10 BPE
tokens per sentence in Phase 1 testing). This has two consequences to
confirm before Phase 2 training begins:
1. Context length must be set large enough for the phonetic tokenizer's
   longest sequences, which increases compute cost for that run relative
   to BPE, even with an identical layer/hidden-size config.
2. This ties directly into the still-open vocab-size-matching question
   (see NEXT_SESSION notes / professor email) — resolving how the phonetic
   vocab is built will also affect how long its token sequences are.

## Evaluation plan (Phase 2-3, once trained)

- Perplexity / loss comparison between Run A and Run B on held-out data
- Fine-tune both on Hinglish-24k sentiment task
- Evaluate both on clean vs. spelling-perturbed test sets (headline
  robustness result)