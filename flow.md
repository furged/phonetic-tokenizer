flowchart TD
    A["L3Cube-HingCorpus archive<br/>(2.49 GB, roman)"] --> B["data/prepare_subset.py<br/>reservoir sample 50k, seed 42"]
    B --> C["data/processed/train.txt<br/>50,000 lines"]
    C --> D["experiments/train.py<br/>make_split(): 90/10, seed 42"]
    D --> E["split_train.txt<br/>45k"]
    D --> F["split_test.txt<br/>5k held-out"]
    E --> G["baselines/bpe_tokenizer.py<br/>BPETokenizer"]
    E --> H["tokenizer/phonetic_g2p.py<br/>build_phonetic_bpe()"]
    H --> H2["espeak_phonemes()<br/>Phonemizer -> eSpeak NG -> IPA"]
    G --> I["results/tokenizer_bpe.json"]
    H --> J["results/tokenizer_phonetic.json"]
    I --> K["model/nanogpt.py<br/>GPT + GPTConfig"]
    J --> K
    K --> L["train_model(): 5 epochs<br/>save_checkpoint() each epoch"]
    L --> M["results/ckpt_*.pt<br/>results/model_*_final.pt"]
    M --> N["compute_bpc()<br/>on split_test.txt"]
    F --> N
    N --> O["results/tables/bpc.csv"]
    C --> P["experiments/tokenizer_stats.py<br/>tokens per sentence"]
    P --> Q["results/tables/tokenizer_stats.csv"]
    R["data/raw/spelling_variants.csv<br/>114 groups"] --> S["experiments/collision_rate.py<br/>read_groups(), collides()"]
    S --> T["results/tables/collision_rate.csv"]
    R --> U["experiments/check_espeak.py<br/>eSpeak agreement"]
    U --> V["results/tables/espeak_check.csv"]
    O --> W["experiments/plot_bpc.py<br/>plot_collision.py, plot_tokens.py,<br/>plot_vocab.py, plot_collision_grid.py,<br/>plot_sound_mapping.py, plot_epoch_loss*.py"]
    Q --> W
    T --> W
    W --> X["results/plots/*.png"]
