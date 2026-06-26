# cs336 assignments

my implementations for stanford cs336 (language modeling from scratch).

## assignment 1: basics

### Dataset Used
- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main) 🤗

### progress
- [x] phase 1: bpe tokenizer training 
- [x] phase 2: tokenizer encode and decode
- [x] phase 3: model components (rmsnorm, swiglu, rope, multi-head attention, transformer block, transformerlm)
- [x] phase 4: training utilities (cross-entropy, gradient clipping, cosine lr schedule, custom adamw optimizer)
- [x] phase 5: data loading & checkpointing (batching, save/load checkpoints)
- [x] training loop (implemented and integrated)

### tasks left
- [ ] phase 6: text generation (sampling & temperature)
- [ ] phase 7: scaling law experiments (only the final evaluation/experiments phase remains!)
