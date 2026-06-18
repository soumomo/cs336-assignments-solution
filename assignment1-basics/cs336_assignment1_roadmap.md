# CS336 Assignment 1 — Complete Roadmap

> **Total: ~120 points across ~30 problems. Everything goes in `cs336_basics/` (currently empty). Wire it up via `tests/adapters.py`.**

---

## How This Assignment Works

```
cs336_basics/          ← You write ALL your code here (from scratch)
tests/adapters.py      ← You fill in thin "glue" functions that call YOUR code
tests/test_*.py        ← Pre-written tests that call adapters (DON'T edit these)
```

You implement a component → fill in the matching adapter → run the test → move on.

---

## Phase 1: BPE Tokenizer Training (Section 2) — ~19 pts

**This is the first big chunk. Build it before touching the model.**

### Step 1.1 — Written questions on Unicode (2 problems, ~4 pts)
- `unicode1` (1 pt): Answer questions about `chr(0)`, `__repr__` vs `print`, etc.
- `unicode2` (3 pts): UTF-8 vs UTF-16/32, broken decoder function, invalid byte sequences.
- **No code needed** — just written answers for `writeup.pdf`.

### Step 1.2 — BPE Training Function (15 pts) ⭐ BIG ONE
**What to build:** A function that takes a text file and produces a **vocabulary** and **merges list**.

Three sub-steps inside this:
1. **Pre-tokenize** the corpus using the GPT-2 regex pattern (use `regex` package)
2. **Count byte pairs** across pre-tokens
3. **Iteratively merge** the most frequent pair, update counts, repeat

Key details:
- Initial vocab = 256 bytes + special tokens
- Handle special tokens as hard boundaries (split on them before pre-tokenizing)
- Break ties by lexicographic order (`max()` on tuples)
- Parallelize pre-tokenization with `multiprocessing` (starter code provided in [pretokenization_example.py](file:///Users/soumodeep/Resources/Courses/CS336/assignment1-basics/cs336_basics/pretokenization_example.py))
- Optimize merging with incremental pair count updates

**Adapter:** `run_train_bpe` in [adapters.py](file:///Users/soumodeep/Resources/Courses/CS336/assignment1-basics/tests/adapters.py)  
**Test:** `uv run pytest tests/test_train_bpe.py`

### Step 1.3 — Train BPE on Datasets (written, ~4 pts)
- `train_bpe_tinystories` (2 pts): Train on TinyStories, vocab_size=10,000. Report time, memory, longest token.
- `train_bpe_expts_owt` (2 pts): Train on OpenWebText, vocab_size=32,000. Compare tokenizers.

---

## Phase 2: BPE Tokenizer Encoding & Decoding (Section 2.6) — ~19 pts

### Step 2.1 — Tokenizer Class (15 pts) ⭐ BIG ONE
**What to build:** A `Tokenizer` class with these methods:

| Method | What it does |
|---|---|
| `__init__(vocab, merges, special_tokens)` | Load vocab + merges |
| `from_files(cls, ...)` | Class method to load from serialized files |
| `encode(text) → list[int]` | Text → token IDs (pre-tokenize, then apply merges in order) |
| `encode_iterable(iterable) → Iterator[int]` | Memory-efficient streaming encode |
| `decode(ids) → str` | Token IDs → text (concat bytes, decode UTF-8 with `errors='replace'`) |

Key details:
- Handle special tokens during encoding (split on them first)
- Apply merges **in order of creation** during encoding
- `encode_iterable` must be a generator for memory efficiency

**Adapter:** `get_tokenizer`  
**Test:** `uv run pytest tests/test_tokenizer.py`

### Step 2.2 — Tokenizer Experiments (written, 4 pts)
- `tokenizer_experiments` (4 pts): Compression ratios, cross-domain tokenization, throughput estimates, encode datasets to `uint16` numpy arrays for training.

---

## Phase 3: Transformer Model Components (Section 3) — ~24 pts

**Build these bottom-up, each one is a `torch.nn.Module`. Each has a test.**

> [!IMPORTANT]
> You may NOT use `torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.functional`, or `torch.optim` (except `Optimizer` base class, `nn.Parameter`, and container classes).

### Step 3.1 — Linear Module (1 pt)
- Implements `y = Wx` (no bias)
- Init: truncated normal with σ² = 2/(d_in + d_out)
- **Test:** `uv run pytest -k test_linear`

### Step 3.2 — Embedding Module (1 pt)
- Index into a `(vocab_size, d_model)` matrix
- Init: truncated normal with σ² = 1
- **Test:** `uv run pytest -k test_embedding`

### Step 3.3 — RMSNorm (1 pt)
- Root Mean Square Layer Normalization with learnable gain
- Upcast to float32, compute, downcast back
- **Test:** `uv run pytest -k test_rmsnorm`

### Step 3.4 — SwiGLU Feed-Forward Network (2 pts)
- `FFN(x) = W2(SiLU(W1·x) ⊙ W3·x)` — three weight matrices
- d_ff ≈ (8/3)·d_model, rounded to multiple of 64
- You can use `torch.sigmoid`
- **Test:** `uv run pytest -k test_swiglu`

### Step 3.5 — Softmax (1 pt)
- Numerically stable softmax (subtract max trick)
- **Test:** `uv run pytest -k test_softmax_matches_pytorch`

### Step 3.6 — RoPE — Rotary Position Embeddings (2 pts)
- Apply pairwise rotations to query/key vectors based on position
- Pre-compute sin/cos buffers in `__init__` with `register_buffer`
- Must handle arbitrary batch dimensions
- **Test:** `uv run pytest -k test_rope`

### Step 3.7 — Scaled Dot-Product Attention (5 pts)
- `Attention(Q, K, V) = softmax(QK^T / √d_k) V`
- Support optional boolean mask
- Handle arbitrary batch-like dimensions
- **Test:** `uv run pytest -k test_scaled_dot_product_attention` and `test_4d_scaled_dot_product_attention`

### Step 3.8 — Causal Multi-Head Self-Attention (5 pts)
- Project x → Q, K, V with learned weights (3 matmuls total)
- Split into heads, apply RoPE to Q and K, apply attention with causal mask, concat, project output
- **Test:** `uv run pytest -k test_multihead_self_attention`

### Step 3.9 — Transformer Block (3 pts)
- Pre-norm: `y = x + MHSA(RMSNorm(x))`, then `y = z + FFN(RMSNorm(z))`
- **Test:** `uv run pytest -k test_transformer_block`

### Step 3.10 — Full Transformer LM (3 pts)
- Token embedding → num_layers × Transformer blocks → final RMSNorm → linear head → logits
- **Test:** `uv run pytest -k test_transformer_lm`

### Step 3.11 — Resource Accounting (written, 5 pts)
- `transformer_accounting`: Count parameters, FLOPs for forward pass across GPT-2 sizes.

---

## Phase 4: Training Infrastructure (Section 4) — ~9 pts

### Step 4.1 — Cross-Entropy Loss (1 pt)
- Numerically stable: subtract max, cancel log/exp
- **Test:** `uv run pytest -k test_cross_entropy`

### Step 4.2 — AdamW Optimizer (2 pts)
- Subclass `torch.optim.Optimizer`
- Implement Algorithm 1 from the handout (weight decay, bias correction, moment estimates)
- **Test:** `uv run pytest -k test_adamw`

### Step 4.3 — AdamW Accounting (written, 2 pts)
- Memory analysis, MFU calculation, training time estimation

### Step 4.4 — LR Tuning Question (written, 1 pt)
- Try different LR values with the toy SGD example

### Step 4.5 — Cosine LR Schedule with Warmup (1 pt)
- Warmup → cosine decay → constant minimum
- **Test:** `uv run pytest -k test_get_lr_cosine_schedule`

### Step 4.6 — Gradient Clipping (1 pt)
- Clip global gradient norm to max value
- **Test:** `uv run pytest -k test_gradient_clipping`

---

## Phase 5: Training Loop (Section 5) — ~7 pts

### Step 5.1 — Data Loader (2 pts)
- Sample random (input, target) pairs from a flat token array
- Shape: `(batch_size, context_length)`
- Support `np.memmap` for large datasets
- **Test:** `uv run pytest -k test_get_batch`

### Step 5.2 — Checkpointing (1 pt)
- Save/load model state, optimizer state, and iteration number
- **Test:** `uv run pytest -k test_checkpointing`

### Step 5.3 — Training Script (4 pts)
- Wire everything together: data loading, forward pass, loss, backward, optimizer step, LR schedule, gradient clipping, logging, checkpointing
- Make hyperparameters configurable (command-line args or config file)

---

## Phase 6: Text Generation (Section 6) — ~4 pts

### Step 6.1 — Decoding Function (3 pts)
- Autoregressive generation: feed prompt, sample next token, repeat
- Support temperature scaling and top-p (nucleus) sampling
- Stop at `<|endoftext|>` or max tokens

### Step 6.2 — Generate Text (1 pt)
- Generate ≥256 tokens from your trained model, report and comment on fluency

---

## Phase 7: Experiments & Leaderboard (Section 7) — ~20 pts

### Step 7.1 — Experiment Logging (3 pts)
- Build infrastructure to track loss curves vs. steps and wall-clock time
- Keep an experiment log document

### Step 7.2 — Learning Rate Sweep (3 pts)
- Sweep LRs, find the best, investigate "edge of stability"
- Target: validation loss ≤ 1.45 on TinyStories

### Step 7.3 — Batch Size Experiments (1 pt)
- Vary batch size from 1 to GPU limit, discuss findings

### Step 7.4 — Ablations (4 pts total)
| Ablation | Points | What to do |
|---|---|---|
| `layer_norm_ablation` | 1 | Remove all RMSNorms, train, observe |
| `pre_norm_ablation` | 1 | Switch to post-norm, train, compare |
| `no_pos_emb` | 1 | Remove RoPE entirely, train, compare |
| `swiglu_ablation` | 1 | Replace SwiGLU with plain SiLU FFN (matched params), compare |

### Step 7.5 — OpenWebText Experiment (2 pts)
- Train on OWT with same architecture, compare to TinyStories
- Generate text, discuss quality difference

### Step 7.6 — Leaderboard (6 pts)
- Free-form: minimize OWT validation loss within 45 min on B200
- Must beat 5.0 loss baseline

---

## 🗺️ Suggested Execution Order

```
Week 1:  Phase 1 (BPE training) → Phase 2 (Tokenizer encode/decode)
         Serialize trained tokenizers & tokenized datasets to disk.

Week 2:  Phase 3 (Model components, Steps 3.1 → 3.10)
         Build bottom-up, test each piece as you go.

Week 3:  Phase 4 (Loss, optimizer, LR schedule, grad clipping)
       → Phase 5 (Data loader, checkpointing, training loop)
       → Phase 6 (Text generation)
         First successful training run on TinyStories!

Week 4:  Phase 3.11 + Phase 4.3 (Written accounting questions)
       → Phase 7 (Experiments, ablations, OWT, leaderboard)
         Written questions for writeup.pdf.
```

---

## 📋 Quick Test Checklist

Run these in order as you complete each component:

```
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_tokenizer.py
uv run pytest -k test_linear
uv run pytest -k test_embedding
uv run pytest -k test_rmsnorm
uv run pytest -k test_swiglu
uv run pytest -k test_softmax_matches_pytorch
uv run pytest -k test_rope
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
uv run pytest -k test_multihead_self_attention
uv run pytest -k test_transformer_block
uv run pytest -k test_transformer_lm
uv run pytest -k test_cross_entropy
uv run pytest -k test_adamw
uv run pytest -k test_get_lr_cosine_schedule
uv run pytest -k test_gradient_clipping
uv run pytest -k test_get_batch
uv run pytest -k test_checkpointing
```

> [!TIP]
> **Debug strategy:** Start each module by overfitting a tiny example. Use your IDE debugger to inspect tensor shapes. Check activation/gradient norms for sanity.

> [!TIP]
> **Low-resource:** If you don't have a GPU, the whole assignment is doable on Apple Silicon MPS or even CPU — just use fewer total tokens (40M instead of 328M) and relax the target loss to 2.0.
