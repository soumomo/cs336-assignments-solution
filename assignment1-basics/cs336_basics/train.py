"""
CS336 Assignment 1 — Training Script (Phase 5.3 + Phase 6 + Phase 7)

Wire everything together into a complete training pipeline.

1. Training Loop (4 pts)
   - Load tokenized dataset from disk (as numpy memmap or array)
   - Initialize TransformerLM with chosen hyperparameters
   - Initialize AdamW optimizer
   - For each training step:
     a. Sample a batch using get_batch(dataset, batch_size, context_length, device)
     b. Forward pass: logits = model(x)
     c. Compute cross-entropy loss between logits and targets
     d. Backward pass: loss.backward()
     e. Gradient clipping
     f. Optimizer step + zero gradients
     g. Update learning rate via cosine schedule
     h. Log loss periodically (every N steps)
     i. Evaluate on validation set periodically
     j. Save checkpoints periodically
   - Make all hyperparameters configurable via command-line args (argparse)
     Suggested hyperparams: batch_size, context_length, d_model, num_heads,
     num_layers, d_ff, max_lr, min_lr, warmup_iters, max_iters,
     weight_decay, grad_clip_norm, eval_interval, checkpoint_interval

2. Text Generation / Decoding (3 pts)
   - Autoregressive generation function:
     a. Start with a prompt (token IDs)
     b. Feed through model to get logits for next token
     c. Apply temperature scaling: logits = logits / temperature
     d. Apply top-p (nucleus) sampling: keep only tokens whose cumulative
        probability mass is <= p, zero out the rest, renormalize
     e. Sample from the distribution, append token to sequence
     f. Repeat until max_tokens reached or <|endoftext|> generated
   - Generate at least 256 tokens from trained model (1 pt)

3. Experiments & Leaderboard (Phase 7)
   - Experiment logging infrastructure: track loss vs steps and wall-clock time (3 pts)
   - Learning rate sweep: find best LR, investigate edge of stability (3 pts)
     Target: validation loss <= 1.45 on TinyStories
   - Batch size experiments: vary from 1 to GPU memory limit (1 pt)
   - Ablations (4 pts total):
     a. Remove all RMSNorms, train, observe what happens
     b. Switch to post-norm (normalize after residual), compare
     c. Remove RoPE entirely, compare
     d. Replace SwiGLU with plain SiLU FFN (matched params), compare
   - OpenWebText experiment: train on OWT, compare to TinyStories (2 pts)
   - Leaderboard: minimize OWT validation loss within 45 min on B200 (6 pts)
     Must beat 5.0 loss baseline
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.training import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.nn_utils import cross_entropy, lr_cosine_schedule, gradient_clipping
import time


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a Transformer model with memory-mapped data."
    )

    # data paths and loading ;-;
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train.npy",
        help="Path to training .npy file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/validation.npy",
        help="Path to validation .npy file",
    )
    parser.add_argument(
        "--mmap",
        type=str,
        default="r",
        choices=["r", "r+", "w+", "c", "None"],
        help="Memory mapping mode",
    )

    # model architecture (^ ^')
    parser.add_argument(
        "--d_model", type=int, default=512, help="Embedding dimension size"
    )
    parser.add_argument(
        "--num_heads", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, help="Base value for RoPE calculation"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=1344,
        help="Dimension of feed-forward network",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=10000, help="Vocabulary size"
    )

    # training loop configs t_t
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size per training step"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_iters", type=int, default=10000, help="Total training iterations"
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold",
    )

    # lr and scheduler +_+
    parser.add_argument(
        "--max_lr", type=float, default=6e-4, help="Peak learning rate"
    )
    parser.add_argument(
        "--min_lr", type=float, default=6e-5, help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=2000,
        help="Number of iterations for LR warmup",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay factor"
    )

    # evaluation and checkpointing :/
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="How often to run evaluation",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=200,
        help="Number of batches to run during evaluation",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="How often to save model checkpoints",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="How often to print training logs",
    )

    return parser.parse_args()


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    args = get_args()

    mmap_mode = None if args.mmap == "None" else args.mmap

    train_data = np.load(args.train_path, mmap_mode=mmap_mode)
    val_data = np.load(args.val_path, mmap_mode=mmap_mode)

    print(f"data successfully mapped!!")
    print(f"Model configured with d_model={args.d_model}, heads={args.num_heads}")
    print(f"Training will run for {args.max_iters} iterations.")

    model = TransformerLM(args.vocab_size , args.context_length , args.d_model, args.num_layers , args.num_heads , args.d_ff , args.rope_theta)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr = args.max_lr,
        weight_decay=args.weight_decay
    )

    model.train()
    start_time = time.time()
    
    for step in range(args.max_iters):
        lr = lr_cosine_schedule(
            step, args.max_lr, args.min_lr,args.warmup_iters, args.max_iters - args.warmup_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(
            train_data, args.batch_size, args.context_length, device
        )

        optimizer.zero_grad()
        logits = model(x)
        #logits shape : [batch_size , context_length ,vocab_size]
        # y shape: [batch_size , context_length]
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()

        gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip_norm)
        optimizer.step()

        # logging
        if step % args.log_interval == 0 or step == args.max_iters - 1:
          elapsed = time.time() - start_time
          print(
              f"step {step:5d} | loss: {loss.item():6.4f} | lr: {lr:.2e} | time: {elapsed:.1f}s"
          )

        # evaluation
        if step % args.eval_interval == 0 and step > 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for _ in range(args.eval_iters):
                    x_val, y_val = get_batch(val_data, args.batch_size, args.context_length, device)
                    logits_val = model(x_val)
                    loss_val = cross_entropy(logits_val.reshape(-1, logits_val.shape[-1]), y_val.reshape(-1))
                    val_losses.append(loss_val.item())
            
            val_loss = np.mean(val_losses)
            print(f"\n > < Evaluation at step {step} | Val Loss: {val_loss:6.4f} > <")
            
            model.train()

          
          # checkpointing
        if step % args.checkpoint_interval == 0 and step > 0:
            checkpoint_path = f"checkpoint_step_{step}.pt"
            save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path} (^ ^)")


if __name__ == "__main__":
    main()



