"""
CS336 Assignment 1 — Training Utilities

1. Cross-Entropy Loss
   - Function: cross_entropy_loss(inputs, targets) -> scalar tensor
   - inputs: (batch_size, vocab_size) unnormalized logits
   - targets: (batch_size,) integer class indices
   - Use log-sum-exp trick for numerical stability: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
   - Return the AVERAGE loss across the batch
   - Do NOT use torch.nn.functional or torch.nn.CrossEntropyLoss
   - Test: pytest -k test_cross_entropy

2. Gradient Clipping
   - Function: gradient_clipping(parameters, max_l2_norm) -> None
   - Compute the GLOBAL L2 norm across all parameter .grad tensors
   - If the global norm exceeds max_l2_norm, scale ALL gradients down by (max_l2_norm / global_norm)
   - Modify gradients IN-PLACE
   - Test: pytest -k test_gradient_clipping

3. Cosine Learning Rate Schedule (with linear warmup)
   - Function: lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters) -> float
   - Phase 1 (it < warmup_iters): Linear warmup from 0 to max_lr
   - Phase 2 (warmup_iters <= it < cosine_cycle_iters): Cosine decay from max_lr to min_lr
   - Phase 3 (it >= cosine_cycle_iters): Constant min_lr
   - Test: pytest -k test_lr_cosine_schedule
"""
