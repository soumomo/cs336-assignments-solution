"""
CS336 Assignment 1 — AdamW Optimizer

1. AdamW (subclass of torch.optim.Optimizer)
   - Implements Adam with DECOUPLED weight decay (not L2 regularization)
   - Constructor: __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
   - Must implement step() method:
     a. For each parameter group and each parameter with a gradient:
     b. Update biased first moment estimate:  m = beta1 * m + (1 - beta1) * grad
     c. Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
     d. Bias-correct: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
     e. Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + eps)
     f. Apply weight decay SEPARATELY: param = param - lr * weight_decay * param
   - Store m and v in self.state for each parameter
   - Track step count (t) per parameter
   - Do NOT use torch.optim.AdamW or any built-in optimizer implementation
   - Test: pytest -k test_adamw
"""
