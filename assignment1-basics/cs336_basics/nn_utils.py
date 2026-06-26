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
import torch
import torch.nn as nn
import math

def cross_entropy(inputs, targets) -> torch.Tensor:

    #preparing to prevent overflow
    max_inp = torch.max(inputs , dim = -1, keepdim=True).values
    stable_inp = inputs - max_inp
    # calculating logits
    sum_exp = torch.sum(torch.exp(stable_inp), dim = -1 , keepdim = True)
    log_softmax = stable_inp - torch.log(sum_exp)
    target_log_probs = torch.gather(log_softmax, dim = -1 , index = targets.unsqueeze(-1))
    return -target_log_probs.mean()

def gradient_clipping(parameters, max_l2_norm) -> None:
    #extracting all the gradients
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
         return
    
    l2_norm = (sum(torch.sum(g.detach() ** 2).item() for g in grads)) ** 0.5
    if l2_norm > max_l2_norm:
        clip_coef = max_l2_norm / (l2_norm + 1e-6 )
        for g in grads:
            g.detach().mul_(clip_coef) #in-place operation :)
    

def lr_cosine_schedule(it, max_learning_rate, min_learning_rate,warmup_iters, cosine_cycle_iters) -> float:
    alpha_max = max_learning_rate
    alpha_min = min_learning_rate
    T_w = warmup_iters
    T_c = cosine_cycle_iters

    # warmup phase
    if it < T_w:
      alpha_t = (it / T_w) * alpha_max

    # cosine annealing phase

    elif T_w <= it <= T_c:
        #division by 0 case 
        if T_c == T_w:
            alpha_t = alpha_min
        else:
            cos_arg = ((it - T_w) / (T_c - T_w)) * math.pi
            alpha_t = alpha_min + 0.5*(1 + math.cos(cos_arg)) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
   
    return alpha_t
            

    


