import torch
import torch.nn as nn
import timeit
import argparse
import numpy as np
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


MODEL_CONFIGS = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
    '10B': {'d_model': 4608, 'd_ff': 12288, 'num_layers': 50, 'num_heads': 36},
}



def benchmark(args):

    #looking for the size information dictionary 
    config = MODEL_CONFIGS[args.model_size]

    #architecture parameters
    d_model = config['d_model']
    d_ff = config['d_ff']
    num_layers = config['num_layers']
    num_heads = config['num_heads']

    # defining the model
    model = TransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = d_model,
        num_layers = num_layers,
        num_heads = num_heads,
        d_ff = d_ff,
        rope_theta = 10000.0
    ).to(args.device)

    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device = args.device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device = args.device)

    if args.mode == 'full':
        optimizer = AdamW(model.parameters(), lr = 1e-4)
    else:
        optimizer = None

    def run_step():
        if args.mode == "forward":
            with torch.no_grad():
                _ = model(x)
                
        elif args.mode == "forward_backward":
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            model.zero_grad()
            loss.backward()
            
        elif args.mode == "full":
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.device == 'cuda':
            torch.cuda.synchronize()

    for _ in range(args.warmup_iters):
        run_step()

    times = []
    for _ in range(args.measurement_steps):
        start_time = timeit.default_timer()
        run_step()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Mean step time: {mean_time:.6f} seconds")
    print(f"Std dev: {std_time:.6f} seconds")
        
            
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking the Transformer")
    parser.add_argument('--model_size', type=str, choices=list(MODEL_CONFIGS.keys()), default='small')
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--warmup_iters', type=int, default=5)
    parser.add_argument('--measurement_steps', type=int, default=10)
    parser.add_argument('--mode', type=str, choices=['forward', 'forward_backward', 'full'], default='forward')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    benchmark(args)
