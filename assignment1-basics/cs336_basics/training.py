"""
CS336 Assignment 1 — Data Loading & Checkpointing

1. get_batch(dataset, batch_size, context_length, device)
   - dataset: 1D numpy array of integer token IDs
   - Randomly sample batch_size starting indices from [0, len(dataset) - context_length)
   - For each start index i:
     x[i] = dataset[start : start + context_length]       (input)
     y[i] = dataset[start + 1 : start + context_length + 1] (label, shifted by 1)
   - Return (x, y) as torch.LongTensors on the specified device
   - Test: pytest -k test_get_batch

2. save_checkpoint(model, optimizer, iteration, out)
   - Serialize model.state_dict(), optimizer.state_dict(), and iteration to a file
   - Use torch.save with a dictionary containing all three
   - out can be a file path string or a file-like object
   - Test: pytest -k test_checkpoint

3. load_checkpoint(src, model, optimizer)
   - Load checkpoint from src (file path or file-like object) using torch.load
   - Restore model state with model.load_state_dict()
   - Restore optimizer state with optimizer.load_state_dict()
   - Return the saved iteration number
   - Test: pytest -k test_checkpoint
"""
import torch
import torch.nn as nn
import numpy as np

def get_batch(dataset, batch_size, context_length, device):
    rng = np.random.default_rng()
    upper_bound = len(dataset) - context_length
    start_indices = rng.integers(0 , upper_bound , batch_size)

    # shape: (batch_size, 1) + (context_length,) -> (batch_size, context_length)
    grid_indices = start_indices[:, None] + np.arange(context_length) #broadcasting 
    x_np = dataset[grid_indices]
    y_np = dataset[grid_indices + 1]
    x = torch.from_numpy(x_np).to(device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device, dtype=torch.long)
    return (x , y)



def save_checkpoint(model,optimizer,iteration,out):
   checkpoint = {}
   checkpoint["model"] = model.state_dict()
   checkpoint["optimizer"] = optimizer.state_dict()
   checkpoint["iteration"] = iteration 

   return torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
   checkpoint = torch.load(src , map_location='cpu')
   model.load_state_dict(checkpoint["model"])
   optimizer.load_state_dict(checkpoint["optimizer"])

   return checkpoint["iteration"] 






    

