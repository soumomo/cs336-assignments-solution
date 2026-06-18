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
