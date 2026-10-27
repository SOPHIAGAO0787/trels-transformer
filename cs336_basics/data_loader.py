import numpy as np
import torch

def data_loading(x, batch_size, context_length, device):
    n = len(x)
    max_start = n-context_length-1
    
    start_indices = np.random.randint(0,max_start+1,size=batch_size)
    
    sample_batch = np.stack([x[i:i+context_length] for i in start_indices])
    target_batch = np.stack([x[i+1:i+context_length+1] for i in start_indices])
    
    sample_batch = torch.tensor(sample_batch, dtype=torch.long, device=device)
    target_batch = torch.tensor(target_batch, dtype=torch.long, device=device)
    
    return sample_batch, target_batch



