import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = x.max(dim=dim, keepdim=True)[0]
    x_exp=torch.exp(x - max_val) 
    partition = x_exp.sum(dim,keepdim=True)
    return x_exp / partition