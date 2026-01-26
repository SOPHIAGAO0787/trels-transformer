import torch 
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        #y=Wx
        self.W = nn.Parameter(torch.empty(out_features, in_features), device=device, dtype=dtype)
        #sigma
        n = in_features
        #trunc
        bound = 1.0 / math.sqrt(n)
        with torch.no_grad():
            self.W.uniform_ (-bound, bound)
            
        #previous initialization using Xavier for sigmoid
        #nn.init.trunc_normal_(self.W,
                              #mean=0.0,
                              #std=sigma,
                              #a=-3*sigma,
                              #b=3*sigma)
                 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...j, ij -> ...i', x, self.W)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        ))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)
        
    def forward(self, tokens_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[tokens_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        #float32 refrain from overload
        x = x.to(torch.float32)
        
        #RMS(a) = sqrt((1/d) * sum(xi^2) + eps)
        ms = x.pow(2).mean(dim=-1,keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        
        x_normed = x / rms
        result = x_normed * self.weight
        
        return result.to(in_dtype)
        
class SwiGlu(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 8 / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (SiLU(xW1) * xW3) W2
        # SiLU(x) = x * sigmoid(x)
        gate = self.w1(x)
        swish = gate * torch.sigmoid(gate)
        
        # element-wise multiplication w2
        return self.w2(swish * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__
        self.d_k = d_k
        inv_freq = 1 / theta**(torch.arange(0, d_k, 2).float()/d_k)
        t = torch.arange(max_seq_len).float()
        
        freqs = torch.outer(t,inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        #x as left and right
        x1 = x[..., : self.d_k // 2] #first half dim
        x2 = x[..., self.d_k // 2 :] #second half dim
        
        x_rotated = torch.cat((-x2, x1), dim=-1)
        
        return (x*cos) +(x_rotated*sin)