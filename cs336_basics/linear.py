import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # Store W of shape (out_features, in_features); forward computes xW^T via einsum
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # Spec §3.4.1: N(0, 2/(d_in+d_out)) truncated at [-3σ, 3σ]
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...j, ij -> ...i', x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        # Spec §3.4.1: N(0, 1) truncated at [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, tokens_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[tokens_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # Spec §3.4.1: RMSNorm gain initialized to 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # RMS(a) = sqrt(1/d * sum(ai^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)


class SwiGlu(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 8 / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        # Must use our custom Linear, not nn.Linear
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = W2(SiLU(W1 x) ⊙ W3 x)
        # SiLU(x) = x * sigmoid(x)
        gate = self.w1(x)
        swish = gate * torch.sigmoid(gate)
        return self.w2(swish * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        # θ_{i,k} = i / Θ^{(2k-2)/d}  for k=1..d/2
        # inv_freq[k] = 1 / Θ^{2k/d} for k=0,2,...,d-2
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)          # (max_seq_len, d_k/2)
        emb = torch.cat((freqs, freqs), dim=-1)   # (max_seq_len, d_k) — split-half convention
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        cos = self.cos_cached[token_positions]   # (..., seq_len, d_k)
        sin = self.sin_cached[token_positions]

        # Split-half rotation: x_rotated = (-x2, x1)
        x1 = x[..., : self.d_k // 2]
        x2 = x[..., self.d_k // 2 :]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + x_rotated * sin


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - max_val)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None
) -> torch.Tensor:
    dk = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)
