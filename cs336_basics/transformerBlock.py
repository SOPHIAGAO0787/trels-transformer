import torch
import torch.nn as nn
from .linear import Linear, Embedding, RMSNorm, SwiGlu
from .attention import MultiHeadSelfAttention
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=10000, max_seq_len=2048):
        super().__init__()
        
        #first sublayer: multi-head sel-attention
        self.norm1 = RMSNorm(d_model, device=None, dtype=None)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device=None, dtype=None)
        
        #second sublayer: feed-forward
        self.norm2 = RMSNorm(d_model, device=None, dtype=None)
        self.ffn = SwiGlu(d_model, d_ff)
        
    def forward(self, x, token_positions) -> torch.Tensor:
        x = x+self.attn(self.norm1(x), token_positions)
        x = x+self.ffn(self.norm2(x))
        
        return x
    
    
    