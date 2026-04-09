import torch
import torch.nn as nn
from .linear import Linear, RotaryPositionalEmbedding, scaled_dot_product_attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta=10000, max_seq_len = 2048, use_rope=True, device = None,dtype = None):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads #dimension per head
        
        #Q, K, V projections for all heads
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)
        
        #Rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            token_positions (torch.Tensor): _description_

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        #Project to Q,K,V
        Q = self.W_q(x) #(batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        #reshape for multi-head: split d_model into (num_heads, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        #rearaange to (batch_size, num_heads, seq_len, d_k)       
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        #apply RoPE to Q and K
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        
        #casual masking: lower triangular boolean(seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        #(batch)size, num_heads, seq_len, d_k)
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        
        #merge heads back: (batch_size, seq_len, d_models)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(attn_output)
    