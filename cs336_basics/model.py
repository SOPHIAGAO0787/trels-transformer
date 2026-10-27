import torch
import torch.nn as nn
from .linear import Linear, Embedding, RMSNorm
from .transformerBlock import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int,
                 rope_theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta=rope_theta, max_seq_len=context_length)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch_size, seq_len)
        seq_len = token_ids.shape[1]
        token_positions = torch.arange(seq_len, device=token_ids.device)

        x = self.token_embeddings(token_ids)       # (batch, seq_len, d_model)
        for block in self.layers:
            x = block(x, token_positions)          # (batch, seq_len, d_model)
        x = self.ln_final(x)                       # (batch, seq_len, d_model)
        return self.lm_head(x)                     # (batch, seq_len, vocab_size)
