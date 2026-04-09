import torch
import torch.nn as nn
from .linear import Linear, Embedding, RMSNorm, SwiGlu
from .attention import MultiHeadSelfAttention

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(*args, **kwargs)