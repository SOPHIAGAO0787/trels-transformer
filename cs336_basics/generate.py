
from __future__ import annotations

import torch


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str | torch.device = 'cpu',
) -> list[int]:
    
    device = torch.device(device)
    model.eval()

    # Build initial context tensor — shape (1, seq_len)
    context = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits = model(context)      
        next_logits = logits[0, -1, :]  

        if temperature != 1.0:
            next_logits = next_logits / temperature

        probs = _softmax(next_logits)

        if top_p < 1.0:
            probs = _top_p_filter(probs, top_p)

        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break
        
        next_token_t = torch.tensor([[next_token]], dtype=torch.long, device=device)
        context = torch.cat([context, next_token_t], dim=1)

    return generated



def _softmax(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over a 1-D logit tensor."""
    logits = logits - logits.max()
    exp = torch.exp(logits)
    return exp / exp.sum()


def _top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus (top-p) filtering.

    Keep the smallest set V(p) of tokens whose cumulative probability ≥ p,
    zero out the rest, and renormalise.
    """
    # Sort in descending probability order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    # Remove tokens once cumulative mass exceeds p.
    # Shift by 1 so the token that *crosses* p is still included.
    remove = cumulative_probs - sorted_probs > p
    sorted_probs[remove] = 0.0

    # Put back in original order and renormalise
    filtered = torch.zeros_like(probs)
    filtered[sorted_indices] = sorted_probs
    total = filtered.sum()
    if total > 0:
        filtered = filtered / total
    return filtered


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device = 'cpu',
) -> str:
    # Resolve <|endoftext|> token id if the tokeniser knows it
    eos_id = tokenizer.special_tokens.get('<|endoftext|>', None)

    prompt_ids = tokenizer.encode(prompt)
    new_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_id,
        device=device,
    )
    return tokenizer.decode(new_ids)
