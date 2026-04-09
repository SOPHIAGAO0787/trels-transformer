import torch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable cross-entropy loss.

    Args:
        logits: (..., vocab_size)  — unnormalized scores
        targets: (...)             — integer class indices

    Returns:
        Scalar mean loss.
    """
    # Subtract max for numerical stability (cancels in log-softmax)
    M = logits.max(dim=-1, keepdim=True).values
    logits = logits - M

    # Gather the logit at the correct class
    # targets is a tensor, so we need len(targets) not range(targets)
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    correct_logits = flat_logits[torch.arange(len(flat_targets), device=logits.device), flat_targets]

    logsumexp = torch.log(torch.exp(flat_logits).sum(dim=-1))
    loss = -correct_logits + logsumexp
    return loss.mean()


def gradient_clipping(parameters, max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip combined gradient l2-norm of all parameters to max_l2_norm in-place."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.norm(p=2) ** 2 for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)


def get_lr_cosine_schedule(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing with linear warmup (§4.4)."""
    import math
    if it < warmup_iters:
        return (it / warmup_iters) * max_lr
    if it > cosine_cycle_iters:
        return min_lr
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)
