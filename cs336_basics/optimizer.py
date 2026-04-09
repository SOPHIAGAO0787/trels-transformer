import math
import torch


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer as described in Algorithm 1 of Loshchilov & Hutter (2019)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        beta1, beta2 = betas
        defaults = {
            'lr': lr,
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data
                state = self.state[p]

                # Initialise state on first step
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                t = state['t'] + 1          # Algorithm says t starts at 1
                m = state['m']
                v = state['v']

                # Update biased first and second moment estimates
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g ** 2

                # Bias-corrected learning rate
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # Parameter update
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

                # Weight decay uses the BASE learning rate α, not the bias-corrected α_t
                # (this is the key difference between AdamW and Adam+L2)
                p.data -= lr * weight_decay * p.data

                state['t'] = t
                state['m'] = m
                state['v'] = v

        return loss
