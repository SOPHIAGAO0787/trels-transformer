
import argparse
import csv
import os
import time

import numpy as np
import torch

from .model import TransformerLM
from .optimizer import AdamW
from .data_loader import data_loading
from .checkpoint import save_checkpoint, load_checkpoint
from .train import cross_entropy_loss, gradient_clipping, get_lr_cosine_schedule



class Logger:
    """Writes metrics to stdout, a local CSV, and (optionally) Weights & Biases."""

    def __init__(self, log_path: str, use_wandb: bool = False,
                 wandb_project: str = '', wandb_run: str = ''):
        self.log_path = log_path
        self._csv_file = open(log_path, 'w', newline='')
        fieldnames = ['step', 'train_loss', 'val_loss', 'lr', 'tokens_seen', 'wallclock_s']
        self._writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames, extrasaction='ignore')
        self._writer.writeheader()

        self.wandb = None
        if use_wandb:
            try:
                import wandb  # type: ignore
                wandb.init(project=wandb_project, name=wandb_run or None)
                self.wandb = wandb
            except ImportError:
                print("wandb not installed — skipping W&B logging.")

    def log(self, metrics: dict):
        self._writer.writerow(metrics)
        self._csv_file.flush()

        # W&B
        if self.wandb is not None:
            self.wandb.log(metrics)

    def close(self):
        self._csv_file.close()
        if self.wandb is not None:
            self.wandb.finish()


def evaluate_val_loss(model, val_data, batch_size, context_length, device,
                      num_batches: int = 20) -> float:
    """Average cross-entropy over *num_batches* random validation batches."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = data_loading(val_data, batch_size, context_length, device)
            logits = model(x)
            total += cross_entropy_loss(logits, y).item()
    model.train()
    return total / num_batches



def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer LM (CS336 A1)")

    # Data
    p.add_argument('--train-path', type=str, required=True)
    p.add_argument('--val-path',   type=str, default=None)

    # Model architecture
    p.add_argument('--vocab-size',     type=int,   default=10_001)
    p.add_argument('--context-length', type=int,   default=256)
    p.add_argument('--d-model',        type=int,   default=512)
    p.add_argument('--num-layers',     type=int,   default=4)
    p.add_argument('--num-heads',      type=int,   default=16)
    p.add_argument('--d-ff',           type=int,   default=1344)
    p.add_argument('--rope-theta',     type=float, default=10_000.0)

    # Optimiser
    p.add_argument('--lr-max',       type=float, default=3e-4)
    p.add_argument('--lr-min',       type=float, default=3e-5)
    p.add_argument('--warmup-iters', type=int,   default=200)
    p.add_argument('--beta1',        type=float, default=0.9)
    p.add_argument('--beta2',        type=float, default=0.95)
    p.add_argument('--eps',          type=float, default=1e-8)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--grad-clip',    type=float, default=1.0)

    # Training schedule
    p.add_argument('--batch-size',    type=int, default=32)
    p.add_argument('--num-iters',     type=int, default=5_000)
    p.add_argument('--log-interval',  type=int, default=100)
    p.add_argument('--val-interval',  type=int, default=500)
    p.add_argument('--save-interval', type=int, default=1_000)

    # Checkpointing
    p.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    p.add_argument('--resume',         type=str, default=None,
                   help='Checkpoint path to resume from')

    # Logging (Section 7.1)
    p.add_argument('--log-file',       type=str, default='train_log.csv')
    p.add_argument('--wandb-project',  type=str, default='')
    p.add_argument('--wandb-run',      type=str, default='')

    # Device
    p.add_argument('--device', type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    train_data = np.load(args.train_path, mmap_mode='r')
    val_data   = np.load(args.val_path,   mmap_mode='r') if args.val_path else None
    print(f"Train tokens: {len(train_data):,}")
    if val_data is not None:
        print(f"Val   tokens: {len(val_data):,}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
 
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    use_wandb = bool(args.wandb_project)
    logger = Logger(
        log_path=args.log_file,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
    )

    model.train()
    t_start = time.time()

    for it in range(start_iter, args.num_iters):

        # 1. Learning-rate schedule (cosine with warmup, §4.4)
        #    Schedule decays to lr_min exactly at num_iters.
        lr = get_lr_cosine_schedule(
            it=it,
            max_lr=args.lr_max,
            min_lr=args.lr_min,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.num_iters,
        )
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 2. Sample batch (§5.1)
        x, y = data_loading(train_data, args.batch_size, args.context_length, device)

        # 3. Forward + loss (§4.1)
        optimizer.zero_grad()
        logits = model(x)                        # (B, T, V)
        loss = cross_entropy_loss(logits, y)

        # 4. Backward
        loss.backward()

        # 5. Gradient clipping (§4.5)
        gradient_clipping(model.parameters(), args.grad_clip)

        # 6. Parameter update
        optimizer.step()

        # ── Logging (Section 7.1) 
        step = it + 1
        elapsed = time.time() - t_start
        tokens_seen = step * args.batch_size * args.context_length

        if step % args.log_interval == 0:
            metrics = {
                'step': step,
                'train_loss': round(loss.item(), 6),
                'lr': lr,
                'tokens_seen': tokens_seen,
                'wallclock_s': round(elapsed, 2),
            }
            # Validation loss
            if val_data is not None and step % args.val_interval == 0:
                val_loss = evaluate_val_loss(
                    model, val_data, args.batch_size, args.context_length, device
                )
                metrics['val_loss'] = round(val_loss, 6)

            logger.log(metrics)
            val_str = f"  val {metrics.get('val_loss', '–'):.4f}" if 'val_loss' in metrics else ''
            print(
                f"step {step:6d} | train {loss.item():.4f}{val_str} | "
                f"lr {lr:.2e} | tokens {tokens_seen:,} | {elapsed:.1f}s"
            )

        elif val_data is not None and step % args.val_interval == 0:
            # Validate even if not a log step so we don't miss checkpoints
            val_loss = evaluate_val_loss(
                model, val_data, args.batch_size, args.context_length, device
            )
            logger.log({'step': step, 'val_loss': round(val_loss, 6),
                        'wallclock_s': round(elapsed, 2)})
            print(f"step {step:6d} | val_loss {val_loss:.4f}")

        if step % args.save_interval == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"ckpt_{step:07d}.pt")
            save_checkpoint(model, optimizer, step, ckpt)
            print(f"  >> checkpoint: {ckpt}")

    # Final checkpoint
    final = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.num_iters, final)
    logger.close()
    print(f"Done. Final checkpoint: {final}")


if __name__ == '__main__':
    main()
