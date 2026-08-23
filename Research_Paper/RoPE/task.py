"""
A position-critical toy task + a training loop, used to compare position encodings.

THE TASK - "offset copy":

    [PAD] * offset  s_1 s_2 ... s_L  <SEP>  s_1 s_2 ... s_L
    |------ shift -------||-- prompt --|   |----- target -----|

The model must emit s_i at the position L + 1 + i after the separator. To do that it
has to attend from a target slot to the prompt slot exactly (L + 1) tokens to its left.
So the task is solvable *only* from relative position - which makes it the right probe
for the claim RoPE makes.

`offset` shifts the whole pattern to different absolute positions while leaving every
relative distance untouched. Train with small offsets, test with large ones, and an
encoding that truly encodes relative position should not care.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

N_SYMBOLS = 20
PAD, SEP = N_SYMBOLS, N_SYMBOLS + 1
VOCAB_SIZE = N_SYMBOLS + 2
IGNORE = -100


def make_batch(batch_size: int, length: int = 16, offset: int = 0,
               generator: torch.Generator | None = None, device=None):
    """Returns (inputs, targets). Targets are IGNORE everywhere except the copy span."""
    total = offset + 2 * length + 1
    syms = torch.randint(0, N_SYMBOLS, (batch_size, length),
                         generator=generator, device=device)

    x = torch.full((batch_size, total), PAD, dtype=torch.long, device=device)
    x[:, offset : offset + length] = syms
    x[:, offset + length] = SEP
    x[:, offset + length + 1 :] = syms

    # next-token prediction: y[t] is what the model must output at position t
    y = torch.full((batch_size, total), IGNORE, dtype=torch.long, device=device)
    y[:, offset + length : offset + 2 * length] = syms
    return x, y


def evaluate(model, length: int = 16, offset: int = 0, batch_size: int = 256,
             seed: int = 0, device=None):
    """Per-token accuracy and loss on the copy span."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x, y = make_batch(batch_size, length, offset, generator=g)
    x, y = x.to(device), y.to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE)
        mask = y != IGNORE
        acc = (logits.argmax(-1)[mask] == y[mask]).float().mean()
    model.train(was_training)
    return acc.item(), loss.item()


def train(model, steps: int = 1500, batch_size: int = 64, length: int = 16,
          max_train_offset: int = 4, lr: float = 3e-4, seed: int = 0,
          device=None, log_every: int = 250, quiet: bool = False):
    """Plain AdamW loop. Returns the history of (step, train_loss, val_acc)."""
    device = device or next(model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 3, total_steps=steps)

    history, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        offset = int(torch.randint(0, max_train_offset + 1, (1,), generator=g))
        x, y = make_batch(batch_size, length, offset, generator=g)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == steps:
            acc, val = evaluate(model, length, offset=0, seed=1234, device=device)
            history.append((step, loss.item(), acc))
            if not quiet:
                print(f"    step {step:>5}  train loss {loss.item():.4f}"
                      f"   val acc {acc:6.2%}   [{time.time() - t0:5.1f}s]")
    return history
