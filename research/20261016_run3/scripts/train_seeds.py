"""Train new 2nd-argmax RNNs (16 neurons, 10 inputs) with ARC's AlgZoo recipe, several seeds at once.

Recipe copied from alg_zoo/training.py (sources/alg-zoo-repo): one-layer ReLU RNN without biases, PyTorch default init
(uniform +-1/sqrt(16) for both RNN matrices, +-1/sqrt(16) for the readout), softmax cross-entropy on the position of the
second-largest of 10 standard Gaussians, AdamW (lr 0.04 / sqrt(16), weight decay 0.001), warmup over 2^14 steps then
exponential decay to 0.01 of the base rate by half-way, batch 1024, 2^30 sequences in total. The S models share nothing
except the random generator stream: each gets its own initial weights and its own input batches. Elementwise AdamW keeps
their updates independent. Usage: train_seeds.py <first_seed> <n_models> [n_steps]
"""
import math
import sys
import time

import torch as th

first, S = int(sys.argv[1]), int(sys.argv[2])
N_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2 ** 30 // 1024
d, n, B = 16, 10, 1024
dev = "cuda"
th.manual_seed(first)
k = 1 / math.sqrt(d)
W_ih = ((th.rand(S, d, device=dev) * 2 - 1) * k).requires_grad_()
W_hh = ((th.rand(S, d, d, device=dev) * 2 - 1) * k).requires_grad_()
W_oh = ((th.rand(S, n, d, device=dev) * 2 - 1) * k).requires_grad_()
opt = th.optim.AdamW([W_ih, W_hh, W_oh], lr=0.04 / math.sqrt(d), weight_decay=0.001)
TOTAL = 2 ** 30 // B


def lr_mult(step):
    return min(step / 2 ** 14, 1) * max(0.01 ** (2 * step / TOTAL), 0.01)


sched = th.optim.lr_scheduler.LambdaLR(opt, lr_mult)


def step_fn():
    x = th.randn(S, B, n, device=dev)
    tgt = x.argsort(-1)[..., -2]
    h = th.zeros(S, B, d, device=dev)
    for t in range(n):
        h = th.relu(th.bmm(h, W_hh.transpose(1, 2)) + x[..., t, None] * W_ih[:, None, :])
    logits = th.bmm(h, W_oh.transpose(1, 2))
    loss = th.nn.functional.cross_entropy(logits.reshape(-1, n), tgt.reshape(-1), reduction="none").view(S, B).mean(1)
    loss.sum().backward()
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    return loss.detach(), (logits.argmax(-1) == tgt).float().mean(1)


t0 = time.time()
acc_run = th.zeros(S, device=dev)
for step in range(N_STEPS):
    loss, acc = step_fn()
    acc_run = 0.999 * acc_run + 0.001 * acc
    if (step + 1) % 2 ** 16 == 0 or step + 1 == N_STEPS:
        print(f"step {step + 1}: {time.time() - t0:.0f} s; running accuracy " + " ".join(f"{a:.3f}" for a in acc_run.tolist()), flush=True)
if N_STEPS == TOTAL:
    for i in range(S):
        sd = {"rnn.weight_ih_l0": W_ih[i].detach().cpu()[:, None].clone(), "rnn.weight_hh_l0": W_hh[i].detach().cpu().clone(),
              "linear.weight": W_oh[i].detach().cpu().clone()}
        th.save(sd, f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{first + i}.pth")
    print("saved")
