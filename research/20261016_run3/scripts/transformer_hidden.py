"""Goal 2 pilot (review 7 direction 3): does the hidden-signal test find anything in a transformer?

ARC's longest-cycle models: 2-layer attention-only transformers, 1 head, no biases, read a map x: {0..n-1} -> {0..n-1}
and output the longest cycle's length minus 1 at the last position. Forward re-implemented from ARC's architectures.py
(torch MultiheadAttention semantics, no mask). All n^n inputs are enumerated, so accuracies are exact.

Hidden part = directions the unembedding cannot tell apart (orthogonal complement of the row space of the unembedding
with its mean row removed). For the first layer's attention output a1 (at every position), replace the hidden component
by its mean over inputs (per position), or the visible component, or the component in a random subspace of the same
size as the hidden one (median of 20), and measure exact accuracy."""
import itertools
import sys

import numpy as np
import torch as th

from algzoo.core import load  # noqa: F401  (paths)
from algzoo.paths import MODELS


def attn(s, Win, Wo):
    d = s.shape[-1]
    q, k, v = s @ Win[:d].T, s @ Win[d:2 * d].T, s @ Win[2 * d:].T
    a = th.softmax(q @ k.transpose(-1, -2) / d ** 0.5, -1)
    return (a @ v) @ Wo.T


def task(x):
    n = x.shape[1]
    start = th.arange(n)[None].expand(*x.shape); pos = start; paths = []
    for _ in range(n):
        pos = x.gather(1, pos); paths.append(pos)
    return (th.stack(paths, -1) == start[..., None]).int().argmax(-1).amax(-1)


def run(sd, x, edit=None):
    E, P = sd["embed.weight"], sd["pos_embed.weight"]
    s0 = E[x] + P[None]
    a1 = attn(s0, sd["attns.0.in_proj_weight"], sd["attns.0.out_proj.weight"])
    if edit is not None:
        a1 = edit(a1)
    s1 = s0 + a1
    s2 = s1 + attn(s1, sd["attns.1.in_proj_weight"], sd["attns.1.out_proj.weight"])
    return (s2[:, -1] @ sd["unembed.weight"].T).argmax(-1), a1


def proj_mean(a1, B):
    """Replace the component of a1 in the span of orthonormal columns B by its mean over inputs (per position)."""
    c = a1 @ B
    return a1 - c @ B.T + c.mean(0, keepdim=True) @ B.T


rng = np.random.default_rng(0)
for d, n in [(6, 5), (8, 5), (6, 6), (8, 6)]:
    x = th.tensor(list(itertools.product(range(n), repeat=n)))
    y = task(x)
    for seed in range(5):
        sd = {k: v.double() for k, v in th.load(MODELS / "zoo" / f"longest_cycle_{d}_{n}_1073741824_{seed}.pth", weights_only=True, map_location="cpu").items()}
        pred, a1 = run(sd, x)
        acc = (pred == y).double().mean().item()
        U = sd["unembed.weight"].numpy(); Uc = U - U.mean(0)
        _, S, Vt = np.linalg.svd(Uc); r = int((S > 1e-8 * S[0]).sum())
        Bv = th.tensor(Vt[:r].T); Bh = th.tensor(Vt[r:].T)
        A = a1.reshape(-1, d)
        share = (((A - A.mean(0)) @ Bh).pow(2).sum() / (A - A.mean(0)).pow(2).sum()).item()
        acc_h = (run(sd, x, lambda a: proj_mean(a, Bh))[0] == y).double().mean().item()
        acc_v = (run(sd, x, lambda a: proj_mean(a, Bv))[0] == y).double().mean().item()
        rand = []
        for _ in range(20):
            Q, _ = np.linalg.qr(rng.standard_normal((d, d - r)))
            rand.append((run(sd, x, lambda a: proj_mean(a, th.tensor(Q)))[0] == y).double().mean().item())
        print(f"d={d} n={n} seed {seed}: accuracy {100 * acc:.1f}%; hidden directions {d - r} of {d}; share of layer-1 output variance in them {share:.2f}; "
              f"accuracy with hidden part averaged {100 * acc_h:.1f}%, visible part averaged {100 * acc_v:.1f}%, random part of the same size {100 * np.median(rand):.1f}%", flush=True)


def r2(F, Y):
    F = np.c_[F, np.ones(len(F))]
    c, *_ = np.linalg.lstsq(F, Y, rcond=None)
    return 1 - ((Y - F @ c) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


if __name__ == "__main__" and "--reveal" in sys.argv:
    print("Does layer 2 reveal what layer 1 hid? Share of the visible part of layer 2's output at the last position explained "
          "(linear fit, all n^n inputs) by the hidden part of layer 1's output at all positions, against its visible part:")
    for d, n in [(8, 5), (8, 6)]:
        x = th.tensor(list(itertools.product(range(n), repeat=n)))
        for seed in range(5):
            sd = {k: v.double() for k, v in th.load(MODELS / "zoo" / f"longest_cycle_{d}_{n}_1073741824_{seed}.pth", weights_only=True, map_location="cpu").items()}
            E, P = sd["embed.weight"], sd["pos_embed.weight"]
            s0 = E[x] + P[None]
            a1 = attn(s0, sd["attns.0.in_proj_weight"], sd["attns.0.out_proj.weight"])
            a2 = attn(s0 + a1, sd["attns.1.in_proj_weight"], sd["attns.1.out_proj.weight"])
            U = sd["unembed.weight"].numpy(); _, S, Vt = np.linalg.svd(U - U.mean(0)); r = int((S > 1e-8 * S[0]).sum())
            Bv, Bh = Vt[:r].T, Vt[r:].T
            Y = a2[:, -1].numpy() @ Bv
            Fh = (a1.numpy() @ Bh).reshape(len(x), -1); Fv = (a1.numpy() @ Bv).reshape(len(x), -1)
            print(f"  d={d} n={n} seed {seed}: from hidden part {r2(Fh, Y):.2f} ({Fh.shape[1]} features), from visible part {r2(Fv, Y):.2f} ({Fv.shape[1]} features)", flush=True)
