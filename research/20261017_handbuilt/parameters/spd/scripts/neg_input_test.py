"""Direct check in ARC (no SPD): how much does the output change if negative numbers use different input weights?
Variants: SPD's negative-number input vector; ARC's input weights times a factor for negative numbers only."""
import sys
from pathlib import Path
import numpy as np
import torch as th
sys.path.insert(0, str(Path(__file__).parent))
from spd_rnn import N_STEPS, arc_params, kl, run_target  # noqa: E402
from algzoo.core import target  # noqa: E402

W = arc_params("cuda")
w = W["hi"][:, 0]
u_neg = th.tensor(np.loadtxt(Path(__file__).parents[1] / "results/main_b1e-3_input_split.log", skiprows=2)[:, 2],
                  dtype=w.dtype, device="cuda")

def run(w_pos, w_neg, x):
    h = x.new_zeros(x.shape[0], 16)
    for t in range(N_STEPS):
        xt = x[:, t, None]
        h = th.relu(h @ W["hh"].T + th.where(xt > 0, xt * w_pos, xt * w_neg))
    return h @ W["oh"].T

g = th.Generator(device="cuda").manual_seed(5)
x = th.randn(2**20, N_STEPS, generator=g, device="cuda")
tgt, _ = run_target(W, x)
y = target(x)
print(f"ARC accuracy {(tgt.argmax(-1) == y).float().mean().item():.4f}")
hiding = th.zeros(16, device="cuda"); hiding[[1, 6, 7, 8]] = 1
for name, wn in [("SPD's negative-number vector", u_neg), ("0.33 x ARC's", 0.33 * w), ("0.67 x ARC's", 0.67 * w),
                 ("zero", 0 * w), ("hiding entries x 0.33, rest ARC's", w * (1 - 0.67 * hiding))]:
    out = run(w, wn, x)
    print(f"negative numbers use {name:36s}: KL {kl(tgt, out).item():.4f}, accuracy {(out.argmax(-1) == y).float().mean().item():.4f}")
