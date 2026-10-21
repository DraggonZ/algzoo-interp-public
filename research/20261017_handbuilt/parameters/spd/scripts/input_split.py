"""Input-weight pieces of a decomposition: which part of ARC's input weights is used for negative numbers, and which
only for positive ones. Usage: python input_split.py NAME"""
import sys
from pathlib import Path
import numpy as np
import torch as th
sys.path.insert(0, str(Path(__file__).parent))
from analyze import load  # noqa: E402

m, W, d = load(sys.argv[1])
U = m.U["hi"].detach().cpu().double().numpy()
V = m.V["hi"].detach().cpu().double().numpy()[:, 0]
xs = th.linspace(-3, 3, 13, device="cuda")
with th.no_grad():
    ci = m.gate("hi", xs[:, None] * m.V["hi"][:, 0]).clamp(0, 1).cpu().numpy()  # (13, C)
neg = ci[xs.cpu().numpy() < -0.4].mean(0) > 0.5
pos = ci[xs.cpu().numpy() > 0.4].mean(0) > 0.5
w = W["hi"][:, 0].cpu().double().numpy()
u_neg = (U[:, neg] * V[neg]).sum(1)
u_pos = (U[:, pos] * V[pos]).sum(1)
print(f"pieces on for negative numbers: {np.nonzero(neg)[0].tolist()}; on for positive numbers: {np.nonzero(pos)[0].tolist()}")
print("neuron  ARC input weight  used for x<0  used for x>0")
for i in range(16):
    print(f"{i:6d}  {w[i]:+15.3f}  {u_neg[i]:+12.3f}  {u_pos[i]:+12.3f}")
