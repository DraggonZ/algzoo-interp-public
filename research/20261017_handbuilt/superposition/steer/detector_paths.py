"""What is the detector balanced against? Scaling neuron 5 by (1+a) at step t changes the final lead D (second over
third) by about a x sum_j W_hh[j,5] h5_t dD/dz_{t+1}[j] (or its readout at the last step). Split that by the group j
it feeds, sum over steps, on near-tie lists; also by step."""
import sys; sys.path.insert(0, "../../family")
import torch
from common import *
from memory_gain import X, l, s, t3, MEM
GR = {"store 2,4": [2, 4], "memory": MEM, "hiding 1,6,7,8": [1, 6, 7, 8], "detector 5": [5], "gate 9": [9]}
W = ARC; Xm = X[:100000]; sm, tm = s[:100000], t3[:100000]; n = torch.arange(len(Xm), device=DEV)
H, Z = [], []
h = torch.zeros(len(Xm), 16, device=DEV, dtype=dt)
for t in range(10):
    z = (h @ W[1].T + Xm[:, t:t + 1] * W[0]).requires_grad_(True); Z.append(z); h = torch.relu(z); H.append(h)
sc = h @ W[2].T; D = sc[n, sm] - sc[n, tm]
G = torch.autograd.grad(D.sum(), Z)                     # dD/dz_t, n x 16
tot = {k: torch.zeros(len(Xm), device=DEV, dtype=dt) for k in GR}; by_step = []
for t in range(9):
    f = H[t][:, 5].detach()[:, None] * W[1][:, 5][None] * G[t + 1]   # n x 16: path via neuron j at step t+1
    row = {}
    for k, g in GR.items(): c = f[:, g].sum(1); tot[k] += c; row[k] = c.abs().mean().item()
    by_step.append(row)
direct = H[9][:, 5].detach() * (W[2][sm, 5] - W[2][tm, 5])
print(f"near-tie lists {len(Xm):,}; final lead |D| median {D.abs().median():.2f}")
print("effect on D of scaling the detector by 1 (i.e. per unit a), summed over steps, by the group it feeds:")
for k, c in tot.items(): print(f"   via {k:16s} mean {c.mean():+8.2f}  mean |.| {c.abs().mean():8.2f}")
s_all = sum(tot.values()) + direct
print(f"   direct readout      mean {direct.mean():+8.2f}  mean |.| {direct.abs().mean():8.2f}")
print(f"   total               mean {s_all.mean():+8.2f}  mean |.| {s_all.abs().mean():8.2f}; share of lists where 5% flips the sign of D (first order): {100*((D + 0.05*s_all).sign() != D.sign()).double().mean():.1f}%")
print("mean |path| by step (via memory / hiding / store):")
for t, r in enumerate(by_step): print(f"   step {t}->{t+1}: {r['memory']:7.2f} {r['hiding 1,6,7,8']:7.2f} {r['store 2,4']:7.2f}")
