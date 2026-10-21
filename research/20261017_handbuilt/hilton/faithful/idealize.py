"""Idealized weights from the trained network itself, by simplifications our explanation licenses (no fitting):
 (A) block: the store (2) and detector (4) read only each other and the input (all their other incoming weights 0);
 (B) tiny: every recurrent and readout weight whose effect (|weight| x RMS activity of the neuron it reads) is below
     a fraction f of the largest effect in its row is set to 0; f = 0.01, 0.02, 0.05, 0.1;
 (A+B) both.
For each idealized network I: its own accuracy; and Hilton's prediction of the trained network's accuracy with the
gap T - I as noise (12 copies I + e, each weight's e normal with SD |T - I|). Accuracies on the three-largest grid
without sampling (40 x 48 x 48; the trained network gives 95.23% on it)."""
import sys
import numpy as np, torch
sys.path.insert(0, "../propagate")
import full_mean
from algzoo.core import example
DEV = "cuda"; dt = torch.float64
T = [torch.tensor(np.asarray(p, float), device=DEV, dtype=dt) for p in example()]
V, wt = full_mean.grid(40, 48)
Vt = torch.tensor(V, device=DEV, dtype=dt); wtt = torch.tensor(wt, device=DEV, dtype=dt)


def acc(W):
    nV = len(wtt); tot = 0.0
    for i in range(0, 720, 16):
        tt = full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = Vt.repeat(len(tt), 1)
        tot += (full_mean.win_hard(W, pos, val) * wtt.repeat(len(tt))).sum().item()
    return tot / (wtt.sum().item() * 720)


# RMS activity of each neuron on random lists (for the effect sizes)
X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(4)); h = torch.zeros(200000, 16, device=DEV, dtype=dt); hs = []
for t in range(10): h = torch.relu(h @ T[1].T + X[:, t:t + 1] * T[0]); hs.append(h)
rms = torch.stack(hs, 1).pow(2).mean((0, 1)).sqrt()


def block(W):
    wi, wh, wo = (w.clone() for w in W)
    for r in (2, 4):
        keep = torch.zeros(16, dtype=torch.bool, device=DEV); keep[[2, 4]] = True; wh[r, ~keep] = 0
    return [wi, wh, wo]


def tiny(W, f):
    wi, wh, wo = (w.clone() for w in W)
    for M in (wh, wo):
        eff = M.abs() * rms[None, :]; M[eff < f * eff.max(1, keepdim=True).values] = 0
    return [wi, wh, wo]


def hilton(I, n=12):
    g = torch.Generator(device=DEV).manual_seed(2); a = []
    for _ in range(n):
        a.append(acc([i + (t - i).abs() * torch.randn(i.shape, device=DEV, dtype=dt, generator=g) for i, t in zip(I, T)]))
    return np.mean(a), min(a), max(a)


if __name__ == "__main__":
  print(f"trained network: {100*acc(T):.2f}%")
  cands = [("block", block(T))] + [(f"tiny f={f}", tiny(T, f)) for f in (0.01, 0.02, 0.05, 0.1)] + [(f"block + tiny f={f}", tiny(block(T), f)) for f in (0.02, 0.05)]
  for name, I in cands:
      nz = sum(int(((i != t)).sum()) for i, t in zip(I, T)); gap = np.sqrt(sum(float(((i - t) ** 2).sum()) for i, t in zip(I, T)) / nz) if nz else 0
      m, lo, hi = hilton(I)
      print(f"{name:20s} weights changed {nz:3d} of 432 (RMS change {gap:.4f}); idealized network {100*acc(I):6.2f}%;  Hilton's prediction (gap as noise) {100*m:6.2f}% (copies {100*lo:.1f}-{100*hi:.1f})", flush=True)
