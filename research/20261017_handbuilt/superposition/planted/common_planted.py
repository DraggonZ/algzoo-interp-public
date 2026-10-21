"""Planted test of the sparse-autoencoder method: the same TopK SAE recipe as ../sae/train_sae.py, applied to hand-built
networks whose roles are known by construction (v21, 28 neurons; v19 with kappa 0.5, 16 neurons) and to a random
16-dimensional linear projection of v21's 28-dimensional state (more roles than dimensions).

Setups:
  v21       SAE input = v21's state, each neuron scaled to unit standard deviation (as for ARC).
  v19       the same for v19.
  v21proj   P @ (scaled v21 state), P a fixed random 16 x 28 Gaussian matrix, then each of the 16 dimensions scaled
            to unit standard deviation. Roles to recover are still v21's 28 neurons.

Role groups. Rotating groups have no preferred basis: v21's clocks C0-C2, E0-E2, QC0-QC2, QE0-QE2 (45- and 90-degree
phases, each three neurons carrying a 2-D phase plus a level) and v19's clock0-2. For those the ground truth is the
three-dimensional subspace. The memory delay lines (v21 Y0-Y8, v19 mem0-7) are always on once the largest so far is
positive, so they run in the ReLU's linear regime and any positive mixture of slots is an equivalent network. We
therefore report memory both per slot and as one subspace."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/superposition/sae")
from common import *
from train_sae import SAE                       # identical SAE class (TopK, unit-norm decoder); d is passed explicitly
from algzoo.handbuilt_v21 import build_v21, NAMES as V21_NAMES
from algzoo.handbuilt_v19 import build_v19
F32 = torch.float32
V19_NAMES = ["store", "detect", "keep", "flip", "clock0", "clock1", "clock2", "flag"] + [f"mem{j}" for j in range(8)]
NETS = {"v21": (T(build_v21()), V21_NAMES), "v19": (T(build_v19(kappa=0.5).params), V19_NAMES)}
assert len(NETS["v19"][0][0]) == 16
def _idx(names, pre): return [i for i, n in enumerate(names) if any(n == p or (n.startswith(p) and n[len(p):].isdigit()) for p in pre)]
ROT = {"v21": {"clock 45 (C)": _idx(V21_NAMES, ["C"]), "clock 90 (E)": _idx(V21_NAMES, ["E"]),
               "reveal 45 (QC)": _idx(V21_NAMES, ["QC"]), "reveal 90 (QE)": _idx(V21_NAMES, ["QE"])},
       "v19": {"clock": _idx(V19_NAMES, ["clock"])}}
MEM = {"v21": _idx(V21_NAMES, ["Y"]), "v19": _idx(V19_NAMES, ["mem"])}
for nm in ROT:
    for g in ROT[nm].values(): assert len(g) == 3, (nm, g)
assert len(MEM["v21"]) == 9 and len(MEM["v19"]) == 8

def states(W, X):
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt); H = []
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0]); H.append(h)
    return torch.stack(H, 1)                    # [n, step, d]
PROJ = torch.tensor(np.random.default_rng(2026).standard_normal((16, 28)) / np.sqrt(28), device=DEV, dtype=dt)

# shared quantities removed before matching: as ../sae/eval_roles_fair.py (largest so far, current number, step), plus
# the largest so far before the current number. Without it the memory slots of v19, which each carry the level
# lam * m1(t-1), share one component (the latest jump) that is over 99% of their remaining variance, and any latent
# carrying that jump would 'match' every slot.
def shared(X, prev=True):
    m = torch.zeros(len(X), device=DEV, dtype=dt); F = []
    for t in range(10):
        mp = m; m = torch.maximum(m, X[:, t])
        F.append(torch.cat([m[:, None], X[:, t:t + 1], torch.nn.functional.one_hot(torch.full((len(X),), t, device=DEV), 10).to(dt)] + ([mp[:, None]] if prev else []), 1))
    return torch.stack(F, 1).reshape(len(X) * 10, -1)
def resid(U, F): return U - F @ torch.linalg.lstsq(F, U).solution
