"""Causal superposition test with distributed alignment search. Variable: the value of the number at position s, as held
in the final state (step 9, just before the readout). Base list Xb, unrelated source list Xs; the edited list Xe is Xb
with x_s taken from Xs. Swap a learned d-dimensional subspace (orthonormal R) of the final state from the source run into
the base run: h = h_b + R R^T (h_s - h_b). Success when the network's answer equals its own answer on Xe (interchange
intervention accuracy, IIA, held out; reported overall and on pairs where the edit changes the answer). For ARC (16
neurons) and v19 (16). Then the overlap of the learned subspaces across positions: for d = 1, the |cosine| between the
directions for positions s and s'; for d = 2, the mean squared cosine of principal angles. Low overlap with high IIA: one
private direction per position; high overlap: positions share directions (superposition, in a causal sense)."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
from algzoo.handbuilt_v19 import build_v19
NETS = {"ARC": ARC, "v19": T(build_v19(kappa=0.5).params)}
def final(W, X):
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    return h
def data(W, s, n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed); Xb = torch.randn(n, 10, device=DEV, dtype=dt, generator=g); Xs = torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
    Xe = Xb.clone(); Xe[:, s] = Xs[:, s]
    hb, hs = final(W, Xb), final(W, Xs); y = (final(W, Xe) @ W[2].T).argmax(1); y0 = (hb @ W[2].T).argmax(1)
    sd = hb.std(0) + 1e-9
    return hb / sd, hs / sd, y, y0 != y, sd
def train(W, hb, hs, y, sd, d, steps=1500):
    lin = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(16, 16, bias=False).to(DEV).double()); opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(steps):
        i = torch.randint(0, len(hb), (8192,), device=DEV); R = lin.weight[:, :d]
        hp = (hb[i] + (hs[i] - hb[i]) @ R @ R.T) * sd; lg = hp @ W[2].T
        loss = torch.nn.functional.cross_entropy(lg / lg.std(), y[i]); opt.zero_grad(); loss.backward(); opt.step()
    return lin.weight[:, :d].detach()
def iia(W, R, hb, hs, y, ch, sd):
    a = (((hb + (hs - hb) @ R @ R.T) * sd) @ W[2].T).argmax(1) == y; return a.double().mean().item(), a[ch].double().mean().item()
if __name__ == "__main__":
    subs = {}
    for nm, W in NETS.items():
        for s in range(10):
            hb, hs, y, ch, sd = data(W, s, 200000, 600 + s); tb, ts, ty, tch, _ = data(W, s, 100000, 700 + s)
            tb, ts = tb * _ / sd, ts * _ / sd
            row = []
            for d in (1, 2, 3):
                R = train(W, hb, hs, y, sd, d); subs[(nm, s, d)] = R; a, c = iia(W, R, tb, ts, ty, tch, sd); row.append(f"d{d} {100*a:5.1f}%/{100*c:5.1f}%")
            Q, _q = torch.linalg.qr(torch.randn(16, 3, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(s))); a, c = iia(W, Q, tb, ts, ty, tch, sd)
            print(f"{nm} position {s}: edit changes the answer on {100*tch.double().mean():4.1f}% | IIA all/changed: " + "  ".join(row) + f"  | random 3-d {100*a:5.1f}%/{100*c:5.1f}%", flush=True)
    torch.save({f"{k[0]}_{k[1]}_{k[2]}": v.cpu() for k, v in subs.items()}, "das_positions.pt")
    for nm in NETS:
        D1 = torch.stack([subs[(nm, s, 1)][:, 0] for s in range(10)], 1); C = (D1.T @ D1).abs().cpu().numpy()
        np.set_printoptions(precision=2, suppress=True, linewidth=150)
        print(f"\n{nm}: |cosine| between the d=1 directions for positions 0-9 (scaled state coordinates):"); print(C)
        off = []
        for s in range(10):
            for s2 in range(s + 1, 10):
                A, B = subs[(nm, s, 2)], subs[(nm, s2, 2)]; off.append((torch.linalg.svdvals(A.T @ B) ** 2).mean().item())
        print(f"{nm}: mean squared cosine of principal angles between d=2 subspaces of different positions: {np.mean(off):.2f} (random 2-d subspaces in 16-d: about {2/16:.2f})")
