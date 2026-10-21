"""What does ARC's memory (neurons 0, 3, 10-15) hold that v7 + z + ninth slot's roles do not? At every step, fit ARC's
memory activity from the v7z roles plus the current number (held out); then explain the leftover with added list
features, one family at a time: (a) the numbers seen so far, by position; (b) their positive parts; (c) the positive
parts split by whether they came before or after the current largest; (d) the largest and previous largest by age
(one-hot age times value); (e) the numbers since the largest by steps since they arrived, positive part; (f) all of
these. Share of the leftover's variance each family explains."""
import sys, numpy as np, torch
sys.path.insert(0, "../family")
from common import *
from role_map import states
from reverse_map import ridge_r2
from baseline_v7z import nets
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
def listfeats(X):
    n = len(X); ar = torch.arange(n, device=DEV); out = {k: [] for k in "abcde"}
    m = torch.zeros(n, device=DEV, dtype=dt); Lp = torch.full((n,), -1, device=DEV, dtype=torch.long); Pv = torch.zeros(n, device=DEV, dtype=dt)
    for t in range(10):
        x = X[:, t]; new = x > m; Pv = torch.where(new, m, Pv); Lp = torch.where(new, t, Lp); m = torch.maximum(m, x)
        pos = torch.arange(10, device=DEV)[None]; seen = pos <= t
        past = torch.where(seen, X, 0); pp = past.clamp(min=0)
        out["a"].append(past); out["b"].append(pp)
        before = pos < Lp[:, None]; out["c"].append(torch.cat([pp * before, pp * (~before)], 1))
        age = (t - Lp).clamp(min=0, max=9); oh = torch.nn.functional.one_hot(age, 10).to(dt)
        out["d"].append(torch.cat([oh * m[:, None], oh * Pv[:, None]], 1))
        since = (pos > Lp[:, None]) & seen; lag = (t - pos).clamp(min=0)          # steps since that number arrived
        e = torch.zeros(n, 10, device=DEV, dtype=dt); e.scatter_add_(1, lag, pp * since); out["e"].append(e)
    return {k: torch.stack(v, 1) for k, v in out.items()}
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(79); X = torch.randn(160000, 10, device=DEV, dtype=dt, generator=g)
    tr, te = slice(0, 80000), slice(80000, None)
    Ha = states(ARC, X)[0][..., MEM]; W = nets()["v7 + z + ninth slot (18)"]; R = states(W, X)[0]
    R = R / (R[tr].reshape(-1, R.shape[2]).std(0) + 1e-12)
    base = torch.cat([R, X[:, :, None]], 2); fl = lambda A, s: A[s].reshape(-1, A.shape[2])
    Y = Ha; r0 = ridge_r2(fl(base, tr), fl(base, te), fl(Y, tr), fl(Y, te))
    print("ARC memory from v7z roles, R2 by neuron (0,3,10-15): " + " ".join(f"{v:.2f}" for v in r0.tolist()))
    F = listfeats(X); F["f"] = torch.cat([F[k] for k in "abcde"], 2)
    for k, nm in zip("abcdef", ["numbers seen", "positive parts", "positive parts before / after the largest", "largest and previous largest by age", "numbers since the largest, by lag", "all of these"]):
        A = torch.cat([base, F[k] / (F[k][tr].reshape(-1, F[k].shape[2]).std(0) + 1e-12)], 2)
        r = ridge_r2(fl(A, tr), fl(A, te), fl(Y, tr), fl(Y, te))
        gain = (r - r0) / (1 - r0)
        print(f"+ {nm:45s}: R2 " + " ".join(f"{v:.2f}" for v in r.tolist()) + " | share of leftover explained " + " ".join(f"{v:.2f}" for v in gain.tolist()))
