"""The simplest deductive estimate: one linear map per arrangement of the three largest. For each of the 720
arrangements, run the network once on a canonical list (the three largest at the expected 1st-3rd of ten normals,
1.539 / 1.001 / 0.656; the seven others at the mean of a normal below the third, -0.348... computed) and take its exact
linear map J_a (scores = J_a x on that list's piece). The deduced model answers argmax(J_a x) on any list of
arrangement a. Its accuracy is the estimate (scored here by sampling, as a diagnostic only). Also: how often the true
network's piece has the same on/off sequence as the canonical one."""
import sys, itertools; sys.path.insert(0, "../../../family")
import numpy as np, torch
from scipy.stats import norm
from common import *
v1, v2, v3 = 1.539, 1.001, 0.656
rest = -norm.pdf(v3) / norm.cdf(v3)            # E[X | X < v3]
A = list(itertools.permutations(range(10), 3)); idx = {a: i for i, a in enumerate(A)}
C = torch.full((720, 10), rest, device=DEV, dtype=dt)
for i, (l, s, t) in enumerate(A): C[i, l], C[i, s], C[i, t] = v1, v2, v3
def run(W, name, N=2_000_000, seed=3):
    J = jac(W, C)                                   # 720 x 10 x 10
    g = torch.Generator(device=DEV).manual_seed(seed); X = torch.randn(N, 10, device=DEV, dtype=dt, generator=g)
    l, s, t = top3(X); a = torch.tensor([idx[k] for k in zip(l.tolist(), s.tolist(), t.tolist())], device=DEV)
    pred = torch.einsum("noi,ni->no", J[a], X).argmax(1); true = scores(W, X).argmax(1)
    acc_true = (true == s).double().mean().item(); acc_ded = (pred == s).double().mean().item()
    agree = (pred == true).double().mean().item()
    wr = true != s; same_mist = ((pred == true) & wr).double().sum().item() / wr.sum().item()
    print(f"{name:22s} true {acc_true:.4f} | deduced (one map per arrangement) {acc_ded:.4f} | answers agree {agree:.4f} | network's mistakes the deduced model repeats {same_mist:.2f}", flush=True)
    return acc_true, acc_ded
if __name__ == "__main__":
    print(f"canonical values: {v1}, {v2}, {v3}, others {rest:.3f}; cost 720 runs + 720 x 10 tangent passes")
    out = [run(ARC, "ARC")]
    for sd in list(range(100, 116)) + list(range(200, 216)):
        try: out.append(run(trained(sd), f"seed {sd}", N=500_000))
        except Exception as e: print(sd, e)
    o = np.array(out); e = 100 * (o[:, 1] - o[:, 0])
    print(f"error of the deduced estimate, points: ARC {e[0]:+.2f}; other {len(e)-1} networks mean {e[1:].mean():+.2f}, RMSE {np.sqrt((e[1:]**2).mean()):.2f}, range {e[1:].min():+.2f} to {e[1:].max():+.2f}")
