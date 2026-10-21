"""Numbers for the propagation explainer (propagation.html), all from ARC's trained network on random lists.
Writes data.json: the largest so far (sampled against its exact formula), when it arrived, the store against the largest
by age, the leftovers of two store rules, and the right answer's score against the largest's score."""
import json
import numpy as np
from scipy.stats import norm
from algzoo.core import example

wi, wh, wo = (np.asarray(p, float) for p in example())
rng = np.random.default_rng(11); N = 200000; X = rng.standard_normal((N, 10))
h = np.zeros((N, 16)); HS = []
for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); HS.append(h)
HS = np.stack(HS, 1); L = h @ wo.T
rk = np.argsort(-X, 1); y = rk[:, 1]
acc = float((L.argmax(1) == y).mean())
M = np.maximum.accumulate(np.maximum(X, 0), 1)                 # largest so far after step t (0 if none positive yet)
rec = np.full(N, -1); arg = np.zeros((N, 10), int)
for t in range(10):
    rec = np.where(X[:, t] > (M[:, t - 1] if t else 0), t, rec); arg[:, t] = rec
D = {"acc": acc, "n": N}

# 1. largest so far: histogram against the exact density t phi(a) Phi(a)^(t-1); share still at 0 (no positive yet)
edges = np.linspace(0, 3.5, 36); mid = (edges[1:] + edges[:-1]) / 2; grid = np.linspace(0.005, 3.5, 140)
D["max"] = [{"t": t + 1, "hist": (np.histogram(M[:, t][M[:, t] > 0], edges)[0] / N / np.diff(edges)).round(4).tolist(),
             "zero": float((M[:, t] == 0).mean()), "zero_exact": 0.5 ** (t + 1),
             "dens": ((t + 1) * norm.pdf(grid) * norm.cdf(grid) ** t).round(4).tolist()} for t in range(10)]
D["max_edges"] = edges.round(3).tolist(); D["max_grid"] = grid.round(3).tolist()
# when did the largest arrive (given some positive number so far): exact 1/t each
D["arrive"] = [[float((arg[:, t][arg[:, t] >= 0] == j).mean()) for j in range(t + 1)] for t in range(10)]
D["indep"] = {"mean_max_if_early": float(M[:, 9][(arg[:, 9] >= 0) & (arg[:, 9] < 5)].mean()),
              "mean_max_if_late": float(M[:, 9][arg[:, 9] >= 5].mean())}

# 2-3. store (neuron 2) after step t holds about 17.6 x the largest of the steps before t (it lags one step)
Sv, Mv, Av = [], [], []
for t in range(1, 10):
    prev = M[:, t - 1]; age = t - arg[:, t - 1]                 # steps since that largest arrived (>= 1)
    ok = (arg[:, t - 1] >= 0) & (prev > 0.3)
    Sv.append(HS[ok, t, 2]); Mv.append(prev[ok]); Av.append(age[ok])
Sv, Mv, Av = map(np.concatenate, (Sv, Mv, Av))
ratio = Sv / Mv
f = {int(a): float(np.median(ratio[Av == a])) for a in range(1, 10)}
D["store_f"] = f
pick = rng.choice(len(Sv), 2400, replace=False)
D["store_pts"] = np.c_[Mv[pick], Sv[pick], Av[pick]].round(3).tolist()
q = [0.1, 0.25, 0.5, 0.75, 0.9]
D["left"] = {"ideal": [np.quantile(ratio[Av == a] - 17.6, q).round(3).tolist() for a in range(1, 10)],
             "fade": [np.quantile(ratio[Av == a] - f[a], q).round(3).tolist() for a in range(1, 10)],
             "share": [float((Av == a).mean()) for a in range(1, 10)]}

# 4. the right answer's score against the largest's score, real and with the two scored separately (shuffled)
ry, r1 = L[np.arange(N), y], L[np.arange(N), rk[:, 0]]
perm = rng.permutation(N); pick = rng.choice(N, 1800, replace=False)
D["scores"] = {"real": np.c_[ry[pick], r1[pick]].round(2).tolist(), "shuf": np.c_[ry[pick], r1[perm][pick]].round(2).tolist(),
               "p_real": float((ry > r1).mean()), "p_shuf": float((ry > r1[perm]).mean()), "corr": float(np.corrcoef(ry, r1)[0, 1])}
lead = L[np.arange(N), y][:, None] - L
p_each = [float((lead[np.arange(N), rk[:, k]] > 0).mean()) for k in range(10) if k != 1]
D["leads"] = {"p_each": p_each, "product": float(np.prod(p_each)), "joint": acc}
json.dump(D, open("data.json", "w"))
print("acc", acc, "f", {a: round(v, 2) for a, v in f.items()})
print("left ideal medians", [r[2] for r in D["left"]["ideal"]]); print("left fade 10-90", [(r[0], r[4]) for r in D["left"]["fade"]])
print("scores", {k: v for k, v in D["scores"].items() if k.startswith("p") or k == "corr"}, "leads", D["leads"]["product"])
print("indep", D["indep"], "zero t=3", D["max"][2]["zero"], D["max"][2]["zero_exact"])
print("score ranges", np.quantile(ry, [.01, .99]), np.quantile(r1, [.01, .99]))
