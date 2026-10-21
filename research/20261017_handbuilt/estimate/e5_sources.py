"""E5: (1) the E4 integral (grid 24 x 20 x 20) on ARC and the 32 sisters, against truth and the surrogate's sampled
accuracy; (2) which neuron groups' fixed switching makes the bias: each group put back on its own sign, the rest on
the canonical-run pattern (sampled, diagnostic). Roles from ../parameters/universal/roles.json."""
import json
from elib import *
from e4_integrate import maps, estimate, PERM
truth = np.load("/data/output/research/algzoo/20261017_handbuilt/hilton/propagate/across.npy")[:, 0] * 100
roles = json.load(open("/data/output/research/algzoo/20261017_handbuilt/parameters/universal/roles.json"))
names = ["ARC"] + [str(s) for s in list(range(100, 116)) + list(range(200, 216))]
X = lists(1_000_000, 102); o = X.argsort(1, descending=True)[:, :3]
LUT = torch.full((1000,), -1, dtype=torch.long, device=DEV); LUT[(PERM[:, 0] * 10 + PERM[:, 1]) * 10 + PERM[:, 2]] = torch.arange(720, device=DEV)
ai = LUT[(o[:, 0] * 10 + o[:, 1]) * 10 + o[:, 2]]
rows = []
for i, nm in enumerate(names):
    W = arc() if nm == "ARC" else sister(int(nm)); r = roles["ARC" if nm == "ARC" else f"s{nm}"]
    M, on = maps(W); e, _, _ = estimate(M, 24, 20)
    S0, _ = run(W, X); a0 = acc(S0, X)
    Ss = torch.einsum("nij,nj->ni", M[ai], X); asur = acc(Ss, X)
    grp = {"memory": r["memory"], "hiding": r["hiding"], "store+detector": [r["store"]] + r["detectors"], "gates": r["late"] + r["keep"]}
    grp = {k: [n for n in v if n is not None] for k, v in grp.items()}
    allg = sorted(set(sum(grp.values(), []))); grp["unassigned"] = [n for n in range(16) if n not in allg]
    rec = {}
    for g, ns in grp.items():
        m = torch.zeros(16, dtype=torch.bool, device=DEV); m[ns] = True
        S, _ = run(W, X, lambda t, pre: torch.where(m, pre > 0, on[ai, t])); rec[g] = acc(S, X) - asur
    rows.append((truth[i], e, asur, a0, rec))
    print(f"{nm:>4s}: truth {truth[i]:.2f} estimate {e:.2f} ({e - truth[i]:+.2f}); surrogate sampled {asur:.2f} (integral off by {e - asur:+.2f}); "
          "recovered by own sign: " + ", ".join(f"{g} ({len(grp[g])}) {v:+.2f}" for g, v in rec.items()), flush=True)
err = np.array([r[1] - r[0] for r in rows]); ie = np.array([r[1] - r[2] for r in rows])
print(f"sisters: estimate RMSE {np.sqrt((err[1:] ** 2).mean()):.2f}, mean {err[1:].mean():+.2f}; integral against sampled surrogate RMSE {np.sqrt((ie[1:] ** 2).mean()):.2f}")
for g in rows[0][4]:
    v = np.array([r[4][g] for r in rows[1:]]); print(f"   {g}: mean recovery {v.mean():+.2f}, median {np.median(v):+.2f}")
