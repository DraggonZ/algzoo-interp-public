"""Development check (sampled lists): where does the propagated model's accuracy deficit come from? Model per node
(regime and grid cells of L and S, gate sub-regimes pooled) against sampled accuracy at the same nodes; the deficit
sum_nodes w (p_model - p_sampled) split by kind of list."""
import sys, json, numpy as np, torch
from algzoo.core import example
from prop4 import Prop, DEV, dt
G = int(sys.argv[1]); KW = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
W = [np.asarray(w, float) for w in example()]
P = Prop(W, G=G, **KW); acc = P.run(); nd = P.node
K = 10 * 11 * G * (G + 1)
code = ((nd["pL"] * 11 + nd["pS"] + 1) * G + nd["iL"]) * (G + 1) + nd["iS"]
mw = torch.zeros(K, device=DEV, dtype=dt).index_add_(0, code, nd["w"]); mr = torch.zeros(K, device=DEV, dtype=dt).index_add_(0, code, nd["w"] * P.p_right)
edges = torch.tensor(P.edges[1:-1], device=DEV, dtype=dt)
cnt = torch.zeros(K, device=DEV, dtype=dt); rt = torch.zeros(K, device=DEV, dtype=dt); N = 0
wi, wh, wo = P.wi, P.wh, P.wo
for sd in range(5):
    X = torch.randn(2_000_000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(100 + sd))
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    o = X.argsort(1, descending=True); pL, pS = o[:, 0], o[:, 1]
    L = X.gather(1, pL[:, None])[:, 0]; S = X.gather(1, pS[:, None])[:, 0]
    iL = torch.bucketize(L, edges); iS = torch.bucketize(S, edges) + 1
    c = ((pL * 11 + pS + 1) * G + iL) * (G + 1) + iS
    right = ((h @ wo.T).argmax(1) == pS).to(dt)
    cnt.index_add_(0, c, torch.ones_like(right)); rt.index_add_(0, c, right); N += len(X)
    if sd == 0: X0, pT0 = X, o[:, 2]
print(f"G {G} {KW}: model {100*acc:.2f}%, sampled {100*rt.sum().item()/N:.2f}%")
pt = cnt / N; ok = cnt > 0
# deficit per node: model's right mass minus sampled right mass (sampled weights used for both, model's per-node p)
pm = torch.where(mw > 0, mr / mw.clamp_min(1e-300), torch.zeros_like(mw)); ps = torch.where(ok, rt / cnt.clamp_min(1), torch.zeros_like(rt))
print(f"node weights: total |model - sampled| = {(mw - pt).abs().sum().item():.4f}")
print(f"accuracy at sampled weights with model's per-node p: {100*(pt*pm).sum().item():.2f}%")
idx = torch.arange(K, device=DEV); iS_ = idx % (G + 1); r = idx // (G + 1); iL_ = r % G; r = r // G; pS_ = r % 11 - 1; pL_ = r // 11
d = pt * (pm - ps)
def show(name, lab, vals):
    print(name)
    for v in vals:
        m = lab == v; print(f"   {str(v):>10s}: share of lists {100*pt[m].sum().item():5.1f}%, deficit {100*d[m].sum().item():+.2f} points, model {100*(pt*pm)[m].sum().item()/max(1e-12,pt[m].sum().item()):.1f}% vs sampled {100*(pt*ps)[m].sum().item()/max(1e-12,pt[m].sum().item()):.1f}%")
show("by order of the two largest (S before L = 1)", (pS_ < pL_).long(), [0, 1])
show("by gap L-S in cells (0 = same cell)", (iL_ - (iS_ - 1)).clamp(0, 6), list(range(7)))
show("by position of the second-largest", pS_, list(range(10)))
show("by position of the largest", pL_, list(range(10)))
show("by cell of S (grid index)", iS_ - 1, list(range(0, G, max(1, G // 10))))
