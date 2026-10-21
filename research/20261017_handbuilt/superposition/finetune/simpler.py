"""Does continued training make close calls simpler, or only rarer? Fine-tune a copy of ARC (big batch 65,536, AdamW
1e-3, 10,000 steps, as in probe.py), save it, then repeat two measures on ARC and the fine-tuned copy: the number of
distinct second-over-third lead maps at 3% precision covering half of 200,000 lists (pieces3.py), and how well a tree
on positions and two ratios predicts the near-tie decision (contrast.py). Prediction (written first): the close calls
get rarer but stay dense (thousands of maps, trees well below the hand-built 98-99%)."""
import sys; sys.path.insert(0, "../sensitive"); sys.path.insert(0, "../../family")
import numpy as np, torch
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from restore import w0, unflat
from common import DEV, dt, jac, top3, scores, ARC
def sc(v, X):
    wi, wh, wo = unflat(v); h = torch.zeros(len(X), 16, device=DEV, dtype=X.dtype)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return h @ wo.T
v = w0.clone().float().requires_grad_(True); opt = torch.optim.AdamW([v], lr=1e-3, weight_decay=0.001); gt = torch.Generator(device=DEV).manual_seed(99)
for step in range(10000):
    x = torch.randn(65536, 10, device=DEV, generator=gt); loss = torch.nn.functional.cross_entropy(sc(v, x), x.argsort(1)[:, -2]); opt.zero_grad(); loss.backward(); opt.step()
wf = v.detach().double(); np.save("finetuned_10k.npy", wf.cpu().numpy())
nets = {"ARC": unflat(w0), "ARC + 10,000 steps": unflat(wf)}
g = torch.Generator(device=DEV).manual_seed(7); X = torch.randn(200_000, 10, device=DEV, dtype=dt, generator=g); l, s, t = top3(X); n = torch.arange(len(X), device=DEV)
G = torch.Generator(device=DEV).manual_seed(101); Y = torch.randn(4_000_000, 10, device=DEV, dtype=dt, generator=G); o = Y.argsort(1, descending=True); vv = Y.sort(1, descending=True).values
near = (vv[:, 1] > 0) & ((vv[:, 1] - vv[:, 2]) < 0.05 * vv[:, 1]); Y, o, vv = Y[near][:400000], o[near][:400000], vv[near][:400000]
F = np.c_[o[:, :3].cpu().numpy().astype(float), (vv[:, 2] / vv[:, 1]).cpu().numpy(), (vv[:, 0] / vv[:, 1]).cpu().numpy()]
for name, W in nets.items():
    W = [w.contiguous() for w in W]
    J = jac(W, X); d = J[n, s] - J[n, t]; d = d / d.norm(dim=1, keepdim=True)
    c = Counter(map(bytes, torch.round(d / 0.03).to(torch.int64).cpu().numpy())); cnt = np.array(sorted(c.values(), reverse=True))
    y = (scores(W, Y).argmax(1) == o[:, 2]).cpu().numpy()
    clf = HistGradientBoostingClassifier(max_iter=400, max_leaf_nodes=63).fit(F[:300000], y[:300000])
    print(f"{name:20s}: accuracy {100*(scores(W, X).argmax(1) == s).double().mean().item():.2f}% | lead maps at 3%: {len(c):,} distinct, {np.searchsorted(np.cumsum(cnt) / len(X), 0.5) + 1:,} cover half | near ties: names the third {100*y.mean():.1f}%, tree on top three predicts {100*(clf.predict(F[300000:]) == y[300000:]).mean():.1f}% (always-guess {100*max(y.mean(), 1-y.mean()):.1f}%)", flush=True)
