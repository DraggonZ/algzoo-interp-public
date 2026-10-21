"""Development check (sampled lists): does the propagated mixture match the real state distribution per regime and
step? For each step, weighted over regimes: weight error, mean error of h (relative to the state's size), SD ratio,
and at the end the per-regime win probability against the sampled one."""
import sys, json, numpy as np, torch
from algzoo.core import example
from prop2 import Prop, D
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
from common import ARC, DEV, dt
kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
W = [np.asarray(w, float) for w in example()]
P = Prop(W, **kw)
X = torch.randn(1_000_000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(7))
wi, wh, wo = ARC; h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
L = torch.full((len(X),), -1e9, device=DEV, dtype=dt); S = L.clone(); pL = torch.full((len(X),), -1, device=DEV); pS = pL.clone()
c = {(-1, -1, ()): (1.0, np.zeros(D), np.zeros((D, D)))}
for t in range(10):
    x = X[:, t]; h = torch.relu(h @ wh.T + x[:, None] * wi)
    rec = x > L; sec = ~rec & (x > S)
    S = torch.where(rec, L, torch.where(sec, x, S)); pS = torch.where(rec, pL, torch.where(sec, t, pS))
    L = torch.where(rec, x, L); pL = torch.where(rec, t, pL)
    c = P.step(c, t)
    key = (pL * 10 + pS).cpu().numpy(); hn = h.cpu().numpy()
    werr = merr = 0; sdr = []
    for k, (w, mu, Sg) in c.items():
        m = key == k[0] * 10 + k[1]; fw = m.mean()
        werr += abs(fw - w)
        if m.sum() < 200: continue
        hm = hn[m].mean(0); hs = hn[m].std(0); scale = np.sqrt((hn[m] ** 2).mean(0).sum())
        merr += fw * np.linalg.norm(mu[:16] - hm) / scale
        sdr.append((fw, np.sqrt(np.maximum(np.diag(Sg)[:16], 0)), hs))
    big = np.argsort(-np.array([s[0] for s in sdr]))[:1]
    print(f"step {t}: regimes {len(c)}, total weight error {werr:.4f}, weighted mean error {100*merr:.1f}% of state size", flush=True)
    for i in big:
        fw, ms, ts = sdr[i]; print("   biggest regime SD model/true per neuron: " + " ".join(f"{a:.1f}/{b:.1f}" for a, b in zip(ms, ts)))
# final per-regime win
sc = (h @ wo.T).argmax(1); right = (sc == pS).cpu().numpy(); key = (pL * 10 + pS).cpu().numpy()
P.accuracy(c); rows = []
for k, (w, p) in P.per.items():
    m = key == k[0] * 10 + k[1]
    if m.sum(): rows.append((w, p, right[m].mean(), k))
rows.sort(key=lambda r: -r[0] * abs(r[1] - r[2]))
print(f"model accuracy {100*sum(r[0]*r[1] for r in rows):.2f}%, sampled {100*right.mean():.2f}%")
for w, p, q, k in rows[:12]: print(f"  regime largest at {k[0]}, second at {k[1]}: weight {w:.4f}, model win {p:.3f}, sampled {q:.3f}")
