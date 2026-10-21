"""Question: does v18 still work if its delay-line memory wraps around (slot 7 feeds slot 0), as the trained memory
seems to (rotating pairs near a cyclic shift's angles)? If the wrapped content is harmless, the memory is a cyclic
shift, which a change of basis turns into rotating pairs like the trained network's."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18

rng = np.random.default_rng(0); X = rng.standard_normal((200000, 10)); y = np.argsort(X, 1)[:, -2]
def run(P, X, keep=False):
    wi, wh, wo = P; h = np.zeros((len(X), len(wi))); hs = []
    for t in range(X.shape[1]):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return (h @ wo.T, np.stack(hs, 1)) if keep else h @ wo.T
net = build_v18(); I = net.idx; P0 = [np.array(p, float) for p in net.params]
Y = [I[f"mem{j}"] for j in range(8)]; S, D = I["store"], I["detect"]
lg0, H0 = run(P0, X, True); print("v18", (lg0.argmax(1) == y).mean())
# what slot 7 holds beyond its level, per step (this is what would wrap into slot 0)
lam = 10.0; m1 = H0[:, :, S] / 20 + H0[:, :, D] / 1000
extra = H0[:, :, Y[7]] - lam * m1
print("slot 7 minus level, mean |.| by step:", np.round(np.abs(extra).mean(0), 4))
wi, wh, wo = (p.copy() for p in P0)
wh[Y[0], S] -= lam / 20; wh[Y[0], D] -= lam / 1000      # drop slot 0's own level ...
wh[Y[0], Y[7]] += 1.0; wh[Y[0], D] += lam / 1000         # ... take slot 7 (level lam m1(t-1)) plus the jump's level
lg1 = run((wi, wh, wo), X); print("cyclic memory", (lg1.argmax(1) == y).mean(), "agree with v18", (lg1.argmax(1) == lg0.argmax(1)).mean())
ev = np.linalg.eigvals(wh[np.ix_(Y, Y)]); print("memory block eigen angles", np.sort(np.round(np.degrees(np.angle(ev)), 1)), "moduli", np.round(np.abs(ev), 3))
np.savez("cyclic_v18.npz", wi=wi, wh=wh, wo=wo)
