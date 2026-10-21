"""Second try at a wrap-around memory for v18. The detector's injection already writes the record's 8-periodic wave
into all 8 slots (offsets -1..-8, and offset -8 is the record's own slot once it wraps). In a wrap-around register
that covers the forward half too, so v18's forward copies from the clock and flip into slot 0 double it. Variant:
wrap slot 7 into slot 0, drop the forward copies, keep everything else. Expected difference from v18: an overtaken
record's wave also stays at positions after the overtaking (v18 stops writing it there)."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
exec(open("cyclic_memory.py").read().split("net = build_v18()")[0].split("from algzoo")[0])
rng = np.random.default_rng(0); X = rng.standard_normal((200000, 10)); y = np.argsort(X, 1)[:, -2]
def run(P, X):
    wi, wh, wo = P; h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T
net = build_v18(); I = net.idx; P0 = [np.array(p, float) for p in net.params]
Y = [I[f"mem{j}"] for j in range(8)]; S, D, F, Z = I["store"], I["detect"], I["flip"], I["flag"]; C = [I[f"clock{k}"] for k in range(3)]
lam = 10.0
lg0 = run(P0, X); a0 = lg0.argmax(1)
wi, wh, wo = (p.copy() for p in P0)
wh[Y[0], S] -= lam / 20; wh[Y[0], D] -= lam / 1000; wh[Y[0], Y[7]] += 1.0; wh[Y[0], D] += lam / 1000
before = wh[Y[0]].copy()
wh[Y[0], C] = 0.0; wh[Y[0], F] = 0.0                          # forward copies of the clock and flip parts removed
print("removed weights into slot 0:", {k: round(before[i] - wh[Y[0], i], 3) for k, i in I.items() if abs(before[i] - wh[Y[0], i]) > 1e-12})
lg1 = run((wi, wh, wo), X); a1 = lg1.argmax(1)
print("v18", (a0 == y).mean(), " wrap-around, no forward copies", (a1 == y).mean(), " agree", (a0 == a1).mean())
# where they differ: was the largest overtaken? by kind of list
amax = X.argmax(1); m = np.maximum.accumulate(X, 1); nrec = (X[:, 1:] > m[:, :-1]).sum(1)
for k in range(0, 5):
    s = nrec == k
    print(f"  {k} records after the first number: share {s.mean():.2f}  v18 {(a0[s]==y[s]).mean():.3f}  wrap {(a1[s]==y[s]).mean():.3f}")
np.savez("cyclic2_v18.npz", wi=wi, wh=wh, wo=wo)
