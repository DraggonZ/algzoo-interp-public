"""Role of each of the ten other neurons.

1. Its place on the wheel: the age at which it peaks after a single input.
2. Silencing it (held at zero every step): two-spike accuracy and full accuracy (sampled, 2^22 inputs each).
3. Is the slide possible without the ten? Cosine between output row j and row j-1 times the step matrix restricted to
   the six comparing neurons.
"""
import numpy as np
import torch as th

from algzoo.core import example

w_hi, w_hh, w_oh = example()
C = [1, 2, 4, 6, 7, 8]
Dn = [i for i in range(16) if i not in C]
dev = "cuda"
W = [th.as_tensor(w, device=dev, dtype=th.float32) for w in (w_hi, w_hh, w_oh)]


def acc(x, mask):
    h = th.zeros(x.shape[0], 16, device=dev)
    for t in range(10):
        h = th.relu(h @ W[1].T + x[:, t, None] * W[0]) * mask
    pred = (h @ W[2].T).argmax(-1)
    tgt = x.argsort(-1)[:, -2]
    return (pred == tgt).float().mean().item()


g = th.Generator(device=dev).manual_seed(1)
N = 2 ** 22
x = th.randn(N, 10, device=dev, generator=g)
top2 = x.argsort(-1)[:, -2:]
x2 = th.zeros_like(x).scatter(1, top2, x.gather(1, top2))
h = np.maximum(0, w_hi); H = [h]
for k in range(1, 12):
    h = np.maximum(0, w_hh @ h); H.append(h)
H = np.array(H)
full = th.ones(16, device=dev)
print(f"baseline: two-spike {acc(x2, full):.4f}, full {acc(x, full):.4f}")
for i in Dn + C:
    m = full.clone(); m[i] = 0
    peak = int(np.argmax(H[1:10, i])) + 1
    print(f"silence n{i:<2d} (peaks at age {peak}, size {H[1:10, i].max():5.1f}): two-spike {acc(x2, m):.4f}, full {acc(x, m):.4f}")
on = (w_hh @ H[5] > 0).astype(float)
M = on[:, None] * w_hh
Wc = w_oh - w_oh.mean(0)
for lab, keep in [("all 16", list(range(16))), ("comparing only", C), ("ten other only", Dn)]:
    Mk = np.zeros_like(M); Mk[np.ix_(keep, keep)] = M[np.ix_(keep, keep)]
    cs = [np.dot(Wc[j], Wc[j - 1] @ Mk) / (np.linalg.norm(Wc[j]) * np.linalg.norm(Wc[j - 1] @ Mk) + 1e-12) for j in range(1, 10)]
    print(f"slide relation with step matrix on {lab}: cosines {np.round(cs, 2)}")
