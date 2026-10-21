"""Which kinds of mistakes do noisy copies of ARC's network add? Noise: each weight times (1 + sigma * N(0,1)),
as in noisy_copy_mistakes.py (check). Kinds: A at most one positive; B answer = third-largest; C answer = largest;
D anything else. For C and D also how far below the second the answer's number is (are they close calls?)."""
import numpy as np
import torch as th

from algzoo.core import example, forward

P0 = example(); rng = np.random.default_rng(0)
g = th.Generator(device="cuda").manual_seed(5)
X = th.randn(2_000_000, 10, device="cuda", generator=g, dtype=th.float64)
o = X.argsort(-1, descending=True); few = (X > 0).sum(-1) <= 1
v1, v2, v3 = (X.gather(1, o[:, i:i + 1])[:, 0] for i in range(3))
gap12 = ((v1 - v2) / v1.abs()).cpu().numpy(); gap23 = ((v2 - v3) / v2.abs()).cpu().numpy()
print("noise  copy | error | A  B  C  D (points) | C with largest-second gap > 10% | B with second-third gap > 10%")
for sigma in (0.0, 0.005, 0.01, 0.02):
    for c in range(3 if sigma else 1):
        P = tuple(w * (1 + sigma * rng.standard_normal(w.shape)) for w in P0)
        ans = forward(P, X).argmax(-1); wrong = ans != o[:, 1]
        A = wrong & few; B = wrong & ~few & (ans == o[:, 2]); C = wrong & ~few & (ans == o[:, 0]); D = wrong & ~few & ~(ans == o[:, 2]) & ~(ans == o[:, 0])
        k = lambda t: 100 * t.double().mean().item()
        Cn = C.cpu().numpy(); Bn = B.cpu().numpy()
        print(f"{sigma:.3f}  {c}    | {k(wrong):5.2f} | {k(A):.2f} {k(B):.2f} {k(C):.2f} {k(D):.2f} | {100*np.mean(Cn & (gap12 > 0.1)):.2f} | {100*np.mean(Bn & (gap23 > 0.1)):.2f}")
