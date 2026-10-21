"""Behavioural prediction from the corrected account: the store holds the largest and discounts later arrivals by it,
so the outcome of a near tie between the second and third should depend on how big the largest is. Prediction
(written first): with the largest arriving before both contenders, making it bigger (x1.1) helps the earlier
contender; smaller (x0.9) helps the later one. With the largest between them or after both, the effect should be
smaller or absent (the discount then hits only the contender after it, or neither)."""
from harness import *
X = lists(8_000_000, seed=51); o = X.argsort(1, descending=True); l, s, t = o[:, 0], o[:, 1], o[:, 2]
v = X.sort(1, descending=True).values; near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1])
e = torch.minimum(s, t); late = torch.maximum(s, t)
for name, m in (("largest before both", near & (l < e)), ("largest between them", near & (l > e) & (l < late)), ("largest after both", near & (l > late))):
    Xm = X[m]; em = e[m]; lm = l[m]; n = torch.arange(len(Xm), device=DEV); row = []
    for f in (0.9, 1.0, 1.1, 1.3):
        Xc = Xm.clone(); Xc[n, lm] *= f; a = run(ARC, Xc).argmax(1)
        row.append(f"x{f}: {100*(a == em).double().mean().item():5.1f}%")
    print(f"{name:22s} n {len(Xm):7,}: earlier contender wins, largest scaled " + " | ".join(row), flush=True)
