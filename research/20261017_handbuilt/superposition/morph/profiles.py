"""Per arrangement the mean morph is stable (split-half cosine 0.97), though single morphs agree only weakly (0.26). Is
that stable direction compact? Model every smaller number's mean morph in an arrangement as a(offset from the second)
+ b(offset from the third) + c(offset from the largest), 3 x 19 numbers shared by all arrangements, least squares over
arrangement means; share of the arrangement means' variance explained, held out on half the arrangements. Then compare
with the single morphs: share of their variance the arrangement means explain."""
import numpy as np, torch
from morph import d, sm, Lm, Sm, Tm, DEV
pos = torch.arange(10, device=DEV)[None].expand_as(d)
key = (Lm * 100 + Sm * 10 + Tm)
rows, ys, ks = [], [], []
for k in torch.unique(key).tolist():
    sel = key == k
    if sel.sum() < 40: continue
    mean = d[sel].mean(0); l, s, t = k // 100, (k // 10) % 10, k % 10
    for j in range(10):
        if j in (l, s, t): continue
        f = np.zeros(57); f[j - s + 9] = 1; f[19 + j - t + 9] = 1; f[38 + j - l + 9] = 1
        rows.append(f); ys.append(mean[j].item()); ks.append(k)
F, y, ks = np.array(rows), np.array(ys), np.array(ks)
u = np.unique(ks); rng = np.random.default_rng(1); tr = np.isin(ks, rng.choice(u, len(u) // 2, replace=False))
for lab, cols in (("second only", slice(0, 19)), ("second + third", slice(0, 38)), ("second + third + largest", slice(0, 57))):
    coef, *_ = np.linalg.lstsq(F[tr][:, cols], y[tr], rcond=None); p = F[~tr][:, cols] @ coef
    print(f"profiles by offset from {lab:26s}: held-out share of arrangement-mean variance explained {1 - np.var(y[~tr] - p) / np.var(y[~tr]):.2f}")
# how much of single morphs the arrangement means explain
mean_of = {k: d[key == k].mean(0) for k in torch.unique(key).tolist()}
M = torch.stack([mean_of[k] for k in key.tolist()])
print(f"share of single morphs' variance (smaller numbers only) explained by their arrangement's mean morph: {1 - ((d - M)[sm]).var().item() / d[sm].var().item():.2f}")
