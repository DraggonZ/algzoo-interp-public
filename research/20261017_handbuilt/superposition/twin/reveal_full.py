"""Does ARC reveal an overtaken largest fully? On lists where the second was a new largest when it arrived (so it was
hidden, then revealed when the largest overtook it), the exact lead map of second over third: the coefficient on the
second itself and on the number it overtook (the record before it). Full reveal: about the same coefficient on the
second as for a second that never was a largest, and near zero on the record before it. If part of the hiding stays
in memory, the second scores partly by its predecessor: a lower coefficient on itself and a positive one on the
predecessor. Same for v23 with ARC's settings."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
import torch
from common import DEV, dt, jac, top3, T, ARC
from algzoo.handbuilt_v23 import build_v23
g = torch.Generator(device=DEV).manual_seed(81); X = torch.randn(300_000, 10, device=DEV, dtype=dt, generator=g); L, S, T3 = top3(X); n = torch.arange(len(X), device=DEV)
run_max = torch.cat([torch.zeros(len(X), 1, device=DEV, dtype=dt), X.cummax(1).values[:, :-1]], 1).clamp(min=0)
srec = X[n, S] > run_max[n, S]
# position of the record the second overtook: the argmax of numbers before it (if that was positive)
before = torch.where(torch.arange(10, device=DEV)[None] < S[:, None], X, torch.full_like(X, -1e9)); pp = before.argmax(1); has_pp = srec & (S >= 1) & (X[n, pp] > 0)
for name, W in (("ARC", ARC), ("v23 ARC settings", T(build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False).params))):
    J = jac(W, X); d = J[n, S] - J[n, T3]
    print(f"{name}: coefficient of the lead on the second itself, median: second never a largest {d[n, S][~srec].median():.3f}, second an overtaken largest {d[n, S][has_pp].median():.3f}; on the record it overtook {d[n, pp][has_pp].median():+.3f}")
