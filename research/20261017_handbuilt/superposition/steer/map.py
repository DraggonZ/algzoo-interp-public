"""Sensitivity map: each group at each step, x0.98 and x1.02. Answer changes and accuracy change on random lists,
real against first order. Large gaps between real and first order mark switching."""
from harness import *
X = lists(500_000, seed=1); S0 = run(ARC, X); a0 = S0.argmax(1); k0 = kinds(S0, X)
print(f"base: right {100*k0['right']:.2f}%, mistakes naming largest {100*k0['largest']:.2f}, third {100*k0['third']:.2f}, smaller {100*k0['smaller']:.2f} (points)")
print("group      step  | answers changed % at x0.98 / x1.02: real (first order) | accuracy change, points, x0.98 / x1.02 | largest-mistake change x1.02")
for name, g in GROUPS.items():
    for t in range(10):
        row = []; accs = []
        for f in (0.98, 1.02):
            S = run(ARC, X, [(g, {t}, f)]); F = first_order(ARC, X, [(g, {t}, f)])
            row.append(f"{100*(S.argmax(1) != a0).double().mean().item():5.2f} ({100*(F.argmax(1) != a0).double().mean().item():5.2f})")
            k = kinds(S, X); accs.append(100 * (k["right"] - k0["right"]))
        print(f"{name:10s} {t:3d}   | {row[0]} / {row[1]} | {accs[0]:+6.2f} / {accs[1]:+6.2f} | {100*(k['largest']-k0['largest']):+6.2f}", flush=True)
