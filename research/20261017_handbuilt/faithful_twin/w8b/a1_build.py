"""w8b analysis read: the rebuilt clock reads, and how far w8b's function is from w7b's (accuracy, answers that differ,
in which case they differ). Not a candidate score."""
import numpy as np
from algzoo.handbuilt_w7b import build_w7b
from algzoo.handbuilt_w8b import build_w8b, special_parts

A = build_w7b(); B = build_w8b(); idx = A[3]; name = {v: k for k, v in idx.items()}
print("special parts (age1, age2, wi, w_store) relative to C7:")
for k, v in special_parts(B[0], B[1], idx).items(): print(f"  {k:3s} " + " ".join(f"{x:+.4g}" for x in v))
print("reads of (h7, c1, c6, c7): w7b -> w8b")
for r, (w, abg) in B[4].items(): print(f"  {name[r]:7s} {np.array2string(w, precision=6)} -> {np.array2string(abg, precision=6)} 0")

rng = np.random.default_rng(3); n = 400_000; x = rng.standard_normal((n, 10))
def run(W):
    wi, wh, wo = W[:3]; h = np.zeros((n, len(wi))); P = []
    for t in range(10):
        p = h @ wh.T + x[:, t:t + 1] * wi; P.append(p); h = np.maximum(p, 0)
    return (h @ wo.T).argmax(1), np.stack(P, 1)
ya, Pa = run(A); yb, Pb = run(B); true = np.argsort(-x, 1)[:, 1]
print(f"accuracy w7b {100 * (ya == true).mean():.2f}  w8b {100 * (yb == true).mean():.2f}  answers differ {100 * (ya != yb).mean():.2f}%")
print(f"  fixed {100 * ((ya != true) & (yb == true)).mean():.2f}%  broken {100 * ((ya == true) & (yb != true)).mean():.2f}%")
# where the clocks' pre-activations differ: by age of the largest and whether the number after the record is the new second
cl = [idx[f"clock{k}"] for k in range(3)]
dif = np.abs(Pa[:, :, cl] - Pb[:, :, cl]).max(2)
m = np.maximum.accumulate(x, 1); age = np.zeros((n, 10), int)
for t in range(1, 10): age[:, t] = np.where(x[:, t] >= m[:, t - 1], 0, age[:, t - 1] + 1)
for a in range(0, 5):
    sel = age[:, 1:] == a; d = dif[:, 1:][sel]
    print(f"  age {a} (of the largest at the step the clocks read): clock pre-activation change median {np.median(d):.2e}, share > 1e-3: {100 * (d > 1e-3).mean():.1f}%")
