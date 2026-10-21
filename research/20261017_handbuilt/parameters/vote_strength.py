"""Per position p of the second-largest: accuracy, the lead (score of p minus best other score), its slope on the
gap x2 - x3 and the scatter around that line, on 2^22 lists. README experiment 13."""
import numpy as np
import torch as th
from algzoo.core import example, forward

dev = "cuda"
P = example()
X = th.randn(2**22, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev, dtype=th.float64)
S = forward(P, X)
srt = X.sort(-1)
p2 = srt.indices[:, -2]; gap = srt.values[:, -2] - srt.values[:, -3]
s_own = S.gather(1, p2[:, None])[:, 0]
S_other = S.clone(); S_other.scatter_(1, p2[:, None], -1e9)
lead = s_own - S_other.max(1).values
rival = S_other.argmax(1)
largest = srt.indices[:, -1]; third = srt.indices[:, -3]
print(" p  lists   acc   mean lead  slope on gap  scatter  slope/scatter  rival is the largest / third / other")
for p in range(10):
    m = p2 == p
    g, l = gap[m].cpu().numpy(), lead[m].cpu().numpy()
    A = np.stack([g, np.ones_like(g)], 1); co, *_ = np.linalg.lstsq(A, l, rcond=None)
    sc = (l - A @ co).std()
    wrong = m & (lead < 0)
    nw = wrong.sum().item()
    rl = (rival[wrong] == largest[wrong]).sum().item() / max(nw, 1); rt = (rival[wrong] == third[wrong]).sum().item() / max(nw, 1)
    print(f"{p:2d} {m.sum().item():7d} {1 - nw / m.sum().item():.4f} {l.mean():9.2f} {co[0]:12.2f} {sc:8.2f} {co[0] / sc:12.3f}"
          f"   errors: {100 * rl:4.0f}% / {100 * rt:4.0f}% / {100 * (1 - rl - rt):4.0f}%")

print("\nsplit by situation: revealed = the largest comes after p (the second was an overtaken record); kept = largest before p")
print(" p  situation  lists   acc   slope on gap  scatter  slope/scatter")
for p in range(10):
    for lab, cond in [("revealed", largest > p2), ("kept", largest < p2)]:
        m = (p2 == p) & cond
        n = m.sum().item()
        if n < 20000:
            continue
        g, l = gap[m].cpu().numpy(), lead[m].cpu().numpy()
        A = np.stack([g, np.ones_like(g)], 1); co, *_ = np.linalg.lstsq(A, l, rcond=None)
        sc = (l - A @ co).std()
        print(f"{p:2d}  {lab:9s} {n:7d} {(l > 0).mean():.4f} {co[0]:12.2f} {sc:8.2f} {co[0] / sc:12.3f}")

print("\nby the vote's age at the end: kept = 9 - p; revealed = 9 - position of the largest")
print(" age  situation  lists   acc   slope on gap  scatter  slope/scatter")
for age in range(10):
    for lab, m in [("revealed", (largest > p2) & (9 - largest == age)), ("kept", (largest < p2) & (9 - p2 == age))]:
        n = m.sum().item()
        if n < 20000:
            continue
        g, l = gap[m].cpu().numpy(), lead[m].cpu().numpy()
        A = np.stack([g, np.ones_like(g)], 1); co, *_ = np.linalg.lstsq(A, l, rcond=None)
        sc = (l - A @ co).std()
        print(f"{age:3d}  {lab:9s} {n:7d} {(l > 0).mean():.4f} {co[0]:12.2f} {sc:8.2f} {co[0] / sc:12.3f}")
