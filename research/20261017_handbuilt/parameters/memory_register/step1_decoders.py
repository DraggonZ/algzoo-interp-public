"""Step 1: compare the two decoders. Prints example lists, then agreement statistics."""
import numpy as np
from decoders import *
np.set_printoptions(precision=1, suppress=True, linewidth=220)
rng = np.random.default_rng(3)
X = rng.standard_normal((100_000, 10)); H = run(X)
on, frac = common_pattern(H)
print(f"most common on/off pattern at steps 4-8: {100*frac:.1f}% of steps; off: {np.where(~on)[0].tolist()}")
allon = np.ones(16, bool)
for i in range(3):
    x = X[i]; print(f"\n=== list {i}: {np.round(x,2)}  answer {np.argsort(x)[-2]}  largest {np.argmax(x)}")
    for t in range(1, 11):
        C = zero_split(H[i:i+1, t], t)[0]
        g = {k: C[v].sum(0) for k, v in GROUPS.items()}
        pb = pushback(t, on, MEM) @ H[i, t, MEM]
        print(f" t={t:2d} total {C.sum(0)}\n       memory{g['memory']}\n       hiding{g['hiding']}\n       store {g['store']}  gates{g['gates']}\n       pushback(mem){pb}")

def corr(a, b):  # per-list Pearson over 10 slots, averaged; and global relative error
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-30)
    return np.nanmean(r), np.linalg.norm(a - b) / np.linalg.norm(a)

print("\nagreement of pushback with the zero continuation's memory share (mean per-list correlation over slots; relative error)")
n = 20000
for t in range(1, 11):
    C = zero_split(H[:n, t], t); zm = C[:, MEM].sum(1); zmh = C[:, MEM + HID].sum(1); ztot = C.sum(1)
    row = []
    for name, P, blk, tgt in [("mem/common", pushback(t, on, MEM), MEM, zm), ("mem/all-on", pushback(t, allon, MEM), MEM, zm),
                              ("mem+hid/common", pushback(t, on, MEM + HID), MEM + HID, zmh),
                              ("full/common, mem cols", pushback_full(t, on)[:, MEM], MEM, zm)]:
        c, e = corr(tgt, H[:n, t][:, blk] @ P.T)
        row.append(f"{name} r={c:.2f} err={e:.2f}")
    share = np.linalg.norm(zm) / np.linalg.norm(ztot)
    print(f" t={t:2d} memory share of total size {share:.2f} | " + " | ".join(row))
