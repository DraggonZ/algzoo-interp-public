"""ARC's mistakes that a generic imprecise solver never makes: naming a position that holds none of the three largest
('another'), and naming the third when it is more than 10% below the second. Where are they? Probability-weighted
shares by the named position, by its offset from the largest's position, and by the order of the three largest."""
import numpy as np, torch
from collections import Counter
import store_ladder as sl
DEV = "cuda"; dt = torch.float64
if __name__ == "__main__":
    nV = len(sl.wtt); rows = []
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = sl.run(X).argmax(1)
        gap = (val[:, 1] - val[:, 2]) / val[:, 1].abs()
        oth = (a0 != pos[:, 0]) & (a0 != pos[:, 1]) & (a0 != pos[:, 2]); thr = (a0 == pos[:, 2]) & (gap > 0.1)
        for nm, m in (("another", oth), ("third, gap > 10%", thr)):
            if m.any(): rows.append((nm, a0[m].cpu().numpy(), pos[m].cpu().numpy(), w[m].cpu().numpy(), val[m].cpu().numpy()))
    for nm in ("another", "third, gap > 10%"):
        a = np.concatenate([r[1] for r in rows if r[0] == nm]); p = np.concatenate([r[2] for r in rows if r[0] == nm])
        w = np.concatenate([r[3] for r in rows if r[0] == nm]); v = np.concatenate([r[4] for r in rows if r[0] == nm]); W = w.sum()
        def top(keys, k=8):
            c = Counter(); [c.__setitem__(key, c[key] + wi) for key, wi in zip(keys, w)]
            return ", ".join(f"{key}: {100*x/W:.0f}%" for key, x in c.most_common(k))
        print(f"== ARC names {nm}")
        print("  named position:", top(a.tolist()))
        print("  named minus largest's position:", top((a - p[:, 0]).tolist()))
        print("  order of arrival (L=largest, S=second, T=third):", top(["".join("LST"[j] for j in np.argsort(pp)) for pp in p]))
        print("  largest at step:", top(p[:, 0].tolist()), "| second at step:", top(p[:, 1].tolist(), 5))
        print(f"  median values: largest {np.median(v[:,0]):.2f}, second {np.median(v[:,1]):.2f}, third {np.median(v[:,2]):.2f}")
