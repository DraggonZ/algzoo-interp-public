"""Are tie offsets (scripts/tie_bias_stability.py: where the network switches between second and third for each of
720 placements of the top three) more sensitive to 0.5% weight noise in networks that cancel more? 22 networks,
3 noisy copies each. Cancellation = median lead/pushes from robust_across.log."""
import itertools, re
from pathlib import Path
import numpy as np
import torch as th
from scipy.stats import spearmanr
from algzoo.core import example, forward, zoo, load

pl = np.array(list(itertools.permutations(range(10), 3))); N = len(pl); r = np.arange(N)


def bias(P, m1=1.6, bg=-0.46):
    base = np.full((N, 10), bg); base[r, pl[:, 0]] = m1; base[r, pl[:, 1]] = 1.0
    lo, hi = np.full(N, 0.8), np.full(N, 1.2)
    for _ in range(30):
        mid = (lo + hi) / 2; x = base.copy(); x[r, pl[:, 2]] = mid
        s = forward(P, th.tensor(x, device="cuda")).cpu().numpy()
        tw = s[r, pl[:, 2]] > s[r, pl[:, 1]]; hi = np.where(tw, mid, hi); lo = np.where(tw, lo, mid)
    return (lo + hi) / 2 - 1


mp = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\S+)\s+acc [0-9.]+ cutoff [0-9.]+ margin/pushes ([0-9.]+)", open("robust_across.log").read(), re.M)}
models = {"ARC": example()}
for s in range(5):
    models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
rng = np.random.default_rng(0); rows = []
for name, P in models.items():
    B0 = bias(P); ch, cs = [], []
    for _ in range(3):
        Q = tuple(w * (1 + 0.005 * th.as_tensor(rng.standard_normal(tuple(w.shape)), dtype=w.dtype, device=w.device)) if th.is_tensor(w)
                  else w * (1 + 0.005 * rng.standard_normal(w.shape)) for w in P)
        Bq = bias(Q); ch.append(np.median(np.abs(Bq - B0))); cs.append(np.corrcoef(B0, Bq)[0, 1])
    rows.append((name, mp[name], np.median(np.abs(B0)), np.mean(ch), np.mean(cs)))
    print(f"{name:7s} lead/pushes {mp[name]:.3f}; typical |offset| {rows[-1][2]:.4f}; change at 0.5% weight noise {rows[-1][3]:.4f}; pattern correlation {rows[-1][4]:.2f}", flush=True)
R = np.array([x[1:] for x in rows])
for j, lab in ((1, "typical |offset|"), (2, "offset change at 0.5% noise"), (3, "pattern correlation")):
    rho, p = spearmanr(R[:, 0], R[:, j]); print(f"rank correlation, lead/pushes vs {lab}: {rho:.2f} (p {p:.3f})")
