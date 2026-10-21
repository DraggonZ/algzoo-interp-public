"""Sharper test: for each rotating eigenpair (turn theta, shrink r) of D W_hh (D = most common on/off pattern at
steps 4-8), read its output phases c_k = (W_oh[k] - mean) . v for positions 0-8. A dial for this pair needs the
phases to advance by theta per position, in a straight line. Fit phase(k) = phi + s * k (circular, grid search
over s), and report s, theta and the fit (mean cos of residuals, 1 = perfect). Lists pairs with r > 0.5."""
from collections import Counter
from pathlib import Path
import numpy as np
from algzoo.core import example, load, zoo

models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["s" + f.stem.split("_")[-1]] = load(str(f))
X = np.random.default_rng(17).standard_normal((100_000, 10))
slopes = np.radians(np.arange(-180, 180, 0.5))
summary = []
for name, P in models.items():
    wi, wh, wo = (np.asarray(w, float) for w in P); wi = wi.reshape(-1)
    h = np.zeros((len(X), len(wi))); masks = []
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); masks.append(h > 0)
    M = np.concatenate(masks[4:9]); top, k = Counter(map(lambda r: r.tobytes(), M)).most_common(1)[0]
    D = np.frombuffer(top, dtype=bool).astype(float)
    ev, V = np.linalg.eig(np.diag(D) @ wh)
    W = wo - wo.mean(0)
    out = []
    for i in np.flatnonzero((ev.imag > 1e-6) & (np.abs(ev) > 0.5)):
        c = (W @ V[:, i])[:9]; ph = np.angle(c); wgt = np.abs(c) / np.abs(c).sum()
        best = max(((np.abs((wgt * np.exp(1j * (ph - s * np.arange(9)))).sum()), s) for s in slopes))
        theta = np.degrees(np.angle(ev[i]))
        out.append((best[0], np.degrees(best[1]), theta, abs(ev[i])))
    out.sort(reverse=True)
    line = "; ".join(f"turn {th:5.1f} shrink {r:.2f}: spokes step {s:6.1f}, fit {f:.2f}" for f, s, th, r in out[:3])
    print(f"{name:6s} {line}")
    # does the best-fitting pair's spoke step match its own turn (spokes step -theta or +theta depending on convention)?
    if out:
        f, s, th, r = out[0]
        summary.append((name, th, r, s, f))
print("\nbest-fitting pair per network: |spoke step| vs |turn| (deg), fit")
for name, th, r, s, f in summary:
    print(f"  {name:6s} turn {abs(th):5.1f}  spokes {abs(s):5.1f}  shrink {r:.2f}  fit {f:.2f}")
