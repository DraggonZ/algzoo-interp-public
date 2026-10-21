"""Is the trained timer a fading shift register? (1) All eigenvalues of D W_hh for ARC's most common pattern.
(2) Are the output rows propagated by the dynamics: cosine between W_oh[k] and W_oh[k+1] @ M (and the reverse),
rows centred over positions, for every network; baseline: cosine between W_oh[k] and W_oh[k+1] itself."""
from collections import Counter
from pathlib import Path
import numpy as np
from algzoo.core import example, load, zoo

models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["s" + f.stem.split("_")[-1]] = load(str(f))
X = np.random.default_rng(17).standard_normal((100_000, 10))
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
for name, P in models.items():
    wi, wh, wo = (np.asarray(w, float) for w in P); wi = wi.reshape(-1)
    h = np.zeros((len(X), len(wi))); masks = []
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); masks.append(h > 0)
    top, _ = Counter(map(lambda r: r.tobytes(), np.concatenate(masks[4:9]))).most_common(1)[0]
    D = np.diag(np.frombuffer(top, dtype=bool).astype(float)); M = D @ wh
    W = wo - wo.mean(0)
    back = [cos(W[k], W[k + 1] @ M) for k in range(0, 8)]
    fwd = [cos(W[k + 1], W[k] @ M) for k in range(0, 8)]
    base = [cos(W[k], W[k + 1]) for k in range(0, 8)]
    if name == "ARC":
        ev = np.linalg.eigvals(M); ev = ev[np.argsort(-np.abs(ev))]
        print("ARC eigenvalues (modulus, angle in degrees):", ", ".join(f"{abs(e):.2f} {np.degrees(np.angle(e)):+.0f}" for e in ev if abs(e) > 0.3))
    print(f"{name:6s} cos(W[k], W[k+1] M) mean {np.mean(back):+.2f} (min {np.min(back):+.2f}) | W[k+1] vs W[k] M {np.mean(fwd):+.2f} | plain neighbours {np.mean(base):+.2f}")
