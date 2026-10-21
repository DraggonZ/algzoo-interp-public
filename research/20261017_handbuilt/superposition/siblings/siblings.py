"""Are dense close calls the rule for this training recipe, or is ARC unlucky? For every trained network we have (16,
20 and 32 neurons) and the hand-built ones: accuracy, share of near ties where it names the third, lead maps at 3%
precision covering half of 100,000 lists, and how much a tree on the top three (positions and two ratios) beats
always-guessing on near ties (the share of the possible gain it reaches: (tree - guess) / (100 - guess)). Prediction
(written first): all trained 16-neuron networks are dense like ARC; the 32-neuron ones, more accurate, may be cleaner."""
import sys; sys.path.insert(0, "../norule"); sys.path.insert(0, "../../family")
import numpy as np, torch
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from nr_lib import net_paths
from common import DEV, dt, jac, top3, scores, T
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v21 import build_v21
paths = net_paths(); paths["hand-built v19"] = lambda: T(build_v19(kappa=0.5).params); paths["hand-built v21"] = lambda: T(build_v21())
g = torch.Generator(device=DEV).manual_seed(7); X = torch.randn(100_000, 10, device=DEV, dtype=dt, generator=g); l, s, t = top3(X); n = torch.arange(len(X), device=DEV)
G = torch.Generator(device=DEV).manual_seed(101); Y = torch.randn(3_000_000, 10, device=DEV, dtype=dt, generator=G); o = Y.argsort(1, descending=True); vv = Y.sort(1, descending=True).values
near = (vv[:, 1] > 0) & ((vv[:, 1] - vv[:, 2]) < 0.05 * vv[:, 1]); Y, o, vv = Y[near][:250000], o[near][:250000], vv[near][:250000]
F = np.c_[o[:, :3].cpu().numpy().astype(float), (vv[:, 2] / vv[:, 1]).cpu().numpy(), (vv[:, 0] / vv[:, 1]).cpu().numpy()]
rows = []
for name, fn in paths.items():
    W = [w.contiguous() for w in fn()]
    acc = (scores(W, X).argmax(1) == s).double().mean().item()
    J = jac(W, X); d = J[n, s] - J[n, t]; d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-30)
    c = Counter(map(bytes, torch.round(d / 0.03).to(torch.int64).cpu().numpy())); cnt = np.array(sorted(c.values(), reverse=True)); half = np.searchsorted(np.cumsum(cnt) / len(X), 0.5) + 1
    y = (scores(W, Y).argmax(1) == o[:, 2]).cpu().numpy(); guess = max(y.mean(), 1 - y.mean())
    clf = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63).fit(F[:200000], y[:200000]); tr = (clf.predict(F[200000:]) == y[200000:]).mean()
    gain = (tr - guess) / max(1 - guess, 1e-9)
    rows.append((name, acc, y.mean(), half, gain))
    print(f"{name:16s} accuracy {100*acc:5.2f}% | names the third on {100*y.mean():4.1f}% of near ties | lead maps covering half {half:6,} | tree reaches {100*gain:3.0f}% of the possible gain over guessing", flush=True)
np.save("siblings.npy", np.array([(r[1], r[2], r[3], r[4]) for r in rows]))
tr16 = [r for r in rows if not r[0].startswith(("hand", "20", "32", "zoo32"))]
print(f"trained 16-neuron networks ({len(tr16)}): lead maps covering half {min(r[3] for r in tr16):,}-{max(r[3] for r in tr16):,}, tree gain {100*min(r[4] for r in tr16):.0f}-{100*max(r[4] for r in tr16):.0f}%; correlation of tree gain with accuracy {np.corrcoef([r[1] for r in tr16], [r[4] for r in tr16])[0,1]:.2f}")
