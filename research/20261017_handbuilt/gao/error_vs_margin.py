"""Do wrong and slightly-off accounts fall on one curve once the size of their output error is measured against the
decision margin? For each account: per list, the largest change in any of the ten output scores divided by that list's
gap between its top two true scores; summarized by the median over lists. Twin accounts from twin_graded.py runs saved
with SAVE (exact rules plus noise, conceptual misreadings with and without noise, slightly-off rules with noise);
trained network: its true values plus noise (as noise_tolerance.py) and scaled by 1 + e (as scale_error.py).
Args: the two saved .npz files. Writes error_vs_margin.json for the figure."""
import sys, json, re
import numpy as np
import simlib


def summary(X, L, Ls):
    srt = np.sort(L, 1); margin = srt[:, -1] - srt[:, -2]
    r = np.abs(Ls - L).max(1) / np.maximum(margin, 1e-12)
    return float(np.median(r)), float((Ls.argmax(1) == X.argsort(1)[:, -2]).mean())


def kind(label):
    if label.startswith("exact"):
        return "exact rules + noise"
    if label.startswith(("wrong", "flip")):
        return "conceptual misreading"
    return "right idea, one constant off"


rows = []
for f in sys.argv[1:3]:
    D = np.load(f); X, L = D["X"], D["L"]
    for k in D.files:
        if k in ("X", "L"):
            continue
        r, a = summary(X, L, D[k]); rows.append({"net": "twin", "label": k, "kind": kind(k), "err": r, "acc": a})
X = simlib.lists(808, 20_000); L = simlib.run(X)[0]; rng = np.random.default_rng(0)
_, _, H = simlib.run(X); typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) for n in range(16)])
for s in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0); h = np.maximum(h + s * typ * rng.standard_normal(h.shape) * (h > 0), 0)
    r, a = summary(X, L, h @ simlib.WO.T); rows.append({"net": "trained", "label": f"true values + noise {s:g}", "kind": "trained: true values + noise", "err": r, "acc": a})
for e in (0.01, 0.02, 0.05, 0.1):
    for ns, lab in (([7], "neuron 7"), (list(range(16)), "all 16")):
        h = np.zeros((len(X), 16)); fct = np.ones(16); fct[ns] = 1 + e
        for t in range(10):
            h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0) * fct
        r, a = summary(X, L, h @ simlib.WO.T); rows.append({"net": "trained", "label": f"{lab} scaled {1 + e:.2f}", "kind": "trained: true values scaled", "err": r, "acc": a})
for row in sorted(rows, key=lambda r: (r["net"], r["err"])):
    print(f"{row['net']:7s} {row['kind']:30s} {row['label']:42s} median error/margin {row['err']:10.3g}  accuracy {row['acc']:.3f}")
json.dump(rows, open("error_vs_margin.json", "w"), indent=1)
