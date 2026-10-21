"""Sign test: do the account's neuron roles predict the signs of ARC's weights? Rules are stated in README.md
(experiment 7) for whole groups and applied here mechanically. Importance: census 1% nudge sensitivity (mean KL)."""
import csv
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(p, float) for p in example())
hiding = [1, 6, 7, 8]
pred = {}  # (source, target) -> +1 / -1; source 'input' for input weights
for j in hiding:
    pred[("input", j)] = -1
    pred[(2, j)] = +1
for j in [7, 1]:
    pred[(4, j)] = +1
for j in [1, 6]:
    pred[(9, j)] = -1
pred[(2, 2)] = +1; pred[(4, 2)] = +1
pred[("input", 4)] = +1; pred[(2, 4)] = -1; pred[(4, 4)] = -1
for i in hiding:
    pred[(i, 5)] = -1; pred[(i, 9)] = -1
pred[(4, 5)] = +1; pred[(4, 9)] = +1
pred[(9, 0)] = -1

census = {r["weight"]: r for r in csv.DictReader(open("census.csv"))}
imp = {k: 0.5 * (float(r["up1_kl"]) + float(r["down1_kl"])) for k, r in census.items()}
total = sum(imp.values())
right = wrong = 0; imp_pred = 0
print("predicted sign, ARC's weight, nudge sensitivity (share of total), zeroing cost (points)")
for (i, j), s in pred.items():
    name = f"input->{j}" if i == "input" else f"{i}->{j}"
    w = wi[j] if i == "input" else wh[j, i]
    ok = np.sign(w) == s
    right += ok; wrong += not ok; imp_pred += imp[name]
    print(f"  {name:9s} predicted {'+' if s > 0 else '-'}  ARC {w:+7.3f}  {'right' if ok else 'WRONG'}  "
          f"{100 * imp[name] / total:5.2f}%  {-100 * float(census[name]['zero_dacc']):6.2f}")
print(f"\n{len(pred)} predicted signs ({100 * len(pred) / 432:.1f}% of the 432 weights): {right} right, {wrong} wrong")
print(f"share of summed 1% nudge sensitivity in predicted weights: {100 * imp_pred / total:.1f}%")
# where the rest of the sensitivity sits
groups = {"store/det": [2, 4], "gates": [5, 9], "hiding": hiding, "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
g = lambda n: next(k for k, v in groups.items() if n in v)
rest = {}
for k, v in imp.items():
    src, dst = k.split("->")
    key_ = ("input" if src == "input" else g(int(src))) + " -> " + ("readout" if dst.startswith("out") else g(int(dst)))
    sname = None
    if dst.startswith("out"):
        pass
    elif (("input" if src == "input" else int(src)), int(dst)) in pred:
        continue
    rest[key_] = rest.get(key_, 0) + v
print("unpredicted sensitivity by source -> target group (share of total):")
for k, v in sorted(rest.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {k:24s} {100 * v / total:5.1f}%")
