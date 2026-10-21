"""Step 3a: compress the order rule. (a) cells keyed by the three ages without the step; (b) one small decision tree
per neuron on (step, age of the largest, of the second, of the third), with at most L leaves, fitted to the network's
own on/off on seed 32 (each list and step one sample), applied closed loop on seed 31. Fitted numbers of a tree with
L leaves: L-1 splits (feature and threshold, 2 numbers each) plus L on/off leaves."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from swlib import *

X = lists(TEST_SEED, 200_000); K = kinds(X)
S0, P0, H0 = run(X); base = score(X, S0, S0, K); true_off = P0[:, :, MEM] <= 0
Xf = lists(FIT_SEED, 200_000); _, Pf, _ = run(Xf); off_f = Pf[:, :, MEM] <= 0
ag, _ = order_keys(X); agf, _ = order_keys(Xf)
T = np.broadcast_to(np.arange(10)[None, :, None], ag.shape[:2] + (1,))
F = np.concatenate([T, ag], 2).reshape(-1, 4); Ff = np.concatenate([np.broadcast_to(np.arange(10)[None, :, None], agf.shape[:2] + (1,)), agf], 2).reshape(-1, 4)
NAMES = ["step", "age1", "age2", "age3"]

# (a) no step
tab = {}; kf = [tuple(r) for r in Ff[:, 1:]]; d = {}
for i, k in enumerate(kf): d.setdefault(k, []).append(i)
of = off_f.reshape(-1, 8)
for k, idx in d.items():
    if len(idx) >= 30: tab[k] = of[idx].mean(0) > 0.5
kt = [tuple(r) for r in F[:, 1:]]
pred = np.array([tab.get(k, np.zeros(8, bool)) for k in kt]).reshape(len(X), 10, 8)
S, offs = machine(X, lambda t, c: pred[:, t]); report(f"cells of three ages, no step ({len(tab)} cells)", X, S0, S, offs, true_off, base)

for L in [2, 3, 4, 6, 8, 12, 16, 32]:
    pred = np.zeros((len(X) * 10, 8), bool); trees = []
    for j in range(8):
        tr = DecisionTreeClassifier(max_leaf_nodes=L, random_state=0).fit(Ff, of[:, j]); trees.append(tr)
        pred[:, j] = tr.predict(F)
    pred = pred.reshape(len(X), 10, 8)
    nfit = sum(3 * tr.get_n_leaves() - 2 for tr in trees)
    S, offs = machine(X, lambda t, c: pred[:, t]); report(f"trees, at most {L} leaves per neuron", X, S0, S, offs, true_off, base, nfit=nfit)
    if L in (4, 8):
        for j, tr in enumerate(trees):
            print(f"  -- neuron {MEM[j]}:\n" + "\n".join("     " + l for l in export_text(tr, feature_names=NAMES, class_names=["on", "off"]).splitlines()))
