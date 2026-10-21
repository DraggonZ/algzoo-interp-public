"""Which lists need neuron 0's (and neuron 15's) switching? Accuracy by kind of list with its ReLU removed.
Kinds: second-largest is an overtaken record (it was the largest when it arrived, and the largest came later);
second came after the largest (kept number); second came before the largest but was never a record."""
import numpy as np
from simlib import lists, run
from linearize import run_lin
X = lists(202); L = run(X)[0]; o = X.argsort(1); sec, lar = o[:, -2], o[:, -1]; B = len(X)
rec = np.array([X[np.arange(B), i] > np.max(np.c_[np.zeros(B), X[:, :i]], 1) for i in range(10)]).T
kind = np.where(sec > lar, "second after largest", np.where(rec[np.arange(B), sec], "second an overtaken record", "second before largest, never a record"))
for lab, lin in (("none", []), ("neuron 0", [0]), ("neuron 15", [15]), ("memory except 0, 15", [3, 10, 11, 12, 13, 14])):
    Lx = run_lin(X, lin) if lin else L
    print(f"no ReLU in {lab:20s}: " + "; ".join(f"{k} ({(kind == k).mean():.0%}) {(Lx.argmax(1)[kind == k] == sec[kind == k]).mean():.3f}" for k in np.unique(kind)))
