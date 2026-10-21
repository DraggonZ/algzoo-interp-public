"""w8b analysis read (for the strict count of jobs 6 and 7): in w7b and w8b,
(i) silence one hiding slot alone one step after the largest (as score_mechanism2's hiding test does for all four):
    how often does the network then name the largest?
(ii) remove the late gate's push on slot 6 (C6) or slot 1 (C1) (its weight from the late gate and the cancel pair
    that gates it): accuracy. Does each push decide anything?"""
import sys
import numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/faithful_twin")
from score_wide import load, Net, lists, DEV  # noqa: E402

def silenced(net, slots):
    X = lists(400_000, 31); L = X.argmax(1); keep = L <= 8; X, L = X[keep], L[keep]
    kill = torch.zeros(len(X), 10, dtype=torch.bool, device=DEV); kill[torch.arange(len(X), device=DEV), L + 1] = True
    Hw, _ = net.run(X, kill=(net.S(slots), kill)); return ((Hw[:, -1] @ net.Wt[2].T).argmax(1) == L).double().mean().item()

def acc(W, M):
    net = Net(W, M); X = lists(400_000, 77); Hw, _ = net.run(X)
    return ((Hw[:, -1] @ net.Wt[2].T).argmax(1) == X.argsort(1, descending=True)[:, 1]).double().mean().item()

for cand in ("candidates/w7b_twopair.py", "candidates/w8b_gapread.py"):
    net = load(cand); import importlib.util as iu
    spec = iu.spec_from_file_location("c", cand); m = iu.module_from_spec(spec); spec.loader.exec_module(m)
    from algzoo.handbuilt_w7b import build_w7b
    idx = build_w7b()[3]
    print(f"== {cand}")
    print("  silence one slot one step after the largest -> names the largest: " +
          "  ".join(f"slot {s}: {100 * silenced(net, [s]):.1f}%" for s in (1, 6, 7, 8)) + f"  all four: {100 * silenced(net, [1, 6, 7, 8]):.1f}%")
    W = [np.array(w, float) for w in m.weights()]
    print(f"  accuracy intact {100 * acc(W, m.MAP):.2f}")
    for nm in ("c6", "c1"):
        W2 = [w.copy() for w in W]; k = idx[nm]
        for src in ("late", "cancel"): W2[1][k, idx[src]] = 0.0
        W2[1][k, idx["detect"]] = W2[1][idx["c7"], idx["detect"]] if nm == "c1" else 0.0   # back to the plain comparator
        print(f"  without the late gate's push on {nm}: accuracy {100 * acc(W2, m.MAP):.2f}")
