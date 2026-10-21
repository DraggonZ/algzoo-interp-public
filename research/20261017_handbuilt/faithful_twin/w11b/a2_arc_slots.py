"""w11b analysis read: in ARC itself, silence one hiding neuron (1, 6, 7, 8) one step after the largest (as
score_mechanism2's hiding test does for all four): how often does ARC then name the largest? Sets the bar for
"slot 8 hides" in the strict count. Also pairs, and ARC's neuron 8 silenced at age 1 vs at every age >= 1."""
import sys
import numpy as np, torch
FT = "/data/output/src/algzoo/research/20261017_handbuilt/faithful_twin"
sys.path.insert(0, FT)
from score_wide import load, lists, DEV  # noqa: E402

X = lists(400_000, 31); L = X.argmax(1); k_ = L <= 8; X, L = X[k_], L[k_]


def silenced(net, slots, every=False):
    kill = torch.zeros(len(X), 10, dtype=torch.bool, device=DEV); kill[torch.arange(len(X), device=DEV), L + 1] = True
    if every:
        steps = torch.arange(10, device=DEV)[None, :]; kill = steps >= (L[:, None] + 1)
    Hw, _ = net.run(X, kill=(net.S(slots), kill)); return 100 * ((Hw[:, -1] @ net.Wt[2].T).argmax(1) == L).double().mean().item()


for cand in ("ARC", "candidates/w8b_gapread.py"):
    net = load(cand); print(f"== {cand}")
    print("  one step after: " + "  ".join(f"{s}: {silenced(net, [s]):.1f}%" for s in (1, 6, 7, 8)) +
          f"  all four: {silenced(net, [1, 6, 7, 8]):.1f}%")
    print("  three of four (leave one in): " + "  ".join(f"without {s}: {silenced(net, [t for t in (1, 6, 7, 8) if t != s]):.1f}%" for s in (1, 6, 7, 8)))
    print("  every step from one after: " + "  ".join(f"{s}: {silenced(net, [s], every=True):.1f}%" for s in (1, 6, 7, 8)))
