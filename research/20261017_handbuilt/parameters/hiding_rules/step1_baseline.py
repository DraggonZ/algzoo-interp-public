"""Reproduce the account's errors, then split each hiding neuron's input by source group and ask how well the
account's features explain each part."""
import numpy as np
from harness import *  # noqa
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/gao")
from sim_groups import account_features

print("account's formulas (should match gao/sim_tolerance.log):")
evaluate(account_features)
G = {"input": None, "store": [2], "detector": [4], "keep gate": [5], "late gate": [9], "hiding": [1, 6, 7, 8],
     "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
Hprev = np.concatenate([np.zeros((len(Xs), 1, 16)), Hs[:, :-1]], 1)
for n in HID:
    F = account_features(Xs, n); A = F.reshape(-1, F.shape[-1])
    print(f"\nneuron {n}: parts of its input (rms over all steps), and how much of each part the account's features explain")
    for g, src in G.items():
        part = Xs * simlib.WI[n] if src is None else np.einsum("btk,k->bt", Hprev[:, :, src], simlib.WH[n, src])
        co = np.linalg.lstsq(A, part.reshape(-1), rcond=None)[0]
        res = part.reshape(-1) - A @ co
        print(f"  {g:10s} rms {np.sqrt((part ** 2).mean()):7.3f}   unexplained rms {np.sqrt((res ** 2).mean()):6.3f}")
