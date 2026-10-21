"""Cancelling neurons (1, 6, 7, 8) as a ReLU block on few inputs. At every step their pre-activation is rebuilt from
chosen sources: the input x, the store (2), the detector (4), their own block, the memory (0, 3, 10-15), the gates
(5, 9). Dropped sources are replaced by zero or by their average contribution at that step (300,000 random lists).
Optionally the store and detector values they read come from short rules (store: 17.6 x the largest by age table plus
the previous largest, ../hilton/topdown/store_short.py medians; detector: 10.16 (x - largest)+). The rest of the
network runs on its real weights and sees the rebuilt values. Scored by ARC's mistakes reproduced."""
import numpy as np, torch
from common import *
import score
wi, wh, wo = ARC
CAN = [1, 6, 7, 8]; GROUPS = {"store": [2], "detector": [4], "self": CAN, "memory": [0, 3, 10, 11, 12, 13, 14, 15], "gates": [5, 9]}
# per-step average contribution of each group to the cancellers' pre-activation
_X = torch.randn(300000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(11)); _h = torch.zeros(len(_X), 16, device=DEV, dtype=dt)
MEAN = {g: [] for g in GROUPS}
for t in range(10):
    for g, idx in GROUPS.items(): MEAN[g].append((_h[:, idx] @ wh[CAN][:, idx].T).mean(0))
    _h = torch.relu(_h @ wh.T + _X[:, t:t + 1] * wi)
def runner(keep, drop="zero"):
    def f(X):
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
        for t in range(10):
            pre = h @ wh.T + X[:, t:t + 1] * wi
            pc = X[:, t:t + 1] * wi[CAN]
            for g, idx in GROUPS.items():
                if g in keep: pc = pc + h[:, idx] @ wh[CAN][:, idx].T
                elif drop == "mean": pc = pc + MEAN[g][t]
            pre[:, CAN] = pc; h = torch.relu(pre)
        return h @ wo.T
    return f
if __name__ == "__main__":
    allg = set(GROUPS)
    models = {"real (all sources)": [runner(allg)],
              "drop gates (zero)": [runner(allg - {"gates"})], "drop gates (average)": [runner(allg - {"gates"}, "mean")],
              "drop memory (zero)": [runner(allg - {"memory"})], "drop memory (average)": [runner(allg - {"memory"}, "mean")],
              "x, store, detector, self only (avg)": [runner({"store", "detector", "self"}, "mean")],
              "x, store, self only (avg)": [runner({"store", "self"}, "mean")]}
    score.report(models, family=False)
