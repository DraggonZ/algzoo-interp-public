"""Which memory switches matter? ARC with each memory neuron in one of three forms: 'linear' (ReLU removed: activation =
pre-activation), 'rule' (pre-activation where its simple rule says on, else 0: neuron 0 off at age 2; 10 off at age 6;
11 off at age 5; ages are steps since the largest arrived), 'real' (its ReLU). Everything else real. Ladder scored by
accuracy and ARC's mistakes reproduced (grid, ../../family/score.py)."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
import score
MEM = [0, 3, 10, 11, 12, 13, 14, 15]; RULE_AGE = {0: 2, 10: 6, 11: 5}
def runner(forms):
    wi, wh, wo = ARC
    def f(X):
        n = len(X); h = torch.zeros(n, 16, device=DEV, dtype=dt); m = torch.zeros(n, device=DEV, dtype=dt); Lp = torch.full((n,), -1, device=DEV, dtype=torch.long)
        for t in range(10):
            new = X[:, t] > m; Lp = torch.where(new, t, Lp); m = torch.where(new, X[:, t], m); age = torch.where(Lp >= 0, t - Lp, -1)
            z = h @ wh.T + X[:, t:t + 1] * wi; hn = torch.relu(z)
            for j, form in forms.items():
                if form == "linear": hn[:, j] = z[:, j]
                elif form == "rule": hn[:, j] = torch.where(age == RULE_AGE[j], 0, z[:, j])
            h = hn
        return h @ wo.T
    return f
if __name__ == "__main__":
    lin = {j: "linear" for j in MEM}
    L = {"all memory linear": lin,
         "0 by rule, rest linear": {**lin, 0: "rule"},
         "0, 10, 11 by rule, rest linear": {**lin, 0: "rule", 10: "rule", 11: "rule"},
         "0, 10, 11 by rule, 15 real": {**lin, 0: "rule", 10: "rule", 11: "rule", 15: "real"},
         "0, 10, 11 by rule, 3 and 15 real": {**lin, 0: "rule", 10: "rule", 11: "rule", 3: "real", 15: "real"},
         "0, 10, 11 by rule, 3, 12, 15 real": {**lin, 0: "rule", 10: "rule", 11: "rule", 3: "real", 12: "real", 15: "real"},
         "0, 10, 11 by rule, all others real": {0: "rule", 10: "rule", 11: "rule"},
         "0, 10, 11 real, 15 real, rest linear": {**lin, 0: "real", 10: "real", 11: "real", 15: "real"}}
    score.report({k: [runner(v)] for k, v in L.items()}, family=False)
