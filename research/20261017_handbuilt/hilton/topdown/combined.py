"""Top-down test, several blocks replaced together. One runner applies each block's rule in the same run; everything
not replaced runs on its own weights. Scored on the fixed grid lists of grid_lists.py (16 x 16 x 16, no sampling) by
agreement with the intact trained network and by accuracy.
Blocks and rules:
  S  store (2): age table + straight-line residual (store_ladder2.py)
  D  detector (4): own three weights + straight-line residual (detector_ladder.py)
  C  cancelling neurons (1, 6, 7, 8): the hiding agent's rule (../../hiding/rulelib.py; PM + age + left + since + clip)
  K  keep gate (5): the gates agent's pooled rule; G late gate (9): its short 7-number rule (../../gates/g28_final_rules.py)
     (the gate rules are input-only: computed from the list before the run)
  M  memory (0, 3, 10-15): the real weights with the memory agent's switching rule (../../memory/sign_rules3_fresh.py);
     its features read the keep gate's on/off, taken from the keep-gate rule when K is replaced, else from the intact run.
"""
import sys, os, pickle, itertools
import numpy as np
BASE = "/data/output/research/algzoo/20261017_handbuilt"
for d in ("hiding", "gates", "memory"): sys.path.insert(0, f"{BASE}/{d}")
sys.path.insert(0, f"{BASE}/hilton/propagate"); sys.path.insert(0, f"{BASE}/hilton/topdown")
import node_common as nc
WI, WH, WO = nc.WI, nc.WH, nc.WO
# store and detector coefficients (fitted on random lists in the torch modules)
import store_ladder as sl, store_ladder2 as s2, detector_ladder as dl
R_TAB = sl.r.cpu().numpy(); SCO = {a: c.cpu().numpy() for a, c in s2.coef.items()}; DCO = {a: c.cpu().numpy() for a, c in dl.coef.items()}
# cancelling neurons
cwd = os.getcwd(); os.chdir(f"{BASE}/hiding")
import rulelib as RL
GC = ["PM", "age", "left", "since", "clip"]; CO_C = RL.fit_rule(RL.lists(3, 200_000), GC)
# gates
os.chdir(f"{BASE}/gates")
from gaterules3 import rule_values, HJ
CO_G = pickle.load(open("g28_tables.pkl", "rb"))
D5K = "gates|-kept|-forgot|-neg|-rec|-p"
KEEP = {**{f"r{j}": "ages" for j in HJ}, "D5": D5K, "D9": "ages"}
LATE = {"r1": "off", "r4": "basic", "r6": "off", "r7": "const:1.0", "r8": "off", "D9": "basic", "D5": "basic"}
# memory
os.chdir(f"{BASE}/memory")
exec(open("sign_rules3_fresh.py").read().split("X = lists(99")[0])
from sign_rules2 import build_rules, apply_rules
MEMR = build_rules(lists(12, 200_000)); MEM = [0, 3, 10, 11, 12, 13, 14, 15]
os.chdir(cwd)
C = [1, 6, 7, 8]; CN = -WI[C]


def feats(X):
    n = len(X); L = np.zeros((n, 10)); A = np.zeros((n, 10), int); P = np.zeros((n, 10)); M2 = np.zeros((n, 10))
    m = np.zeros(n); p = np.zeros(n); m2 = np.zeros(n); r = np.full(n, -1)
    for t in range(10):
        L[:, t] = m; A[:, t] = np.where(r >= 0, t - r, 0); P[:, t] = p; M2[:, t] = m2
        x = X[:, t]; new = x > m; m2 = np.where(new, m, np.maximum(m2, x)); p = np.where(new, m, p); m = np.where(new, x, m); r = np.where(new, t, r)
    return L, A, P, M2


def design(X, L, P, M2, t, a, idx):
    since = [X[idx, t - a + 1 + j] for j in range(a - 1)]
    return np.stack([L[idx, t], P[idx, t], M2[idx, t]] + since + [np.maximum(s, 0) for s in since] + [np.ones(len(idx))], 1)


def run(X, blocks):
    n = len(X); h = np.zeros((n, 16)); L, A, P, M2 = feats(X)
    V = rule_values(X, {"K": KEEP, "G": LATE}["K" if "K" in blocks else "G"], CO_G)[0] if ("K" in blocks or "G" in blocks) else None
    if "K" in blocks and "G" in blocks:
        V5 = rule_values(X, KEEP, CO_G)[0][5]; V9 = rule_values(X, LATE, CO_G)[0][9]
    else:
        V5 = V[5] if V is not None and "K" in blocks else None; V9 = V[9] if V is not None and "G" in blocks else None
    if "M" in blocks:
        S0, P0, H0 = nc_run(X)
        Hk = H0.copy()
        if V5 is not None: Hk[:, :, 5] = V5
        off = apply_rules(MEMR, X, task2(X), Hk)
    if "C" in blocks:
        VC = RL.list_vars(X); Hc = np.zeros((n, 10, 4)); E = np.zeros((n, 10, 4))
    for t in range(10):
        pre = h @ WH.T + X[:, t:t + 1] * WI; hn = np.maximum(pre, 0)
        if "M" in blocks: hn[:, MEM] = np.where(off[:, t], 0.0, pre[:, MEM])
        if V5 is not None: hn[:, 5] = V5[:, t]
        if V9 is not None: hn[:, 9] = V9[:, t]
        if "C" in blocks:
            cref = np.zeros((n, 4))
            if t > 0:
                for rr in range(-1, t):
                    idx = np.where(VC["R"][:, t] == rr)[0]
                    if len(idx) == 0 or (t, rr) not in CO_C: continue
                    Fm = RL.features(X, VC, t, rr, idx, Hc, E, GC)
                    if Fm.shape[1]: cref[idx] = Fm @ CO_C[(t, rr)]
            prer = cref - CN * X[:, t:t + 1]; hn[:, C] = np.maximum(prer, 0); Hc[:, t] = hn[:, C]; E[:, t] = np.maximum(-prer, 0)
        if "D" in blocks:
            v = 10.157 * X[:, t] - 0.492 * h[:, 2] - 0.986 * h[:, 4]
            for a in range(2, min(9, t + 1)):
                idx = np.where(A[:, t] == a)[0]
                if len(idx): v[idx] += design(X, L, P, M2, t, a, idx) @ DCO[a]
            hn[:, 4] = np.maximum(v, 0)
        if "S" in blocks:
            v = R_TAB[np.minimum(A[:, t], 9)] * L[:, t]
            for a in range(2, min(9, t + 1)):
                idx = np.where(A[:, t] == a)[0]
                if len(idx): v[idx] += design(X, L, P, M2, t, a, idx) @ SCO[a]
            hn[:, 2] = np.maximum(v, 0)
        h = hn
    return h @ WO.T


def nc_run(X):
    h = np.zeros((len(X), 16)); Ps, Hs = [], []
    for t in range(10):
        pre = h @ WH.T + X[:, t:t + 1] * WI; h = np.maximum(pre, 0); Ps.append(pre); Hs.append(h.copy())
    return h @ WO.T, np.stack(Ps, 1), np.stack(Hs, 1)


if __name__ == "__main__":
    combos = sys.argv[1:] or ["SD", "C", "KG", "M", "SDC", "SDCKG", "SDCKGM"]
    X, w, Y = nc.X, nc.w, nc.Y
    for blocks in combos:
        agr = acc = mw = zw = 0.0
        for i in range(0, len(X), 100_000):
            sl_ = slice(i, i + 100_000); Xc = X[sl_]; a0 = nc.intact(Xc).argmax(1); a1 = run(Xc, blocks).argmax(1); wr = a0 != Y[sl_]
            agr += (w[sl_] * (a1 == a0)).sum(); acc += (w[sl_] * (a1 == Y[sl_])).sum(); mw += (w[sl_] * (wr & (a1 == a0))).sum(); zw += (w[sl_] * wr).sum()
        print(f"blocks {blocks:7s} replaced: agreement {100*agr:6.2f}%   mistakes reproduced {100*mw/zw:5.1f}%   accuracy {100*acc:6.2f}%  (intact accuracy 94.35%)", flush=True)
