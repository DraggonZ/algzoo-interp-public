"""Gate rules and the substitution harness.

A gate rule gives neuron g's value one step after each x_t from named quantities. Structure (exact, g05_structure.py):
    pre_g(t+1) = D_g(t) + sum_j s_gj * hinge_j(x_t - r_j(t)),
hinge = (x - r)^+ for the detector 4 and (r - x)^+ for hiding neurons 1, 6, 7, 8, s_gj = W_gj |w_j| (fixed numbers,
e.g. neuron 5: +12.06 (x - r4)^+ - 7.81 (r7 - x)^+ - 10.46 (r8 - x)^+), r_j the switch points, and D_g the rest.
Step 0 (no history): pre_g(0) = w_g x_0 exactly (w_5 = -0.28, w_9 = -0.23).

Each part (r_j, D_g) can come from:
  'true'  - the running network's own state (the substituted run), for diagnosis;
  'table' - a coefficient table by role and age (rolefeat.py), fitted on the trained network (seed 6),
            computed from the list and the rule's own past decisions only.
"""
import numpy as np
from gatelib import WI, WH, run
from rolefeat import role_features

HJ = [1, 4, 6, 7, 8]
HINGE_SIGN = {1: -1, 4: +1, 6: -1, 7: -1, 8: -1}   # +1: (x - r)^+, -1: (r - x)^+


def slope(g, j):
    return WH[g, j] * abs(WI[j])


def hinges(g, x, r):
    tot = 0
    for j in HJ:
        s = slope(g, j)
        if abs(s) < 1e-9:
            continue
        tot = tot + s * (np.maximum(x - r[j], 0) if HINGE_SIGN[j] > 0 else np.maximum(r[j] - x, 0))
    return tot


def extra_cols(X, t, prev_own, which):
    cols = [X[:, t], X[:, t + 1]]
    if "own" in which:
        cols.append(prev_own)
    return np.stack(cols, 1)


def fit_tables(X, gates=(5, 9), extras=None):
    """Fit role tables for r_j (j in HJ) and D_g on the network's own run. Returns dict of coefficient vectors."""
    extras = extras or {5: ("own",), 9: ("own",)}
    L, P, H = run(X)
    keep = np.zeros((len(X), 10), bool); keep[:, :9] = H[:, 1:, 5] > 0
    rows = {k: ([], []) for k in [f"r{j}" for j in HJ] + [f"D{g}" for g in gates]}
    for t in range(1, 9):
        F = role_features(X, t, keep)[0]
        u = H[:, t - 1] @ WH.T
        for j in HJ:
            rows[f"r{j}"][0].append(F); rows[f"r{j}"][1].append(-u[:, j] / WI[j])
        for g in gates:
            D = P[:, t + 1, g] - sum(WH[g, j] * H[:, t, j] for j in HJ)
            rows[f"D{g}"][0].append(np.c_[F, extra_cols(X, t, H[:, t, g], extras[g])]); rows[f"D{g}"][1].append(D)
    co = {}
    for k, (A, b) in rows.items():
        co[k] = np.linalg.lstsq(np.concatenate(A), np.concatenate(b), rcond=None)[0]
    co["extras"] = extras
    return co


def substitute(X, g, co, refs="table", D="table", keep_from=5, params=None):
    """Run the network with neuron g replaced by its rule at every step. refs / D: 'table' or 'true'.
    The rule's own keep decisions (for neuron 5) come from its own values; for neuron 9 the keep decisions in its
    features come from the running network's neuron 5."""
    B = len(X)
    keep = np.zeros((B, 10), bool)
    state = {"hprev": np.zeros((B, 16)), "own_prev": np.zeros(B), "vals": np.zeros((B, 10))}
    extras = co["extras"][g]

    def hook(t, h_prev, pre, h):
        # h_prev = state after input t-1, i.e. after x_{t-1}; this step's gate value judges x_{t-1}
        if t == 0:
            v = np.maximum(WI[g] * X[:, 0], 0)
        else:
            s = t - 1                               # the number being judged
            if s == 0:                              # no history: switch points 0, D from x_0 and x_1 exactly
                h0 = np.maximum(X[:, :1] * WI, 0)
                Dv = sum(WH[g, j] * h0[:, j] for j in range(16) if j not in HJ) + WI[g] * X[:, 1]
                if D == "true":
                    Dv = pre[:, g] - sum(WH[g, j] * h_prev[:, j] for j in HJ)
                v = np.maximum(Dv + hinges(g, X[:, 0], {j: np.zeros(B) for j in HJ}), 0)
                h[:, g] = v
                keep[:, 0] = (v > 0) if g == 5 else (h[:, 5] > 0)
                state["hprev"] = h_prev.copy(); state["own_prev"] = v; state["vals"][:, t] = v
                return h
            if refs == "table" or D == "table":
                F = role_features(X, s, keep)[0]
            if refs == "true":
                u = state["hprev"] @ WH.T
                r = {j: -u[:, j] / WI[j] for j in HJ}
            else:
                r = {j: F @ co[f"r{j}"] for j in HJ}
            if D == "true":
                Dv = pre[:, g] - sum(WH[g, j] * h_prev[:, j] for j in HJ)
            else:
                Dv = np.c_[F, extra_cols(X, s, state["own_prev"], extras)] @ co[f"D{g}"]
            v = np.maximum(Dv + hinges(g, X[:, s], r), 0)
        h[:, g] = v
        if g == 5 and t >= 1:
            keep[:, t - 1] = v > 0
        if g != 5 and t >= 1:
            keep[:, t - 1] = h[:, 5] > 0
        state["hprev"] = h_prev.copy(); state["own_prev"] = v; state["vals"][:, t] = v
        return h

    Lr, Pr, Hr = run(X, params=params, hook=hook)
    return Lr, Hr
