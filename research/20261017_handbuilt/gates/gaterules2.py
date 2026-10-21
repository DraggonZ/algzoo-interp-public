"""Input-only gate rules, computed from the list alone (no network run), and their fitting.

Rule for gate g (5 or 9), one step after x_t:
    v_g(t+1) = ReLU( D_g(t) + sum_j s_gj hinge_j(t) ),     hinge_j(t) = (x_t - r_j(t))^+ for j = 4, (r_j(t) - x_t)^+ for j = 1, 6, 7, 8
with s_gj = W_gj |w_j| (fixed by the weights) and
    r_j(t) = switch point of neuron j: a coefficient table by role and age (rolefeat.py),
    D_g(t) = rest: a table over
        'roles'   - role features + x_t, x_{t+1}, own previous value,
        'hinges'  - plus the five hinge values of each past number, by age (the writes the memory receives from the
                    detector and hiding neurons),
        'gates'   - plus both gates' values one step after each past number, by age.
At step 0 the switch points are 0 and D_g is the exact closed form in x_0 and x_1.
The rule runs on its own: the hinge values, keep decisions and gate values in its features are its own earlier outputs.
Tables are least-squares fits on the trained network's own run (analysis fit; the network itself is untouched)."""
import numpy as np
from gatelib import WI, WH, run
from rolefeat import role_features

HJ = [1, 4, 6, 7, 8]
SIGN = {1: -1, 4: +1, 6: -1, 7: -1, 8: -1}
S = {g: np.array([WH[g, j] * abs(WI[j]) for j in HJ]) for g in (5, 9)}


def hinge_vals(x, r):
    """x (B,), r (B, 5) -> (B, 5) hinge values in HJ order."""
    out = np.empty_like(r)
    for k, j in enumerate(HJ):
        out[:, k] = np.maximum(x - r[:, k], 0) if SIGN[j] > 0 else np.maximum(r[:, k] - x, 0)
    return out


def d_features(X, t, F, own_prev, HG, G, level):
    B = len(X)
    cols = [F, X[:, t:t + 1], X[:, t + 1:t + 2], own_prev[:, None]]
    if level in ("hinges", "gates"):
        hz = np.zeros((B, 9 * 5))
        for a in range(1, 10):
            if t - a >= 0:
                hz[:, (a - 1) * 5:a * 5] = HG[:, t - a]
        cols.append(hz)
    if level == "gates":
        gz = np.zeros((B, 9 * 2))
        for a in range(1, 10):
            if t - a >= 0:
                gz[:, (a - 1) * 2] = G[5][:, t - a + 1]; gz[:, (a - 1) * 2 + 1] = G[9][:, t - a + 1]
        cols.append(gz)
    return np.concatenate(cols, 1)


def step0_D(X, g):
    h0 = np.maximum(X[:, :1] * WI, 0)
    return sum(WH[g, j] * h0[:, j] for j in range(16) if j not in HJ) + WI[g] * X[:, 1]


def fit(X, level="gates"):
    L, P, H = run(X); B = len(X)
    keep = np.zeros((B, 10), bool); keep[:, :9] = H[:, 1:, 5] > 0
    HG = np.zeros((B, 10, 5)); R = np.zeros((B, 10, 5))
    for s in range(10):
        u = (H[:, s - 1] if s > 0 else np.zeros((B, 16))) @ WH.T
        R[:, s] = np.stack([-u[:, j] / WI[j] for j in HJ], 1)
        HG[:, s] = hinge_vals(X[:, s], R[:, s])
    G = {5: H[:, :, 5], 9: H[:, :, 9]}
    rows = {k: ([], []) for k in ["R", "D5", "D9"]}
    for t in range(1, 9):
        F = role_features(X, t, keep)[0]
        rows["R"][0].append(F); rows["R"][1].append(R[:, t])
        for g in (5, 9):
            A = d_features(X, t, F, H[:, t, g], HG, G, level)
            D = P[:, t + 1, g] - (HG[:, t] * S[g]).sum(1)
            rows[f"D{g}"][0].append(A); rows[f"D{g}"][1].append(D)
    co = {"level": level}
    for k, (A, b) in rows.items():
        co[k] = np.linalg.lstsq(np.concatenate(A), np.concatenate(b), rcond=None)[0]
    return co


def rule_values(X, co, true_parts=None):
    """Both gates' rule values (B, 10), computed from X alone. Also returns the switch points and D values used."""
    B = len(X); level = co["level"]
    V = {5: np.zeros((B, 10)), 9: np.zeros((B, 10))}
    keep = np.zeros((B, 10), bool); HG = np.zeros((B, 10, 5)); R = np.zeros((B, 10, 5)); Dv = {5: np.zeros((B, 10)), 9: np.zeros((B, 10))}
    for g in (5, 9):
        V[g][:, 0] = np.maximum(WI[g] * X[:, 0], 0)
    # number x_0: switch points 0
    HG[:, 0] = hinge_vals(X[:, 0], R[:, 0])
    for g in (5, 9):
        Dv[g][:, 0] = step0_D(X, g)
        V[g][:, 1] = np.maximum(Dv[g][:, 0] + (HG[:, 0] * S[g]).sum(1), 0)
    keep[:, 0] = V[5][:, 1] > 0
    for t in range(1, 9):
        F = role_features(X, t, keep)[0]
        R[:, t] = F @ co["R"]
        HG[:, t] = hinge_vals(X[:, t], R[:, t])
        for g in (5, 9):
            A = d_features(X, t, F, V[g][:, t], HG, V, level)
            Dv[g][:, t] = A @ co[f"D{g}"]
            V[g][:, t + 1] = np.maximum(Dv[g][:, t] + (HG[:, t] * S[g]).sum(1), 0)
        keep[:, t] = V[5][:, t + 1] > 0
    return V, R, Dv
