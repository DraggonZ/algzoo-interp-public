"""Gate rules with a choice of detail for each part (generalizes gaterules2.py).

v_g(t+1) = ReLU( D_g(t) + sum_j s_gj hinge_j(t) ), hinge_4 = (x_t - r_4)^+, hinge_j = (r_j - x_t)^+ for j = 1, 6, 7, 8.
Each switch point r_j and each history part D_g is a least-squares formula over one of these feature sets:
  'basic'    - the largest and second-largest so far (m1, m2), one coefficient each
  'ages'     - m1 by its age, m2 by both ages (beaten record or kept), as in rolefeat groups m1, m2
  'roles'    - all rolefeat groups (also p, kept, forgotten, negative and older records, by age)
  'hinges'   - roles plus the five hinge values of each past number, by age        (D only)
  'gates'    - hinges plus both gates' values one step after each past number      (D only)
D formulas always also get x_t, x_{t+1} and the gate's own previous value.
'const:<value>' fixes r_j to value x m1 (e.g. r7 = m1). Fitted on the trained network's own run; the rule then runs on
its own outputs (input-only)."""
import numpy as np
from gatelib import WI, WH, run
from rolefeat import role_features
from gaterules2 import HJ, S, hinge_vals, step0_D

GROUPS = {"basic": None, "ages": ("m1", "m2"), "m1ages": ("m1",), "roles": ("m1", "m2", "p", "kept", "forgot", "neg", "rec")}


def parse(kind):
    """'gates|-kept|-hz' = kind 'gates' without role group 'kept' and without the hinge features ('hz'), gate
    features ('gz') or own previous value ('own')."""
    parts = kind.split("|"); base = parts[0]; drop = {p[1:] if p.startswith("-") else p for p in parts[1:]}
    return base, drop   # a part 'step' (no minus) means separate coefficients for each step t


def base_feats(X, t, keep, kind):
    base, drop = parse(kind)
    if base == "basic":
        _, _, info = role_features(X, t, keep, groups=())
        return np.c_[info["m1"], info["m2"]], info
    groups = GROUPS["roles" if base in ("hinges", "gates") else base]
    F, _, info = role_features(X, t, keep, groups=tuple(g for g in groups if g not in drop))
    return F, info


def d_feats(X, t, keep, kind, own_prev, HG, G):
    F, info = base_feats(X, t, keep, kind)
    base, drop = parse(kind)
    B = len(X); cols = [F, X[:, t:t + 1], X[:, t + 1:t + 2]]
    if "own" not in drop:
        cols.append(own_prev[:, None])
    if base in ("hinges", "gates") and "hz" not in drop:
        hz = np.zeros((B, 45))
        for a in range(1, 10):
            if t - a >= 0:
                hz[:, (a - 1) * 5:a * 5] = HG[:, t - a]
        cols.append(hz)
    if base == "gates" and "gz" not in drop:
        gz = np.zeros((B, 18))
        for a in range(1, 10):
            if t - a >= 0:
                gz[:, (a - 1) * 2] = G[5][:, t - a + 1]; gz[:, (a - 1) * 2 + 1] = G[9][:, t - a + 1]
        cols.append(gz)
    return np.concatenate(cols, 1)


def network_parts(X):
    L, P, H = run(X); B = len(X)
    keep = np.zeros((B, 10), bool); keep[:, :9] = H[:, 1:, 5] > 0
    HG = np.zeros((B, 10, 5)); R = np.zeros((B, 10, 5))
    for s in range(10):
        u = (H[:, s - 1] if s > 0 else np.zeros((B, 16))) @ WH.T
        R[:, s] = np.stack([-u[:, j] / WI[j] for j in HJ], 1)
        HG[:, s] = hinge_vals(X[:, s], R[:, s])
    return dict(P=P, H=H, keep=keep, HG=HG, R=R)


def fit_part(X, NP, part, kind):
    """part: 'r<j>' or 'D5'/'D9'. Returns coefficient vector."""
    if kind.startswith("const") or kind == "off":
        return None
    AtA, Atb = 0, 0          # normal equations, accumulated step by step to keep memory small
    for t in range(1, 9):
        if part[0] == "r":
            A, _ = base_feats(X, t, NP["keep"], kind); b = NP["R"][:, t, HJ.index(int(part[1:]))]
        else:
            g = int(part[1:])
            A = d_feats(X, t, NP["keep"], kind, NP["H"][:, t, g], NP["HG"], {5: NP["H"][:, :, 5], 9: NP["H"][:, :, 9]})
            b = NP["P"][:, t + 1, g] - (NP["HG"][:, t] * S[g]).sum(1)
        if "step" in parse(kind)[1]:
            AtA, Atb = {} if not isinstance(AtA, dict) else AtA, {} if not isinstance(Atb, dict) else Atb
            AtA[t] = A.T @ A; Atb[t] = A.T @ b
        else:
            AtA = AtA + A.T @ A; Atb = Atb + A.T @ b
    if isinstance(AtA, dict):
        return {t: np.linalg.lstsq(AtA[t], Atb[t], rcond=1e-12)[0] for t in AtA}
    return np.linalg.lstsq(AtA, Atb, rcond=1e-12)[0]


def rule_values(X, spec, co):
    """spec: {'r1': kind, ..., 'r8': kind, 'D5': kind, 'D9': kind}; co: {(part, kind): coefficients}."""
    B = len(X)
    V = {5: np.zeros((B, 10)), 9: np.zeros((B, 10))}
    keep = np.zeros((B, 10), bool); HG = np.zeros((B, 10, 5)); R = np.zeros((B, 10, 5))
    for g in (5, 9):
        V[g][:, 0] = np.maximum(WI[g] * X[:, 0], 0)
    HG[:, 0] = hinge_vals(X[:, 0], R[:, 0])
    for g in (5, 9):
        V[g][:, 1] = np.maximum(step0_D(X, g) + (HG[:, 0] * S[g]).sum(1), 0)
    keep[:, 0] = V[5][:, 1] > 0
    for t in range(1, 9):
        cache = {}
        for k, j in enumerate(HJ):
            kind = spec[f"r{j}"]
            if kind == "off":                      # hinge never active (its neuron's push left out)
                R[:, t, k] = 1e9 if j == 4 else -1e9
            elif kind.startswith("const"):
                _, _, info = role_features(X, t, keep, groups=())
                R[:, t, k] = float(kind.split(":")[1]) * info["m1"]
            else:
                if kind not in cache:
                    cache[kind] = base_feats(X, t, keep, kind)[0]
                c = co[(f"r{j}", kind)]
                R[:, t, k] = cache[kind] @ (c[t] if isinstance(c, dict) else c)
        HG[:, t] = hinge_vals(X[:, t], R[:, t])
        for g in (5, 9):
            kind = spec[f"D{g}"]
            A = d_feats(X, t, keep, kind, V[g][:, t], HG, V)
            c = co[(f"D{g}", kind)]
            V[g][:, t + 1] = np.maximum(A @ (c[t] if isinstance(c, dict) else c) + (HG[:, t] * S[g]).sum(1), 0)
        keep[:, t] = V[5][:, t + 1] > 0
    return V, R
