"""Causal scrubbing of the account's claims (Gao's "circuit faithfulness"; Chan et al. 2022). A claim says which
features of the list a neuron group depends on at step t. Test: at each step t, build a second list that agrees with
the real one on those features and is redrawn elsewhere (numbers redrawn from the normal distribution below a bound,
so that nothing the claim keeps changes); run the trained network on it up to step t and give the group that step's
value in the run on the real list. If the claim holds, the output does not move.

Claims (positions are of the prefix 0..t; m1, m2 = largest and second-largest before the step in question):
  store+detector at t: depend on x_t and the value of m1 only (m1 moved to a random earlier position, the rest below m1).
  cancellers at t:     depend on x_t, the value and position of m1 (its age sets the timer) and the value of m2 (m2 moved
                       to a random position other than m1's; the rest below m2).
  keep gate at t:      depends on x_t, x_{t-1} and the values of m1 and m2 before t-1 (positions random, rest below m2).
  late gate at t:      depends on x_t, x_{t-1} and the value of m1 before t-1 (position random, rest below m1).
  memory at t:         depends on the kept numbers only, at their positions (kept: positive and >= 0.75 x the
                       second-largest before it); every forgotten number is redrawn below its keep threshold.
Control for each: the same number of positions kept, but chosen at random, the rest redrawn freely."""
import numpy as np
from scipy.special import ndtr, ndtri
from simlib import GROUPS, WI, WH, lists, run, score

rng = np.random.default_rng(7)


def below(c):
    """Standard normal draws truncated to (-inf, c)."""
    u = rng.uniform(1e-12, 1, size=np.shape(c)); return ndtri(u * ndtr(c))


PARAMS = None   # set to (wi, wh, wo) to scrub another network


def state_at(X2, t):
    """Run the trained network (or PARAMS) on X2 up to step t; return the activations after step t."""
    wi, wh = (WI, WH) if PARAMS is None else PARAMS[:2]
    h = np.zeros((len(X2), 16))
    for s in range(t + 1):
        h = np.maximum(h @ wh.T + X2[:, s:s + 1] * wi, 0)
    return h


def top2(X, t):
    """Values and positions of the largest and second-largest of X[:, :t], floored at 0 (no position then: -1)."""
    B = len(X)
    if t == 0:
        z = np.zeros(B); return z, np.full(B, -1), z, np.full(B, -1)
    P = np.c_[X[:, :t], np.zeros((B, 2))]; o = np.argsort(-P, 1)
    i1, i2 = o[:, 0], o[:, 1]; v1, v2 = P[np.arange(B), i1], P[np.arange(B), i2]
    return v1, np.where(i1 < t, i1, -1), v2, np.where(i2 < t, i2, -1)


def place(X2, pos_range, vals, has, b_idx):
    """Put vals at a random position in [0, pos_range) for lists where has; returns the chosen positions."""
    B = len(X2); p = np.floor(rng.uniform(0, 1, B) * np.maximum(pos_range, 1)).astype(int)
    X2[b_idx[has], p[has]] = vals[has]; return p


def claim_list(X, t, group, opts=()):
    """A second list agreeing with X on the claimed features of `group` at step t (only columns 0..t matter).
    opts add features to the claim: 'm1pos' keeps m1 at its position, 'm2' keeps m2's value and 'm2pos' its position,
    'prev' keeps x_{t-1}, 'prev2' keeps x_{t-2}, 'thr=0.6' sets the memory's keep threshold."""
    B = len(X); b = np.arange(B); X2 = X.copy()
    if "asmemory" in opts:   # use the memory's claim (kept numbers at their places) for this group
        group = "memory"
    if group == "store+detector":
        v1, i1, v2, i2 = top2(X, t)
        bound = v2 if "m2" in opts else v1
        X2[:, :t] = below(np.repeat(bound[:, None], t, 1)) if t else X2[:, :t]
        if "m1pos" in opts:
            X2[b[i1 >= 0], i1[i1 >= 0]] = v1[i1 >= 0]
        elif "m2" in opts and t >= 2:   # move m1 to a random place other than m2's (fix 2026-10-18: it could overwrite m2's place)
            p = np.floor(rng.uniform(0, 1, B) * (t - (i2 >= 0))).astype(int); p = np.where((i2 >= 0) & (p >= i2), p + 1, p)
            ok = i1 >= 0; X2[b[ok], p[ok]] = v1[ok]
        else:
            place(X2, np.full(B, t), v1, i1 >= 0, b)
        if "m2" in opts:
            ok = i2 >= 0; X2[b[ok], i2[ok]] = v2[ok]
    elif group == "cancellers":
        v1, i1, v2, i2 = top2(X, t)
        X2[:, :t] = below(np.repeat(v2[:, None], t, 1)) if t else X2[:, :t]
        X2[b[i1 >= 0], i1[i1 >= 0]] = v1[i1 >= 0]
        if "prevrec" in opts and t >= 1:   # also keep the previous record (largest before m1 arrived), value and place
            pos = np.arange(t)[None, :]
            pre_mask = pos < np.where(i1 >= 0, i1, 0)[:, None]
            Pm = np.where(pre_mask, X[:, :t], -np.inf); jr = Pm.argmax(1); vr = Pm.max(1)
            okr = (i1 >= 1) & (vr > 0)
            bnd = np.where(pre_mask, np.where(okr, vr, 0.0)[:, None], v2[:, None])
            red = below(bnd); fixed = np.zeros_like(red, bool)
            fixed[b[i1 >= 0], i1[i1 >= 0]] = True; fixed[b[i2 >= 0], i2[i2 >= 0]] = True; fixed[b[okr], jr[okr]] = True
            X2[:, :t] = np.where(fixed, X[:, :t], red)
        elif t >= 2 and "m2pos" in opts:
            ok = i2 >= 0; X2[b[ok], i2[ok]] = v2[ok]
        elif t >= 2:   # m2 at a random position other than m1's
            p = np.floor(rng.uniform(0, 1, B) * (t - 1)).astype(int); p = np.where((i1 >= 0) & (p >= i1), p + 1, p)
            ok = i2 >= 0; X2[b[ok], p[ok]] = v2[ok]
    elif group in ("keep gate", "late gate"):
        if t >= 1:
            v1, i1, v2, i2 = top2(X, t - 1); keep2 = group == "keep gate"
            bound = v2 if keep2 else v1
            if t >= 2:
                X2[:, :t - 1] = below(np.repeat(bound[:, None], t - 1, 1))
                if "m1pos" in opts:
                    X2[b[i1 >= 0], i1[i1 >= 0]] = v1[i1 >= 0]; ok = (i2 >= 0) & keep2; X2[b[ok], i2[ok]] = v2[ok]
                p1 = None if "m1pos" in opts else place(X2, np.full(B, t - 1), v1, i1 >= 0, b)
                if keep2 and t >= 3 and p1 is not None:
                    p = np.floor(rng.uniform(0, 1, B) * (t - 2)).astype(int); p = np.where(p >= p1, p + 1, p)
                    ok = (i2 >= 0) & (i1 >= 0); X2[b[ok], p[ok]] = v2[ok]
    elif group == "memory":
        for s in range(t + 1):
            _, _, v2, _ = top2(X, s)
            f = next((float(o[4:]) for o in opts if o.startswith("thr=")), 0.75)
            thr = np.where(v2 > 0, f * v2, 0.0); kept = (X[:, s] > 0) & (X[:, s] >= thr)
            X2[:, s] = np.where(kept, X[:, s], below(thr))
    if "after" in opts and group == "cancellers":   # also keep the number right after the current largest
        _, i1, _, _ = top2(X, t); ok = (i1 >= 0) & (i1 + 1 < t)
        X2[b[ok], i1[ok] + 1] = X[b[ok], i1[ok] + 1]
    if "cur" in opts:
        X2[:, t] = X[:, t]
    if "prev" in opts and t >= 1:
        X2[:, t - 1] = X[:, t - 1]
    if "prev2" in opts and t >= 2:
        X2[:, t - 2] = X[:, t - 2]
    return X2


def kept_count(X, X2, t):
    return (X[:, :t + 1] == X2[:, :t + 1]).sum(1)


def control_list(X, X2, t):
    """Same number of unchanged positions as the claim list, at random positions; the rest redrawn freely."""
    B = len(X); k = kept_count(X, X2, t); X3 = X.copy(); X3[:, :t + 1] = rng.standard_normal((B, t + 1))
    order = np.argsort(rng.uniform(size=(B, t + 1)), 1); keep = np.arange(t + 1)[None, :] < k[:, None]
    rows = np.repeat(np.arange(B)[:, None], t + 1, 1)
    X3[rows[keep], order[keep]] = X[rows[keep], order[keep]]
    return X3


def scrubbed_values(X, groups, control=False, opts=None):
    sub = {}; opts = opts or {}
    for g in groups:
        ns = GROUPS[g]; V = np.zeros((len(X), 10, len(ns)))
        for t in range(10):
            X2 = claim_list(X, t, g, opts.get(g, ()))
            if control:
                X2 = control_list(X, X2, t)
            V[:, t] = state_at(X2, t)[:, ns]
        for i, n in enumerate(ns):
            sub[n] = V[:, :, i]
    return sub


if __name__ == "__main__":
    X = lists(202, 100_000); L = run(X)[0]
    print(f"trained accuracy {(L.argmax(1) == X.argsort(1)[:, -2]).mean():.3f} on these 100,000 lists")
    print(f"{'group':16s} {'claim: KL':>9s} {'acc':>6s} {'agree':>6s} {'control: KL':>11s} {'acc':>6s}")
    for name in list(GROUPS) + ["all"]:
        gs = list(GROUPS) if name == "all" else [name]
        a = score(X, L, run(X, scrubbed_values(X, gs))[0]); c = score(X, L, run(X, scrubbed_values(X, gs, True))[0])
        print(f"{name:16s} {a['kl']:9.3f} {a['acc']:6.3f} {a['agree']:6.3f} {c['kl']:11.3f} {c['acc']:6.3f}", flush=True)
