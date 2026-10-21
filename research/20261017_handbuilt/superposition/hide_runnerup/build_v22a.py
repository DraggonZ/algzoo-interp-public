"""Hand-built v22a: v19 (kappa 0.5) plus 'hide the current largest down to the runner-up's vote, minus a margin that
grows with the largest'. 50 neurons: v19's 16 unchanged, plus 34 below. See README.md for results.

Goal. At the end, the position of the current largest shows (1 - c)(m2 - margin(m1, m2)), where m1 is the largest and
m2 the runner-up (the second-largest of the list), whatever order they arrived in. v19 shows (1 - c)(p - kappa d)
there (p = the largest before the record, d = its jump), which ignores a runner-up that arrives after the largest.

The margin (in values; times 1 - c = 0.25 in score). Two shapes, both read off ARC, neither fitted:
 - LINEAR_SHAPE (first design): margin = 0.5 m1. errmodel/gap_by_order.log fits ARC's D_L per arrival order as
   a x_L + b x_S + c x_T; a is -0.44, -0.62, -0.32, -0.61, -0.43, -0.54 (mean -0.49), b about 0.
 - ARC_SHAPE (final, the default): ARC's D_L / x_S against r = x_L / x_S - 1 (look_arc_dl_rel.log, S before L, rows
   with x_S >= 0.6, where the curve no longer depends on scale) is steep near a tie and flattens: through the points
   (0, 0.08), (0.075, 0.26), (0.2, 0.40), (2.5, 1.25), slopes 2.4, 1.1, 0.37. Written homogeneously in (m1, m2), as a
   bias-free network needs: margin = a m2 + k1 (m1 - m2) - sum_i dk_i relu(m1 - (1 + r_i) m2), a = 0.08, k1 = 2.4,
   kinks at r = 0.075 (slope drop 1.3) and r = 0.2 (drop 0.73). The linear shape breaks ARC's near-tie behaviour (it
   puts the largest 0.5 votes per unit below the second at a tie, ARC 0.08); this one keeps it.

Mechanism (every weight follows from these rules):
 1. Runner-up store S2, U (2 neurons, like v19's store and detector): S2(t) = m2(t-1), U(t) = relu(x_t - m2(t-1)),
    m2(t) = S2 + U - D / a, because relu(x - m2) - relu(x - m1) = (min(x, m1) - m2)+ is the runner-up's rise.
 2. Kinks (2 per kink): R_i(t+1) = relu(m1(t) - (1 + r_i) m2(t)), one step late; Rr_i(t) = relu(x_t - (1 + r_i) m1(t-1)),
    which is the same kink at a record (m1 = x_t, m2 = the previous largest), fresh, and zero for a non-record.
 3. Record level Lv(t) = relu(g (m1(t-1) - x_t)) (1 neuron): zero exactly at a record, g x the gap otherwise.
 4. Largest's-position line E0-E7 (8 neurons, laid out like v19's memory: slot j holds position t-2-j at step t).
    E0(t+1) = relu(B m1(t) - WL Lv(t)) is B x the record's value one step after a record, else 0 (no biases, so every
    constant scales with the input); each later slot copies the previous one and is erased by a newer record (- WE D).
    At step 9, E holds B m1 at the current largest's position if it is in 0..7, zero elsewhere.
 5. Correction F0-F8 and partners H0-H8 (18 neurons), read only at the end. At step 10, F_p computes
    relu(T - V_p + g (m1(9) - x10) + gate_p) and H_p = relu(g (m1(9) - x10) + gate_p); the readout adds F_p - H_p at p.
    T is the target score, V_p is v19's own score at p as a linear map of (h9, x10) (v19's step-10 update with the
    detector and flag dropped: both are zero unless x10 is a record). gate_p = E_p - B m1(9) for p <= 7 (zero only at
    the current largest; - B d if step 9 was a newer record), - WL Lv(9) for p = 8 (zero only if step 9 was a record).
    The level g (m1(9) - x10) keeps F's argument positive when x10 is not a record and drives F and H to zero when it
    is one (the old largest is then revealed, as v19's clock does). Net effect: the largest's position shows T,
    every other position v19's score. (One shared partner subtracted everywhere fails: the largest keeps the level.)
    For p = 8, T uses the fresh m1(9), m2(9) and the fresh kinks Rr_i(9). For p <= 7 the kinks are one step late, so T
    also reads the runner-up one step late (m2(8) = S2(9)). Mixing a fresh m2 with stale kinks lifted the largest above
    the runner-up and cost 6 points of accuracy (88.5%).
 6. Position 9 (a record at the last step): v19 never names it (-1000 D). v22a gives it the same rule with m1 = x10,
    m2 = p: v19's (1 - c)(p - kappa d) plus (1 - c)(kappa - k1) d (from D), plus the fresh kinks Rr_i(10), minus
    a (1 - c) p at a record, which a pair gives: M9 = relu(a (1 - c) m1(9) + g (x10 - m1(9))) minus g D / a.
Limit (depth): the runner-up is a max over inputs, the kinks a relu of it, and the gate another relu; the last step
leaves one layer. So a runner-up arriving at step 10 (and, with kinks, at step 9) is not seen by the largest's
position: it shows the runner-up before it.

Constants (gate sizes, not fitted): g = 1e4 (v19's clock gain); B = 1e9 so that B m1 exceeds the level
g (m1 - x10) unless |x10| > 1e5 m1; WL = 1e11 so that WL Lv > B m1 for a non-record more than 1e-6 m1 below the
largest; WE = 1e10 so that a record whose jump exceeds 1e-4 of its value erases the line."""
import numpy as np
from algzoo.handbuilt_v19 import build_v19

MARGIN = 0.5            # first design: margin = 0.5 m1 (linear); final: ARC_SHAPE (see the docstring)
ARC_SHAPE = dict(a=0.08, k1=2.4, knots=((0.075, 2.4 - 1.1), (0.2, 1.1 - 0.37)))
LINEAR_SHAPE = dict(a=MARGIN, k1=MARGIN, knots=())


class Built:
    def __init__(self, params, idx):
        self.params, self.idx = params, idx


def build_v22a(shape=ARC_SHAPE, kappa=0.5, g=1e4, B=1e9, WL=1e11, WE=1e10, keep_v19_pos9=False, **kw):
    v = build_v19(kappa=kappa, **kw)
    wi0, wh0, wo0 = v.params
    a, s, c = 1000.0, 20.0, 0.75            # v19 defaults (detector gain, store gain, 1 - vote per unit)
    A0 = 1 - c
    idx = dict(v.idx)
    n0 = len(wi0)
    sa, k1, knots = shape["a"], shape["k1"], shape["knots"]
    names = [f"R{i}" for i in range(len(knots))] + [f"Rr{i}" for i in range(len(knots))] + ["S2", "U", "Lv"] + [f"E{j}" for j in range(8)] + [f"F{p}" for p in range(9)] + [f"H{p}" for p in range(9)] + ["M9"]
    for k, nm in enumerate(names):
        idx[nm] = n0 + k
    d = n0 + len(names)
    wi = np.zeros(d); wh = np.zeros((d, d)); wo = np.zeros((10, d))
    wi[:n0] = wi0; wh[:n0, :n0] = wh0; wo[:, :n0] = wo0
    S, D, Z = idx["store"], idx["detect"], idx["flag"]
    S2, U, Lv, M9 = idx["S2"], idx["U"], idx["Lv"], idx["M9"]
    E = [idx[f"E{j}"] for j in range(8)]; F = [idx[f"F{p}"] for p in range(9)]
    m1 = np.zeros(d); m1[S] = 1 / s; m1[D] = 1 / a            # m1(t) read from h(t)
    m2 = np.zeros(d); m2[S2] = 1; m2[U] = 1; m2[D] = -1 / a   # m2(t) read from h(t)

    # 1. runner-up store
    wh[S2] = m2                     # S2(t+1) = relu(m2(t)) = m2(t)
    wh[U] = -m2; wi[U] = 1.0        # U(t+1) = relu(x - m2(t))
    # 1b. margin kinks: R_i(t+1) = relu(m1(t) - (1 + r_i) m2(t)) (one step late, global); Rr_i(t) = relu(x_t - (1 + r_i)
    #     m1(t-1)): at a record this is the same kink with m2 = the previous largest and m1 = x_t, fresh; zero otherwise
    R = [idx[f"R{i}"] for i in range(len(knots))]; Rr = [idx[f"Rr{i}"] for i in range(len(knots))]
    for i, (r, dk) in enumerate(knots):
        wh[R[i]] = m1 - (1 + r) * m2
        wh[Rr[i]] = -(1 + r) * m1; wi[Rr[i]] = 1.0
    # 2. record level
    wh[Lv] = g * m1; wi[Lv] = -g    # relu(g (m1(t) - x_{t+1}))
    # 3. largest's-position line
    wh[E[0]] = B * m1; wh[E[0], Lv] += -WL   # E0(t+1) = relu(B m1(t) - WL Lv(t)): B m1 one step after a record
    for j in range(1, 8):
        wh[E[j], E[j - 1]] = 1.0
        wh[E[j], D] = -WE           # erased one step after a newer record
    # 4. correction neurons
    target = A0 * ((1 - sa + k1) * m2 - k1 * m1)           # T's linear part as a function of h9: (1 - c)(m2 - margin)
    level_h = g * m1; level_x = -g                         # g (m1(9) - x10)
    wo19_nd = wo0.copy(); wo19_nd[:, D] = 0; wo19_nd[:, Z] = 0   # v19 readout without detector and flag
    Vh_all, Vx_all, tgt_all = {}, {}, {}
    m2lag = np.zeros(d); m2lag[S2] = 1.0                   # m2(t-1) read from h(t)
    for p in range(9):
        Vh = np.zeros(d); Vh[:n0] = wo19_nd[p] @ wh0        # v19's score at p as a linear map of h9 ...
        Vx = wo19_nd[p] @ wi0                              # ... and x10
        Vh_all[p], Vx_all[p] = Vh, Vx
        tgt = target
        if knots and p <= 7:
            # the kinks are one step late (they read m2(8)), so the linear part must read the same m2(8) = S2(9):
            # mixing a fresh m2 with stale kinks lifted the largest above the runner-up (88.5% accuracy, check_v22a)
            tgt = target + A0 * (1 - sa + k1) * (m2lag - m2)
        wh[F[p]] = tgt - Vh + level_h
        wi[F[p]] = -Vx + level_x
        tgt_all[p] = tgt
        for i, (r, dk) in enumerate(knots):                # the kinks: lagged global ones for p <= 7, fresh at a record at 9
            wh[F[p], R[i] if p <= 7 else Rr[i]] += A0 * dk
        if p <= 7:
            jp = 7 - p                                     # slot of position p at step 9 (slot j holds 9-2-j)
            wh[F[p], E[jp]] += 1.0
            wh[F[p]] += -B * m1                            # E_p - B m1(9)
        else:
            wh[F[p], Lv] += -WL
        wo[p, F[p]] += 1.0
    # partners: the level alone under the same gates
    Hn = [idx[f"H{p}"] for p in range(9)]
    for p in range(9):
        wh[Hn[p]] = wh[F[p]] - tgt_all[p] + Vh_all[p]
        for i, (r, dk) in enumerate(knots):
            wh[Hn[p], R[i] if p <= 7 else Rr[i]] -= A0 * dk
        wi[Hn[p]] = wi[F[p]] + Vx_all[p]
        wo[p, Hn[p]] += -1.0
    # 5. position 9
    if not keep_v19_pos9:
        wo[9, D] += 1000.0                                 # remove v19's -1000 D
        # v19 shows (1 - c)(p - kappa d) for a record at 9; the rule wants (1 - c)((1 - a) p - k1 d + sum dk Rr_i)
        wo[9, D] += A0 * (kappa - k1) / a                  # + (1 - c)(kappa - k1) d
        for i, (r, dk) in enumerate(knots):
            wo[9, Rr[i]] += A0 * dk
        wh[M9] = (sa * A0 - g) * m1; wi[M9] = g            # relu(a (1 - c) m1(9) + g (x10 - m1(9))) = a (1 - c) p + g d at a record
        wo[9, M9] += -1.0
        wo[9, D] += g / a                                  # + g d: the pair's second half
    return Built((wi, wh, wo), idx)
