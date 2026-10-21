"""Hand-built w4b (wider track, up to 24 neurons): r3b's network (handbuilt_v27b, level_exact=True: store, detector,
keep gate cutting at 0.75 of the runner-up, runner-up comparator Q, linear route, four hiding-wave neurons, seven delay
cells) plus extra neurons, each added for one of ARC's jobs that the 16-neuron build had no room for
(research/20261017_handbuilt/faithful_twin/LEDGER.md, branch w4b).

Timing (state h_t after reading x_t): the store holds S_t / s = largest through x_{t-1}; the detector D_t / a = d_t, the
amount x_t beats the largest so far (0 if not a record); m1(t) = S_t / s + D_t / a.

Extra neurons (options of build_w4b):
  late=True   LG, late gate (ARC slot 9): LG_{t+1} = relu(D_t / a) = d_t, on exactly one step after a record. It takes
              over r1a's switch-off push: memory cell 0 reads -eps LG (before: the linear route carried +eps d and cell 0
              read -L); position 8's readout reads +eps LG so the function is unchanged. The linear route, which held
              ARC's slot 9 in the 16-neuron builds, becomes a helper (late only) or, once Q leaves the memory slots
              (level=True), takes Q's memory slot (it is the ninth stored position).
  level=True  C7, comparator with the largest (ARC slot 7): C7_{t+1} = g relu(m1(t) - x_{t+1}). Every reader that took
              the clocks' shared comparator level (Q, keep gate, linear route, readout rows) now reads C7 instead:
              each reader's clock weights w become w - mean(w) (the wave part only) and C7 gets sum(w). Exact while
              the clocks are on (their wave parts sum to zero over the three clocks).
  refs=True   C6 and C1, comparators with ARC's references, reading the late gate's push:
              C6_{t+1} = g relu(S_t / s - c6 LG_t - x_{t+1}): one step after a record it compares with the previous
                  largest, two steps after with the largest minus c6 of the jump (ARC's neuron 6: 1.01, then 0.23).
              C1_{t+1} = g relu(m1(t) - c1 LG_t - x_{t+1}): one step after a record the largest, two steps after the
                  largest minus c1 of the jump, i.e. about the previous largest (ARC's neuron 1: 0.18, then 0.95).
              c6 = 0.23, c1 = 0.95: ARC's measured weights of the previous largest two steps after a record
              (score_mechanism.py's constructed lists; two constants read off ARC). Without margin=True nothing
              reads C1 or C6: they hold ARC's references and receive the late gate's push, but decide nothing.
  cancel=True CR, a helper in no slot: CR_{t+1} = relu(D_t / a + k (x_{t+1} - m1(t))), k = a = 1000 (the detector's
              gain): d_t + k d_{t+1} when x_{t+1} is itself a record, about 0 otherwise (unless x_{t+1} comes within
              d_t / k of the largest). C1 and C6 read + c CR - c k D / a, which removes the late gate's push exactly
              when the number after a record is itself a record: the push then belongs to a largest already overtaken.
  margin=True The hiding margin (kappa = 0.5 of the jump, the twin's value) enters the wave through C1 two steps
              after a record, instead of from the detector one step after: (C7 - C1) / (g c1) = d exactly two steps
              after a record that still stands, 0 at other steps; if the number then clips C1 (it is the new
              second-largest), the margin shrinks with it. The readout reads it straight from C1 and C7 (record at step
              7) or from the gated late gate (record at step 8). Replacing C1 by C7 costs 20 points (LEDGER w4b-4).
  share6>0    (failed, LEDGER w4b-5) a share of the margin through C6 via a helper DV6 one step later; a record
              three steps after the first one then receives the old record's share. Not used by any candidate.
  Q goes to ARC's slot 8 when level=True (Q compares with the runner-up; ARC's 8 with the previous largest at both
  delays). The four hiding-wave neurons (flip, three clocks) are then helpers in no ARC slot.

MAP: one wider neuron per ARC slot, scaled to ARC's RMS activity on random lists (analysis read, changes no decision).
Free memory slots 12-15 get delay cells 1, 2, 5 and the linear route by activity correlation (match_twin's rule).
"""
import numpy as np
import algzoo.handbuilt_v23 as v23
from algzoo.handbuilt import FROM_PQ, _rot
from algzoo.handbuilt_v27b import build_v27b

C6_ARC, C1_ARC = 0.23, 0.95


def build_w4b(late=True, level=True, refs=True, cancel=False, margin=False, share6=0.0, k_cancel=1000.0, kappa=0.5, eps=0.1, a=1000.0, s=20.0, g=1e4, c6=C6_ARC, c1=C1_ARC):
    net = build_v27b(level_exact=True, eps=eps, a=a, s=s, g=g)
    wi0, wh0, wo0 = [np.asarray(p, float) for p in net.params]
    idx = dict(net.idx); idx["Q"] = idx.pop("mem7")
    extra = (["late"] if late else []) + (["c7"] if level else []) + (["c6", "c1"] if refs else []) + (["cancel"] if cancel else []) + (["dv6"] if margin and share6 > 0 else [])
    n = 16 + len(extra)
    for k, name in enumerate(extra): idx[name] = 16 + k
    wi = np.zeros(n); wh = np.zeros((n, n)); wo = np.zeros((10, n))
    wi[:16] = wi0; wh[:16, :16] = wh0; wo[:, :16] = wo0
    S, D, L, Y0 = idx["store"], idx["detect"], idx["linear"], idx["mem0"]
    C = [idx[f"clock{k}"] for k in range(3)]
    wave_neurons = set(C) | {idx["flip"]}
    if late:
        LG = idx["late"]
        wh[LG, D] = 1 / a
        wh[L, D] -= eps / a                 # the switch-off push leaves the linear route ...
        wh[Y0, LG] += -eps                  # ... and reaches memory cell 0 from the late gate (cell 0 reads -L)
        wo[8, LG] += eps                    # position 8 read L's push; now reads it from the late gate
    if level:
        assert np.allclose(FROM_PQ.sum(0), 0)
        C7 = idx["c7"]
        wh[C7, S] = g / s; wh[C7, D] = g / a; wi[C7] = -g
        for r in range(n):
            if r in wave_neurons or r == C7: continue
            w = wh[r, C].copy()
            if np.abs(w).sum() == 0: continue
            # each clock = level + wave part (wave parts sum to 0); C7 = level, so sum(w) moves onto C7
            wh[r, C7] += w.sum(); wh[r, C] = w - w.mean()
        for p in range(10):
            w = wo[p, C].copy(); wo[p, C7] += w.sum(); wo[p, C] = w - w.mean()
    if refs:
        assert late
        LG = idx["late"]
        for name, cc, det in (("c6", c6, False), ("c1", c1, True)):
            k = idx[name]
            wh[k, S] = g / s; wh[k, D] = g / a if det else 0.0; wh[k, LG] = -cc * g; wi[k] = -g
    if cancel:
        # CR_{t+1} = relu(d_t + k (x_{t+1} - m1(t))): d_t + k d_{t+1} if x_{t+1} is a record, about 0 otherwise
        CR, LG = idx["cancel"], idx["late"]
        wh[CR, D] = (1 - k_cancel) / a; wh[CR, S] = -k_cancel / s; wi[CR] = k_cancel
        for name, cc in (("c6", c6), ("c1", c1)):
            k = idx[name]
            wh[k, CR] += cc * g; wh[k, D] += -cc * g * k_cancel / a
    if margin:
        # The hiding margin (kappa of the jump) reaches the wave through C1 two steps after the record, instead of
        # from the detector one step after. C1 is pushed down by c1 x (gated late gate); C7 is not, so while both are
        # on, (C7 - C1) / (g c1) = d, the record's jump, exactly two steps after a record that still stands, and 0 at
        # every other step. If the number two steps after the record clips C1 (it lies within c1 d of the largest,
        # so it is the new second-largest), the delivery shrinks to (M - x) / c1, and the margin with it; the largest
        # is then hidden below the previous largest, which is below that number, so no answer depends on it.
        assert refs and cancel and level
        P_, C1, C7, CR, LG = idx["flip"], idx["c1"], idx["c7"], idx["cancel"], idx["late"]
        K = 1 + kappa; coef = v23.wave_coef("forward_safe"); Ap = np.hypot(coef[0], coef[1]); Af = coef[2]
        R = _rot(v23.TURN); e1 = np.array([1.0, 0.0])
        # detector starts only 1 x d of the wave (was K x d)
        wh[np.ix_(C, [D])] -= kappa * (FROM_PQ @ (R.T @ e1))[:, None] / a
        wh[P_, D] -= kappa * (Af / Ap) / a
        # the kappa part enters at r + 3 from the state at r + 2, turned on by two steps
        deliv = {C7: 1 / (g * c1), C1: -1 / (g * c1)}
        for j, v in deliv.items():
            wh[C, j] += v * (1 - share6) * kappa * FROM_PQ @ (R @ e1)
            wh[P_, j] += v * (1 - share6) * kappa * (Af / Ap)
        if share6 > 0:
            # a share of the margin through C6: DV6_{t+1} = relu((C7_t - C6_t) / g - LG_t) = c6 d three steps after a
            # record that stands (C6 holds c6 x the gated push, LG_t removes the detector term C7 has and C6 lacks).
            # The ReLU drops the negative values one step after a record when C6 clips (P < x < M) and after
            # back-to-back records. It enters the wave one step later (r + 4), turned by three steps.
            DV, C6 = idx["dv6"], idx["c6"]
            wh[DV, C7] = 1 / g; wh[DV, C6] = -1 / g; wh[DV, LG] = -1.0
            wh[C, DV] += share6 * kappa / c6 * FROM_PQ @ (R @ R @ e1)
            wh[P_, DV] += -share6 * kappa / c6 * (Af / Ap)
            # the linear route writes the wave's forward copy of position r + 3 from the clocks at r + 3, which do not
            # hold C6's share yet: it reads that share from DV6 (the only reader of the wave outside the wave neurons
            # and the readout)
            wh[L, DV] += share6 * kappa / c6 * (wh[L, C] @ FROM_PQ @ (R @ e1) + wh[L, P_] * Af / Ap)
        # readout: a record at step 7 or 8 has not had its kappa part enter the wave by the end; read it directly
        gated = {LG: 1.0, CR: -1.0, D: k_cancel / a}            # = d_8 at the end if step 9 is not a record
        for p in range(10):
            miss7 = wo[p, C] @ FROM_PQ @ (kappa * e1) - wo[p, P_] * kappa * Af / Ap         # state r + 2: clock holds d e1
            miss8 = wo[p, C] @ FROM_PQ @ (kappa * R.T @ e1) + wo[p, P_] * kappa * Af / Ap   # state r + 1
            for j, v in deliv.items(): wo[p, j] += miss7 * v
            for j, v in gated.items(): wo[p, j] += miss8 * v
            if share6 > 0:                                   # record at step 6: C6's share is still in DV6
                miss6 = share6 * kappa * (wo[p, C] @ FROM_PQ @ (R @ e1) + wo[p, P_] * Af / Ap)
                wo[p, DV] += miss6 / c6
    return wi, wh, wo, idx


def arc_map(wi, wh, wo, idx, n_lists=20000, seed=0):
    """MAP [16, n]: fixed slots from the design, slots 12-15 by activity correlation, RMS scale per row."""
    import torch as th
    from scipy.optimize import linear_sum_assignment
    from algzoo.core import example
    n = len(wi)
    x = th.randn(n_lists, 10, generator=th.Generator().manual_seed(seed), dtype=th.float64).numpy()
    def acts(Wi, Wh):
        h = np.zeros((len(x), len(Wi))); out = []
        for t in range(10):
            h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi, 0); out.append(h)
        return np.stack(out[1:], 1).reshape(-1, len(Wi))
    A = [np.asarray(p, float) for p in example()]
    Aa, Ah = acts(A[0], A[1]), acts(wi, wh)
    lay = {2: "store", 4: "detect", 5: "keep", 0: "mem0", 11: "mem3", 10: "mem4", 3: "mem6"}
    free = ["mem1", "mem2", "mem5"]
    if "late" in idx:
        lay[9] = "late"
        if "c7" in idx: free.append("linear")      # Q leaves the memory slots, the linear route takes its place
    else: lay[9] = "linear"
    if "c7" in idx:
        lay[7] = "c7"; lay[8] = "Q"
        if "c6" in idx: lay[6] = "c6"; lay[1] = "c1"
        else: lay[6] = "clock1"; lay[1] = "flip"
    else:
        lay.update({1: "flip", 7: "clock0", 6: "clock1", 8: "clock2"}); free.append("Q")
    slots = [12, 13, 14, 15]
    Cm = np.array([[abs(np.corrcoef(Ah[:, idx[f]], Aa[:, j])[0, 1]) for j in slots] for f in free])
    r, c = linear_sum_assignment(-Cm)
    for a_, b_ in zip(r, c): lay[slots[b_]] = free[a_]
    M = np.zeros((16, n))
    for j in range(16):
        k = idx[lay[j]]
        M[j, k] = np.sqrt((Aa[:, j] ** 2).mean()) / max(np.sqrt((Ah[:, k] ** 2).mean()), 1e-12)
    return M, lay
