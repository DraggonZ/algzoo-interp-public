"""Hand-built v4a: the twin's rules with a two-dimensional hiding wave (no flip neuron), which frees one neuron for a
late gate. Research notes: research/20261017_handbuilt/faithful_twin/LEDGER.md, branch r4a.

Base: the v23 twin at ARC's settings (imprint U = 1, keep share 0.25, memory fade 0.99, exact level bookkeeping, no
memory loop), with r3b's runner-up comparator Q and keep cut (handbuilt_v27b) and position 0 read from the runner-up.
Every rule not restated here is v23's or v27b's.

The hiding wave, general form. The three clock neurons hold the comparator level gc (m - x) on the shared direction
(1, 1, 1) and a 2-D wave state z on the other two (FROM_PQ / TO_PQ). A record x_r (jump d) starts z = b d one step
later; z then moves by z -> A z each step; the next record wipes it (the comparator clips all three).
  copy:     the linear route L(t+1) carries c z_t, so the slot of x_t holds g(t - r) d with g(k) = c A^(k-1) b;
  readout:  position p subtracts c A^(p-9) z_9 = g(p - r) d, so copies cancel while the largest stands and the largest
            itself (no copy: the clocks were wiped when it arrived) is hidden by g(0) d;
  backward: one step after the record the detector writes g(-(j+1)) d into delay cell j (pre-faded, as v23).
After the largest is overtaken the clocks are wiped and everything it wrote stays (spill). Rules for the spill:
  g(0) = K = 1 + kappa (hide below the previous largest by kappa d, v23's margin);
  g(k) <= 0 for k = 1..8 (positions after an overtaken largest hold smaller numbers and are never the answer; lowering
         them is harmless, raising them is what hurts);
  g(-k) < 1 for k >= 1 (positions before it are at least one jump below it).
The twin meets these with a 45-degree rotation plus a sign-flip part (four hiding neurons). Two dimensions meet them
with a growing rotation A = rho R(theta):  g(k) = K rho^k cos(theta k + phi) / cos(phi).
  theta = 20 deg, phi = 70 deg: forward angles 90..230 deg, so g(1) = 0 and g(2..8) < 0 (needs 8 theta + phi <= 270);
  rho = 4: backward g(-1) = 0.70, g(-2) = 0.24, then smaller. Round values fixed by the rules above, not by a score.
The wave grows to about rho^8 = 65,536 x the jump, so the clocks' comparator gain is gc = 1e10 (they clip only when a
number comes within about 1e-5 x the jump of the largest), and the linear route's level is lf = 2e5 so that it stays
positive with the most negative copy (|g(8)| = 1.6e5 x the jump; every jump is at most the largest so far).

The freed neuron, the late gate LG (ARC's slot 9): LG(t+1) = D(t) / a = d_t, on one step after a record only.
It carries v25's switch-off push: delay cell 0 reads -eps x LG, so the record's own cell switches off two steps after
it (ARC's neuron 0 at age 2). In the twin the same push went through the linear route.

Options (LEDGER.md, r4a-c, r4a-d):
  bw_late: the late gate also writes the record's backward wave: LG(t+1) writes g(-(j+1)) d into delay cell j+1, the
      values the detector wrote one step earlier into cell j (the cells shift, so the slots get the same numbers); a
      record at step 8 is completed by the readout reading LG in the final state. The network then needs the gate.
  ring_w: memory ring, last delay cell back into the linear route with weight w (the looped shared level removed through
      the store); the ring's modes turn by 22.5, 67.5, 112.5, 157.5 deg with modulus (w 0.99^6)^(1/8).
  ring_comp: the readout adds back the current largest's backward wave that the ring carries out of the not-yet-used
      cells into later slots (slot p gets -w 0.99^6 g(p - r - 8) d).
  lam: the memory (linear route and delay cells) stored spread across its neurons (orthogonal mix, level lam).
"""
import numpy as np
from algzoo.handbuilt import Net, FROM_PQ, TO_PQ, _rot

NM = 8


def wave_2d(theta_deg=20.0, phi_deg=70.0, rho_w=4.0, K=1.5):
    """A, b, c with g(k) = c A^(k-1) b = K rho^k cos(theta k + phi) / cos phi."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    A = rho_w * _rot(th)
    b = np.array([1.0, 0.0])
    c = K * rho_w / np.cos(ph) * np.array([np.cos(th + ph), -np.sin(th + ph)])
    return A, b, c


def g_of(A, b, c, k):
    return float(c @ np.linalg.matrix_power(A, k - 1) @ b) if k >= 1 else float(c @ np.linalg.matrix_power(np.linalg.inv(A), 1 - k) @ b)


def build_v4a(q=True, late=True, eps=0.1, share=0.25, kappa=0.5, rho=0.99, U=1.0, a=1000.0, s=20.0, gc=1e10,
              gq=1e4, lf=2e5, delta=0.01, cut=0.75, theta_deg=20.0, phi_deg=70.0, rho_w=4.0, bw_late=False,
              ring_w=0.0, ring_comp=False, lam=None, mix_seed=0):
    A0 = 1.0
    beta = share * A0
    alpha = A0 - beta
    K = 1 + kappa
    Aw, bw, cw = wave_2d(theta_deg, phi_deg, rho_w, K)
    Ai = np.linalg.inv(Aw)
    sign = -1.0
    net = Net(d=16)
    S, D, G, L = (net.add(k) for k in ("store", "detect", "keep", "linear"))
    C = [net.add(f"clock{k}") for k in range(3)]
    ncell = NM - 1 if q else NM
    Y = [net.add(f"mem{j}") for j in range(ncell)]
    Qn = net.add("runnerup") if q else None
    LG = net.add("late") if late else None
    wi, wh, wo = net.params

    def add(row, terms, cf=1.0, M=wh):
        for j, v in terms.items():
            M[row, j] += cf * v

    m1 = {S: 1 / s, D: 1 / a}                         # largest so far, read at the same step
    lev = {k: 1 / 3 for k in C}                       # clock level = gc (m(t-1) - x_t)
    xprev = {**m1, **{k: -v / gc for k, v in lev.items()}}   # the last input

    # largest so far
    wh[S, S] = 1; wh[S, D] = s / a
    wi[D] = a; wh[D, D] = -1; wh[D, S] = -a / s
    # clocks: level, 2-D wave z -> A z, started by the jump (z = b d one step after a record)
    wh[np.ix_(C, C)] = FROM_PQ @ Aw @ TO_PQ
    for k in C:
        add(k, m1, gc)
    wi[C] = -gc
    wh[C, D] += FROM_PQ @ bw / a
    zread = TO_PQ                                     # z from the clock neurons (blind to the level)
    # keep gate
    if q:
        xr = dict(xprev)
        for j, v in xr.items():
            wh[G, j] += beta * v
        wh[G, Qn] = -beta * (cut / (1 - cut)) / gq
        # runner-up comparator: Q(t+1) = gq relu(s_t - x_{t+1}), s_t = store/s + Q/gq - level/gc
        wh[Qn, S] = gq / s; wh[Qn, Qn] = 1.0; wh[Qn, C] = -gq / (3 * gc); wi[Qn] = -gq
    else:
        add(G, xprev, beta)
    # linear route: alpha (x - m1) + lf m1 + copy c z
    add(L, {k: -alpha / gc * v for k, v in lev.items()})
    add(L, m1, lf)
    wh[L, C] += cw @ zread
    # late gate and the switch-off push
    if late:
        wh[LG, D] = 1.0 / a
        push_src, push_w = LG, 1.0
    else:
        push_src, push_w = None, 0.0
    # memory
    for j in range(1, ncell):
        wh[Y[j], Y[j - 1]] = rho
    wh[Y[0], L] += sign; wh[Y[0], G] += sign
    add(Y[0], {S: 1 / s}, U - sign * (lf - alpha))
    if late:
        wh[Y[0], LG] += -eps                          # record's cell: -eps d two steps after the record
    else:
        wh[L, D] += eps / a                           # v25's route for the same push
    for j, y in enumerate(Y):
        wh[y, D] += U * rho ** j / a                  # level follows each jump (exact bookkeeping)
        if not bw_late:
            wh[y, D] += sign * g_of(Aw, bw, cw, -(j + 1)) * rho ** j / a   # backward wave
        elif j + 1 < ncell:                           # the same write one step later, one cell further, by the late gate
            wh[Y[j + 1], LG] += sign * g_of(Aw, bw, cw, -(j + 1)) * rho ** (j + 1)
    if ring_w:   # memory ring: last delay cell back into the linear route; the looped shared level removed via the store
        wh[L, Y[-1]] += ring_w
        wh[L, S] += -ring_w * U * rho ** (ncell - 1) / s

    def hide(p):
        u = cw @ np.linalg.matrix_power(Ai, 9 - p)
        wo[p, C] += -(u @ zread)
    for p in range(1 if q else 0, NM):
        wo[p, Y[7 - p]] += sign * rho ** -(7 - p)
        add(p, {S: 1 / s}, -sign * U, wo)
        hide(p)
    wo[8, L] += 1.0; wo[8, G] += 1.0
    add(8, {S: 1 / s}, -(lf - alpha), wo)
    hide(8)
    add(9, xprev, A0, wo)
    wo[9, D] += -A0 * K / a
    if ring_w and ring_comp:   # the ring carries the current largest's backward wave out of the not-yet-used cells into
        # later slots: slot p gets -w rho^(n-1) g(p - r - 8) d; the readout adds it back from the wave state
        for p in range(1 if q else 0, NM + 1):
            u = ring_w * rho ** (ncell - 1) * (cw @ np.linalg.matrix_power(Ai, 17 - p))
            wo[p, C] += u @ zread
    if bw_late:   # a record at step 8: its late gate is on in the final state; the readout adds the writes it would make
        for p in range(1 if q else 0, NM):
            wo[p, LG] += g_of(Aw, bw, cw, p - 8)
    if q:   # position 0 scores (1 - delta) x the final runner-up s_9 = store/s + Q/gq - level/gc
        for j, v in {S: 1 / s, Qn: 1 / gq, **{k: -1 / (3 * gc) for k in C}}.items():
            wo[0, j] += (1 - delta) * v
    if lam is not None:
        net = spread(net, [L] + Y, lam, mix_seed, s, a)
    return net


def spread(net, M, lam, seed, s, a):
    """The memory (linear route and delay cells) stored spread across its neurons: new values z = Qm y + lam m(t-1) 1,
    Qm an orthogonal mix that keeps the shared direction (v23.spread_memory's rule). Exact while every neuron is on;
    lam large keeps them on (the forward copies reach about 2e5 x the largest). Nothing outside M reads M except the
    readout."""
    wi, wh, wo = net.params
    S, D = net.idx["store"], net.idx["detect"]
    n = len(M); O = [k for k in range(len(wi)) if k not in M]
    assert np.allclose(wh[np.ix_(O, M)], 0)
    one = np.ones(n); P1 = np.outer(one, one) / n
    rng = np.random.default_rng(seed)
    B = np.linalg.qr(np.c_[one, rng.standard_normal((n, n - 1))])[0][:, 1:]
    Rr = np.linalg.qr(rng.standard_normal((n - 1, n - 1)))[0]
    Qm = P1 + B @ Rr @ B.T
    Wyy, Wyo, Wout = wh[np.ix_(M, M)].copy(), wh[np.ix_(M, O)].copy(), wo[:, M].copy()
    Am = Qm @ Wyy @ Qm.T
    wh[np.ix_(M, M)] = Am
    wh[np.ix_(M, O)] = Qm @ Wyo
    wh[M, S] += -lam * (Am @ one) / s + lam / s
    wh[M, D] += lam / a
    wi[M] = Qm @ wi[M]
    wo[:, M] = Wout @ Qm.T
    wo[:, S] += -lam * (Wout @ Qm.T @ one) / s
    return net


LAYOUT = {2: "store", 4: "detect", 5: "keep", 9: "late", 8: "runnerup", 1: "clock0", 6: "clock1", 7: "clock2",
          0: "mem0", 11: "mem3", 10: "mem4", 3: "mem6"}


def to_arc(net, free=("mem1", "mem2", "mem5", "linear"), slots=(12, 13, 14, 15), n=20000, seed=0):
    """Relabel into ARC's slots (LAYOUT; the four memory-like neurons without a switching job go to 12-15 by one-to-one
    activity correlation, match_twin.py's rule) and scale each neuron to ARC's partner's RMS activity."""
    import torch as th
    from scipy.optimize import linear_sum_assignment
    from algzoo.core import example, forward
    H = [np.asarray(p, float) for p in net.params]
    Ar = [np.asarray(p, float) for p in example()]
    x = th.randn(n, 10, generator=th.Generator().manual_seed(seed), dtype=th.float64)
    acts = lambda W: forward(W, x, return_hidden=True)[1][:, 1:].reshape(-1, 16).numpy()
    lay = dict(LAYOUT)
    Aa, Ah = acts(Ar), acts(H)
    ours = [net.idx[k] for k in free]
    Cm = np.array([[abs(np.corrcoef(Ah[:, i], Aa[:, j])[0, 1]) for j in slots] for i in ours])
    r, c = linear_sum_assignment(-np.nan_to_num(Cm))
    names = {v: k for k, v in net.idx.items()}
    for a_, b_ in zip(r, c):
        lay[slots[b_]] = names[ours[a_]]
    P = np.array([net.idx[lay[j]] for j in range(16)])
    M = (H[0][P], H[1][np.ix_(P, P)], H[2][:, P])
    rms = lambda W: np.sqrt((acts(W) ** 2).mean(0))
    k = rms(Ar) / np.maximum(rms(M), 1e-12)
    return M[0] * k, M[1] * k[:, None] / k[None, :], M[2] / k[None, :]
