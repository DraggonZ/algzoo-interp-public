"""Hand-built v7: the trained network's layout neuron group by neuron group (16 neurons).

store, detect (ARC 2, 4): largest so far.
keep (ARC 5): relu((1 - c) x_prev - c (second - x_prev)+ - Kr * detect): a vote relative to the runner-up, one step
      late; zero for forgotten numbers and new largests. x_prev is read from the clock's dip.
late (ARC 9): relu(x_prev - K * clock level / g): fires one step after a new largest (when the clock has been wiped),
      with its value.
clock (3) + second (1) (ARC 1, 6, 7, 8): input weight -g. The clock hides the current largest's wave (phase
      injected by 'late'); 'second' stores the second-largest so far.
memory (8) (ARC 0, 3, 10-15): an 8-slot cyclic shift holding votes two steps old and the current largest's wave;
      its mean is removed each step and its level is fed from the clock's level minus w * ab * largest-so-far, so the
      step after a new largest dips it (trimming the old largest's wave to a spike, clearing older votes).
Positions 8 and 9 are read directly: 8 from 'keep' (its vote is still there), 9 from its relative vote.
"""
import numpy as np
from algzoo.handbuilt import Net, FROM_PQ, TO_PQ, _rot

NM = 8; TURN = 2 * np.pi / NM; TH = TURN * np.arange(NM)


def set_slots(n):
    """Number of dial angles = memory neurons (8 in v7; 9 puts position 8 on the dial too)."""
    global NM, TURN, TH
    NM = n; TURN = 2 * np.pi / NM; TH = TURN * np.arange(NM)


def build_v7(a=1000.0, s=20.0, g=1e4, g2=1e4, c=0.75, w=0.8, K=1000.0, Kr=1.0, lam_c=1000.0, kappa=1.0, clock="ringup", alpha=1.0, delta=1e-3, second=True, c1=0.0):
    ab = kappa * (1 - c) / (1 - w)
    net = Net(d=7 + NM + (clock == "z") + second)
    S, D, G, L = (net.add(k) for k in ("store", "detect", "keep", "late"))
    C = [net.add(f"clock{k}") for k in range(3)]
    M2 = net.add("second") if second else None
    Y = [net.add(f"mem{j}") for j in range(NM)]
    wi, wh, wo = net.params
    m1 = {S: 1 / s, D: 1 / a}
    clev = {k: 1 / (3 * g) for k in C}
    xprev = {S: 1 / s, D: 1 / a, **{k: -v for k, v in clev.items()}}
    wh[S, S] = 1; wh[S, D] = s / a
    wi[D] = a; wh[D, D] = -1; wh[D, S] = -a / s
    V = {j: (1 - c) * v for j, v in xprev.items()}
    if second: V[M2] = V.get(M2, 0) - c / g2
    for j, v in clev.items(): V[j] = V.get(j, 0) - c1 * v      # forget relative to the largest: -c1 (m1 - x_prev)
    for j, v in V.items(): wh[G, j] += v
    wh[G, D] += -Kr
    for j, v in xprev.items(): wh[L, j] += v
    for j, v in clev.items(): wh[L, j] += -K * v
    # clock: phase turns TURN; receives 'late' at the angle of a two-step-old number; level g m1; wiped by x > m1
    wh[np.ix_(C, C)] = FROM_PQ @ _rot(TURN) @ TO_PQ
    for j, v in m1.items(): wh[C, j] += g * v
    R = _rot(TURN); Ri = R.T; e1 = np.array([np.cos(TH[0]), np.sin(TH[0])])
    if clock in ("late", "z"):
        alpha = 0.0
    if clock == "z":
        Z = net.add("z")
        wi[Z] = -g
        for j, v in m1.items(): wh[Z, j] += g * (1 + delta) * v
        # m1_prev * [no record] = (z/g - clock level/g) / delta; gate1 = m1 - that
        gate1 = {S: 1 / s, D: 1 / a, Z: -1 / (g * delta), **{k: v / delta for k, v in clev.items()}}
        # the clock receives the new largest's value one step after it, at the angle it will have one step later
        for j, v in gate1.items(): wh[C, j] += FROM_PQ @ (_rot(TURN).T @ np.array([1.0, 0.0])) * v
    if alpha < 1 and clock != "z":
        wh[C, L] += (1 - alpha) * FROM_PQ @ e1
    if alpha > 0:
        # driven by the largest so far, restarted by a wipe: phase_k = centre + R^(k-2) e1 m1 for k >= 1 steps after the
        # reset, matching a wave delivered two steps after the new largest. b = (I - R^-1) R^-1 e1, centre = -R^-2 e1 m1
        b = (np.eye(2) - Ri) @ Ri @ e1
        centre = -Ri @ Ri @ e1
        for j, v in m1.items(): wh[C, j] += alpha * (FROM_PQ @ b) * v
    wi[C] = -g
    if second:
        wh[M2, M2] = 1.0
        for j, v in {S: 1 / s, **{k: -v for k, v in clev.items()}}.items(): wh[M2, j] += g2 * v
        wi[M2] = -g2
    Psh = np.zeros((NM, NM))
    for j in range(NM): Psh[j, (j - 1) % NM] = 1
    wh[np.ix_(Y, Y)] = Psh - np.ones((NM, NM)) / NM
    for k, y in enumerate(Y):
        for j, v in clev.items(): wh[y, j] += lam_c * v
        wh[y, S] += -w * ab / s
        wh[y, L] += ab * np.cos(TH[k] - TH[0])           # the new largest's wave, centred where its vote would sit
    wh[Y[0], G] += kappa
    # readout: memory neuron j holds a number from position 7 - j at the end
    for p in range(NM):
        jp = 7 - p
        wo[p, Y[jp]] += 1.0; wo[p, Y] -= 1.0 / NM
        wo[p, C] += -ab * np.array([np.cos(TH[jp]), np.sin(TH[jp])]) @ TO_PQ
        if alpha > 0:
            for j, v in m1.items(): wo[p, j] += alpha * ab * (np.array([np.cos(TH[jp]), np.sin(TH[jp])]) @ centre) * v
            # a new largest at step 8 leaves the clock one step into its rebuild, hiding a wave the memory does not
            # hold yet; 'late' fires at the end only then, with that value (ARC's neuron 9 readout does this)
            wo[p, L] += alpha * ab * np.array([np.cos(TH[jp]), np.sin(TH[jp])]) @ (Ri @ e1)
        if clock == "z":
            # a new largest at step 8 sits in the clock one step early, not yet in the memory
            wo[p, L] += ab * np.array([np.cos(TH[jp]), np.sin(TH[jp])]) @ (Ri @ e1)
    wo[8, G] += kappa
    for j, v in V.items(): wo[9, j] += kappa * v
    wo[9, D] += -1000 * kappa
    return net


def scores(net, X):
    wi, wh, wo = net.params
    h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    net = build_v7()
    for vals in ({3: 1.0}, {3: 1.0, 6: 1.5}, {1: .8, 3: 1.0, 5: .7, 6: 1.5, 8: .9}, {2: .9, 4: 1.0, 7: .95}, {3: 1.0, 5: 1.5, 9: 1.2}):
        x = np.full(10, -0.5)
        for k, v in vals.items(): x[k] = v
        print(vals, scores(net, x[None])[0])
    X = np.random.default_rng(27).standard_normal((300_000, 10))
    P = scores(net, X).argmax(1); sec = X.argsort(1)[:, -2]
    print("accuracy", (P == sec).mean(), "| by position of the second:", [round((P[sec == p] == p).mean(), 2) for p in range(10)])
