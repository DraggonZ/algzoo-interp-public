"""v6b: the trained network's arrangement, approximate. One gate = keep gate (relu of the relative vote), as ARC's
neuron 5. Clock and memory are both driven every step by the largest so far (no new-largest signal); a new largest
wipes the clock (reset), so the memory keeps the old largest's wave while the clock restarts. The memory's mean is
removed every step and its level fed from the clock's level minus w*ab*largest, so a new largest dips it.
Votes enter the memory two steps after arrival (through the gate); positions 8 and 9 are read directly."""
import numpy as np
from algzoo.handbuilt import Net, FROM_PQ, TO_PQ, _rot

N9 = 9; TURN = 2 * np.pi / N9; TH = TURN * np.arange(N9)


def build_v6b(a=1000.0, s=20.0, g=1e4, g2=1e4, c=0.75, w=0.8, Kr=1.0, lam_c=1000.0, kappa=1.0, drive_mem=True):
    ab = kappa * (1 - c) / (1 - w)
    net = Net(d=16)
    S, D, G = (net.add(k) for k in ("store", "detect", "keep"))
    C = [net.add(f"clock{k}") for k in range(3)]
    M2 = net.add("second")
    Y = [net.add(f"mem{j}") for j in range(N9)]
    wi, wh, wo = net.params
    m1 = {S: 1 / s, D: 1 / a}
    clev = {k: 1 / (3 * g) for k in C}
    xprev = {S: 1 / s, D: 1 / a, **{k: -v for k, v in clev.items()}}
    wh[S, S] = 1; wh[S, D] = s / a
    wi[D] = a; wh[D, D] = -1; wh[D, S] = -a / s
    # keep gate: relu((1-c) x_prev - c (second - x_prev)+ - Kr * detect)
    V = {j: (1 - c) * v for j, v in xprev.items()}; V[M2] = -c / g2
    for j, v in V.items(): wh[G, j] += v
    wh[G, D] += -Kr
    # clock: driven ring-up. phase' = R phase + b m1, b = (I - R^-1) e0 * ab; level g m1; wiped by x > m1
    R = _rot(TURN); e0 = np.array([1.0, 0.0]); b = (np.eye(2) - R.T) @ e0
    wh[np.ix_(C, C)] = FROM_PQ @ R @ TO_PQ
    for j, v in m1.items(): wh[C, j] += g * v + 0
    for j, v in m1.items(): wh[C, j] += (FROM_PQ @ b) * v
    wi[C] = -g
    # second store
    wh[M2, M2] = 1.0
    for j, v in {S: 1 / s, **{k: -v for k, v in clev.items()}}.items(): wh[M2, j] += g2 * v
    wi[M2] = -g2
    # memory: shift, remove mean, level from the clock minus w*ab*m1_prev; driven like the clock (cos pattern)
    Psh = np.zeros((N9, N9))
    for j in range(N9): Psh[j, (j - 1) % N9] = 1
    wh[np.ix_(Y, Y)] = Psh - np.ones((N9, N9)) / N9
    pat = np.stack([np.cos(TH), np.sin(TH)], 1)            # memory neuron j <-> phase angle TH[j]
    for k, y in enumerate(Y):
        for j, v in clev.items(): wh[y, j] += lam_c * v
        wh[y, S] += -w * ab / s
        if drive_mem:
            for j, v in m1.items(): wh[y, j] += ab * (pat[k] @ b) * v
    wh[Y[0], G] += kappa
    # readout. Memory neuron j holds age j + 2 at the end -> position 7 - j (positions 0-7).
    centre = -R.T @ e0
    for p in range(8):
        jp = 7 - p
        wo[p, Y[jp]] += 1.0; wo[p, Y] -= 1.0 / N9
        spoke = pat[jp] @ np.linalg.matrix_power(R, 0)
        wo[p, C] += -ab * pat[jp] @ TO_PQ
        pass  # memory carries the same offset as the clock
    wo[8, G] += kappa                                          # position 8: its vote is still in the gate
    for j, v in V.items(): wo[9, j] += kappa * v               # position 9: its relative vote, read directly
    wo[9, D] += -1000 * kappa
    return net


def scores(net, X):
    wi, wh, wo = net.params
    h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    net = build_v6b()
    for vals in ({3: 1.0}, {3: 1.0, 6: 1.5}, {1: .8, 3: 1.0, 5: .7, 6: 1.5, 8: .9}, {2: .9, 4: 1.0, 7: .95}):
        x = np.full(10, -0.5)
        for k, v in vals.items(): x[k] = v
        print(vals, scores(net, x[None])[0])
    X = np.random.default_rng(27).standard_normal((300_000, 10))
    print("accuracy", (scores(net, X).argmax(1) == X.argsort(1)[:, -2]).mean())
