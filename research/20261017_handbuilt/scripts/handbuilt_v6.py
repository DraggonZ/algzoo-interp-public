"""Hand-built v6: the trained network's layout (16 neurons).

store, detect: largest so far (as v1).
gate: fires one step late with the new largest's value: relu(x_prev - K * clock level / g); x_prev is read from the
      clock's dip, so no clip neuron. Delivers the new largest's wave to the clock and the memory two steps after it
      arrived.
clock (3) + second (1): four cancelling neurons with input weight -g. The clock hides the current largest's wave.
      'second' holds g2 * (second-largest - x)+, the second-largest store (ARC's neuron 8).
memory (9): shifts one neuron per step; its mean is removed every step and replaced by a level fed from the clock's
      level minus w * Ab * (largest so far). A new largest wipes the clock, so on the next step the level is
      exactly -w * Ab * P: a dip that trims the old largest's wave to a spike of size (1 - c) * P and clears older
      votes. Near ties the dip is partial.
Votes: every number votes (1 - c) x - c (second - x)+ on its own position (relative to the runner-up), new largests
excepted; the true second always gets (1 - c) * its value, every other number less.
"""
import numpy as np
from algzoo.handbuilt import Net, FROM_PQ, TO_PQ, _rot

N9 = 9; TURN = 2 * np.pi / N9; TH = TURN * np.arange(N9)


def build_v6(a=1000.0, s=20.0, g=1e4, g2=1e4, c=0.75, w=0.8, K=1000.0, Kr=1.0, lam_c=1000.0, kappa=1.0):
    ab = kappa * (1 - c) / (1 - w)
    net = Net(d=16)
    S, D, G = (net.add(k) for k in ("store", "detect", "gate"))
    C = [net.add(f"clock{k}") for k in range(3)]
    M2 = net.add("second")
    Y = [net.add(f"mem{j}") for j in range(N9)]
    wi, wh, wo = net.params
    m1 = {S: 1 / s, D: 1 / a}                       # largest after the previous step, from the previous state
    clev = {k: 1 / (3 * g) for k in C}              # clock level / g = (m1_prev - x)+ of that step
    # x of the previous step, from the current state: store/s - clock level/g + detect/a
    xprev = {S: 1 / s, D: 1 / a, **{k: -v for k, v in clev.items()}}
    # running max
    wh[S, S] = 1; wh[S, D] = s / a
    wi[D] = a; wh[D, D] = -1; wh[D, S] = -a / s
    # gate: relu(x_prev - K * clock level / g): the new largest's value one step late
    for j, v in xprev.items(): wh[G, j] += v
    for j, v in clev.items(): wh[G, j] += -K * v
    # clock: level g*m1, phase turns TURN, receives the gate at the angle of a two-step-old number, wiped by x > m1
    wh[np.ix_(C, C)] = FROM_PQ @ _rot(TURN) @ TO_PQ
    for j, v in m1.items(): wh[C, j] += g * v
    wh[C, G] += FROM_PQ @ np.array([np.cos(TH[1]), np.sin(TH[1])])
    wi[C] = -g
    # second-largest store: pre = M2 + g2 * min(x_prev, m1_prevprev) - g2 * x
    wh[M2, M2] = 1.0
    for j, v in {S: 1 / s, **{k: -v for k, v in clev.items()}}.items(): wh[M2, j] += g2 * v
    wi[M2] = -g2
    # memory: shift, remove the mean, add the level
    Psh = np.zeros((N9, N9))
    for j in range(N9): Psh[j, (j - 1) % N9] = 1
    wh[np.ix_(Y, Y)] = Psh - np.ones((N9, N9)) / N9
    for y in Y:
        for j, v in clev.items(): wh[y, j] += lam_c * v * g / g      # lam_c * (m1_prev - x)+
        wh[y, S] += -w * ab / s                                        # - w * ab * m1_prev (store lags a step)
        wh[y, G] += ab * np.cos(TH[Y.index(y)] - TH[1])                # the new largest's wave, two steps late
    # relative vote of the previous input into mem0; new largests suppressed by the detector
    for j, v in xprev.items(): wh[Y[0], j] += kappa * (1 - c) * v
    wh[Y[0], M2] += -kappa * c / g2
    wh[Y[0], D] += -kappa * Kr
    # readout
    for p in range(9):
        jp = 8 - p
        wo[p, Y[jp]] += 1.0
        wo[p, Y] -= 1.0 / N9
        wo[p, C] += -ab * np.array([np.cos(TH[jp]), np.sin(TH[jp])]) @ TO_PQ
    for j, v in xprev.items(): wo[9, j] += kappa * (1 - c) * v
    wo[9, M2] += -kappa * c / g2
    wo[9, D] += -kappa * Kr * 1000
    return net


def scores(net, X):
    wi, wh, wo = net.params
    h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    net = build_v6()
    for vals in ({3: 1.0}, {3: 1.0, 6: 1.5}, {1: .8, 3: 1.0, 5: .7, 6: 1.5, 8: .9}, {3: 1.0, 6: 1.5, 9: 1.2}, {2: .9, 4: 1.0, 7: .95}):
        x = np.full(10, -0.5)
        for k, v in vals.items(): x[k] = v
        print(vals, scores(net, x[None])[0])
    X = np.random.default_rng(27).standard_normal((300_000, 10))
    print("accuracy", (scores(net, X).argmax(1) == X.argsort(1)[:, -2]).mean())
