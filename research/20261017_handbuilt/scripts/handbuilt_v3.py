"""Hand-built v3: the trained network's way of voting, set by hand (17 neurons).

Every positive non-record number leaves a sharp vote in a shared 9-neuron memory that shifts one neuron per step.
The current largest sits in the same memory as a broad wave, hidden by a 3-neuron cancelling clock. When a bigger
number arrives, the clock is wiped (the wave shows) and, one step later, the whole memory dips by w * Ab * P
(P = old largest): older votes are cleared and the wave is trimmed to a spike of size P at the old largest's
position. The new largest's wave is injected one step after that. Position 9 is read from the last input.

Neurons: S, D (running max, as v2), K (clip, gives gate1 = D - K = new largest's value on a record),
Lst = relu(x), G = gate1 one step late, C0-2 (cancelling clock: level + phase), Y0-8 (vote memory).
"""
import numpy as np
from algzoo.handbuilt import Net, FROM_PQ, TO_PQ, _rot

N9 = 9
TURN9 = 2 * np.pi / N9
THETA = TURN9 * np.arange(N9)


def build_v3(a=1000.0, s=20.0, g=1000.0, w=0.8, lam=None, kappa=1.0, k=1.0):
    Ab = kappa / (1 - w)
    lam = 2 * Ab if lam is None else lam
    net = Net(d=17)
    S, D, K, Lst, G = (net.add(nm) for nm in ("S", "D", "K", "Lst", "G"))
    C = [net.add(f"C{i}") for i in range(3)]
    Y = [net.add(f"Y{j}") for j in range(N9)]
    wi, wh, wo = net.w_hi, net.w_hh, net.w_oh
    m1 = {S: 1 / s, D: 1 / a}                       # m1 after step t-1, read from h_{t-1}
    gate1 = {D: 1.0, K: -1.0}
    # running max
    wh[S, S] = 1; wh[S, D] = s / a
    wi[D] = a; wh[D, D] = -1; wh[D, S] = -a / s
    # clip: (a-1) x - a m1
    wi[K] = a - 1
    for j, c in m1.items(): wh[K, j] = -a * c
    wi[Lst] = 1
    for j, c in gate1.items(): wh[G, j] = c
    # cancelling clock: level g*m1, phase turns TURN9, gets the new largest's value at angle 0, wiped by x > m1
    wh[np.ix_(C, C)] = FROM_PQ @ _rot(TURN9) @ TO_PQ
    for j, c in m1.items(): wh[C, j] += g * c
    for j, c in gate1.items(): wh[C, j] += FROM_PQ @ np.array([1.0, 0.0]) * c
    wi[C] = -g
    # vote memory: shift Y_{j-1} -> Y_j
    for j in range(N9): wh[Y[j], Y[(j - 1) % N9]] = 1
    for jj in Y:
        # level change lam * (G + D/a - D + K): level = lam * (m1 - gate1) of the previous step
        wh[jj, G] += lam; wh[jj, D] += lam * (1 / a - 1); wh[jj, K] += lam
        # dip -w*Ab*P on the step after a record: P = gate1 - D/a
        wh[jj, D] += -w * Ab * (1 - 1 / a); wh[jj, K] += w * Ab
    # sharp vote of the previous input (records suppressed) into Y0
    wh[Y[0], Lst] += kappa; wh[Y[0], D] += -kappa * k
    # the new largest's wave, two steps after its arrival, centred on Y1
    for j in range(N9): wh[Y[j], G] += Ab * np.cos(THETA[j] - THETA[1])
    # readout
    for p in range(9):
        jp = 8 - p
        wo[p, Y[jp]] += 1
        wo[p, S] += -lam / s; wo[p, G] += lam                     # remove the level lam * (m1 - gate1)
        wo[p, C] += -Ab * np.array([np.cos(THETA[jp]), np.sin(THETA[jp])]) @ TO_PQ   # cancel the wave
        wo[p, G] += Ab * np.cos(THETA[jp] - THETA[0])             # record at step 8: wave not yet injected
    wo[9, Lst] += kappa; wo[9, D] += -kappa * k
    net.Ab, net.lam = Ab, lam
    return net


def scores(net, X):
    wi, wh, wo = net.params
    h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T


if __name__ == "__main__":
    rng = np.random.default_rng(10)
    X = rng.standard_normal((400_000, 10))
    tgt = X.argsort(1)[:, -2]
    for w in (0.8, 0.7, 0.6):
        net = build_v3(w=w)
        P = scores(net, X).argmax(1)
        print(f"w = {w}: accuracy {(P == tgt).mean():.4f}")
