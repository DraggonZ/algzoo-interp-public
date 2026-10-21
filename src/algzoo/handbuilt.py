"""A 16-neuron 2nd-argmax RNN whose weights are set by hand from design rules.

Same form as AlgZoo's trained models: h <- ReLU(W_hh h + W_hi x), scores = W_oh h,
no biases. Each part below adds named neurons and the weights its rule implies.
Nothing here is fitted to accuracy or loss.
"""
import numpy as np

N_HIDDEN = 16
SEQ_LEN = 10


class Net:
    """Named neurons on top of the three weight arrays."""

    def __init__(self, d=N_HIDDEN, n=SEQ_LEN):
        self.w_hi = np.zeros(d)
        self.w_hh = np.zeros((d, d))
        self.w_oh = np.zeros((n, d))
        self.idx = {}

    def add(self, name):
        i = len(self.idx)
        if i >= len(self.w_hi):
            raise ValueError(f"no room for neuron {name!r}")
        self.idx[name] = i
        return i

    @property
    def params(self):
        return self.w_hi, self.w_hh, self.w_oh


def run(params, x):
    """Hidden states for inputs x [..., n]; returns array [..., n+1, d] (h_0 = 0 first)."""
    w_hi, w_hh, _ = params
    x = np.asarray(x, dtype=float)
    h = np.zeros(x.shape[:-1] + (len(w_hi),))
    hs = [h]
    for t in range(x.shape[-1]):
        h = np.maximum(h @ w_hh.T + x[..., t, None] * w_hi, 0.0)
        hs.append(h)
    return np.stack(hs, -2)


# Part 1: running maximum and new-largest detector.
#
# Rule (the exact form of ARC's excess detector + accumulator, handcrafted.py
# neurons 19 and 20): the detector D fires a * (x_t - m_{t-1}) when x_t beats the
# running maximum m_{t-1} = max(0, x_1..x_{t-1}), and is 0 otherwise. The store S
# adds the detector's output one step later, so S_t = s * m_{t-1}. Because S lags
# a step, D also subtracts its own last output (self-weight -1); S/s + D/a is then
# the true running maximum. The account of ARC's trained model has a detector
# input weight of about 10. v1 uses 100, because the gates below switch within
# 1/a of a tie (design v1, research/algzoo/20261017_handbuilt/design_v1.md).
DETECT_GAIN = 100.0  # a
STORE_GAIN = 20.0   # s


def add_running_max(net, a=DETECT_GAIN, s=STORE_GAIN):
    store, det = net.add("store"), net.add("detect")
    net.w_hh[store, store] = 1.0
    net.w_hh[store, det] = s / a
    net.w_hi[det] = a
    net.w_hh[det, det] = -1.0
    net.w_hh[det, store] = -a / s
    return store, det


def running_max_readout(net, h, a=DETECT_GAIN, s=STORE_GAIN):
    """The true running maximum max(0, x_1..x_t) read from hidden states h [..., d]."""
    return h[..., net.idx["store"]] / s + h[..., net.idx["detect"]] / a


def m1_prev(net, a=DETECT_GAIN, s=STORE_GAIN):
    """Coefficients on h_{t-1} that give m1_{t-1}, the running maximum after step t-1."""
    return {net.idx["store"]: 1 / s, net.idx["detect"]: 1 / a}


# Part 2: the second-largest so far, and two gates.
#
# m2 = second-largest of {0, 0, x_1..x_t}. A new number raises it by
# ReLU(x - m2) - ReLU(x - m1): by x - m2 for a new second, and by m1 - m2 when x
# is a new largest (the old largest becomes the second). detect2 fires
# a * ReLU(x - m2); store2 holds s * m2 one step late, as in part 1.
# Gate: detect - clip1 = min(a (x - m1), x) is x for a clear new largest and 0
# for anything at or below m1; it ramps across a band of width m1/a above m1.
# detect2 - clip2 is the same for m2. So (detect2 - clip2) - (detect - clip1) is
# about x for a new second, 0 otherwise.


def add_second_max(net, a=DETECT_GAIN, s=STORE_GAIN):
    store2, det2, clip1, clip2 = (net.add(k) for k in ("store2", "detect2", "clip1", "clip2"))
    det = net.idx["detect"]
    net.w_hh[store2, store2] = 1.0
    net.w_hh[store2, det2] = s / a
    net.w_hh[store2, det] = -s / a
    # pre-activation a x - a m2_{t-1}, with m2_{t-1} = store2/s + (detect2 - detect)/a
    for k, gain in ((det2, a), (clip2, a - 1)):
        net.w_hi[k] = gain
        net.w_hh[k, store2] = -a / s
        net.w_hh[k, det2] = -1.0
        net.w_hh[k, det] = 1.0
    # clip1: (a-1) x - a m1_{t-1}
    net.w_hi[clip1] = a - 1
    for j, c in m1_prev(net, a, s).items():
        net.w_hh[clip1, j] = -a * c
    last = net.add("last")
    net.w_hi[last] = 1.0
    return store2, det2, clip1, clip2, last


def m2_prev(net, a=DETECT_GAIN, s=STORE_GAIN):
    return {net.idx["store2"]: 1 / s, net.idx["detect2"]: 1 / a, net.idx["detect"]: -1 / a}


def second_max_readout(net, h, a=DETECT_GAIN, s=STORE_GAIN):
    return sum(h[..., j] * c for j, c in m2_prev(net, a, s).items())


def gates(net, h):
    """(new-largest gate, new-second gate) in input units, from hidden states h."""
    i = net.idx
    g1 = h[..., i["detect"]] - h[..., i["clip1"]]
    g2 = h[..., i["detect2"]] - h[..., i["clip2"]] - g1
    return g1, g2


# Part 3: rotating registers and the readout.
#
# A register is 3 neurons at angles 0, 120, 240 degrees. Neuron k holds
# level + Re(z * exp(-i theta_k)): a constant level (for gating) plus a phase z.
# Phase convention: a vote of value v that is `age` steps old is
# z = KAPPA * v * exp(i * TURN * age). The score of slot j reads
# Re(z * exp(-i TURN (n-1-j))) / KAPPA, which is largest at the slot whose age
# matches. With TURN = 36 degrees the ten ages are spread evenly round the circle.
# All three registers turn by TURN per step; the output never reads the level.
TURN = 2 * np.pi / SEQ_LEN
KAPPA = 1.0
LEVEL = 100.0  # g: the level is g * (value it guards); input weight -g wipes it

_TH = 2 * np.pi * np.arange(3) / 3
PHI = np.stack([np.ones(3), np.cos(_TH), np.sin(_TH)], 1)  # neurons = PHI @ (level, p, q)
TO_PQ = np.linalg.inv(PHI)[1:]  # (p, q) from neurons
FROM_PQ = PHI[:, 1:]


def _rot(w):
    return np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])


def _turn_map(sign=1.0, turn=TURN):
    """Neurons -> neurons: take the phase of one register, turn it, put it in another."""
    return sign * FROM_PQ @ _rot(turn) @ TO_PQ


def _inject(v_coef, turn=TURN, kappa=KAPPA):
    """Neuron weights that add phase kappa * value * exp(i turn) (a one-step-old vote)."""
    return FROM_PQ @ (kappa * np.array([np.cos(turn), np.sin(turn)])) * v_coef


def add_registers(net, g=LEVEL, turn=TURN, kappa=KAPPA, a=DETECT_GAIN, s=STORE_GAIN):
    i = net.idx
    A = [net.add(f"clockA{k}") for k in range(3)]
    B = [net.add(f"clockB{k}") for k in range(3)]
    V = [net.add(f"vote{k}") for k in range(3)]
    ix = np.ix_
    gate1 = {i["detect"]: 1.0, i["clip1"]: -1.0}
    gate2 = {i["detect2"]: 1.0, i["clip2"]: -1.0, i["detect"]: -1.0, i["clip1"]: 1.0}
    # clockA: level g*m1, turns, receives the new largest's phase, wiped by x > m1
    net.w_hh[ix(A, A)] = _turn_map(1, turn)
    # clockB: one step behind A, turned forward, so it holds A's phase; never wiped
    net.w_hh[ix(B, A)] = _turn_map(1, turn)
    for R in (A, B):
        for j, c in m1_prev(net, a, s).items():
            net.w_hh[R, j] += g * c
        for j, c in gate1.items():
            net.w_hh[R, j] += _inject(c, turn, kappa)
    net.w_hi[A] = -g
    # vote: level g*m2, turns, receives (B - A) turned (the old largest's phase when
    # A was just wiped) and the new second, wiped by x > m2
    net.w_hh[ix(V, V)] = _turn_map(1, turn)
    net.w_hh[ix(V, B)] = _turn_map(1, turn)
    net.w_hh[ix(V, A)] = _turn_map(-1, turn)
    for j, c in m2_prev(net, a, s).items():
        net.w_hh[V, j] += g * c
    for j, c in gate2.items():
        net.w_hh[V, j] += _inject(c, turn, kappa)
    net.w_hi[V] = -g
    # readout: vote + (B - A) + the last input's own vote (last - new-largest gate)
    n = net.w_oh.shape[0]
    psi = turn * (n - 1 - np.arange(n))
    read = np.stack([np.cos(psi), np.sin(psi)], 1) / kappa  # [n, 2]
    net.w_oh[:, V] = read @ TO_PQ
    net.w_oh[:, B] = read @ TO_PQ
    net.w_oh[:, A] = -read @ TO_PQ
    for j, c in {i["last"]: 1.0, **{k: -v for k, v in gate1.items()}}.items():
        net.w_oh[:, j] += np.cos(psi) * c
    return A, B, V


def build_v1(a=DETECT_GAIN, g=LEVEL):
    """a: detector gain (gates switch within 1/a of a tie); g: register level gain."""
    net = Net()
    add_running_max(net, a)
    add_second_max(net, a)
    add_registers(net, g, a=a)
    assert len(net.idx) == N_HIDDEN
    return net


def build_v2(a=DETECT_GAIN, g=LEVEL):
    """v1 with a split readout. Two votes on the cosine profile can elect a slot
    between them (v1 failure analysis). Stored votes are always at least one step
    old, so the last slot reads only the last input, and the last input votes only
    for the last slot."""
    net = build_v1(a, g)
    i = net.idx
    last_slot = net.w_oh.shape[0] - 1
    direct = [i["last"], i["detect"], i["clip1"]]
    keep = net.w_oh[last_slot, direct].copy()
    net.w_oh[:, direct] = 0.0
    net.w_oh[last_slot, :] = 0.0
    net.w_oh[last_slot, direct] = keep  # last - (detect - clip1), at cos(0) = 1
    return net


def phase(net, h, name):
    """Complex phase z of a register from hidden states h [..., d]."""
    pq = h[..., [net.idx[f"{name}{k}"] for k in range(3)]] @ TO_PQ.T
    return pq[..., 0] + 1j * pq[..., 1]



# v3: the trained network's way of voting (17 neurons).
#
# Every positive number that is not a new largest leaves a sharp vote in a shared
# memory of 9 neurons that shifts one neuron per step. The current largest sits in
# the same memory as a broad wave (a cosine over the 9 neurons), hidden by a
# 3-neuron cancelling clock with the opposite readout. When a bigger number
# arrives, the clock is wiped, so the wave shows. One step later the whole memory
# dips by w * Ab * P (P = the old largest): older, smaller votes are cleared and
# the wave is trimmed to a spike of size P at the old largest's position. The new
# largest's wave is injected one step after that. Position 9 reads the last input.
#
# Neurons: store, detect (running maximum, part 1); clip (gate1 = detect - clip
# is the new largest's value on a record step, as in v1); last = ReLU(x);
# gate_late = gate1 one step later; clockC0-2; mem0-8.
V3_SLOTS = 9
V3_TURN = 2 * np.pi / V3_SLOTS
V3_THETA = V3_TURN * np.arange(V3_SLOTS)


def build_v3(a=DETECT_GAIN * 10, s=STORE_GAIN, g=LEVEL * 10, w=0.8, lam=None, kappa=1.0, clock="gate"):
    """w: dip as a share of the wave's amplitude Ab (w = 0.8 trims the wave to a single
    spike; smaller w leaves spill on neighbours). Ab = kappa / (1 - w) makes the
    trimmed spike equal the old largest's value. lam: level that keeps the memory
    positive (default 2 * Ab). clock="ringup": instead of receiving the new largest's
    value through the gate, the clock is driven every step by the largest so far and
    restarts from zero when a new largest wipes it, as the trained network's cancelling
    neurons rebuild; the output removes the resulting fixed offset using the store."""
    ab = kappa / (1 - w)
    lam = 2 * ab if lam is None else lam
    net = Net(d=17)
    store, det = add_running_max(net, a, s)
    clip, last, late = (net.add(k) for k in ("clip", "last", "gate_late"))
    C = [net.add(f"clockC{k}") for k in range(3)]
    Y = [net.add(f"mem{j}") for j in range(V3_SLOTS)]
    w_hi, w_hh, w_oh = net.params
    m1 = m1_prev(net, a, s)
    gate1 = {det: 1.0, clip: -1.0}
    # clip: (a-1) x - a m1, as in part 2
    w_hi[clip] = a - 1
    for j, c in m1.items():
        w_hh[clip, j] = -a * c
    w_hi[last] = 1.0
    for j, c in gate1.items():
        w_hh[late, j] = c
    # cancelling clock: level g * m1; phase turns V3_TURN per step and receives the
    # new largest's value at angle 0; a number above m1 wipes it
    w_hh[np.ix_(C, C)] = _turn_map(1, V3_TURN)
    for j, c in m1.items():
        w_hh[C, j] += g * c
    e0 = np.array([1.0, 0.0])
    if clock == "gate":
        for j, c in gate1.items():
            w_hh[C, j] += FROM_PQ @ e0 * c
    else:
        # phase' = R phase + b m1. From a reset, phase_k - centre = R^(k-1) e0 m1 with
        # b = (I - R^-1) e0 and centre = (I - R)^-1 b m1 = -R^-1 e0 m1.
        R = _rot(V3_TURN)
        b = (np.eye(2) - R.T) @ e0
        centre = -R.T @ e0
        for j, c in m1.items():
            w_hh[C, j] += FROM_PQ @ b * c
    w_hi[C] = -g
    # memory: shift mem_{j-1} -> mem_j. Level lam * (m1 - gate1) of the previous step,
    # so it vanishes on the step after a record; its change is lam * (late + det/a - gate1).
    # On that step also dip by w * ab * P, with P = gate1 - det/a.
    for j in range(V3_SLOTS):
        w_hh[Y[j], Y[(j - 1) % V3_SLOTS]] = 1.0
    for y in Y:
        w_hh[y, late] += lam
        w_hh[y, det] += lam / a - lam - w * ab * (1 - 1 / a)
        w_hh[y, clip] += lam + w * ab
        # the new largest's wave, two steps after it arrived, centred on mem1
        w_hh[y, late] += ab * np.cos(V3_THETA[Y.index(y)] - V3_THETA[1])
    # sharp vote of the previous input into mem0; a new largest is suppressed
    w_hh[Y[0], last] += kappa
    w_hh[Y[0], det] += -kappa
    # readout: position p ends in mem(8 - p)
    for p in range(V3_SLOTS):
        j = V3_SLOTS - 1 - p
        w_oh[p, Y[j]] += 1.0
        w_oh[p, store] += -lam / s  # remove the level
        w_oh[p, late] += lam
        spoke = np.array([np.cos(V3_THETA[j]), np.sin(V3_THETA[j])])
        w_oh[p, C] += -ab * spoke @ TO_PQ  # cancel the wave
        if clock != "gate":
            # remove the clock's fixed offset, centre * m1; a new largest at the last step has
            # just wiped the clock, so use m1 - gate1 (zero in that case)
            for jj, c in m1.items():
                w_oh[p, jj] += ab * (spoke @ centre) * c
            for jj, c in gate1.items():
                w_oh[p, jj] -= ab * (spoke @ centre) * c
        w_oh[p, late] += ab * np.cos(V3_THETA[j])  # a record at step 8 has no wave in memory yet
    w_oh[V3_SLOTS, last] += kappa
    w_oh[V3_SLOTS, det] += -kappa
    assert len(net.idx) == 17
    return net


def build_v4(a=DETECT_GAIN * 10, s=STORE_GAIN, g=LEVEL * 10, w=0.8, lam=None, kappa=1.0, clock="gate"):
    """v3 without the last-input neuron: 16 neurons, the trained network's size.

    As in the trained network, the cancelling clock doubles as the store of the last
    input. Its input weight -g dips its level to g * (m1_prev - x), so one step later
    x = m1_prev - level / g, with m1_prev read from the store (which lags a step).
    A new largest wipes the clock; then the same expression gives m1_prev, which
    gate1 - detect / a (the old largest on a record step, 0 otherwise) cancels, so
    a new largest leaves no sharp vote."""
    net = build_v3(a, s, g, w, lam, kappa, clock)
    i = net.idx
    store, det, clip, last = i["store"], i["detect"], i["clip"], i["last"]
    C = [i[f"clockC{k}"] for k in range(3)]
    # x of the previous step, as coefficients on the current hidden state
    x_prev = {store: 1 / s, det: -1 + 1 / a, clip: 1.0, **{c: -1 / (3 * g) for c in C}}
    w_hi, w_hh, w_oh = net.params
    mem0 = i["mem0"]
    for W, row in ((w_hh, mem0), (w_oh, V3_SLOTS)):
        c_last = W[row, last]
        W[row, last] = 0.0
        W[row, det] += c_last  # build_v3 suppressed records with -kappa * detect; not needed now
        for j, c in x_prev.items():
            W[row, j] += c_last * c
    # drop the last-input neuron: move the final index into its place
    keep = [k for k in range(17) if k != last]
    small = Net(d=16)
    small.w_hi[:] = w_hi[keep]
    small.w_hh[:] = w_hh[np.ix_(keep, keep)]
    small.w_oh[:] = w_oh[:, keep]
    small.idx = {name: keep.index(k) for name, k in i.items() if k != last}
    return small


def build_v5(a=DETECT_GAIN * 10, s=STORE_GAIN, g=LEVEL * 100, w=0.8, delta=1e-3, lam=None, kappa=1.0, clock="gate"):
    """v4 with the clip neuron replaced by a fourth cancelling neuron, as in the trained network.

    The trained network has no neuron with a large positive input weight besides the
    detector; its four cancelling neurons all have large negative ones. Here the fourth
    one, z, holds g * ((1 + delta) * m1_prev - x): like the clock's level, but wiped only
    by numbers more than delta above the largest. The difference (z - clock level) / (g delta)
    is m1_prev when x is below the largest and 0 when x is a new largest, so
    gate1 = store / s - (z - clock level) / (g delta) + detect / a is the new largest's
    value on a record step, as the clip gave. Layout: store, detect (2); one gate, a step
    late (1); clock and z (4); memory (9)."""
    net = build_v4(a, s, g, w, lam, kappa, clock)
    i = net.idx
    store, det, clip = i["store"], i["detect"], i["clip"]
    C = [i[f"clockC{k}"] for k in range(3)]
    w_hi, w_hh, w_oh = net.params
    # clip = detect - gate1, written with z in its place (z takes over the clip's slot)
    clip_as = {det: 1 - 1 / a, store: -1 / s, clip: 1 / (g * delta), **{c: -1 / (3 * g * delta) for c in C}}
    for W in (w_hh, w_oh):
        col = W[:, clip].copy()
        W[:, clip] = 0.0
        for j, c in clip_as.items():
            W[:, j] += col * c
    z = clip
    w_hh[z, :] = 0.0
    w_hi[z] = -g
    for j, c in m1_prev(net, a, s).items():
        w_hh[z, j] = g * (1 + delta) * c
    net.idx = {("z" if k == "clip" else k): v for k, v in i.items()}
    return net
