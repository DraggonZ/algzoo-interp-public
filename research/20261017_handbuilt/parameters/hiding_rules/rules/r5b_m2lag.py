"""r5b, m2 one step back, on top of r4b_inherit: the second-largest the hiding neurons hear is the one that stood
before input t-1, lag(m2) (r4a's R4a-1 step, here combined with r4b's two links fixed at one). Path: the arriving
number x_t reaches a hiding neuron only through its input weight; the rest arrives from the state at t-1, and the
memory neurons that hold the second-largest feed the hiding neurons from that state, one step behind the rule's
"before t" second-largest.
Structure, declared honestly: 5 x 4 feature bits + 1 shared store (21) + naming which 2 of the 16 feedback links are
fixed at one, log2(16*15/2) = 7, + 1 bit for choosing the lag = 29. Same 28 constants as r4b_inherit.

Docstring of r4b_inherit:
r4b, inherited references: r3b_gates with two feedback links fixed at exactly 1.
Under blind feedback, h_k(t-1) - w_in,k x_{t-1} is neuron k's reference (its input minus the number it compared)
whenever k is on. Two links pass that reference on whole: neuron 7 keeps its own reference from step to step (7 -> 7),
and neuron 1 takes neuron 6's (6 -> 1). Path: ARC's weight 6 -> 1 is +0.995; 7 -> 7 is only +0.08 in ARC, so for 7
the "1" is carried by the combination of paths the blind form stands for (the x_{t-1} part through the detector,
the memory neurons and the hiding neurons), not by one weight.
Constants: 30 - 2 = 28. Structure: r3b_gates' 21 bits + 1 bit for each of the two links named as exactly one = 23.

Docstring of r3b_gates, otherwise unchanged:
r3b, gates: the hiding neurons hear a record's aftermath through the network's two gates, not as list quantities.
The rule runs inside ARC's network as r3b_closed_loop.py: the 12 non-hiding neurons run with ARC's own rows (stated
anyway), reading the rule's hiding activity. Input to hiding neuron j at step t:
    w_in,j x_t + c_j,m2 m2 + a h_2(t-1) + c_j,4 h_4(t-1) + c_j,9 h_9(t-1) + c_j,5 h_5(t-1)
        + sum_k b_j,k (h_k(t-1) - w_in,k x_{t-1})
(store 2, detector 4, late gate 9, early gate 5, all the network's own neurons one step back).

Mechanisms stated:
(1) the timer waves come from the hiding neurons' mutual feedback: no wave features, no angles;
(2) the largest so far reaches the hiding neurons only through the store and detector;
(3) the four hiding neurons take the same share a of the store;
(4) the store and detector are the network's own neurons 2 and 4, driven by all their inputs;
(5) new: what follows a record (the previous jump, the jump and new largest two steps back of earlier rules) reaches
    the hiding neurons through the gates 9 and 5, which fire after records. The previous-jump term of 6 and 8 was a
    stand-in for the store's and detector's missing inputs, and the two-steps-back terms for the late gate.
    Neuron 7 hears neither gate; neuron 8 hears only the late gate.
Constants: 1 shared store + 13 other feature constants + 16 feedback constants = 30.

Structure: one bit per (neuron, feature) choice over the 5 candidate features (m2, store, detector, late gate,
early gate), 5 x 4 = 20 bits, plus 1 bit for the shared store = 21. No hand-set numbers."""
import numpy as np
import torch as th
from common import HID, WI, WH, task_vars, lag

dev = "cuda"
NF = 5
FEATS = ["m2", "store", "detector", "late gate", "early gate"]
NET = [2, 4, 9, 5]                          # network neurons read for features 1..4
MASK = np.array([[1, 1, 1, 1, 1],         # neuron 1
                 [1, 1, 1, 1, 1],         # neuron 6
                 [1, 1, 1, 0, 0],         # neuron 7: no gate
                 [1, 1, 1, 1, 0]], bool)  # neuron 8: late gate only
FULL = np.concatenate([MASK, np.ones((4, 4), bool)], 1)
ONE = [(0, NF + 1), (2, NF + 2)]           # 6 -> 1 and 7 -> 7, fixed at exactly 1
OWN = FULL.copy(); OWN[:, 1] = False
for _i, _f in ONE:
    OWN[_i, _f] = False
NOWN = int(OWN.sum())
SRC = np.full(4 * (NF + 4), 1 + NOWN)
SRC[np.flatnonzero(OWN.ravel())] = 1 + np.arange(NOWN)
SRC[[i * (NF + 4) + 1 for i in range(4)]] = 0
SRC[[i * (NF + 4) + f for i, f in ONE]] = 2 + NOWN
KINDS = np.array(["feature"] + [("feature" if f < NF else "feedback") for i in range(4) for f in range(NF + 4) if OWN[i, f]])
STRUCTURE_BITS = NF * 4 + 1 + 7 + 1
OTHER = [n for n in range(16) if n not in HID]
win = th.tensor(WI[HID], device=dev)
_src = th.tensor(SRC, device=dev)
WH_o = th.tensor(WH[OTHER], device=dev)
WI_o = th.tensor(WI[OTHER], device=dev)
_hid = th.tensor(HID, device=dev); _oth = th.tensor(OTHER, device=dev); _net = th.tensor(NET, device=dev)


def prepare(X):
    return {"F": th.tensor(lag(task_vars(X)["m2"])[..., None], device=dev), "x": th.tensor(X, device=dev)}


def full(C):
    C = C.reshape(-1)
    return th.cat([C, C.new_zeros(1), C.new_ones(1)])[_src].view(4, NF + 4)


def rollout(d, C):
    F, x = d["F"], d["x"]; M = full(C)
    h = th.zeros(F.shape[0], 16, device=dev, dtype=F.dtype); pres = []
    for t in range(10):
        xt = x[:, t:t + 1]
        xp = x[:, t - 1:t] if t else th.zeros_like(xt)
        pre = (win * xt + F[:, t] @ M[:, :1].T + h[:, _net] @ M[:, 1:NF].T
               + (h[:, _hid] - xp * win) @ M[:, NF:].T)
        oth = th.relu(h @ WH_o.T + xt * WI_o)
        h = th.zeros_like(h).index_copy(1, _oth, oth).index_copy(1, _hid, th.relu(pre))
        pres.append(pre)
    return th.stack(pres, 1)


def init_constants(X, P, H):
    """One-step least squares on ARC's states, per neuron; the shared store coefficient starts at the mean of the four
    free values. The scorer's fit then refines all constants."""
    Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
    Fn = np.concatenate([lag(task_vars(X)["m2"])[..., None], Hp[..., NET]], -1)
    blindfb = Hp[..., HID] - lag(X)[..., None] * WI[HID]
    M = np.zeros((4, NF + 4))
    for i, n in enumerate(HID):
        A = np.concatenate([Fn[..., MASK[i]], blindfb], -1)
        on = (P[:, :, n] > 0).reshape(-1)
        c = np.linalg.lstsq(A.reshape(-1, A.shape[-1])[on], (P[:, :, n] - WI[n] * X).reshape(-1)[on], rcond=None)[0]
        M[i, FULL[i]] = c
    return np.concatenate([[M[:, 1].mean()], M[OWN]])


def counterparts(C):
    """For restatement.py: each free constant, its name, and the one ARC weight it corresponds to (None if it
    corresponds to no single weight). Feedback b_j,k <-> ARC's weight k -> j; the gate and detector coefficients <->
    ARC's weight from that neuron; the shared store coefficient stands for the four weights 2 -> j; m2 has none."""
    C = np.asarray(C).reshape(-1); out = [("store (shared)", C[0], None)]; k = 1
    for i, n in enumerate(HID):
        for f in range(NF + 4):
            if not OWN[i, f]:
                continue
            if f == 0:
                name, w = f"m2 -> {n}", None
            elif f < NF:
                name, w = f"{FEATS[f]} ({NET[f - 1]}) -> {n}", WH[n, NET[f - 1]]
            else:
                src = HID[f - NF]; name, w = f"hiding {src} -> {n}", WH[n, src]
            out.append((name, C[k], w)); k += 1
    return out
