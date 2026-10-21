"""Decoders from ARC's state at step t (t = 0..10, h_t after t numbers) to ten position votes.

(a) zero continuation (exact): run the remaining 10 - t steps with input 0 and read the scores. Given the on/off
    masks along that continuation the map h_t -> h_10 is linear, so each score splits exactly into the parts coming
    from each neuron of h_t (sum over source neurons = score). Votes are centred over the ten positions.
(b) linear pushback: P_t = W_oh (D W_hh)^(10 - t) with one fixed on/off pattern D, restricted to a block of neurons
    (memory only, or memory and hiding); applied to that block of h_t. Centred over positions.
Nothing here is fitted."""
from collections import Counter
import numpy as np
from algzoo.core import example

wi, wh, wo = example()
MEM = [0, 3, 10, 11, 12, 13, 14, 15]; HID = [1, 6, 7, 8]; STORE = [2, 4]; GATES = [5, 9]
GROUPS = {"store": STORE, "gates": GATES, "hiding": HID, "memory": MEM}
WOC = wo - wo.mean(0)  # centred readout: the common shift is free


def run(X):
    """States h_0..h_10, shape [n, 11, 16]."""
    h = np.zeros((len(X), 16)); hs = [h]
    for t in range(X.shape[1]):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); hs.append(h)
    return np.stack(hs, 1)


def zero_split(h, t):
    """h: [n,16] state at step t. Returns per-source contributions C [n, 16 sources, 10 positions] (centred),
    with C.sum(1) = centred zero-continuation scores."""
    n = len(h)
    A = np.broadcast_to(np.eye(16), (n, 16, 16)).copy()  # A maps h_t -> current state: cur = A @ h_t
    cur = h.copy()
    for _ in range(10 - t):
        pre = cur @ wh.T
        on = pre > 0
        cur = np.where(on, pre, 0)
        A = on[:, :, None] * np.einsum("ij,njk->nik", wh, A)
    # score_q = sum_k WOC[q] . A[:, :, k] h_t[k]
    return np.einsum("qi,nik,nk->nkq", WOC, A, h)


def common_pattern(H, steps=range(4, 9)):
    top, cnt = Counter(map(lambda r: r.tobytes(), np.concatenate([H[:, t] > 0 for t in steps]))).most_common(1)[0]
    return np.frombuffer(top, dtype=bool).copy(), cnt / (len(H) * len(list(steps)))


def pushback(t, on, block):
    """Fixed 10 x len(block) matrix: centred readout pushed back 10 - t steps through D W restricted to block."""
    Mb = (np.diag(on.astype(float)) @ wh)[np.ix_(block, block)]
    return WOC[:, block] @ np.linalg.matrix_power(Mb, 10 - t)


def pushback_full(t, on):
    """Same through the whole network's D W (all 16 neurons), then restricted to columns later."""
    M = np.diag(on.astype(float)) @ wh
    return WOC @ np.linalg.matrix_power(M, 10 - t)


def contributions(X):
    """Exact split of the state into per-position parts: returns H [n,11,16] and J [n,11,16,10] with
    H[:, t] = sum_p J[:, t, :, p] * X[:, p] (no biases, so this holds exactly)."""
    n, T = X.shape
    h = np.zeros((n, 16)); J = np.zeros((n, 16, T)); Hs = [h]; Js = [J]
    for t in range(T):
        pre = h @ wh.T + X[:, t:t + 1] * wi
        on = pre > 0
        J = np.einsum("ij,njp->nip", wh, J)
        J[:, :, t] += wi
        J = J * on[:, :, None]
        h = np.where(on, pre, 0)
        Hs.append(h); Js.append(J)
    return np.stack(Hs, 1), np.stack(Js, 1)


def labels(X, H):
    """Per position p: 'record', 'kept', 'dropped' (keep gate = neuron 5 in h_{p+2}); and the step at which a record
    is overtaken (index r of the overtaking number, or -1)."""
    n, T = X.shape
    runmax = np.maximum.accumulate(X, 1)
    rec = np.zeros((n, T), bool); rec[:, 0] = True; rec[:, 1:] = X[:, 1:] > runmax[:, :-1]
    gate = np.zeros((n, T), bool)
    for p in range(T - 1):
        gate[:, p] = H[:, p + 2, 5] > 0
    kind = np.where(rec, 0, np.where(gate, 1, 2))  # 0 record, 1 kept, 2 dropped (p = 9: no gate step)
    kind[:, T - 1] = np.where(rec[:, T - 1], 0, 3)
    over = -np.ones((n, T), int)
    for p in range(T - 1):
        later = (X[:, p + 1:] > X[:, p:p + 1]) & rec[:, p:p + 1]
        has = later.any(1)
        over[has, p] = p + 1 + later[has].argmax(1)
    return kind, over
