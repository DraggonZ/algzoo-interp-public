"""Surprise accounting for the hiding-neuron rules (README experiment 23)."""
import math
import numpy as np
from harness import *  # noqa
import step8_lategate as s8
import step11_more as s11
import step5_lean as s5

F = np.load("../fisher_abs.npy")
W0 = s8.W0; w = np.concatenate([x.ravel() for x in W0]); WH = simlib.WH
names = [f"input->{j}" for j in range(16)] + [f"{i}->{j}" for j in range(16) for i in range(16)] + \
        [f"{i}->out{k}" for k in range(10) for i in range(16)]
groups = {"store/det": [2, 4], "gates": [5, 9], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
gof = {n: g for g, ns in groups.items() for n in ns}
block = np.array([gof[int(n.split("->")[0])] + " readout" if n.split("->")[1].startswith("out") else gof[int(n.split("->")[1])] + " in" for n in names])
s_prior = np.zeros(432)
for b in set(block):
    m = block == b; s_prior[m] = np.sqrt((w[m] ** 2).mean())
B = 0.02


def bits(mean, sd, keep=None):
    keep = np.arange(432) if keep is None else np.asarray(keep)
    S = sd[keep]; G = S[:, None] * F[np.ix_(keep, keep)] * S[None, :]
    mu = np.clip(np.linalg.eigvalsh(G), 0, None)
    lo, hi = 1e-20, mu.max()
    for _ in range(200):
        th_ = math.sqrt(lo * hi)
        used = 0.5 * np.sum(mu * np.minimum(1, 2 * th_ / np.maximum(mu, 1e-300)))
        lo, hi = (th_, hi) if used < B else (lo, th_)
    rate = 0.5 * np.sum(np.log2(np.maximum(1, mu / (2 * th_))))
    z = (w[keep] - mean[keep]) / S
    return rate + 0.5 * np.sum(z ** 2) / math.log(2) - 0.5 * len(keep) / math.log(2)


base = bits(np.zeros(432), s_prior)
rows = {n: np.arange(16 + 16 * n, 32 + 16 * n) for n in HID}
hid_idx = np.concatenate(list(rows.values()))
exact = bits(np.zeros(432), s_prior, [k for k in range(432) if k not in hid_idx])
print(f"lookup table: {base:.0f} bits; with the 64 hiding weights known exactly: {exact:.0f} (upper bound on saving {base - exact:.0f})")


def lean(X, n):
    V = task_vars(X)
    return np.concatenate([s5.make(["sd", "xprev", "lategate"])(X, n), np.stack([lag(V["jump"], 2), lag(V["newmax"] * V["x"], 2)], -1)], -1)


def rich(X, n):
    V = task_vars(X)
    return np.concatenate([s11.feat(X, n), np.stack([lag(V["jump"], 2), lag(V["newmax"] * V["x"], 2)], -1)], -1)


def joint_kl(fitted):
    P = [x.copy() for x in W0]
    for n in HID:
        P[1][n] = fitted[n]
    l = s8.run_g(P)
    return (s8.pb * (s8.lb - l)).sum(-1).mean().item()


def quant(c, b):
    c = np.asarray(c, float); out = np.zeros_like(c); nz = c != 0
    e = np.floor(np.log2(np.abs(c[nz]))) + 1
    step = 2.0 ** (e - b); out[nz] = np.round(c[nz] / step) * step
    return out


if __name__ == "__main__":
    for lab, feat, fname, shared in [("rich", rich, "rule_timing_constants.npy", 15), ("lean", lean, "rule_timing_lean_constants.npy", 4 + 1)]:
        Cn = np.load(fname); nconst = Cn.size
        co = {n: Cn[i] for i, n in enumerate(HID)}
        fitted = s8.weights_from_rule(feat, co)
        kl0 = joint_kl(fitted)
        m = np.zeros(432); sd = s_prior.copy()
        for n in HID:
            m[rows[n]] = fitted[n]; sd[rows[n]] = np.sqrt(((fitted[n] - WH[n]) ** 2).mean())
        left = bits(m, sd)
        print(f"\n{lab} rule: {nconst} constants (+{shared} shared, +4 spreads); four rows from it: KL {kl0:.3f}")
        print(f"  bits left given the rule's weights {left:.0f}; saved {base - left:.0f} of at most {base - exact:.0f}")
        print(f"  row spreads (rms gap from ARC): " + ", ".join(f"{n}: {np.sqrt(((fitted[n] - WH[n]) ** 2).mean()):.3f}" for n in HID))
        chosen = None
        for b in [2, 3, 4, 5, 6, 8, 10]:
            coq = {n: quant(co[n], b) for n in HID}
            kq = joint_kl(s8.weights_from_rule(feat, coq))
            print(f"  constants rounded to {b:2d} mantissa bits: joint KL {kq:.3f} ({kq / kl0:.2f}x)", flush=True)
            if chosen is None and kq < 1.1 * kl0:
                chosen = b
        chosen = chosen or 12
        cost = (nconst + shared + 4) * (chosen + 4)
        print(f"  mantissa needed: {chosen} bits; stating the rule: ({nconst} + {shared} + 4) x ({chosen} + 4) = {cost} bits")
        print(f"  NET: saved {base - left:.0f} - cost {cost} = {base - left - cost:+.0f} bits")
