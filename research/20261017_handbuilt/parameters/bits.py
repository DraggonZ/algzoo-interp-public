"""Surprise accounting in bits (README experiment 20). Quadratic picture: curvature of KL(ARC || w) at ARC in absolute
weight coordinates; prior N(m, diag s^2); bits to land within KL budget B = Gaussian rate-distortion (water-filling)
+ mismatch term for the prior mean. Explanations change the prior (exact values, constraints, signs, rule means)."""
import math
import numpy as np
import torch as th
from torch.func import jacrev, vmap
from algzoo.core import example

dev = "cuda"
W0 = [np.asarray(p, float) for p in example()]
w = np.concatenate([x.ravel() for x in W0]); w0 = th.tensor(w, device=dev)
names = [f"input->{j}" for j in range(16)] + [f"{i}->{j}" for j in range(16) for i in range(16)] + \
        [f"{i}->out{k}" for k in range(10) for i in range(16)]
idx = {n: k for k, n in enumerate(names)}


def logits(wv, x):
    wi, wh, wo = wv[:16], wv[16:272].view(16, 16), wv[272:].view(10, 16)
    h = th.zeros(16, dtype=wv.dtype, device=wv.device)
    for t in range(10):
        h = th.relu(wh @ h + wi * x[t])
    return wo @ h


X = th.randn(200_000, 10, generator=th.Generator(device=dev).manual_seed(31), device=dev, dtype=th.float64)
F = th.zeros(432, 432, device=dev, dtype=th.float64)
jac = vmap(jacrev(logits), in_dims=(None, 0))
for chunk in X.split(5000):
    J = jac(w0, chunk)
    lg = vmap(logits, in_dims=(None, 0))(w0, chunk); p = th.softmax(lg, -1)
    Jm = J - (p[:, :, None] * J).sum(1, keepdim=True)
    F += th.einsum("nk,nka,nkb->ab", p, Jm, Jm)
F = (F / len(X)).cpu().numpy()          # KL ~ 1/2 dw^T F dw, absolute coordinates
np.save("fisher_abs.npy", F)

groups = {"store/det": [2, 4], "gates": [5, 9], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
gof = {n: g for g, ns in groups.items() for n in ns}
block = []
for n in names:
    s, d = n.split("->")
    if d.startswith("out"):
        block.append(gof[int(s)] + " readout")
    else:
        block.append(gof[int(d)] + " incoming")
block = np.array(block)
s_prior = np.zeros(432)
for b in set(block):
    m = block == b; s_prior[m] = np.sqrt((w[m] ** 2).mean())
print("reference class: Gaussian per block, sd = " + ", ".join(f"{b} {s_prior[block == b][0]:.2f}" for b in sorted(set(block))))


def bits(prior_mean, prior_sd, B, keep=None, offset_kl=0.0):
    """Bits to land within KL B of ARC, weights ~ N(prior_mean, prior_sd^2) on the coordinates in keep (others exact).
    offset_kl: KL already used by fixed coordinates that differ from ARC (counted against the budget)."""
    keep = np.arange(432) if keep is None else np.asarray(keep)
    Bl = B - offset_kl
    if Bl <= 0:
        return math.inf, math.inf, math.inf
    S = prior_sd[keep]
    G = S[:, None] * F[np.ix_(keep, keep)] * S[None, :]
    mu = np.clip(np.linalg.eigvalsh(G), 0, None)  # KL of unit whitened error along direction: mu/2
    # water-filling: per-direction error variance v_i = min(1, 2 theta / mu_i), with sum mu_i v_i / 2 = Bl
    lo, hi = 1e-20, mu.max()
    for _ in range(200):
        th_ = math.sqrt(lo * hi)
        used = 0.5 * np.sum(mu * np.minimum(1, 2 * th_ / np.maximum(mu, 1e-300)))
        lo, hi = (th_, hi) if used < Bl else (lo, th_)
    rate = 0.5 * np.sum(np.log2(np.maximum(1, mu / (2 * th_))))
    z = (w[keep] - prior_mean[keep]) / S
    mismatch = 0.5 * np.sum(z ** 2) / math.log(2) - 0.5 * len(keep) / math.log(2)  # relative to an ARC-typical draw
    return rate + mismatch, rate, mismatch


B = 0.02
base_total, base_rate, base_mis = bits(np.zeros(432), s_prior, B)
print(f"\nlookup table (no explanation): {base_rate:.0f} bits of precision + {base_mis:.0f} bits of mismatch = {base_total:.0f} bits")
for Bx in [0.005, 0.01, 0.04, 0.08]:
    print(f"  at budget KL {Bx}: {bits(np.zeros(432), s_prior, Bx)[0]:.0f} bits")


def log2comb(n, k):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


# (a) negligible set: 28 weights exactly zero; KL used 0.0038 (restate.log)
neg = "14->7, 1->5, 3->8, 11->out5, 11->out2, 9->7, 13->8, 6->8, 3->4, 10->10, 5->4, 9->out7, 2->3, 9->out2, 15->9, 7->13, 6->9, 9->13, 9->out6, input->2, 9->out3, 5->out2, 9->out9, input->13, 15->4, 2->9, 13->5, 6->5".split(", ")
negi = [idx[n] for n in neg]
keep = [k for k in range(432) if k not in negi]
a_tot = bits(np.zeros(432), s_prior, B, keep, 0.0038)[0]
a_cost = log2comb(432, 28)
print(f"\n(a) negligible set: bits left {a_tot:.0f}, saved {base_total - a_tot:.0f}; naming the 28 costs {a_cost:.0f}; net {base_total - a_tot - a_cost:+.0f}")
# what the 28 cost in the lookup table on their own: bits with only them uncertain
sub = bits(np.zeros(432), s_prior, 0.0038, negi)[0]
print(f"    (for scale: describing just these 28 within KL 0.0038 would cost {sub:.0f} bits)")

# (b) blindness: 13 linear constraints, KL used 0.0185 (restate.log). Model: the constrained directions are exact.
wi, wh = W0[0], W0[1]
hid = [1, 6, 7, 8]; v = wi[hid] / np.linalg.norm(wi[hid])
from collections import Counter
Xs = np.random.default_rng(17).standard_normal((100_000, 10)); h = np.zeros((len(Xs), 16)); masks = []
for t in range(10):
    h = np.maximum(h @ wh.T + Xs[:, t:t + 1] * wi, 0); masks.append(h > 0)
top, _ = Counter(map(lambda r: r.tobytes(), np.concatenate(masks[4:9]))).most_common(1)[0]
D = np.frombuffer(top, dtype=bool).astype(float); u = D * wi; u /= np.linalg.norm(u)
C = []
for j in [0, 1, 2, 4, 7, 14, 15]:
    c = np.zeros(432); c[[16 + 16 * j + i for i in hid]] = v; C.append(c)
for j in [3, 6, 10, 11, 12, 13]:
    c = np.zeros(432); c[16 + 16 * j: 16 + 16 * j + 16] = u; C.append(c)
C = np.array(C)
# restrict to the null space of C (in whitened coordinates the constraint is linear too)
Q, _ = np.linalg.qr(C.T, mode="complete"); N_ = Q[:, 13:]          # basis of the 419-dim subspace (absolute coords)
Fs = N_.T @ F @ N_
# prior in the subspace: project the prior covariance
P = N_.T @ np.diag(s_prior ** 2) @ N_
L = np.linalg.cholesky(P)
G = L.T @ Fs @ L
mu = np.clip(np.linalg.eigvalsh(G), 0, None)
Bl = B - 0.0185
lo, hi = 1e-20, mu.max()
for _ in range(200):
    th_ = math.sqrt(lo * hi)
    used = 0.5 * np.sum(mu * np.minimum(1, 2 * th_ / np.maximum(mu, 1e-300)))
    lo, hi = (th_, hi) if used < Bl else (lo, th_)
rate_b = 0.5 * np.sum(np.log2(np.maximum(1, mu / (2 * th_))))
zb = np.linalg.solve(L, N_.T @ w)
mis_b = 0.5 * np.sum(zb ** 2) / math.log(2) - 0.5 * len(zb) / math.log(2)
b_tot = rate_b + mis_b
b_cost = log2comb(16, 13) + 13  # which 13 rows, and which class each row is in
print(f"\n(b) blindness (13 constraints, KL budget left {Bl:.4f}): bits left {b_tot:.0f}, saved {base_total - b_tot:.0f}; "
      f"stating it costs about {b_cost:.0f}; net {base_total - b_tot - b_cost:+.0f}")
# fairer comparison: the lookup table at the same reduced budget, so the saving is not mixed with budget use
ref = bits(np.zeros(432), s_prior, Bl)[0]
print(f"    lookup table at the same budget {Bl:.4f}: {ref:.0f} bits; saving at equal budget {ref - b_tot:.0f}")

# (c) signs: 27 right, 1 wrong; each right sign halves a Gaussian prior if the weight needs at least 1 bit
print(f"\n(c) signs: at most 27 bits saved (one per right sign) and about 1 bit lost for the wrong one; stating the "
      f"group rules (about 10 rules) costs roughly 10-20 bits; net about +5 to +15")

# (d) hiding rows from the rule fit (rule_fit.py): prior mean = fitted rows, prior sd = each row's rms error
import re
txt = open("rule_fit.log").read()
fitted = {}
for j in [1, 6, 7, 8]:
    blockj = txt.split(f"neuron {j}: rule")[1]
    line = [l for l in blockj.splitlines() if l.strip().startswith("fitted:")][0]
    fitted[j] = np.array([float(x) for x in line.split(":")[1].split()])
m = np.zeros(432); sd = s_prior.copy()
for j, fr in fitted.items():
    sl = slice(16 + 16 * j, 16 + 16 * j + 16)
    m[sl] = fr; sd[sl] = np.sqrt(((fr - wh[j]) ** 2).mean())
d_tot = bits(m, sd, B)[0]
for cost_per in [5, 10]:
    print(f"(d) hiding rows from the rule fit: bits left {d_tot:.0f}, saved {base_total - d_tot:.0f}; 24 constants at "
          f"{cost_per} bits each cost {24 * cost_per}; net {base_total - d_tot - 24 * cost_per:+.0f}")

# hypothetical rules of precision r (relative error per weight), in bits
ev = np.clip(np.loadtxt("curvature_eigenvalues.txt"), 0, None)
print("\nbits still needed beyond a hypothetical rule with relative error r per weight (KL 0.02):")
for r in [0.01, 0.05, 0.25, 1.0]:
    mu = ev * r ** 2
    lo, hi = 1e-20, mu.max()
    for _ in range(200):
        th_ = math.sqrt(lo * hi)
        used = 0.5 * np.sum(mu * np.minimum(1, 2 * th_ / np.maximum(mu, 1e-300)))
        lo, hi = (th_, hi) if used < B else (lo, th_)
    print(f"  r = {100 * r:5.1f}%: {0.5 * np.sum(np.log2(np.maximum(1, mu / (2 * th_)))):.0f} bits")


def bits_general(mean, Sigma, Bl):
    """Bits with a general Gaussian prior N(mean, Sigma) on all 432 weights."""
    ev_, U = np.linalg.eigh(Sigma); ev_ = np.clip(ev_, 1e-30, None)
    L = U * np.sqrt(ev_)
    G = L.T @ F @ L
    mu = np.clip(np.linalg.eigvalsh(G), 0, None)
    lo, hi = 1e-20, mu.max()
    for _ in range(200):
        th_ = math.sqrt(lo * hi)
        used = 0.5 * np.sum(mu * np.minimum(1, 2 * th_ / np.maximum(mu, 1e-300)))
        lo, hi = (th_, hi) if used < Bl else (lo, th_)
    rate = 0.5 * np.sum(np.log2(np.maximum(1, mu / (2 * th_))))
    z = (U.T @ (w - mean)) / np.sqrt(ev_)
    mis = 0.5 * np.sum(z ** 2) / math.log(2) - 0.5 * len(z) / math.log(2)
    return rate + mis, rate, mis


# (b') blindness as a prior: variance along each constraint vector shrunk to sigma^2, sigma one number read off ARC
Cn = C / np.linalg.norm(C, axis=1, keepdims=True)           # rows have disjoint supports per neuron -> orthonormal
dev_c = Cn @ w
sigma = np.sqrt((dev_c ** 2).mean())
S0 = np.diag(s_prior ** 2)
Pp = np.eye(432) - Cn.T @ Cn
Sig = Pp @ S0 @ Pp + sigma ** 2 * Cn.T @ Cn
bp = bits_general(np.zeros(432), Sig, B)
print(f"\n(b') blindness as a prior (each row's component along its blind direction ~ N(0, {sigma:.4f}^2); one number read off):"
      f" bits left {bp[0]:.0f}, saved {base_total - bp[0]:.0f}; stating it about {b_cost + 8:.0f} (rows, classes, sigma); "
      f"net {base_total - bp[0] - b_cost - 8:+.0f}")
print(f"    ARC's components along the blind directions: {np.round(dev_c, 4)}")
