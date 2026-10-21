"""Version 2 (2026-10-21, between rounds 5 and 6; v1 = score_memory_rule.py, unchanged). The only change: the lookup
table's reference class has one spread per (source group -> target group) block, with the input and the readout as
their own source and target: groups input, store/detector, gates, hiding, memory, readout; 4 x 4 recurrent blocks,
4 input blocks, 4 readout blocks = 24 spreads (v1: 8, one per target group with input weights mixed in). Spreads are
not charged in the lookup total, as in v1; with them it would be 24 x 6 = 144 bits more, and a rule's description of
the uncovered weights would pay the same, so net bits do not depend on it.

Frozen scorer for rules on the memory's and the readout's weights (same accounting as
../hiding_rules/score_rule.py, adapted to rules that state weights directly). Do not edit once a loop uses it.

Usage:  python -u score_memory_rule.py rules/<module>.py

A rule module defines:
  COVERED: list of weight names it sets, from "input->j", "i->j" (from neuron i to neuron j), "i->outk" (readout)
  init_constants() -> numpy array C0 (read off ARC's weights or activity; never fitted to answers)
  weights(C) -> torch float64 tensor, one value per COVERED name, differentiable in C
  KINDS: array of C's shape naming each constant's kind (one reference spread per kind)
  STRUCTURE_BITS: bits for the rule's form (about 1 bit per choice, 6 bits per hand-set number)
  optional fit(C0, target, F) -> C. Default: Adam on the KL curvature-weighted squared distance of the rule's
    weights from ARC's, (w(C) - w_ARC)^T F (w(C) - w_ARC) over the covered weights (reading constants off ARC's
    weights, allowed by the project's rules; no accuracy or answers used).

Printed:
  KL of the network with the rule's weights in place of ARC's (2^22 lists, seed 7), and accuracy change;
  net bits = lookup table - (constants + structure + rest given the rule), best of shares 0.001, 0.002, 0.005 of the
  KL budget 0.02 for the constants. Constants: water-filling on J^T F J, absolute reference with one spread per kind.
  Rest: lookup-style bits of all 432 weights with mean = rule's weights on covered ones, spread = rms error of the
  rule per covered row (one row = covered weights into one neuron or one readout position); 6 bits per spread.
"""
import hashlib
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np
import torch as th
sys.path.insert(0, "/data/output/src/algzoo/src")
from algzoo.core import example

FISHER = "/data/output/research/algzoo/20261017_handbuilt/parameters/fisher_abs.npy"
BUDGET = 0.02
SHARES = [0.001, 0.002, 0.005]
SPREAD_BITS = 6
dev = "cuda"
WI, WH, WO = example()
W0 = [WI, WH, WO]
w_flat = np.concatenate([x.ravel() for x in W0])
NAMES = [f"input->{j}" for j in range(16)] + [f"{i}->{j}" for j in range(16) for i in range(16)] + \
        [f"{i}->out{k}" for k in range(10) for i in range(16)]
# check: index of "i->j" is 16 + 16 j + i, i.e. WH[j, i]
assert w_flat[NAMES.index("3->10")] == WH[10, 3] and w_flat[NAMES.index("5->out2")] == WO[2, 5]
IDX = {n: k for k, n in enumerate(NAMES)}
_groups = {"store/det": [2, 4], "gates": [5, 9], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
_gof = {n: g for g, ns in _groups.items() for n in ns}
_src = lambda n: "input" if n.split("->")[0] == "input" else _gof[int(n.split("->")[0])]
_tgt = lambda n: "readout" if n.split("->")[1].startswith("out") else _gof[int(n.split("->")[1])]
_block = np.array([f"{_src(n)} -> {_tgt(n)}" for n in NAMES])
s_prior = np.zeros(432)
for _b in set(_block):
    s_prior[_block == _b] = np.sqrt((w_flat[_block == _b] ** 2).mean())
F_full = np.load(FISHER)


def waterfill(mu, Bb):
    lo, hi = 1e-20, mu.max()
    for _ in range(200):
        th_ = math.sqrt(lo * hi)
        used = 0.5 * np.sum(mu * np.minimum(1, 2 * th_ / np.maximum(mu, 1e-300)))
        lo, hi = (th_, hi) if used < Bb else (lo, th_)
    return 0.5 * np.sum(np.log2(np.maximum(1, mu / (2 * th_))))


def weight_bits(mean, sd, Bb):
    G = sd[:, None] * F_full * sd[None, :]
    mu = np.clip(np.linalg.eigvalsh(G), 0, None)
    z = (w_flat - mean) / sd
    return waterfill(mu, Bb) + 0.5 * np.sum(z ** 2) / math.log(2) - 0.5 * len(z) / math.log(2)


LOOKUP = weight_bits(np.zeros(432), s_prior, BUDGET)


def row_of(name):
    a, b = name.split("->")
    return b if b.startswith("out") else f"in{b}"


@th.no_grad()
def run_g(P, Xg):
    a, b, c = (th.as_tensor(p, dtype=th.float32, device=dev) for p in P)
    h = th.zeros(Xg.shape[0], 16, device=dev)
    for t in range(10):
        h = th.relu(h @ b.T + Xg[:, t, None] * a)
    return th.log_softmax(h @ c.T, -1), h @ c.T


def default_fit(rule, C0, target, F):
    C = th.tensor(C0, dtype=th.float64, requires_grad=True)
    Ft, tg = th.tensor(F), th.tensor(target)
    opt = th.optim.Adam([C], lr=1e-3)
    for _ in range(3000):
        opt.zero_grad(); d = rule.weights(C) - tg; L = d @ Ft @ d; L.backward(); opt.step()
    return C.detach().numpy()


def main():
    path = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path); rule = importlib.util.module_from_spec(spec); spec.loader.exec_module(rule)
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    print(f"score_memory_rule_v2.py sha256 {me}; rule {path.name}", flush=True)
    cov = np.array([IDX[n] for n in rule.COVERED])
    Fcc = F_full[np.ix_(cov, cov)]
    C0 = np.asarray(rule.init_constants(), float)
    C = (rule.fit if hasattr(rule, "fit") else lambda c0, tg, F: default_fit(rule, c0, tg, F))(C0, w_flat[cov], Fcc)
    C = np.asarray(C, float)
    kinds = np.asarray(rule.KINDS).reshape(-1)
    assert kinds.size == C.size, "KINDS must have one entry per constant"
    ct = th.tensor(C, dtype=th.float64)
    wr = rule.weights(ct).detach().numpy()
    J = th.autograd.functional.jacobian(rule.weights, ct).numpy().reshape(len(cov), -1)

    # KL and accuracy with the rule's weights
    wf = w_flat.copy(); wf[cov] = wr
    P = [wf[:16], wf[16:272].reshape(16, 16), wf[272:].reshape(10, 16)]
    Xg = th.randn(2 ** 22, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
    Y = Xg.argsort(-1)[:, -2]
    lb, sb = run_g(W0, Xg); lr, sr = run_g(P, Xg)
    kl = (lb.exp() * (lb - lr)).sum(-1).mean().item()
    acc0, acc1 = (sb.argmax(-1) == Y).float().mean().item(), (sr.argmax(-1) == Y).float().mean().item()
    del Xg, lb, lr, sb, sr; th.cuda.empty_cache()

    # bits
    Fc = J.T @ Fcc @ J
    sd_c = np.zeros_like(C)
    for k in set(kinds):
        sd_c[kinds == k] = np.sqrt((C[kinds == k] ** 2).mean()) + 1e-12
    mu = np.clip(np.linalg.eigvalsh(sd_c[:, None] * Fc * sd_c[None, :]), 0, None)
    mis = 0.5 * np.sum((C / sd_c) ** 2) / math.log(2) - 0.5 * len(C) / math.log(2)
    m = np.zeros(432); sd = s_prior.copy(); m[cov] = wr
    rws = np.array([row_of(rule.COVERED[i]) for i in range(len(cov))])
    for r in set(rws):
        ii = cov[rws == r]; sd[ii] = max(np.sqrt(((wf[ii] - w_flat[ii]) ** 2).mean()), 1e-9)
    struct = int(rule.STRUCTURE_BITS) + SPREAD_BITS * (len(set(rws)) + len(set(kinds)))
    res = []
    for bc in SHARES:
        rc = waterfill(mu, bc) + mis
        left = weight_bits(m, sd, BUDGET - bc)
        res.append((LOOKUP - (rc + struct + left), bc, rc, left))
    best = max(res)
    err = np.linalg.norm(wr - w_flat[cov]) / np.linalg.norm(w_flat[cov])
    print(f"{len(cov)} weights covered, {C.size} constants, {len(set(kinds))} kinds; relative weight error {err:.4f}; "
          f"structure {struct} bits; lookup table {LOOKUP:.0f} bits")
    for net, bc, rc, left in res:
        print(f"  share {bc}: constants {rc:.0f} bits ({rc / C.size:.1f} each) + structure {struct} + rest {left:.0f}; net {net:+.0f}")
    print(f"KL from ARC {kl:.4f} (budget {BUDGET}); accuracy {acc1:.4f} (ARC {acc0:.4f})")
    print(f"NET {best[0]:+.1f}")


if __name__ == "__main__":
    main()
