"""Frozen scoring harness for hiding-neuron rules (autoresearch loop, an internal working note, not included). Do not edit.

Usage:  python -u score_rule.py rules/<module>.py [--fit-seed 101] [--score-seed 202]

A rule module defines (see rules/constrained_lean.py):
  prepare(X) -> dict of float64 cuda tensors, batch first: the list-only inputs the rule reads
  rollout(data, C) -> (B, 10, 4) input (pre-activation) of hiding neurons 1, 6, 7, 8, the rule run on its own outputs;
                      differentiable in the constants C
  init_constants(X, P, H) -> numpy array C0 (P, H: ARC's pre-activations and activations on X, (B, 10, 16))
  KINDS: array of C's shape naming each constant's kind (one reference spread per kind)
  STRUCTURE_BITS: bits for the rule's form (which features, about 1 bit each per neuron; 6 bits per hand-set number)
  optional fit(data, P_hid, C0) -> C (default below: Adam on the rollout loss against ARC's activity)

Printed:
  net bits: fair surprise accounting (as step19_fair_bits.py): the constants' bits by water-filling on their KL
    curvature through constants -> rule activity -> weights (closed-form least squares on the rule's on steps) ->
    KL curvature of the 64 hiding weights; absolute reference class with one spread per kind; plus structure, the
    4 row spreads and the kind spreads (6 bits each), plus the bits left for the rest of the network given the rule's
    weights. Shares 0.001, 0.002, 0.005 of the KL budget 0.02 for the constants; the best is reported.
  checks: activity rms error per neuron (score lists); cosine of the derived weights with ARC's; joint KL of the four
    derived rows (2^22 lists, seed 7); KL with all four neurons replaced by the rule's activity (score lists).
"""
import argparse
import hashlib
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "gao"))
sys.path.insert(0, str(HERE / "rules"))
import simlib  # noqa: E402

FISHER = "/data/output/research/algzoo/20261017_handbuilt/parameters/fisher_abs.npy"
HID = [1, 6, 7, 8]
BASE_RMS = {1: 1.846, 6: 1.968, 7: 0.628, 8: 2.133}   # the account's formulas (gao/sim_tolerance.log)
BUDGET = 0.02
SHARES = [0.001, 0.002, 0.005]
N_CLOSED = 50_000          # fit lists used for the closed-form weights and their Jacobian
N_SUB = 2 ** 22            # lists for the joint KL of the derived rows
SPREAD_BITS = 6            # each stated spread (4 row spreads, one per constant kind)
dev = "cuda"
WI, WH, WO = simlib.WI, simlib.WH, simlib.WO
W0 = [WI, WH, WO]
w_flat = np.concatenate([x.ravel() for x in W0])

# reference class of the weights: Gaussian per block (README experiment 20; as step17_bits.py)
_names = [f"input->{j}" for j in range(16)] + [f"{i}->{j}" for j in range(16) for i in range(16)] + \
         [f"{i}->out{k}" for k in range(10) for i in range(16)]
_groups = {"store/det": [2, 4], "gates": [5, 9], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
_gof = {n: g for g, ns in _groups.items() for n in ns}
_block = np.array([_gof[int(n.split("->")[0])] + " readout" if n.split("->")[1].startswith("out")
                   else _gof[int(n.split("->")[1])] + " in" for n in _names])
s_prior = np.zeros(432)
for _b in set(_block):
    s_prior[_block == _b] = np.sqrt((w_flat[_block == _b] ** 2).mean())
F_full = np.load(FISHER)
rows = {n: np.arange(16 + 16 * n, 32 + 16 * n) for n in HID}
hid_idx = np.concatenate([rows[n] for n in HID])


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


def default_fit(rule, data, Pt, C0):
    """As step18_constrained.py: Adam on the rule's own multi-step run; squared error where ARC's neuron is on,
    max(0, rule input)^2 where it is off."""
    C = th.tensor(C0, device=dev, dtype=th.float64, requires_grad=True)
    opt = th.optim.Adam([C], lr=2e-3); g = th.Generator(device=dev).manual_seed(0)
    B = Pt.shape[0]
    for _ in range(3001):
        idx = th.randint(0, B, (20000,), device=dev, generator=g)
        opt.zero_grad(); pre = rule.rollout({k: v[idx] for k, v in data.items()}, C); tgt = Pt[idx]
        L = th.where(tgt > 0, (pre - tgt) ** 2, th.relu(pre) ** 2).mean(); L.backward(); opt.step()
    return C.detach()


@th.no_grad()
def run_g(P, Xg):
    a, b, c = (th.as_tensor(p, dtype=th.float32, device=dev) for p in P)
    h = th.zeros(Xg.shape[0], 16, device=dev)
    for t in range(10):
        h = th.relu(h @ b.T + Xg[:, t, None] * a)
    return th.log_softmax(h @ c.T, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rule"); ap.add_argument("--fit-seed", type=int, default=101); ap.add_argument("--score-seed", type=int, default=202)
    a = ap.parse_args()
    path = Path(a.rule).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path); rule = importlib.util.module_from_spec(spec); spec.loader.exec_module(rule)
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    print(f"score_rule.py sha256 {me}; rule {path.name}; fit seed {a.fit_seed}, score seed {a.score_seed}", flush=True)

    Xf, Xs = simlib.lists(a.fit_seed), simlib.lists(a.score_seed)
    _, Pf, Hf = simlib.run(Xf)
    Ls, _, Hs = simlib.run(Xs)
    dfit, dsc = rule.prepare(Xf), rule.prepare(Xs)
    C0 = np.asarray(rule.init_constants(Xf, Pf, Hf), float)
    Pt = th.tensor(Pf[:, :, HID], device=dev)
    C = (rule.fit if hasattr(rule, "fit") else lambda d, p, c: default_fit(rule, d, p, c))(dfit, Pt, C0)
    C = th.as_tensor(C, device=dev, dtype=th.float64).detach()
    kinds = np.asarray(rule.KINDS).reshape(-1)
    assert kinds.size == C.numel(), "KINDS must have one entry per constant"

    # activity
    with th.no_grad():
        act = th.relu(rule.rollout(dsc, C)).cpu().numpy()
    rms = {n: np.sqrt(((act[:, :, i] - Hs[:, :, n]) ** 2).mean()) for i, n in enumerate(HID)}
    sub = simlib.score(Xs, Ls, simlib.run(Xs, {n: act[:, :, i] for i, n in enumerate(HID)})[0])

    # weights from the rule, closed form (on/off pattern fixed at the fitted constants)
    dsub = {k: v[:N_CLOSED] for k, v in dfit.items()}
    Hp = th.tensor(np.concatenate([np.zeros((N_CLOSED, 1, 16)), Hf[:N_CLOSED, :-1]], 1), device=dev)
    xs_ = th.tensor(Xf[:N_CLOSED], device=dev)
    with th.no_grad():
        on_mask = rule.rollout(dsub, C) > 0
    shape = C.shape

    def weights_closed(cflat):
        pre = rule.rollout(dsub, cflat.view(shape)); ws = []
        for i, n in enumerate(HID):
            m = on_mask[:, :, i]; A = Hp[m]; y = pre[:, :, i][m] - xs_[m] * WI[n]
            ws.append(th.linalg.solve(A.T @ A, A.T @ y))
        return th.cat(ws)

    c0 = C.reshape(-1)
    wc = weights_closed(c0).detach().cpu().numpy().reshape(4, 16)
    cos = {n: wc[i] @ WH[n] / np.linalg.norm(wc[i]) / np.linalg.norm(WH[n]) for i, n in enumerate(HID)}
    Xg = th.randn(N_SUB, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
    lb = run_g(W0, Xg); pb = lb.exp()
    P = [x.copy() for x in W0]
    for i, n in enumerate(HID):
        P[1][n] = wc[i]
    joint = (pb * (lb - run_g(P, Xg))).sum(-1).mean().item()
    del Xg, lb, pb; th.cuda.empty_cache()

    # fair bits
    J = th.autograd.functional.jacobian(weights_closed, c0).cpu().numpy()        # (64, nconst)
    Fc = J.T @ F_full[np.ix_(hid_idx, hid_idx)] @ J
    cv = c0.cpu().numpy()
    sd_c = np.zeros_like(cv)
    for k in set(kinds):
        sd_c[kinds == k] = np.sqrt((cv[kinds == k] ** 2).mean())
    mu = np.clip(np.linalg.eigvalsh(sd_c[:, None] * Fc * sd_c[None, :]), 0, None)
    mis = 0.5 * np.sum((cv / sd_c) ** 2) / math.log(2) - 0.5 * len(cv) / math.log(2)
    m = np.zeros(432); sd = s_prior.copy()
    for i, n in enumerate(HID):
        m[rows[n]] = wc[i]; sd[rows[n]] = np.sqrt(((wc[i] - WH[n]) ** 2).mean())
    struct = int(rule.STRUCTURE_BITS) + SPREAD_BITS * (4 + len(set(kinds)))
    res = []
    for bc in SHARES:
        rc = waterfill(mu, bc) + mis
        left = weight_bits(m, sd, BUDGET - bc)
        res.append((LOOKUP - (rc + struct + left), bc, rc, left))
    best = max(res)

    print(f"\n{C.numel()} free constants, {len(set(kinds))} kinds ({', '.join(sorted(set(kinds)))}); structure {struct} bits "
          f"({rule.STRUCTURE_BITS} declared + {SPREAD_BITS} x {4 + len(set(kinds))} spreads); lookup table {LOOKUP:.0f} bits")
    for net, bc, rc, left in res:
        print(f"  share {bc}: constants {rc:.0f} bits ({rc / len(cv):.1f} each) + structure {struct} + rest {left:.0f}; net {net:+.0f}")
    print("\n| net bits | share | rms 1, 6, 7, 8 | cosine 1, 6, 7, 8 | joint KL of 4 rows | KL, 4 neurons replaced | constants |")
    print("|---|---|---|---|---|---|---|")
    print(f"| {best[0]:+.0f} | {best[1]} | " + ", ".join(f"{rms[n]:.3f}" for n in HID) + " | " + ", ".join(f"{cos[n]:.4f}" for n in HID) +
          f" | {joint:.3f} | {sub['kl']:.3f} | {C.numel()} |")
    print(f"NET {best[0]:+.1f}")


if __name__ == "__main__":
    main()
