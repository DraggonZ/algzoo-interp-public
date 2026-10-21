"""Pieces of the tie-band estimator.

For each ordered triple (p1, p2, p3) the wrong set is modelled as
    a < A * S23(u, b)   or   b < B * S12(u, a)   or   m2 < 0,
with u the third-largest, a = m2 - m3, b = m1 - m2. A is the band width at the 2nd-3rd tie measured at one node
(the density-weighted centroid of that tie plane), B the same at the 1st-2nd tie, and S a fixed scale law that carries
the width from the node to other (u, gap). The error mass of this set is integrated on a grid with the exact density
(no network runs)."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate")
import full_mean
import common as c

# density-weighted centroids of the two tie planes (u > 0), from scratch centroids computation (see README)
N23 = (1.3679, 0.7877, 0.7877, -0.3945)     # m1, m2 = m3, background
N12 = (1.1374, 1.1374, 0.7659, -0.4039)     # m1 = m2, m3, background
NU, NA = 56, 96
_V, _wt = full_mean.grid(NU, NA)
WT = torch.tensor(_wt.reshape(NU, NA, NA), device=c.DEV, dtype=c.dt); MASS = WT.sum().item() * 720
U = torch.tensor(_V[:, 2].reshape(NU, NA, NA), device=c.DEV, dtype=c.dt)
GA = torch.tensor((_V[:, 1] - _V[:, 2]).reshape(NU, NA, NA), device=c.DEV, dtype=c.dt)
GB = torch.tensor((_V[:, 0] - _V[:, 1]).reshape(NU, NA, NA), device=c.DEV, dtype=c.dt)
BGU = c.bg(U)


def node_inputs(tie, P, node=None):
    """Tie inputs at the node for triples P [n,3]: values placed as (m1, m2, m3), background fixed."""
    n = len(P); ar = torch.arange(n, device=c.DEV)
    nd = node or (N23 if tie == "23" else N12)
    X = torch.full((n, 10), nd[3], device=c.DEV, dtype=c.dt)
    for k in range(3): X[ar, P[:, k]] = nd[k]
    d = torch.zeros_like(X)
    if tie == "23": d[ar, P[:, 1]] = 1; d[ar, P[:, 0]] = 1      # a grows: m2 and m1 up (b fixed)
    else: d[ar, P[:, 0]] = 1                                    # b grows: m1 up
    return X, d


def scale(tie, law):
    """S(u, other gap) on the integration grid [NU, NA, NA] (axes u, a, b)."""
    if law == "const": return torch.ones_like(U)
    if tie == "23": m1, m2, m3, ndd = U + GA + GB, U + GA, U, N23   # evaluate at the tie: a -> 0 but keep grid b
    else: m1, m2, m3, ndd = U + GA + GB, U + GA, U, N12
    if tie == "23": m2 = U; m1 = U + GB
    else: m1 = U + GA; m2 = U + GA
    if law == "norm":
        n = torch.sqrt(m1 ** 2 + m2 ** 2 + m3 ** 2 + 7 * BGU ** 2); n0 = np.sqrt(ndd[0] ** 2 + ndd[1] ** 2 + ndd[2] ** 2 + 7 * ndd[3] ** 2)
        return n / n0
    if law == "proj":
        x0 = np.array(ndd[:3] + (ndd[3],) * 7); n0 = (x0 ** 2).sum()
        return (m1 * x0[0] + m2 * x0[1] + m3 * x0[2] + BGU * x0[3] * 7) / n0
    if law == "spread":   # tied value minus background
        t = m3 if tie == "23" else m2; t0 = ndd[2] if tie == "23" else ndd[1]
        return (t - BGU) / (t0 - ndd[3])
    raise ValueError(law)


def accuracy(A, B, law="norm", low=True, S23=None, S12=None):
    """A, B: [720] widths at the nodes (torch). Returns the modelled accuracy in percent."""
    S23 = scale("23", law) if S23 is None else S23; S12 = scale("12", law) if S12 is None else S12
    tot = 0.0
    lowm = ((U + GA) < 0) if low else torch.zeros_like(U, dtype=torch.bool)
    for t in range(720):
        wr = (GA < A[t] * S23) | (GB < B[t] * S12) | lowm
        tot += (wr.to(c.dt) * WT).sum()
    return 100 * (1 - tot.item() / MASS)


def exact_width(W, tie, P, amax=5.0, n=1200, node=None):
    """Band width at the node by a fine scan along the gap (development reference; 0 if right at the tie)."""
    X0, d = node_inputs(tie, P, node); out = torch.zeros(len(P), device=c.DEV, dtype=c.dt)
    s = torch.linspace(0, 1, n, device=c.DEV, dtype=c.dt); g = amax * s ** 2
    for i in range(0, len(P), 40):
        k = min(40, len(P) - i)
        X = (X0[i:i + k, None, :] + g[None, :, None] * d[i:i + k, None, :]).reshape(-1, 10)
        S, _ = c.scores(W, X); m, _, _ = c.margin_from(S, P[i:i + k].repeat_interleave(n, 0)[:, 1]); m = m.reshape(k, n)
        right = m > 0
        j = torch.where(right.any(1), right.float().argmax(1), torch.full((k,), n - 1, device=c.DEV))
        jm = (j - 1).clamp_min(0); ar = torch.arange(k, device=c.DEV)
        m0, m1_ = m[ar, jm], m[ar, j]; g0, g1 = g[jm], g[j]
        r = torch.where((j > 0) & (m1_ > m0), g0 + (g1 - g0) * (-m0) / (m1_ - m0).clamp_min(1e-12), g[j])
        r = torch.where(right.any(1), r, torch.full_like(r, amax))       # never right on the scan: whole range
        out[i:i + k] = torch.where(j == 0, torch.zeros_like(r), r)
    return out
