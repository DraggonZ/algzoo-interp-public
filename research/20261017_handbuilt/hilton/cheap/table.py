"""The band model's integral as a network-independent table. With the scale law fixed ('norm'), a triple's wrong mass
under 'm2 < 0, or a < A S23(u, b), or b < B S12(u, a)' depends on its two node widths (A, B) only, so it is tabulated
once on a grid of (A, B) (no network involved) and each network then needs 720 table look-ups instead of a 3-D
integral per triple. Check: table look-up against the direct integral, with the exact path widths of choose_law.py."""
import numpy as np, torch
import common as c, bandlib as bl
log = c.Log("table.log")
NODES = np.r_[0.0, np.geomspace(1e-4, 5.0, 120)]
phi = lambda x: np.log(np.asarray(x, float) + 1e-5)


def build():
    S23 = bl.scale("23", "norm"); S12 = bl.scale("12", "norm"); low = (bl.U + bl.GA) < 0
    G = np.zeros((len(NODES), len(NODES)))
    for i, A in enumerate(NODES):
        for j, B in enumerate(NODES):
            wr = (bl.GA < A * S23) | (bl.GB < B * S12) | low
            G[i, j] = (wr.to(c.dt) * bl.WT).sum().item() / bl.MASS
    np.savez("table.npz", nodes=NODES, G=G); return G


def lookup(G, A, B):
    """Bilinear in log(width + 1e-5); widths above the last node are clipped (the grid ends at 5)."""
    x = phi(NODES); xa = np.clip(phi(A), x[0], x[-1]); xb = np.clip(phi(B), x[0], x[-1])
    ia = np.clip(np.searchsorted(x, xa) - 1, 0, len(x) - 2); ib = np.clip(np.searchsorted(x, xb) - 1, 0, len(x) - 2)
    fa = (xa - x[ia]) / (x[ia + 1] - x[ia]); fb = (xb - x[ib]) / (x[ib + 1] - x[ib])
    return ((1 - fa) * (1 - fb) * G[ia, ib] + fa * (1 - fb) * G[ia + 1, ib] + (1 - fa) * fb * G[ia, ib + 1] + fa * fb * G[ia + 1, ib + 1])


def accuracy(A, B, G=None):
    G = np.load("table.npz")["G"] if G is None else G
    return 100 * (1 - lookup(G, A, B).sum())


if __name__ == "__main__":
    import time; t0 = time.time(); G = build(); log(f"table of {len(NODES)} x {len(NODES)} width pairs built in {time.time()-t0:.0f}s (GPU; network-independent)")
    tt = lambda x: torch.tensor(x, device=c.DEV, dtype=c.dt)
    for name in ["ARC", "200", "208", "212"]:
        d = np.load(f"path_widths_{name}.npz")
        log(f"{name:>4s}: direct integral {bl.accuracy(tt(d['A']), tt(d['B']), 'norm'):.3f}%, table look-up {accuracy(d['A'], d['B'], G):.3f}%")
