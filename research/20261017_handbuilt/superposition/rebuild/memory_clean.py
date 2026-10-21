"""Precision of a clean memory. Base: ARC with memory neurons 0, 10, 11, 15 real (ReLU) and 3, 12, 13, 14 linear
(../patterns/which_switches.py: 93.4%, 82% of ARC's mistakes). Replacements, one at a time:
  dynamics: the memory's own 8x8 block B = V diag(lambda) V^-1 with eigenvalues rounded (angle to the nearest 1 or
            5 degrees, size to 0.01 or 0.05), directions V kept;
  readout:  the memory's output weights for positions 0-8 built from one anchor row pushed through the dynamics:
            r_p = r_8 B^(8-p) (votes read alike at every position, account item 7); position 9 kept; anchor row 8 kept
            as ARC's. Also the same with the rows rescaled to ARC's row norms (allows a per-position gain).
Scored by accuracy and ARC's mistakes reproduced."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
import score
MEM = [0, 3, 10, 11, 12, 13, 14, 15]; REAL = [0, 10, 11, 15]; LIN = [3, 12, 13, 14]
def runner(wh_mod=None, wo_mod=None):
    wi, wh, wo = ARC; wh = wh if wh_mod is None else wh_mod; wo = wo if wo_mod is None else wo_mod
    def f(X):
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
        for t in range(10):
            z = h @ wh.T + X[:, t:t + 1] * wi; hn = torch.relu(z); hn[:, LIN] = z[:, LIN]; h = hn
        return h @ wo.T
    return f
def round_eigs(deg, size):
    wh = ARC[1].clone(); B = wh[MEM][:, MEM].cpu().numpy(); ev, V = np.linalg.eig(B)
    ang = np.angle(ev); mag = np.abs(ev)
    if deg: ang = np.radians(np.round(np.degrees(ang) / deg) * deg)
    if size: mag = np.round(mag / size) * size
    Bn = np.real(V @ np.diag(mag * np.exp(1j * ang)) @ np.linalg.inv(V))
    wh[np.ix_(MEM, MEM)] = torch.tensor(Bn, device=DEV, dtype=dt); return wh
def pushed_readout(rescale, anchor=0):
    """r_(p+1) = r_p B (account item 7: position k+1's output weights are position k's pushed one step)."""
    wo = ARC[2].clone(); B = ARC[1][MEM][:, MEM]; Bi = torch.linalg.inv(B); rows = {anchor: wo[anchor, MEM].clone()}
    for p in range(anchor + 1, 9): rows[p] = rows[p - 1] @ B
    for p in range(anchor - 1, -1, -1): rows[p] = rows[p + 1] @ Bi
    for p in range(9):
        v = rows[p]
        if rescale: v = v * wo[p, MEM].norm() / v.norm()
        wo[p, MEM] = v
    cos = [torch.nn.functional.cosine_similarity(rows[p], ARC[2][p, MEM], 0).item() for p in range(9)]
    return wo, cos
if __name__ == "__main__":
    M = {"base (0,10,11,15 real; 3,12,13,14 linear)": [runner()]}
    for deg, size in (() if "--readout" in sys.argv else ((1, None), (5, None), (None, 0.01), (None, 0.05), (1, 0.01), (5, 0.05))):
        M[f"eigenvalues rounded: angle {deg} deg, size {size}"] = [runner(wh_mod=round_eigs(deg, size))]
    for anchor in (0, 4, 8):
        for rs in (False, True):
            wo, cos = pushed_readout(rs, anchor); print(f"readout pushed from row {anchor}: cosine with ARC's rows 0-8: " + " ".join(f"{c:.2f}" for c in cos))
            M[f"readout pushed from row {anchor}{' + row gains' if rs else ''}"] = [runner(wo_mod=wo)]
    score.report(M, family=False)
