"""Are the cancelling neurons' internal weights just a rotation? W_cc (4x4, neurons 1, 6, 7, 8) has eigenvalues
0.937 at +-42.6 degrees and 0.043 at +-17.1. Replace W_cc by its rotation part alone (the rank-2 reconstruction
from the 42.6-degree pair) inside the cancellers' block (fitted drive from the menu, as block_recurrent.py), and in
the full block account (block_full2.py's best version: memory and keep gate block with all ReLUs)."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import block_recurrent as br
    import block_full2 as bf
import simlib

C = [1, 6, 7, 8]; W = simlib.WH[np.ix_(C, C)]
ev, V = np.linalg.eig(W); Vi = np.linalg.inv(V); keep = np.abs(ev) > 0.5
Wrot = np.real(V[:, keep] @ np.diag(ev[keep]) @ Vi[keep, :])
print("rotation part of W_cc:\n", np.round(Wrot, 3), "\nlargest entry change", np.round(np.abs(Wrot - W).max(), 3))


def canc_block(Wcc):
    Xf, Xs, Hf, PREf, Qf, Qs = br.Xf, br.Xs, br.Hf, br.PREf, br.Qf, br.Qs
    prev = np.zeros_like(Hf[:, :, C]); prev[:, 1:] = Hf[:, :-1][:, :, C]; drive = PREf[:, :, C] - prev @ W.T; Bm = []
    for i, n in enumerate(C):
        m = Hf[:, :, n].reshape(-1) > 0
        Bm.append(np.linalg.lstsq(Qf.reshape(-1, Qf.shape[-1])[m], drive[:, :, i].reshape(-1)[m], rcond=None)[0])
    Bm = np.stack(Bm); c = np.zeros((len(Xs), 4)); out = np.zeros((len(Xs), 10, 4))
    for t in range(10):
        c = np.maximum(c @ Wcc.T + Qs[:, t] @ Bm.T, 0); out[:, t] = c
    return {n: out[:, :, i] for i, n in enumerate(C)}


for lab, Wcc in (("trained 4x4 weights", W), ("rotation part only", Wrot)):
    cb = canc_block(Wcc)
    sims = {**cb, **bf.f, **bf.g9}
    blk = bf.run_block(bf.mem + [5], {**bf.true, **sims})
    print(f"cancellers with {lab:20s}: four replaced {br.acc(cb):.3f}; all 16 replaced (memory+gate block) {br.acc({**sims, **blk}):.3f}", flush=True)
