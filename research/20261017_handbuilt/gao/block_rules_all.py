"""All memory switching from rules: every memory neuron's on/off predicted from the sign of a formula of the inputs
(menu + register waves + input, least squares on all steps), neuron 0 by its rule, keep gate ReLU; memory otherwise
linear on its weights. Cancellers as their clipped rotation, store/detector/late gate as formulas (block_full2.py).
All 16 replaced. Also each neuron's on/off agreement."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import block_full2 as bf
import simlib

mem = bf.mem; B = mem + [5]; Hs = bf.Hs
_, PREf2, _ = simlib.run(bf.Xf)
onp = {}
for n in mem:
    co = np.linalg.lstsq(bf.Ff.reshape(-1, bf.Ff.shape[-1]), PREf2[:, :, n].reshape(-1), rcond=None)[0]
    onp[n] = (bf.Fs @ co) > 0
    print(f"memory neuron {n:2d}: on/off agreement {np.mean(onp[n] == (Hs[:, :, n] > 0)):.3f} (on {np.mean(Hs[:, :, n] > 0):.3f} of steps)")
O = [n for n in range(16) if n not in B]; Wbb = simlib.WH[np.ix_(B, B)]; Wbo = simlib.WH[np.ix_(B, O)]
others = {**bf.true, **bf.canc, **bf.f, **bf.g9}
for lab, use in (("neuron 0 by its rule, 15 by formula sign", (15,)), ("all memory neurons by formula sign (0 by its rule)", tuple(n for n in mem if n != 0))):
    c = np.zeros((len(bf.Xs), len(B))); out = np.zeros((len(bf.Xs), 10, len(B)))
    for t in range(10):
        drive = (np.stack([others[n][:, t - 1] for n in O], -1) @ Wbo.T) if t else 0
        c = c @ Wbb.T + drive + bf.Xs[:, t:t + 1] * simlib.WI[B]
        c[:, B.index(5)] = np.maximum(c[:, B.index(5)], 0)
        c[:, B.index(0)] = np.where((bf.age[:, t] == 2) & bf.anypos[:, t], 0.0, c[:, B.index(0)])
        for n in use:
            c[:, B.index(n)] = np.where(onp[n][:, t], c[:, B.index(n)], 0.0)
        out[:, t] = c
    blk = {n: np.maximum(out[:, :, i], 0) for i, n in enumerate(B)}
    print(f"{lab:52s}: all 16 replaced {bf.acc({**{k: v for k, v in others.items() if k in (1, 2, 4, 6, 7, 8, 9)}, **blk}):.3f}", flush=True)
