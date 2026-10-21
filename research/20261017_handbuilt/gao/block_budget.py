"""Where do the most-explained block account's lost points go? Version: cancellers as their clipped rotation with
fitted drive (menu); store and detector from menu formulas; late gate from the account's quantities; memory as a
linear register with neuron 0's rule, neuron 15 by its formula's sign, keep gate ReLU (block_full2.py). Put one part
back to exact at a time and re-run all 16 replaced."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import block_full2 as bf
import simlib

mem = bf.mem; B = mem + [5]


def run(others, exact_switch=(), exact_canc=False):
    O = [n for n in range(16) if n not in B]; Wbb = simlib.WH[np.ix_(B, B)]; Wbo = simlib.WH[np.ix_(B, O)]
    i0, i15, i5 = B.index(0), B.index(15), B.index(5); c = np.zeros((len(bf.Xs), len(B))); out = np.zeros((len(bf.Xs), 10, len(B)))
    for t in range(10):
        drive = (np.stack([others[n][:, t - 1] for n in O], -1) @ Wbo.T) if t else 0
        c = c @ Wbb.T + drive + bf.Xs[:, t:t + 1] * simlib.WI[B]
        c[:, i5] = np.maximum(c[:, i5], 0)
        c[:, i0] = np.maximum(c[:, i0], 0) if 0 in exact_switch else np.where((bf.age[:, t] == 2) & bf.anypos[:, t], 0.0, c[:, i0])
        c[:, i15] = np.maximum(c[:, i15], 0) if 15 in exact_switch else np.where(bf.on15[:, t], c[:, i15], 0.0)
        if "memory" in exact_switch:
            c = np.maximum(c, 0)
        out[:, t] = c
    blk = {n: out[:, :, i] for i, n in enumerate(B)}
    return bf.acc({n: np.maximum(v, 0) for n, v in {**others, **blk}.items() if n not in () } | {n: np.maximum(v, 0) for n, v in blk.items()})


base = {**bf.canc, **bf.f, **bf.g9}
def score(parts, **kw):
    sims = {k: v for k, v in base.items()}
    sims.update(parts)
    others = {**bf.true, **sims}
    return run(others, **kw) if True else None


T = bf.true
print(f"most-explained block account: {score({}):.3f}")
print(f"  cancellers exact:                {score({n: T[n] for n in [1, 6, 7, 8]}):.3f}")
print(f"  store and detector exact:        {score({n: T[n] for n in [2, 4]}):.3f}")
print(f"  late gate exact:                 {score({9: T[9]}):.3f}")
print(f"  neuron 0 switching exact:        {score({}, exact_switch=(0,)):.3f}")
print(f"  neuron 15 switching exact:       {score({}, exact_switch=(15,)):.3f}")
print(f"  all memory switching exact:      {score({}, exact_switch=('memory', 0, 15)):.3f}")
