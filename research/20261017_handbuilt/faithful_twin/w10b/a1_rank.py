"""w10b a1: can a hiding wave with no backward part be carried by few turning modes?

The wave per unit start at offset k (position minus the record's position) is f(k) = sum of the modes' parts. No backward
part means f(k) = 0 for k = -1..-7 (every earlier position the readout reads, positions 1..8) and f(0) = 1. For each set
of modes, solve the least-squares fit of these 8 conditions and report the residual (0 = possible), the forward values
f(1..8) (what the forward copies must cancel and what an overtaken record spills) and the clock neurons the twin's style
needs (3 per turning pair: two wave dimensions and the shared comparator level; 1 for the flip; 1 for a non-turning part,
whose level must be removed through C7).
"""
import numpy as np

K = np.arange(-7, 1)
FWD = np.arange(1, 9)


def basis(modes, k):
    cols = []
    for m in modes:
        if m == "const":
            cols.append(np.ones_like(k, dtype=float))
        elif m == "flip":
            cols.append((-1.0) ** k)
        elif m.startswith("fade"):
            r = float(m[4:]); cols.append(r ** k.astype(float))
        else:
            th = np.radians(float(m)); cols += [np.cos(th * k), np.sin(th * k)]
    return np.stack(cols, 1)


def cost(modes):
    return sum(1 if (m in ("const", "flip") or m.startswith("fade")) else 3 for m in modes)


tgt = (K == 0).astype(float)
SETS = [["45", "135"], ["45", "135", "flip"], ["45", "90", "135"], ["45", "90", "135", "flip"],
        ["45", "90", "135", "flip", "const"], ["45", "90", "135", "flip", "fade0.5"],
        ["38", "84", "127", "flip", "const"], ["38", "84", "127", "flip", "fade0.5"]]
for modes in SETS:
    B = basis(modes, K); c, *_ = np.linalg.lstsq(B, tgt, rcond=None)
    res = np.abs(B @ c - tgt).max(); f_fwd = basis(modes, FWD) @ c
    print(f"modes {'+'.join(modes):28s} dims {B.shape[1]}  clock neurons {cost(modes):2d}  max error on k=-7..0 {res:.3f}"
          f"  forward f(1..8) {np.round(f_fwd, 2)}")
