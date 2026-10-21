"""Warming the state does not remove the position-0 failure (coldstart.log), so it is not the cold start. Which part
under-delivers? Final readout by group at the second's slot minus at the third's slot (the lead), for the failing shape
(second at 0, largest at 9) against the nearest working shapes (second at 0, largest at 8; second at 1, largest at 9)."""
import sys; sys.path.insert(0, "../steer")
from harness import *
from phase import X, S, L, T, wr
for lab, m in (("second 0, largest 9 (fails 4%)", (S == 0) & (L == 9)), ("second 0, largest 8 (works)", (S == 0) & (L == 8)),
               ("second 1, largest 9 (works)", (S == 1) & (L == 9)), ("second 0, largest 7 (fails 1%)", (S == 0) & (L == 7))):
    Xm, Sm, Tm = X[m], S[m], T[m]; nn = torch.arange(len(Xm), device=DEV)
    _, Hs = run(ARC, Xm, keep=True); h = Hs[-1]
    parts = {k: (h[:, g] @ ARC[2][:, g].T) for k, g in GROUPS.items()}
    row = " ".join(f"{k} {(p[nn, Sm] - p[nn, Tm]).median().item():+6.2f}" for k, p in parts.items())
    tot = sum(parts.values()); lead = tot[nn, Sm] - tot[nn, Tm]
    print(f"{lab:32s} lead per unit of second, median {(lead / Xm[nn, Sm]).median().item():+.2f} | parts of the lead (median): {row}")
