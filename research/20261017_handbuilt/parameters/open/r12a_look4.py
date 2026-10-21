"""r12a look 4: variants of the regression push (x = 0 at the extra step), all rows 1-8, scaled by the gain;
and where its residual lies (cosine with readout row 0, with row k-1)."""
import numpy as np
import r12a_look3 as L
X = L.X
WO = X.WO
ALL = list(range(16))
for cols, nm in ((L.TGT, "push cols"), (ALL, "all 16")):
    net, d = X.price(X.Extend(L.R, L.cand("zero", range(1, 9), cols, True)))
    print(f"zero rows 1-8 {nm}: {net - L.net0:+.1f} (gain {d['C'][0]:.3f})")
    for r, (e, s) in sorted(d["rowerr"].items()):
        if r.startswith("out"):
            print(f"   {r} rel err {e / s:.3f}")
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
T = L.TGT
for k in range(1, 10):
    e = WO[k, T] - L.REG["zero", k][T]
    print(f"row {k}: residual size {np.linalg.norm(e) / np.linalg.norm(WO[k, T]):.3f}; cos with row0 {cos(e, WO[0, T]):+.2f}, "
          f"row k-1 {cos(e, WO[k - 1, T]):+.2f}, row k {cos(e, WO[k, T]):+.2f}, mean row {cos(e, WO[:, T].mean(0)):+.2f}")
