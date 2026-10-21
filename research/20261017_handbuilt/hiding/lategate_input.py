"""What the late gate reads from the memory one step after an overtaking. Near-tie overtaking lists (margin_route.sel).
The memory's input to the late gate at q+1, sum over memory neurons of W[9, n] h_n(q), fitted by least squares
(no intercept) on: (a) the old record x_r; (b) x_r and the jump; (c) x_r, the jump and every other number's positive
part. Variance matched and coefficients. Also its value on the same lists with the old record replaced by a number
just below the new largest's predecessor... (not done); and after a first record (no earlier positive number)."""
import numpy as np
from common import WI, WH, WO, GROUPS, lists, run
from margin_route import sel
MEM = GROUPS["memory"]
for rr, qq in ((2, 5), (3, 6), (4, 7)):
    Xs = sel(rr, qq); _, _, H = run(Xs); mi = H[:, qq][:, MEM] @ WH[9, MEM]
    J = Xs[:, qq] - Xs[:, rr]; oth = np.maximum(np.delete(Xs, [rr, qq], 1), 0)
    out = []
    for lab, A in (("x_r", Xs[:, [rr]]), ("x_r, jump", np.c_[Xs[:, rr], J]), ("x_r, jump, others", np.c_[Xs[:, rr], J, oth])):
        co = np.linalg.lstsq(A, mi, rcond=None)[0]; e = mi - A @ co
        out.append(f"{lab}: {1 - e.var() / mi.var():.2f} (x_r {co[0]:+.2f}" + (f", jump {co[1]:+.2f}" if A.shape[1] > 1 else "") + ")")
    print(f"record at {rr}, overtaken at {qq}: memory input to the late gate median {np.median(mi):.2f}; variance matched by " + " | ".join(out))
# first records: the largest so far arrives at q with no positive number before
X = lists(18, 400_000)
for qq in (3, 5):
    ok = (X[:, :qq] < 0).all(1) & (X[:, qq] > 0.3); Xs = X[ok]; _, _, H = run(Xs); mi = H[:, qq][:, MEM] @ WH[9, MEM]
    print(f"first positive number at {qq} (no earlier record to reveal): memory input to the late gate median {np.median(mi):.2f}, late gate at q+1 median {np.median(H[:, qq + 1, 9]):.2f} (x_q median {np.median(Xs[:, qq]):.2f})")
print("late gate's weights from the hiding neurons 1, 6, 7, 8:", " ".join(f"{WH[9, n]:+.2f}" for n in (1, 6, 7, 8)),
      f"| from the detector {WH[9, 4]:+.2f} | its weights onto 1, 6, 7, 8:", " ".join(f"{WH[n, 9]:+.2f}" for n in (1, 6, 7, 8)),
      "| onto memory 0, 3, 10-15:", " ".join(f"{WH[n, 9]:+.2f}" for n in MEM))
