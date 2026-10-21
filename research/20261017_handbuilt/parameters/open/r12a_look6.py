"""r12a look 6: the push read backwards through regression on states: row k-1 = argmin E[(r . h_11 - row_k . h_10)^2],
h_11 = ReLU(W h_10) (x = 0). Is it well-posed, and how close to ARC's row k-1 (all 16 columns and push columns)?"""
import numpy as np
import r12a_look3 as L
X = L.X
WO, WH = X.WO, X.WH
T = L.TGT
N = L.N
h10 = L.h10
for mode in ("zero", "fresh"):
    h11 = np.maximum(h10 @ WH.T + (L.x[:, 10, None].numpy() * X.WI if mode == "fresh" else 0), 0)
    G1 = h11.T @ h11 / N
    print(mode, "cond of E[h11 h11]:", f"{np.linalg.cond(G1):.0f}", " cond of E[h10 h10]:", f"{np.linalg.cond(L.G):.0f}")
    for k in range(1, 10):
        r = np.linalg.pinv(G1, rcond=1e-10) @ (h11.T @ (h10 @ WO[k]) / N)
        e16 = np.linalg.norm(r - WO[k - 1]) / np.linalg.norm(WO[k - 1])
        eT = np.linalg.norm((r - WO[k - 1])[T]) / np.linalg.norm(WO[k - 1, T])
        gb = (r[T] @ WO[k - 1, T]) / (r[T] @ r[T])
        print(f"  row {k - 1} from row {k}: err all 16 {e16:.3f}, push cols {eT:.3f}, best scale {gb:.3f}")
