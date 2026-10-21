"""v19 with vote crosstalk in the readout. ARC's crosstalk, measured: for a kept positive number at q (not the largest)
on random lists, C[p, q] = J[p, q] - mean_p' J[p', q] (exact, centred over positions since adding the same to every score
changes nothing), normalized by C[q, q]; K = median over lists, by (p, q). Built into v19 (kappa 0.5) as score' = K
score, i.e. readout rows mixed: wo' = K wo. K is copied from ARC's measured behaviour (a judgement call, not tuned to
mistakes). Variants: K measured on numbers before the largest only; K with the off-diagonal halved; a control with
the off-diagonal signs shuffled at random. Also the traces of traces.py on top."""
import numpy as np, torch
from common import *
from algzoo.handbuilt_v19 import build_v19
import score
def measure(before_only=False, n=200000):
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(41)); J = jac(ARC, X)
    C = J - J.mean(1, keepdim=True)                         # [n, p, q]
    Lp = X.argmax(1); pos = torch.arange(10, device=DEV)
    K = torch.zeros(10, 10, device=DEV, dtype=dt)
    for q in range(10):
        m = (X[:, q] > 0) & (Lp != q)
        if before_only: m &= Lp > q
        if m.sum() < 100: K[q, q] = 1.0; continue          # no such numbers (a kept number at 9 is never before L)
        r = C[m, :, q] / C[m, q, q][:, None]
        K[:, q] = r.median(0).values
    return K.cpu().numpy()
if __name__ == "__main__":
    K = measure(); Kb = measure(before_only=True)
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print("ARC crosstalk K[p, q] (row p = score position, column q = kept number's position):"); print(K)
    print("before the largest only:"); print(Kb)
    wi, wh, wo = build_v19(kappa=0.5).params
    off = K - np.diag(np.diag(K)); rng = np.random.default_rng(0)
    shuf = np.diag(np.diag(K)) + off * rng.choice([-1, 1], size=off.shape)
    V = {"v19": np.eye(10), "v19 + ARC crosstalk": K, "v19 + crosstalk (before L)": Kb,
         "v19 + crosstalk halved": np.diag(np.diag(K)) + 0.5 * off, "v19 + crosstalk, random signs": shuf}
    models = {nm: [T([wi, wh, M @ wo])] for nm, M in V.items()}
    score.report(models)
