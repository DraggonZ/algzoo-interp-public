"""ARC's timing puts nine positions (0-8) into an eight-dimensional memory: the direct route writes a number into memory
the step after it arrives, so positions 0-8 are all there at the end and only position 9 is read from elsewhere. v23
avoids this with a buffer neuron. With nine labels in eight dimensions the best linear readout (least squares, a rule
computed from the dynamics, nothing fitted to data) is the projection K = I - n n^T, n the one direction the memory
cannot hold: every vote leaks -n_p n_q onto every other position. Question: what does n look like for a regular
45-degree code and for ARC's angles, and do such leaks with keep-or-forget switching give dense close calls?
Prediction (written first): regular grid puts n on positions 0 and 8 only (aliasing), few maps; ARC's angles spread n over
all positions, thousands of maps."""
import numpy as np, torch
from collections import Counter
import ideal_density as I
def null_dir(angles, rho=None):
    th = np.radians(angles); blocks = []
    for a in th:
        if np.isclose(np.sin(a), 0): blocks.append(np.array([[np.cos(a)]]))
        else: blocks.append(np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]))
    d = sum(len(b) for b in blocks); M = np.zeros((d, d)); i = 0
    for b in blocks: M[i:i + len(b), i:i + len(b)] = b; i += len(b)
    if rho is not None: M = M * np.repeat(rho, [len(b) for b in blocks])[:, None]
    b = np.ones(d); V = np.stack([np.linalg.matrix_power(M, 8 - q) @ b for q in range(9)], 1)   # label q at the end
    n = np.linalg.svd(V)[2][-1]; return n / np.linalg.norm(n)
if __name__ == "__main__":
    for name, ang, rho in (("regular 45-degree grid", [0, 45, 90, 135, 180], None), ("ARC's angles", [0, 43, 83, 126, 180], None),
                           ("ARC's angles and sizes", [0, 43, 83, 126, 180], np.array([0.99, 0.94, 0.94, 0.82, 0.90])),
                           ("40-degree grid without 160", [0, 40, 80, 120, 180], None)):
        n = null_dir(ang, rho); K = np.eye(10); K[:9, :9] -= np.outer(n, n)
        print(f"{name:28s} leak direction n over positions 0-8: " + " ".join(f"{v:+.2f}" for v in n))
        I.KARC = torch.tensor(K, device=I.DEV, dtype=I.dt)
        for m in ("broad (ARC angles), all numbers", "broad (ARC angles), positives", "broad (ARC angles), keep or forget"):
            acc, h = I.maps(m); print(f"     votes from {m.split(', ')[1]:15s} accuracy {100*acc:5.1f}%  lead maps covering half {h:6,}")
