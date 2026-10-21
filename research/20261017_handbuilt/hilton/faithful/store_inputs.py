"""Which of the store's (2) and detector's (4) inputs from other neurons matter? Each weight, its effect size
(|w| x RMS activity of the source), and the grid accuracy with that one weight set to 0."""
import numpy as np, torch
import idealize as I
T = I.T; rms = I.rms
for r in (2, 4):
    print(f"== row {r} ({'store' if r == 2 else 'detector'}); input weight {T[0][r].item():+.3f}")
    for j in range(16):
        w = T[1][r, j].item()
        W = [x.clone() for x in T]; W[1][r, j] = 0
        print(f"   from {j:2d}: weight {w:+.3f}  effect {abs(w) * rms[j].item():6.3f}  accuracy without it {100 * I.acc(W):6.2f}%", flush=True)
