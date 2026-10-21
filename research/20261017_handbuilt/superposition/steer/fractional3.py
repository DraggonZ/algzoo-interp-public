"""Continuous steering, done with the right map. fractional2.log showed the shift needs the ReLU and the store's input:
the step that shifts votes runs in the pattern of a 'forgotten number' step (neuron 4 off, detector mostly off), with
the store feeding the rotating neurons. Take M = D W over all 16 neurons with D the on/off pattern of the extra step
on most lists (neurons on in more than half of lists), and fractional powers M^a through its eigenvalues. Predictions
(written first): a = 1 reproduces the extra step (about 94%); a = 0.5 splits between the second and second-1; a = -1
names second+1 on most lists with the second at 0-8."""
import numpy as np
from harness import *
from modes import X, S, h
W = ARC[1].cpu().numpy(); v4 = torch.relu(h @ ARC[1].T); D = ((v4 > 0).double().mean(0) > 0.5).cpu().numpy().astype(float)
M = np.diag(D) @ W; ev, R = np.linalg.eig(M); Li = np.linalg.pinv(R)
print("on in the pattern:", [i for i in range(16) if D[i]], "| eigenvalues (angle:size):", ", ".join(f"{np.degrees(abs(np.angle(e))):.0f}:{abs(e):.2f}" for e in ev if abs(e) > 0.05))
def pw(a):
    lam = ev.astype(complex); lam = np.where(np.abs(lam) > 1e-6, lam ** a, 0)
    return (R @ np.diag(lam) @ Li).real
for a in (0.5, 1.0, 2.0, -0.5, -1.0):
    Ma = torch.tensor(pw(a), device=DEV, dtype=dt); ans = ((h @ Ma.T) @ ARC[2].T).argmax(1)
    f = lambda d: 100 * (ans[(S + d >= 0) & (S + d <= 9)] == (S + d)[(S + d >= 0) & (S + d <= 9)]).double().mean().item()
    print(f"a {a:+.1f}: names second-2 {f(-2):5.1f}% | second-1 {f(-1):5.1f}% | the second {f(0):5.1f}% | second+1 {f(1):5.1f}%", flush=True)
