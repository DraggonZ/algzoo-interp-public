"""Step 3 (i): substitute the memory block by V diag(lambda') V^-1 with ARC's eigenvectors V and changed eigenvalues."""
import numpy as np, torch as th
from decoders import *
dev = "cuda"
Xg = th.randn(2**20, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev, dtype=th.float64)
Y = Xg.argsort(-1)[:, -2]

@th.no_grad()
def logp(Whh):
    a, b, c = (th.as_tensor(p, device=dev) for p in (wi, Whh, wo))
    h = th.zeros(len(Xg), 16, device=dev, dtype=th.float64)
    for t in range(10):
        h = th.relu(h @ b.T + Xg[:, t, None] * a)
    return th.log_softmax(h @ c.T, -1)

l0 = logp(wh); p0 = l0.exp()
def kl(Whh):
    l = logp(Whh); return (p0 * (l0 - l)).sum(-1).mean().item(), (l.argmax(-1) == Y).double().mean().item()

Wm = wh[np.ix_(MEM, MEM)]; ev, V = np.linalg.eig(Wm); Vi = np.linalg.inv(V)
def sub(ev2):
    W = wh.copy(); B = (V @ np.diag(ev2) @ Vi); assert abs(B.imag).max() < 1e-9
    W[np.ix_(MEM, MEM)] = B.real; return W
print("ARC eigenvalues:", [(round(abs(e), 3), round(np.degrees(np.angle(e)), 1)) for e in ev])
print(f"ARC accuracy on these lists {kl(wh)[1]:.4f}")
ang, mod = np.angle(ev), abs(ev)
def snap(a, step):
    return np.where(abs(abs(a) - np.pi) < 1e-9, a, np.sign(a) * np.round(abs(a) / step) * step)
cases = {
    "angles snapped to multiples of 42 deg": mod * np.exp(1j * snap(ang, np.radians(42))),
    "only the 38 deg pair snapped to 42": mod * np.exp(1j * np.where(abs(abs(np.degrees(ang)) - 38) < 3, np.sign(ang) * np.radians(42), ang)),
    "all shrink factors set to their mean": mod.mean() * np.exp(1j * ang),
    "rotating shrink factors set to their mean (0 and 180 deg kept)": np.where(abs(np.sin(ang)) > 1e-6, mod[abs(np.sin(ang)) > 1e-6].mean(), mod) * np.exp(1j * ang),
    "each angle +1 deg (sign kept)": mod * np.exp(1j * (ang + np.where(abs(np.sin(ang)) > 1e-6, np.sign(np.sin(ang)), 0) * np.radians(1))),
    "each angle +0.25 deg": mod * np.exp(1j * (ang + np.where(abs(np.sin(ang)) > 1e-6, np.sign(np.sin(ang)), 0) * np.radians(0.25))),
    "each shrink factor x1.01": 1.01 * ev,
    "each shrink factor x1.0025": 1.0025 * ev,
}
for k, e2 in cases.items():
    d, acc = kl(sub(e2)); print(f"  {k:62s} KL {d:.4f}  accuracy {acc:.4f}")
