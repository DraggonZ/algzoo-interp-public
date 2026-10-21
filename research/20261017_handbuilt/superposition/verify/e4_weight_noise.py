"""50 copies of ARC with N(0, 0.0025^2) added to all 432 weights."""
from common import *
P = arc(); g = gen(9404); f = flat(P); print("number of weights:", f.numel())
x = draw(400_000, g); y = target(x); top = x.argmax(1)
pr = predict(P, x); w = pr != y
print(f"ARC: accuracy {100*(~w).double().mean().item():.2f}%, mistakes naming the largest {100*(pr[w]==top[w]).double().mean().item():.1f}%")
accs, larg = [], []
for c in range(50):
    Q = unflat(f + 0.0025 * torch.randn(f.shape, dtype=DT, device=DEV, generator=g), P)
    q = predict(Q, x); wq = q != y
    accs.append((~wq).double().mean().item()); larg.append((q[wq] == top[wq]).double().mean().item())
accs, larg = np.array(accs), np.array(larg)
print(f"50 copies: mean accuracy {100*accs.mean():.2f}% (sd {100*accs.std():.2f}, range {100*accs.min():.1f}-{100*accs.max():.1f})")
print(f"  mean share of mistakes naming the largest {100*larg.mean():.1f}% (sd {100*larg.std():.1f})")
