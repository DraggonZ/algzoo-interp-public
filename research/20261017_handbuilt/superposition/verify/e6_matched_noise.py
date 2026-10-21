"""Gaussian noise on ARC's hidden state, per neuron variance 0.005 x that neuron's state variance."""
from common import *
P = arc(); g = gen(9606)
H = states(P, draw(200_000, g))                     # [10,B,16]
var = H.reshape(-1, 16).var(0)                       # pooled over all steps
sd = (0.005 * var).sqrt()
print("per-neuron noise sd:", np.round(sd.cpu().numpy(), 4))
N = 1_000_000; x = draw(N, g); y = target(x)
a = lambda pr: 100 * (pr == y).double().mean().item()
print(f"clean: {a(predict(P, x)):.2f}%")
r1, r2, r3 = [], [], []
for c in range(0, N, 250_000):
    xb = x[c:c+250_000]; B = len(xb)
    fn = torch.randn(B, 16, dtype=DT, device=DEV, generator=g) * sd
    r1.append(scores(P, xb, final_noise=fn).argmax(1))
    sn = torch.randn(10, B, 16, dtype=DT, device=DEV, generator=g) * sd
    r2.append(scores(P, xb, step_noise=sn).argmax(1))
    # variant: noise added before the relu at every step
    w_hi, W_hh, W_oh = P; h = torch.zeros(B, 16, dtype=DT, device=DEV)
    for t in range(10):
        h = torch.relu(h @ W_hh.T + xb[:, t:t+1] * w_hi + sn[t])
    r3.append((h @ W_oh.T).argmax(1))
print(f"noise on h_10 only: {a(torch.cat(r1)):.2f}%")
print(f"noise added to the state after the relu at every step: {a(torch.cat(r2)):.2f}%")
print(f"variant, noise added before the relu at every step: {a(torch.cat(r3)):.2f}%")
