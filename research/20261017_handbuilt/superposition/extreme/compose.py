"""Composition. Lists whose top two sit in one half and whose other half is all below q x second. Predictions (written
first): (a) top two in the first five, a quiet second half: the network names the second with high accuracy, falling as
the second half rises toward the second (more near ties with the third) and with the slot-0 weakness when the second
is at 0; (b) top two in the last five, first half quiet: also high, but the late-number effects (direct-route-only
last step) cost something when the second is at 9."""
import sys; sys.path.insert(0, "../steer")
from harness import *
g = torch.Generator(device=DEV).manual_seed(5)
def make(first, q, N=1_000_000):
    top = torch.randn(N, 5, device=DEV, dtype=dt, generator=g).abs() + 0.5            # the half holding the top two
    tv = top.sort(1, descending=True).values; s = tv[:, 1]
    other = s[:, None] * q * torch.rand(N, 5, device=DEV, dtype=dt, generator=g)       # uniform in [0, q x second)
    other = torch.where(torch.rand(N, 5, device=DEV, dtype=dt, generator=g) < 0.3, -torch.randn(N, 5, device=DEV, dtype=dt, generator=g).abs(), other)
    # keep only the top two of 'top' above the other half: the rest of 'top' also below q x second
    rest = top.clone(); rk = top.argsort(1, descending=True)
    for k in range(2, 5): rest[torch.arange(N, device=DEV), rk[:, k]] = s * q * torch.rand(N, device=DEV, dtype=dt, generator=g)
    X = torch.cat([rest, other], 1) if first else torch.cat([other, rest], 1)
    return X
for first in (True, False):
    for q in (0.0, 0.5, 0.9, 0.97):
        X = make(first, q); o = X.argsort(1, descending=True); a = run(ARC, X).argmax(1); s = o[:, 1]
        ok = a == s; row = " ".join(f"{p}:{100*ok[s == p].double().mean().item():.1f}" for p in (range(5) if first else range(5, 10)))
        print(f"top two in the {'first' if first else 'last'} five, others below {q} x second: right {100*ok.double().mean().item():.2f}% | by the second's position {row}", flush=True)
