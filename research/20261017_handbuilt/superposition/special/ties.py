"""Follow-up to P8/P10: a number equal to the stored largest is not a new record, so it is not hidden and its vote wins.
Prediction (written first): put a copy of the largest at another position. If the copy comes after the largest, the
network names the copy (it is the visible one); if the copy comes first, the later original is the 'equal, not new'
one and gets named. Either way it names the later of the two. Then a near-copy L(1-e), which is the true second:
named correctly when it comes after the largest; when it comes first, the largest arrives as a new record barely above
it, the hard case (P5), so misses grow as e shrinks."""
import sys; sys.path.insert(0, "../steer")
from harness import *
R = lists(500_000, seed=13); n = torch.arange(len(R), device=DEV); L = R.argmax(1)
g = torch.Generator(device=DEV).manual_seed(3); P = torch.randint(0, 10, (len(R),), device=DEV, generator=g); ok = P != L
R, n, L, P = R[ok], torch.arange(int(ok.sum()), device=DEV), L[ok], P[ok]
for e in (0.0, 0.001, 0.01, 0.03, 0.1):
    Y = R.clone(); Y[n, P] = R[n, L] * (1 - e); a = run(ARC, Y).argmax(1); later = torch.maximum(L, P)
    aft = P > L
    print(f"copy at {1-e:.3f} x largest: names the later of the two {100*(a == later).double().mean().item():5.1f}% | copy after the largest: names the copy {100*(a[aft] == P[aft]).double().mean().item():5.1f}% | copy before: names the copy {100*(a[~aft] == P[~aft]).double().mean().item():5.1f}%, names the largest {100*(a[~aft] == L[~aft]).double().mean().item():5.1f}%", flush=True)
