"""Hypothesis: the second at position 0 is special because its record jump is its whole value (nothing positive came
before), so its hidden wave is unusually large; overtaking it 7 or 9 steps later catches the wave at a phase where the
reveal fails. Prediction (written first): any second that was a record with no positive number before it (whatever its
position) shows the same elevated error at ages 7 and 9 (age = largest's position - second's position), and a second
at position 0 preceded by nothing shows it only through that. Clear-gap lists as before."""
import sys; sys.path.insert(0, "../steer")
from harness import *
X = lists(32_000_000, seed=88); v = X.sort(1, descending=True).values
gap = (v[:, 1] >= 0.3 * v[:, 0]) & (v[:, 0] >= 1.2 * v[:, 1]) & (v[:, 1] >= 1.2 * v[:, 2]) & (v[:, 2] > 0.2)
X = X[gap]; o = X.argsort(1, descending=True); L, S, T = o[:, 0], o[:, 1], o[:, 2]; a = run(ARC, X).argmax(1); wr = a != S
n = torch.arange(len(X), device=DEV)
prevmax = torch.cat([torch.zeros(len(X), 1, device=DEV, dtype=dt), X.clamp(min=0).cummax(1).values[:, :-1]], 1)[n, S]   # largest positive before the second
fresh = prevmax == 0
age = L - S
print(f"{len(X):,} clear-gap lists")
for lab, base in (("second with no positive number before it", fresh & (S > 0)), ("second with a positive number before it", ~fresh & (S > 0) & (T > S)), ("second at position 0", S == 0)):
    row = []
    for g in range(1, 10):
        m = base & (age == g)
        if m.sum() > 2000: row.append(f"{g}:{100*wr[m].double().mean().item():.2f}")
    print(f"{lab:44s} wrong % by age: " + " ".join(row))
