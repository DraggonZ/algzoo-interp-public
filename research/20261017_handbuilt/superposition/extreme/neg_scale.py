"""Causal test of negatives.log (accuracy falls as the negatives grow relative to the second, but that is confounded
with a small second). Intervention on inputs: multiply only the negative numbers by c, keeping the positives, so the
top three and the right answer are unchanged whenever the second is positive. Prediction: accuracy falls with c, and
c = 0 (negatives set to zero) raises it above 95.3% on these lists."""
import sys; sys.path.insert(0, "../steer")
from harness import *
X = lists(4_000_000, seed=6); o = X.argsort(1, descending=True); s = o[:, 1]; v = X.sort(1, descending=True).values
m = v[:, 2] > 0                                           # top three positive, so scaling negatives keeps the answer
X, s = X[m], s[m]
for c in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
    Y = torch.where(X < 0, X * c, X); a = run(ARC, Y).argmax(1); ok = a == s
    print(f"negatives x{c}: right {100*ok.double().mean().item():.2f}% (lists with the top three positive, {100*m.double().mean().item():.0f}% of all)", flush=True)
