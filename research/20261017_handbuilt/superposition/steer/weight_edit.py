"""Can the bias edit be made permanent? Weights are shared across steps, so 'the hiding neurons' input at the last step'
is not a weight. Candidates and predictions (written first):
(a) hiding input weights x0.99 at every step: also weakens their response to every new largest (push about new - 0.7
    old) at every step, so the largest leaks a little everywhere: predicted accuracy loss of a few tenths of a point,
    bias partly removed.
(b) hiding neurons' readout x(1-c): the readout carries their whole history, not just the last input; predicted to
    shift the largest's cancellation, cost accuracy, bias change unclear.
Conclusion expected: no clean permanent edit; the bias is a property of shared weights serving two jobs."""
from harness import *
X = lists(4_000_000, seed=21); o = X.argsort(1, descending=True); s = o[:, 1]
v = X.sort(1, descending=True).values; near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1])
late9 = near & ((o[:, 1] == 9) | (o[:, 2] == 9))
def show(lab, W):
    a = run(W, X).argmax(1)
    print(f"{lab:34s}: accuracy {100*(a == s).double().mean().item():.3f}% | step-9 contender wins {100*(a[late9] == 9).double().mean().item():5.1f}%", flush=True)
show("unchanged", ARC)
for c in (0.01, 0.02):
    W = [p.clone() for p in ARC]; W[0][[1, 6, 7, 8]] *= (1 - c); show(f"(a) hiding input weights x{1-c}", W)
    W = [p.clone() for p in ARC]; W[2][:, [1, 6, 7, 8]] *= (1 - c); show(f"(b) hiding readout x{1-c}", W)
