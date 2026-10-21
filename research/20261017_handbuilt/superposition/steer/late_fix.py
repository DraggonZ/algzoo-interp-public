"""Edit 2, revised from late_cost.log: the 1% last-step turn-down broke lists whose last number is the new largest,
because the hiding push (about new - 0.7 x old) shrinks faster than the vote. Keep the hiding neurons' input at full
strength and turn down only the other neurons' input at the last step. Prediction (before running): bias near 50%,
and the 0.12 points lost on last-step-largest lists mostly gone, so accuracy at or above 95.31%."""
from harness import *
def run_last(W, X, b, spare=()):
    f = torch.ones(16, device=DEV, dtype=dt) * (1 - b); f[list(spare)] = 1.0
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0] * (f if t == 9 else 1))
    return h @ W[2].T
X = lists(4_000_000, seed=21); s = X.argsort(1, descending=True)[:, 1]
v = X.sort(1, descending=True).values; near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1])
o = X.argsort(1, descending=True); late9 = near & ((o[:, 1] == 9) | (o[:, 2] == 9))
for lab, b, spare in (("unchanged", 0.0, ()), ("all neurons x0.99", 0.01, ()), ("all but hiding x0.99", 0.01, (1, 6, 7, 8)), ("all but hiding x0.98", 0.02, (1, 6, 7, 8))):
    a = run_last(ARC, X, b, spare).argmax(1)
    print(f"{lab:22s}: accuracy {100*(a == s).double().mean().item():.3f}% | near ties with a contender at step 9: it wins {100*(a[late9] == 9).double().mean().item():5.1f}% (truly second {100*(s[late9] == 9).double().mean().item():.1f}%)", flush=True)
print("-- follow-up: only the hiding neurons' input turned down at the last step (prediction: same bias removal and same cost as all neurons, since sparing them removed both)")
def run_only(W, X, b, only):
    f = torch.ones(16, device=DEV, dtype=dt); f[list(only)] = 1 - b
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0] * (f if t == 9 else 1))
    return h @ W[2].T
for only in ((1, 6, 7, 8), (1,), (6,), (7,), (8,)):
    for b in (0.01, 0.02):
        a = run_only(ARC, X, b, only).argmax(1)
        print(f"hiding {str(only):12s} x{1-b:.2f}: accuracy {100*(a == s).double().mean().item():.3f}% | step-9 contender wins {100*(a[late9] == 9).double().mean().item():5.1f}%", flush=True)
