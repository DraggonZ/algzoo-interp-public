"""Shift-register test (account item 7: the readout row for position k+1 equals the row for k pushed one step through
the dynamics). Prediction, written first: running the network for one extra step with input 0 before the readout makes
every vote look one step older, so the network should name the position one earlier (second - 1) on most lists with
the second at 1-9; two extra steps, two earlier. The hidden largest rotates with the rest, so it should stay hidden."""
from harness import *
X = lists(1_000_000, seed=71); o = X.argsort(1, descending=True); L, S = o[:, 0], o[:, 1]
def run_extra(W, X, k, x_extra=0.0):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    for _ in range(k): h = torch.relu(h @ W[1].T + x_extra * W[0])
    return (h @ W[2].T).argmax(1)
for k in (1, 2, 3):
    a = run_extra(ARC, X, k); m = S >= k
    print(f"{k} extra step(s), input 0: names second-{k}: {100*(a[m] == S[m] - k).double().mean().item():5.1f}% | the second: {100*(a[m] == S[m]).double().mean().item():5.1f}% | largest-{k}: {100*(a[m] == L[m] - k).double().mean().item():5.1f}% | largest: {100*(a[m] == L[m]).double().mean().item():5.1f}%", flush=True)
    for p in range(10):
        mm = S == p
        if p >= k: print(f"      second at {p}: names {p-k} on {100*(a[mm] == p - k).double().mean().item():5.1f}%")
print("-- an 11th number appended (prediction, written first: the network treats it as an eleventh number, with every old vote one step older. A huge one becomes the new largest and reveals the old largest, so it should name largest-1; one between the largest and second becomes the second, at the last slot, so it should name position 9; a tiny one changes nothing but the shift, so second-1)")
def run_append(W, X, x):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    h = torch.relu(h @ W[1].T + x[:, None] * W[0]); return (h @ W[2].T).argmax(1)
v = X.sort(1, descending=True).values
for lab, x in (("huge (10)", torch.full((len(X),), 10.0, device=DEV, dtype=dt)), ("twice the largest", 2 * v[:, 0]), ("between largest and second", (v[:, 0] + v[:, 1]) / 2), ("very negative (-3)", torch.full((len(X),), -3.0, device=DEV, dtype=dt))):
    a = run_append(ARC, X, x); mL = L >= 1; mS = S >= 1
    print(f"appended {lab:28s}: names largest-1 {100*(a[mL] == L[mL] - 1).double().mean().item():5.1f}% | position 9 {100*(a == 9).double().mean().item():5.1f}% | second-1 {100*(a[mS] == S[mS] - 1).double().mean().item():5.1f}%", flush=True)
