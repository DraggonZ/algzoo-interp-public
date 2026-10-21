"""Longer lists. The readout reads ten slots, the last ten steps. Prediction from the account (written first): the
store remembers the all-time largest and keeps it hidden even after it leaves the window; votes older than ten steps
are no longer read. So the answer should be the largest number in the window that is below the all-time largest, i.e.
the overall second if it lies in the window. Alternative: second-largest of the window alone (as if the network
forgot everything older). Report how often each is named, split by where the overall largest sits."""
from harness import *
def run_len(W, X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(X.shape[1]): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    return (h @ W[2].T).argmax(1)
for T in (12, 15):
    g = torch.Generator(device=DEV).manual_seed(T); X = torch.randn(1_000_000, T, device=DEV, dtype=dt, generator=g)
    a = run_len(ARC, X) + (T - 10)                       # slot k = step T-10+k
    Lall = X.argmax(1); win = X[:, T - 10:]; n = torch.arange(len(X), device=DEV)
    below = torch.where(win < X[n, Lall][:, None], win, torch.full_like(win, -1e9)); pred = below.argmax(1) + (T - 10)
    wo = win.argsort(1, descending=True); win2 = wo[:, 1] + (T - 10)
    for lab, m in (("largest inside the window", Lall >= T - 10), ("largest before the window", Lall < T - 10)):
        print(f"{T} numbers, {lab:26s} ({100*m.double().mean().item():.0f}% of lists): account's answer {100*(a[m] == pred[m]).double().mean().item():5.1f}% | second of the window {100*(a[m] == win2[m]).double().mean().item():5.1f}%", flush=True)
