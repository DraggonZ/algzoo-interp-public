"""r9a analysis 1: r6a_gap with the hiding wave relayed between the clocks C and a relay set R (handbuilt_v9a), 3 or 2
relays. Own accuracy (1M lists, seed 1000), and the scorer's job-8 and version-2 hiding-test formulas re-implemented
for any set of neurons (copied from score_twin.jobs job 8 and score_mechanism2.names_largest_when_silenced, same
lists and seeds). Checked first on r6a_gap itself with the clocks + runner-up as the hiding set."""
import numpy as np, torch
from algzoo.handbuilt_v9a import build_v9a
DEV, DT = "cuda", torch.float64


def lists(n, seed): return torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(seed))


def T(P): return [torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in P]


def acc(P, n=1_000_000):
    wi, wh, wo = T(P); X = lists(n, 1000); Y2 = X.argsort(1, descending=True)[:, 1]
    h = torch.zeros(n, len(wi), device=DEV, dtype=DT)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return 100 * ((h @ wo.T).argmax(1) == Y2).double().mean().item()


def job8(P, HID):
    wi, wh, wo = T(P); X = lists(200_000, 7); Xp = X[X.argmax(1) <= 3][:50000]
    woc = wo[:, HID] - wo[:, HID].mean(0)
    def st(nudge):
        h = torch.zeros(len(Xp), len(wi), device=DEV, dtype=DT); o = []
        for t in range(10):
            h = torch.relu(h @ wh.T + Xp[:, t:t + 1] * wi)
            if nudge and t == 5: h = h.clone(); h[:, HID] *= 0.9
            o.append(h)
        return o
    a, b = st(False), st(True); d6 = b[6][:, HID] - a[6][:, HID]
    left = (d6.norm(dim=1) / a[6][:, HID].norm(dim=1).clamp(min=1e-9)).median().item() / 0.1
    leftr = ((d6 @ woc.T).norm(dim=1) / (a[6][:, HID] @ woc.T).norm(dim=1).clamp(min=1e-12)).median().item() / 0.1
    return left, leftr


def silenced(P, KILL):
    wi, wh, wo = T(P); X = lists(400_000, 31); L = X.argmax(1); keep = L <= 8; X, L = X[keep], L[keep]
    h = torch.zeros(len(X), len(wi), device=DEV, dtype=DT)
    for t in range(10):
        h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
        m = (L + 1) == t
        if m.any(): h = h.clone(); h[m.nonzero().flatten()[:, None], torch.tensor(KILL, device=DEV)] = 0.0
    named = (h @ wo.T).argmax(1)
    return 100 * (named == L).double().mean().item(), 100 * (named == 0).double().mean().item()


if __name__ == "__main__":
    for k in (3, 2):
        net, P, C = build_v9a(n_relay=k); Q = net.idx["runnerup"]; R = list(range(16, 16 + k))
        if k == 3:
            base = net.params
            print(f"r6a_gap (16): accuracy {acc(base):.2f}", flush=True)
            print(f"  hiding = clocks + runner-up: job 8 activity / readout view {job8(base, C + [Q])}; "
                  f"silenced -> names largest / names position 0 {silenced(base, C + [Q])}", flush=True)
        print(f"relay {k} ({16 + k} neurons): accuracy {acc(P):.2f}", flush=True)
        for name, hid in (("relays + runner-up", R + [Q]), ("relays", R), ("clocks", C), ("clocks + runner-up", C + [Q])):
            print(f"  {name:20s} job 8 activity / readout view " + " ".join(f"{v:.3f}" for v in job8(P, hid)) +
                  "; silenced one step after the largest -> names largest / position 0 " +
                  " ".join(f"{v:.1f}" for v in silenced(P, hid)), flush=True)
