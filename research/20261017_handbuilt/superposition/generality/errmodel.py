"""Question 4: the error model of ../errmodel/shuffle_typical.py per target network. Typical rule: one linear rule per
exact arrangement of the three largest (score_T - score_S and score_L - score_S as a x_L + b x_S + c x_T, fitted on
2,000,000 other random lists, as ../../family/config_rule.py but on the target's own scores). Leftover: the network's two
differences minus the rule's. Typical + leftover borrowed from another list in the same arrangement and ratio bin
(rescaled by the second-largest), against the network's own accuracy among the three largest."""
from common_gen import *
from config_rule import cells
def fit(W, n=2000000):
    g = torch.Generator(device=DEV).manual_seed(31)
    A = torch.zeros(1000, 3, 3, device=DEV, dtype=dt); B = torch.zeros(1000, 3, 2, device=DEV, dtype=dt)
    for _ in range(n // 500000):
        X = torch.randn(500000, 10, device=DEV, dtype=dt, generator=g); o = X.argsort(1, descending=True); ar = torch.arange(len(X), device=DEV)
        v = torch.stack([X[ar, o[:, k]] for k in range(3)], 1); Sc = scores(W, X)
        y = torch.stack([Sc[ar, o[:, 2]] - Sc[ar, o[:, 1]], Sc[ar, o[:, 0]] - Sc[ar, o[:, 1]]], 1); c = cells(X, "exact arrangement")
        A.index_add_(0, c, v[:, :, None] * v[:, None, :]); B.index_add_(0, c, v[:, :, None] * y[:, None, :])
    return torch.linalg.solve(A + 1e-6 * torch.eye(3, device=DEV, dtype=dt), B)
def choice(D): return torch.where((D[:, 0] > 0) & (D[:, 0] > D[:, 1]), 2, torch.where((D[:, 1] > 0) & (D[:, 1] >= D[:, 0]), 0, 1))
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(202); X = torch.randn(1000000, 10, device=DEV, dtype=dt, generator=g)
    o = X.argsort(1, descending=True); L, S, Tq = o[:, 0], o[:, 1], o[:, 2]; ar = torch.arange(len(X), device=DEV)
    v = torch.stack([X[ar, L], X[ar, S], X[ar, Tq]], 1); xs = v[:, 1]
    rb = torch.bucketize(v[:, 0] / xs.abs().clamp(min=1e-3), torch.tensor([1.1, 1.25, 1.5, 2.0, 3.0], device=DEV, dtype=dt))
    cell = cells(X, "exact arrangement") * 6 + rb; near = (xs > 0) & ((xs - v[:, 2]) / xs < 0.05)
    def borrow(Dr, E, seed, within=True):
        gg = torch.Generator(device=DEV).manual_seed(seed)
        if within:
            order = torch.argsort(cell.double() + torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * 0.5); cs = cell[order]
            st, en = torch.searchsorted(cs, cs), torch.searchsorted(cs, cs, right=True)
            pick = torch.minimum(st + (torch.rand(len(X), device=DEV, dtype=dt, generator=gg) * (en - st)).long(), en - 1)
            src = torch.empty_like(order); src[order] = order[pick]
        else: src = torch.randint(0, len(X), (len(X),), device=DEV, generator=gg)
        return Dr + E[src] * (xs.abs() / xs[src].abs().clamp(min=1e-3)).clamp(max=10)[:, None]
    print(f"{'network':10s} {'own, top three':>14s} {'rule alone':>10s} {'rule+leftover':>13s} {'(any list)':>10s} | {'own mistakes repr.: rule':>24s} {'rule+leftover':>13s} | near ties third wins: own, rule, rule+leftover | leftover RMS / decision RMS")
    for nm, W in TARGETS.items():
        sol = fit(W); Sa = scores(W, X)
        Da = torch.stack([Sa[ar, Tq] - Sa[ar, S], Sa[ar, L] - Sa[ar, S]], 1); Dr = (v[:, :, None] * sol[cells(X, "exact arrangement")]).sum(1); E = Da - Dr
        ca = choice(Da); wr = ca != 1
        def st(D):
            c = choice(D); return 100 * (c == 1).double().mean().item(), 100 * ((c == ca) & wr).double().sum().item() / wr.double().sum().item(), 100 * ((c == 2) & near).double().sum().item() / near.double().sum().item()
        a0, _, n0 = st(Da); a1, r1, n1 = st(Dr); b = [st(borrow(Dr, E, s)) for s in range(2)]; a2 = np.mean([x[0] for x in b]); r2 = np.mean([x[1] for x in b]); n2 = np.mean([x[2] for x in b])
        a3, _, _ = st(borrow(Dr, E, 9, False))
        print(f"{nm:10s} {a0:13.2f}% {a1:9.2f}% {a2:12.2f}% {a3:9.2f}% | {r1:23.1f}% {r2:12.1f}% | {n0:5.1f}% {n1:5.1f}% {n2:5.1f}% | {E.pow(2).mean().sqrt():.2f} / {Da.pow(2).mean().sqrt():.2f}", flush=True)
