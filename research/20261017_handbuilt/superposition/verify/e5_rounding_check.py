"""Check whether the k=0 rounding baseline depends on rounding convention, precision or which matrices are rounded."""
from common import *
P = arc(); g = gen(9506); xt = draw(2_000_000, g); yt = target(xt)
def acc(Q, dt=DT):
    Q = tuple(q.to(dt) for q in Q); return 100*(predict(Q, xt.to(dt)) == yt).double().mean().item()
names = ["w_hi", "W_hh", "W_oh"]
for step in (0.01, 0.1):
    R = tuple(torch.round(p/step)*step for p in P)
    Rn = to_t([np.round(np.asarray(p.cpu())/step)*step for p in P])
    Rf = tuple(torch.floor(p/step + 0.5)*step for p in P)
    print(f"step {step}: torch.round {acc(R):.2f}, numpy round {acc(Rn):.2f}, floor(x+0.5) {acc(Rf):.2f}, float32 eval {acc(R, torch.float32):.2f}")
    for i, n in enumerate(names):
        Q = list(P); Q[i] = R[i]
        print(f"   only {n} rounded: {acc(tuple(Q)):.2f}")
    for i, n in enumerate(names):
        Q = list(R); Q[i] = P[i]
        print(f"   all but {n} rounded: {acc(tuple(Q)):.2f}")
