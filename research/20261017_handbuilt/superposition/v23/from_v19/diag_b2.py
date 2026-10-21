"""What does the direct route change, against the gate route only (same layout, same loop gain)?"""
import torch
from check_basic import T, scores, lists, DEV, dt
from algzoo.handbuilt_v23_from_v19 import build_v23f
torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)
if __name__ == "__main__":
    X = lists(100_000, 1)
    s0 = scores(T(build_v23f(r1=0.0, ring=False)), X); s1 = scores(T(build_v23f(ring=False)), X)
    d = s1 - s0
    print("difference half-direct minus gate-only, quantiles by position (5%, 50%, 95%):")
    for p in range(10):
        q = torch.quantile(d[:20000, p], torch.tensor([0.05, 0.5, 0.95], device=DEV, dtype=dt))
        print(f"  position {p}: {q.tolist()}")
    S = X.argsort(1, descending=True)[:, 1]
    fx = (s1.argmax(1) == S) & (s0.argmax(1) != S)
    i = torch.nonzero(fx)[:3, 0]
    for k in i:
        print("list", X[k].cpu().numpy().round(3)); print("  gate only", s0[k].cpu().numpy().round(4)); print("  half dir ", s1[k].cpu().numpy().round(4))
