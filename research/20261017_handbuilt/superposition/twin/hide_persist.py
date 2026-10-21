"""Does a nudge to the hiding neurons fade in ARC? Scale the hiding neurons by 0.9 at step 5 (largest by step 3) and
follow the hiding neurons' change over steps 5-9, as a share of their unnudged activity (median over lists; size measured
as the norm over the group). In v23 the clock rotates without loss, so the share should stay 0.10."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family"); sys.path.insert(0, "..")
import torch
from common import DEV, dt, T, ARC, top3
from battery import ARC_GROUPS, hb_groups
from algzoo.handbuilt_v23 import build_v23
def states(W, X, g=None):
    h = torch.zeros(len(X), W[0].shape[0], device=DEV, dtype=dt); out = []
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        if g is not None and t == 5: h = h.clone(); h[:, g] *= 0.9
        out.append(h)
    return out
gn = torch.Generator(device=DEV).manual_seed(12); X = torch.randn(200_000, 10, device=DEV, dtype=dt, generator=gn); X = X[top3(X)[0] <= 3]
net = build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False)
for name, W, G in (("ARC", ARC, ARC_GROUPS["hiding"]), ("v23 ARC settings (clock and flip, wave part only)", T(net.params), hb_groups(net)["hiding"])):
    a, b = states(W, X), states(W, X, G)
    row = []
    for t in range(5, 10):
        da = (b[t][:, G] - a[t][:, G]); base = a[t][:, G]
        if "v23" in name:   # remove the shared comparator level, which the next step rebuilds from the store and input
            da = da - da.mean(1, keepdim=True); base = base - base.mean(1, keepdim=True)
        row.append((da.norm(dim=1) / base.norm(dim=1).clamp(min=1e-9)).median().item())
    print(f"{name}: change in the hiding neurons as a share of their activity, steps 5-9: " + " ".join(f"{v:.3f}" for v in row))
