"""Do small state errors grow step by step (chaos), or is the sensitivity only at the readout (cancellation)?
Per list: Jacobian of the final state with respect to the state after step s is the product of diag(on/off) W_hh.
Report the growth of the largest singular value per step (Lyapunov-like rate), and how a small random relative
error in the state at step s changes (a) the final state, relative to its size, (b) the second-third margin,
relative to the margin, for near-tie lists."""
import numpy as np, torch as th
from algzoo.core import example, forward
th.manual_seed(0)
nets = {"ARC": [np.asarray(p, float) for p in example()]}
d = np.load("/data/output/research/algzoo/20261017_handbuilt/paper/tweet/handbuilt_matched.npz")
nets["hand-built (Fig. 2)"] = [d["wi"], d["wh"], d["wo"]]
x = th.randn(200000, 10, dtype=th.float64)
srt = x.sort(-1, descending=True)
near = ((srt.values[:, 1] - srt.values[:, 2]) / srt.values[:, 1].abs() < 0.05) & (srt.values[:, 1] > 0)
for name, P in nets.items():
    wi, wh, wo = (th.tensor(p) for p in P)
    lg, hs = forward(P, x, return_hidden=True)            # hs: [N, 11, H]
    on = (hs[:, 1:] > 0).double()                         # on/off pattern at steps 0..9
    i2, i3 = srt.indices[:, 1], srt.indices[:, 2]
    r = th.arange(len(x))
    print(f"\n{name}: accuracy {(lg.argmax(-1) == i2).double().mean():.3f}")
    for s in (2, 5, 8):                                   # perturb state after step s (0-based input index s)
        J = th.eye(wh.shape[0], dtype=th.float64).expand(len(x), -1, -1).clone()
        for t in range(s + 1, 10):
            J = (on[:, t, :, None] * wh) @ J              # d h_t / d h_s
        sv = th.linalg.svdvals(J[:20000])
        k = 9 - s
        rate = (sv[:, 0].log() / k).median().item()
        hs_s, h9 = hs[:, s + 1], hs[:, 10]
        e = th.randn_like(hs_s); e = e / e.norm(dim=-1, keepdim=True) * hs_s.norm(dim=-1, keepdim=True) * 1e-3
        dh9 = (J @ e[..., None])[..., 0]
        state_amp = (dh9.norm(dim=-1) / h9.norm(dim=-1) / 1e-3)
        m = lg[r, i2] - lg[r, i3]
        dl = dh9 @ wo.T
        dm = dl[r, i2] - dl[r, i3]
        marg_amp = (dm.abs() / m.abs() / 1e-3)
        print(f"  error after step {s} ({k} steps to go): top growth per step x{np.exp(rate):.2f}; "
              f"0.1% state error -> final state {th.median(state_amp[near]).item()*0.1:.2f}%, "
              f"near-tie margin {th.median(marg_amp[near]).item()*0.1:.0f}% (median)")
