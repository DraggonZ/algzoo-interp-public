"""The late gate's timing: it fires one step after a new largest and pushes hiding neurons 1 and 6 one step later, i.e.
two steps after the record. Add the record signal two steps back (jump and new-largest value at t-2) to the step11
features; fit to the rule's own run (as step12); weights from the rule (as step8). Is the late gate's weight into 1 and
6 recovered?"""
import numpy as np
import torch as th
from harness import *  # noqa
import step11_more as s11
import step8_lategate as s8


def feat(X, n):
    V = task_vars(X)
    extra = [lag(V["jump"], 2), lag(V["newmax"] * V["x"], 2)]
    return np.concatenate([s11.feat(X, n), np.stack(extra, -1)], -1)


import step12_rollout as s12  # noqa: E402  (reuse its rollout fit by swapping the features)
dev = "cuda"
s12.Ffit = th.tensor(np.stack([feat(Xf, n) for n in HID], 2), device=dev)
s12.Fsc = th.tensor(np.stack([feat(Xs, n) for n in HID], 2), device=dev)
s12.nf = s12.Ffit.shape[-1]
Hprev_f = np.concatenate([np.zeros((len(Xf), 1, 16)), Hf[:, :-1]], 1)[..., HID]
C0 = []
for i, n in enumerate(HID):
    F = np.concatenate([feat(Xf, n), Hprev_f], -1)
    C0.append(np.linalg.lstsq(F.reshape(-1, F.shape[-1]), Pf[:, :, n].reshape(-1), rcond=None)[0])
C = th.tensor(np.stack(C0), device=dev, requires_grad=True)
opt = th.optim.Adam([C], lr=2e-3); g = th.Generator(device=dev).manual_seed(0)
for it in range(3001):
    idx = th.randint(0, len(Xf), (20000,), device=dev, generator=g)
    opt.zero_grad(); L = s12.loss_fn(C, idx); L.backward(); opt.step()
with th.no_grad():
    rec = th.relu(s12.rollout(s12.Fsc, C)).cpu().numpy()
Cn = C.detach().cpu().numpy(); co = {n: Cn[i] for i, n in enumerate(HID)}
print(f"\nwith the record signal two steps back ({s12.nf} constants + 4 feedback per neuron):")
for i, n in enumerate(HID):
    r = np.sqrt(((rec[:, :, i] - Hs[:, :, n]) ** 2).mean())
    print(f"  neuron {n}: rms {r:.3f} ({BASE_RMS[n] / r:.2f}x the account)")
sc = score(Xs, Ls, run(Xs, {n: rec[:, :, i] for i, n in enumerate(HID)})[0])
print(f"  all four replaced: accuracy {sc['acc']:.4f}, KL {sc['kl']:.3f}")
fitted = s8.weights_from_rule(feat, co)
WH = simlib.WH
for n in HID:
    a, f = WH[n], fitted[n]
    print(f"  weights for {n}: cosine {a @ f / np.linalg.norm(a) / np.linalg.norm(f):.3f}, relative distance {np.linalg.norm(f - a) / np.linalg.norm(a):.2f}")
print(f"  late gate into 1 and 6: ARC {WH[1, 9]:+.2f} {WH[6, 9]:+.2f}; from the rule {fitted[1][9]:+.2f} {fitted[6][9]:+.2f}")
for rows in [[1], [6], [7], [8], HID]:
    P = [w.copy() for w in s8.W0]
    for n in rows:
        P[1][n] = fitted[n]
    l = s8.run_g(P)
    print(f"  rows {rows} from the rule: accuracy {100 * ((l.argmax(-1) == s8.Y).float().mean().item() - s8.acc0):+.2f} points, KL {(s8.pb * (s8.lb - l)).sum(-1).mean().item():.3f}")
np.save("rule_timing_constants.npy", Cn); np.save("weights_from_timing_rule.npy", np.stack([fitted[n] for n in HID]))
