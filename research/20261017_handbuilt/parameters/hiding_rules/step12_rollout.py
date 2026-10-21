"""Fit the rule's constants to its own multi-step run (rollout) instead of one step with ARC's previous values.
Loss on the fit lists: squared error of the rule's input against ARC's where ARC's neuron is on, max(0, rule input)^2
where it is off, all four neurons and all steps, with the rule fed its own previous outputs. Start from the one-step
least-squares constants; Adam on the GPU. Feature set as step11 (33 per neuron). Then activity errors and weights from
the rule, as before. Analysis fit to ARC's activity only."""
import numpy as np
import torch as th
from harness import *  # noqa
import step3_recursive as s3
import step8_lategate as s8
import step11_more as s11

dev = "cuda"
feat = s11.feat
Ffit = th.tensor(np.stack([feat(Xf, n) for n in HID], 2), device=dev)      # (B, 10, 4, nf)
Fsc = th.tensor(np.stack([feat(Xs, n) for n in HID], 2), device=dev)
Pfit = th.tensor(Pf[:, :, HID], device=dev)
Hprev_f = np.concatenate([np.zeros((len(Xf), 1, 16)), Hf[:, :-1]], 1)[..., HID]
C0 = []
for i, n in enumerate(HID):
    F = np.concatenate([feat(Xf, n), Hprev_f], -1)
    C0.append(np.linalg.lstsq(F.reshape(-1, F.shape[-1]), Pf[:, :, n].reshape(-1), rcond=None)[0])
C = th.tensor(np.stack(C0), device=dev, requires_grad=True)                 # (4, nf + 4)
nf = Ffit.shape[-1]


def rollout(F, C):
    B = F.shape[0]; prev = th.zeros(B, 4, device=dev, dtype=F.dtype); pres = []
    for t in range(10):
        pre = (F[:, t] * C[None, :, :nf]).sum(-1) + prev @ C[:, nf:].T
        pres.append(pre); prev = th.relu(pre)
    return th.stack(pres, 1)


def loss_fn(C, idx):
    pre = rollout(Ffit[idx], C); tgt = Pfit[idx]; on = tgt > 0
    return th.where(on, (pre - tgt) ** 2, th.relu(pre) ** 2).mean()


opt = th.optim.Adam([C], lr=2e-3)
g = th.Generator(device=dev).manual_seed(0)
for it in range(3001):
    idx = th.randint(0, len(Xf), (20000,), device=dev, generator=g)
    opt.zero_grad(); L = loss_fn(C, idx); L.backward(); opt.step()
    if it % 500 == 0:
        with th.no_grad():
            full = loss_fn(C, th.arange(len(Xf), device=dev)).item()
        print(f"step {it}: rollout loss on the fit lists {full:.4f}", flush=True)
with th.no_grad():
    rec = th.relu(rollout(Fsc, C)).cpu().numpy()
Cn = C.detach().cpu().numpy()
co = {n: Cn[i] for i, n in enumerate(HID)}
print("\nrule fitted to its own run (33 constants + 4 feedback per neuron):")
for i, n in enumerate(HID):
    r = np.sqrt(((rec[:, :, i] - Hs[:, :, n]) ** 2).mean())
    sc = score(Xs, Ls, run(Xs, {n: rec[:, :, i]})[0])
    print(f"  neuron {n}: rms {r:.3f} ({BASE_RMS[n] / r:.2f}x the account); replaced alone: accuracy {sc['acc']:.4f}, KL {sc['kl']:.3f}")
sc = score(Xs, Ls, run(Xs, {n: rec[:, :, i] for i, n in enumerate(HID)})[0])
print(f"  all four replaced: accuracy {sc['acc']:.4f}, KL {sc['kl']:.3f}")
fitted = s8.weights_from_rule(feat, co)
WH = simlib.WH
for n in HID:
    a, f = WH[n], fitted[n]
    print(f"  weights for {n}: cosine {a @ f / np.linalg.norm(a) / np.linalg.norm(f):.3f}, relative distance {np.linalg.norm(f - a) / np.linalg.norm(a):.2f}")
for rows in [[1], [6], [7], [8], HID]:
    P = [w.copy() for w in s8.W0]
    for n in rows:
        P[1][n] = fitted[n]
    l = s8.run_g(P)
    print(f"  rows {rows} from the rule: accuracy {100 * ((l.argmax(-1) == s8.Y).float().mean().item() - s8.acc0):+.2f} points, KL {(s8.pb * (s8.lb - l)).sum(-1).mean().item():.3f}")
np.save("rule_rollout_constants.npy", Cn)
