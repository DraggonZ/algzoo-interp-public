"""Lean rule with explained constants built in (README experiment 24): input weight fixed to ARC's; feedback as
blind differences h_k(t-1) - w_in,k x_{t-1}; free: 9 feature constants (8 without the previous jump) + 4 feedback.
Fitted to the rule's own multi-step run; weights derived from the rule; surprise accounting as step17."""
import numpy as np
import torch as th
from harness import *  # noqa
import step5_lean as s5
import step8_lategate as s8
import step17_bits as s17

dev = "cuda"; WI = simlib.WI; WH = simlib.WH
win = th.tensor(WI[HID], device=dev)


def feats(X, n, with_jump=True):
    V, P = s5.s4.parts(X)
    from step3_recursive import account_features
    A = account_features(X, n)
    F = [A[..., 1], A[..., 2], A[..., 3], A[..., 4]] + P["sd"]
    if with_jump:
        F += P["lategate"]
    F += [lag(V["jump"], 2), lag(V["newmax"] * V["x"], 2)]
    return np.stack(F, -1)


def fit(with_jump):
    Ff = th.tensor(np.stack([feats(Xf, n, with_jump) for n in HID], 2), device=dev)   # (B,10,4,nf)
    Fs = th.tensor(np.stack([feats(Xs, n, with_jump) for n in HID], 2), device=dev)
    xf, xs = th.tensor(Xf, device=dev), th.tensor(Xs, device=dev)
    nf = Ff.shape[-1]
    # initial: teacher-forced least squares on pre - w_in x
    Hp = np.concatenate([np.zeros((len(Xf), 1, 16)), Hf[:, :-1]], 1)[..., HID]
    xprev = np.concatenate([np.zeros((len(Xf), 1)), Xf[:, :-1]], 1)
    blindfb = Hp - xprev[..., None] * WI[HID]
    C0 = []
    for i, n in enumerate(HID):
        A = np.concatenate([Ff[:, :, i].cpu().numpy(), blindfb], -1)
        C0.append(np.linalg.lstsq(A.reshape(-1, A.shape[-1]), (Pf[:, :, n] - WI[n] * Xf).reshape(-1), rcond=None)[0])
    C = th.tensor(np.stack(C0), device=dev, requires_grad=True)
    Pt = th.tensor(Pf[:, :, HID], device=dev)

    def rollout(F, x, C):
        prev = th.zeros(F.shape[0], 4, device=dev, dtype=F.dtype); pres = []
        for t in range(10):
            xp = x[:, t - 1:t] if t else th.zeros_like(x[:, :1])
            pre = win * x[:, t:t + 1] + (F[:, t] * C[None, :, :nf]).sum(-1) + (prev - xp * win) @ C[:, nf:].T
            pres.append(pre); prev = th.relu(pre)
        return th.stack(pres, 1)

    opt = th.optim.Adam([C], lr=2e-3); g = th.Generator(device=dev).manual_seed(0)
    for it in range(3001):
        idx = th.randint(0, len(Xf), (20000,), device=dev, generator=g)
        opt.zero_grad(); pre = rollout(Ff[idx], xf[idx], C); tgt = Pt[idx]
        L = th.where(tgt > 0, (pre - tgt) ** 2, th.relu(pre) ** 2).mean(); L.backward(); opt.step()
    return C.detach(), rollout, Ff, Fs, xf, xs, nf


def derive_weights(C, rollout, Ff, xf):
    with th.no_grad():
        pre = rollout(Ff, xf, C).cpu().numpy()
    Hp = th.tensor(np.concatenate([np.zeros((len(Xf), 1, 16)), Hf[:, :-1]], 1).reshape(-1, 16))
    out = {}
    for i, n in enumerate(HID):
        tgt = th.tensor(pre[:, :, i].reshape(-1)); xin = th.tensor(Xf.reshape(-1)) * WI[n]; on = tgt > 0
        w = th.zeros(16, dtype=th.float64, requires_grad=True)
        opt = th.optim.LBFGS([w], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); p = Hp @ w + xin
            L = th.where(on, (p - tgt) ** 2, th.relu(p) ** 2).mean(); L.backward(); return L
        for _ in range(4):
            opt.step(closure)
        out[n] = w.detach().numpy()
    return out


if __name__ == "__main__":
    for with_jump in [True, False]:
        lab = "constrained lean rule" + ("" if with_jump else ", without the previous jump")
        C, rollout, Ff, Fs, xf, xs, nf = fit(with_jump)
        with th.no_grad():
            rec = th.relu(rollout(Fs, xs, C)).cpu().numpy()
        print(f"\n{lab}: {C.numel()} free constants ({nf} features + 4 feedback per neuron)")
        names = ["m1", "m2", "wave cos", "wave sin", "store", "detector"] + (["prev jump"] if with_jump else []) + ["jump 2 back", "new largest 2 back", "fb 1", "fb 6", "fb 7", "fb 8"]
        for i, n in enumerate(HID):
            r = np.sqrt(((rec[:, :, i] - Hs[:, :, n]) ** 2).mean())
            print(f"  neuron {n}: rms {r:.3f} ({BASE_RMS[n] / r:.2f}x the account); constants: " +
                  ", ".join(f"{a} {b:+.3f}" for a, b in zip(names, C[i].cpu().numpy())))
        sc = score(Xs, Ls, run(Xs, {n: rec[:, :, i] for i, n in enumerate(HID)})[0])
        print(f"  all four replaced: accuracy {sc['acc']:.4f}, KL {sc['kl']:.3f}")
        fitted = derive_weights(C, rollout, Ff, xf)
        kl0 = s17.joint_kl(fitted)
        for n in HID:
            a, f = WH[n], fitted[n]
            print(f"  weights for {n}: cosine {a @ f / np.linalg.norm(a) / np.linalg.norm(f):.3f}")
        m = np.zeros(432); sd = s17.s_prior.copy()
        for n in HID:
            m[s17.rows[n]] = fitted[n]; sd[s17.rows[n]] = np.sqrt(((fitted[n] - WH[n]) ** 2).mean())
        left = s17.bits(m, sd); saved = s17.base - left
        print(f"  four rows from the rule: KL {kl0:.3f}; bits saved {saved:.0f}")
        chosen = None
        for b in [3, 4, 5, 6, 8, 10]:
            Cq = th.tensor(np.stack([s17.quant(c, b) for c in C.cpu().numpy()]), device=dev)
            kq = s17.joint_kl(derive_weights(Cq, rollout, Ff, xf))
            print(f"  constants rounded to {b:2d} mantissa bits: joint KL {kq:.3f} ({kq / kl0:.2f}x)", flush=True)
            if chosen is None and kq < 1.1 * kl0:
                chosen = b
        chosen = chosen or 12
        shared = 5
        cost = (C.numel() + shared + 4) * (chosen + 4)
        print(f"  mantissa needed {chosen}; stating the rule ({C.numel()} + {shared} + 4) x ({chosen} + 4) = {cost} bits; NET {saved - cost:+.0f} bits")
