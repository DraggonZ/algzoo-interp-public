"""Step 5: after the memory block's rotation is simplified (ARC's eigenvectors kept), how much KL can other weights
remove? Sets of free weights (a) readout, (b) + memory's incoming from outside, (c) + memory's outgoing to the other
groups, (d) everything except the memory block. Two re-derivations: least squares on states (analysis, rollout
iterated) and Adam on KL to ARC's outputs (diagnostic upper bound). No answers or accuracy used in any fit.
Usage: python -u step5_compensate.py [case ...]   (default: all cases)"""
import sys
import numpy as np
import torch as th
from decoders import wi, wh, wo, MEM

dev = "cuda"
th.set_default_dtype(th.float64)
OUT = [i for i in range(16) if i not in MEM]
N_FIT, N_EV = 2 ** 17, 2 ** 20
Xf = th.randn(N_FIT, 10, generator=th.Generator(device=dev).manual_seed(1), device=dev)
Xe = th.randn(N_EV, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Ye = Xe.argsort(-1)[:, -2]
P0 = [th.tensor(w, device=dev) for w in (wi, wh, wo)]


def run(P, X):
    """pre-activations [n,10,16] and states h_0..h_10 [n,11,16], and log-probs."""
    a, b, c = P
    h = th.zeros(len(X), 16, device=dev, dtype=X.dtype); hs, pres = [h], []
    for t in range(10):
        pre = h @ b.T + X[:, t, None] * a; h = th.relu(pre); pres.append(pre); hs.append(h)
    return th.stack(pres, 1), th.stack(hs, 1), th.log_softmax(h @ c.T, -1)


with th.no_grad():
    PRE_f, H_f, L_f = run(P0, Xf)
    _, _, L_e = run(P0, Xe)
    S_f = H_f[:, -1] @ P0[2].T; S_f = S_f - S_f.mean(1, keepdim=True)   # ARC's centred final scores


@th.no_grad()
def evaluate(P):
    l = run(P, Xe)[2]
    return (L_e.exp() * (L_e - l)).sum(-1).mean().item(), (l.argmax(-1) == Ye).double().mean().item()


# ---- simplified blocks
Wm = wh[np.ix_(MEM, MEM)]; ev, V = np.linalg.eig(Wm); Vi = np.linalg.inv(V)
ang, mod = np.angle(ev), abs(ev); rot = abs(np.sin(ang)) > 1e-6; sgn = np.sign(np.sin(ang))
mult = np.where(rot, np.round(abs(ang) / np.radians(42)), 0)           # 1, 2, 3 for the 38, 84, 127 pairs
s_best = (abs(ang[rot]) * mult[rot]).sum() / (mult[rot] ** 2).sum()     # least-squares single spacing


def block(e2):
    B = V @ np.diag(e2) @ Vi; assert abs(B.imag).max() < 1e-9; return B.real


CASES = {
    "snap42": mod * np.exp(1j * np.where(rot, sgn * mult * np.radians(42), ang)),
    "plus1deg": mod * np.exp(1j * (ang + sgn * np.radians(1) * rot)),
    "shrinkmean": np.where(rot, mod[rot].mean(), mod) * np.exp(1j * ang),
    "bestspacing": mod * np.exp(1j * np.where(rot, sgn * mult * s_best, ang)),
    "fastsnap": mod * np.exp(1j * np.where(rot & (mult > 1), sgn * mult * np.radians(42), ang)),
}


# ---- free-weight masks over (wi [16], wh [16,16], wo [10,16])
def masks(name):
    mi, mh, mo = np.zeros(16, bool), np.zeros((16, 16), bool), np.ones((10, 16), bool)
    if name in "bcd":
        mi[MEM] = True; mh[np.ix_(MEM, OUT)] = True
    if name in "cd":
        mh[np.ix_(OUT, MEM)] = True
    if name == "d":
        mi[:] = True; mh[:] = True; mh[np.ix_(MEM, MEM)] = False
    return mi, mh, mo


DAMP = 0.5


def states_fit(P, mk, iters=10, log=None):
    """Rollout-iterated least squares. Each neuron row with free entries: ARC's pre-activation minus the fixed part,
    regressed on the changed network's previous states / input in the free entries, on the steps where ARC's neuron
    is on (as the hiding-neuron rules did); damped (half old, half new) and iterated in rollout.
    Readout: ARC's centred final scores on the changed network's final state."""
    mi, mh, _ = mk
    P = [p.clone() for p in P]; hist = []
    rows = [j for j in range(16) if mi[j] or mh[j].any()]
    for it in range(iters if rows else 1):
        with th.no_grad():
            _, H, _ = run(P, Xf)
            Hp = H[:, :-1].reshape(-1, 16); x = Xf.reshape(-1)
            newb, newa = P[1].clone(), P[0].clone()
            for j in rows:
                fc = np.where(mh[j])[0]; fixed = ~th.tensor(mh[j], device=dev)
                on = PRE_f[..., j].reshape(-1) > 0          # only steps where ARC's neuron is on
                tgt = PRE_f[..., j].reshape(-1) - Hp[:, fixed] @ P[1][j, fixed]
                feats = [Hp[:, fc]]
                if mi[j]: feats.append(x[:, None])
                else: tgt = tgt - x * P[0][j]
                F = th.cat(feats, 1)
                sol = th.linalg.lstsq(F[on], tgt[on, None]).solution[:, 0]
                old = th.cat([P[1][j, fc]] + ([P[0][j:j + 1]] if mi[j] else []))
                sol = DAMP * sol + (1 - DAMP) * old                     # damped rollout iteration
                newb[j, fc] = sol[:len(fc)]
                if mi[j]: newa[j] = sol[-1]
            P[0], P[1] = newa, newb
            _, H, _ = run(P, Xf)
            hT = H[:, -1]; hc = hT  # scores centred: regress centred target on state, then centre rows
            sol = th.linalg.lstsq(hT, S_f).solution        # [16, 10]
            P[2] = (sol.T - sol.T.mean(0, keepdim=True)) + P0[2].mean(0, keepdim=True)
        kl, acc = evaluate(P); hist.append(kl)
        if log: log(f"      states fit iteration {it}: KL {kl:.4f} accuracy {acc:.4f}")
    return P, hist


def output_fit(P, mk, steps=3000, lr=3e-3, bs=2 ** 15):
    # float32 for speed (the GPU's float64 rate is low); the result is evaluated in float64
    mt = [th.tensor(m, device=dev, dtype=th.float32) for m in mk]
    P = [p.float() for p in P]; Xf32, Lf32 = Xf.float(), L_f.float()
    D = [th.zeros_like(p, requires_grad=True) for p in P]
    opt = th.optim.Adam(D, lr=lr); sch = th.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = th.Generator(device=dev).manual_seed(3)
    for _ in range(steps):
        idx = th.randint(0, N_FIT, (bs,), device=dev, generator=g)
        Q = [p + d * m for p, d, m in zip(P, D, mt)]
        l = run(Q, Xf32[idx])[2]; lb = Lf32[idx]
        loss = (lb.exp() * (lb - l)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with th.no_grad():
        return [(p + d * m).double() for p, d, m in zip(P, D, mt)]


def moves(P, mk):
    """Per free row: relative change and cosine to ARC's row (over the row's free entries)."""
    mi, mh, mo = mk; out = {}
    WI, WH, WO = (p.cpu().numpy() for p in P)
    WOc, wo_c = WO - WO.mean(0), wo - wo.mean(0)
    for j in range(16):
        if mh[j].any() or mi[j]:
            a = np.r_[wh[j][mh[j]], wi[j] if mi[j] else []]; b = np.r_[WH[j][mh[j]], WI[j] if mi[j] else []]
            out[f"in{j}"] = (np.linalg.norm(b - a) / np.linalg.norm(a), a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    for q in range(10):
        a, b = wo_c[q], WOc[q]
        out[f"out{q}"] = (np.linalg.norm(b - a) / np.linalg.norm(a), a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    return out


def readout_steps(WO):
    """Phase step per position of the centred readout rows in each memory mode (rows times right eigenvectors)."""
    R = (WO - WO.mean(0))[:, MEM] @ V
    res = {}
    for k in np.where(ang > 1e-6)[0]:
        ph = np.degrees(np.angle(R[:, k]))
        res[round(np.degrees(ang[k]))] = np.round((np.diff(ph) + 180) % 360 - 180).astype(int).tolist()
    return res


if __name__ == "__main__":
    print("ARC angles (deg):", np.round(np.degrees(ang), 2).tolist(), " shrink:", np.round(mod, 3).tolist())
    print(f"best single spacing (least squares, multiples 1,2,3): {np.degrees(s_best):.2f} deg")
    print("ARC readout phase steps per position by mode:", readout_steps(wo))
    todo = sys.argv[1:] or list(CASES)
    results = {}
    for case in todo:
        W = wh.copy(); W[np.ix_(MEM, MEM)] = block(CASES[case])
        Pc = [P0[0].clone(), th.tensor(W, device=dev), P0[2].clone()]
        kl0, acc0 = evaluate(Pc)
        print(f"\n=== {case}: angles {np.round(np.degrees(np.angle(CASES[case])), 2).tolist()}; "
              f"no compensation KL {kl0:.4f} accuracy {acc0:.4f}", flush=True)
        for s in "abcd":
            mk = masks(s); nfree = sum(int(m.sum()) for m in mk)
            Ps, hist = states_fit(Pc, mk)
            kls, accs = evaluate(Ps)
            Po = output_fit(Pc, mk)
            klo, acco = evaluate(Po)
            results[(case, s)] = (kls, klo)
            print(f"  set ({s}), {nfree} free weights: states fit KL {kls:.4f} (iterations {[round(h, 3) for h in hist]}), "
                  f"accuracy {accs:.4f};  output fit KL {klo:.4f} accuracy {acco:.4f}", flush=True)
            for tag, P in (("states", Ps), ("output", Po)):
                mv = moves(P, mk)
                ins = {k: v for k, v in mv.items() if k.startswith("in")}
                outs = {k: v for k, v in mv.items() if k.startswith("out")}
                fmt = lambda d: " ".join(f"{k}:{v[0]:.2f}/{v[1]:.3f}" for k, v in d.items())
                print(f"    {tag} fit moves (relative change / cosine) readout: {fmt(outs)}")
                if ins: print(f"    {tag} fit moves incoming rows: {fmt(ins)}")
                print(f"    {tag} fit readout phase steps by mode: {readout_steps(P[2].cpu().numpy())}", flush=True)
            np.savez(f"step5_weights/{case}_{s}.npz", **{f"{t}_{n}": p.cpu().numpy() for t, P in (("states", Ps), ("output", Po))
                                                    for n, p in zip(("wi", "wh", "wo"), P)})
    print("\nsummary: KL (states fit / output fit)")
    for case in todo:
        print(f"  {case:12s} " + "  ".join(f"({s}) {results[(case, s)][0]:.4f} / {results[(case, s)][1]:.4f}" for s in "abcd"))
