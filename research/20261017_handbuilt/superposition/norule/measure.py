"""Per network, per neuron: census tag (as ../generality/patterns_gen.py), block, and the measures for questions 2, 3, 5.
Q2: share of steps 1-9 with |pre-activation| under 10% of its root-mean-square at that step, and under 10% of the summed size of its terms; median |pre-activation| / summed size of its
terms (steps 3-9, as ../../bridge/n3_cancel.py). Q3: the rotating subspace is the span of the complex or negative-real
modes (angle >= 20 deg, size >= 0.5) of the recurrent map in the most common on/off pattern at steps 4-8 (as item 7's
timer analysis); each pre-activation is split into what it reads from that subspace (fixed spectral projector, applied at
every step), the rest of the state, and the input; shares are mean squares over steps 1-9. Positive carrier: least
squares of the pre-activation on the largest so far before the step and the current number; offset ratio = mean of the
largest-so-far part / standard deviation of what is left. Q5: accuracy with this neuron alone linear (no ReLU), 200,000
other lists. Usage: measure.py "<network name>" -> results/<name>.json"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_lib import *
from collections import Counter
name = sys.argv[1]; W = net_paths()[name](); wi, wh, wo = W; d = len(wi)
X = lists(); c = census(W, X); on = [r["on"] for r in c]
Z, H = run(W, X); lab = blocks(W, X, H, on)
m1p, m1 = running(X)
# Q2
Zs = Z[:, 1:]; rms = Zs.pow(2).mean((0, 1)).sqrt(); rms_t = Zs.pow(2).mean(0, keepdim=True).sqrt()
near = (Zs.abs() < 0.1 * rms_t).double().mean((0, 1))           # under 10% of its root-mean-square at that step
Hprev = H[:, :9]; Xc = X[:, 1:10]
canc, near_t = [], []
for n in range(d):
    terms = (Hprev * wh[n]).abs().sum(-1) + (wi[n] * Xc).abs(); r = Z[:, 1:, n].abs() / (terms + 1e-9)
    canc.append(r[:, 2:].median().item()); near_t.append((r < 0.1).double().mean().item())   # canc: steps 3-9
# Q3: dominant pattern of the state before steps 4-8
Pp = (H[:, 3:8] > 0).reshape(-1, d); w2 = 2 ** torch.arange(d, device=DEV, dtype=torch.float64)
code = Counter((Pp.double() * w2).sum(-1).cpu().numpy().tolist()).most_common(1)[0]
onset = np.array([(int(code[0]) >> k) & 1 for k in range(d)], bool); share_dom = code[1] / len(Pp)
O = np.where(onset)[0]; A = wh.cpu().numpy()[np.ix_(O, O)]; lam, V = np.linalg.eig(A); Vi = np.linalg.inv(V)
rot = (np.abs(np.angle(lam)) >= np.radians(20)) & (np.abs(lam) >= 0.5)
Prot_O = np.real(V[:, rot] @ Vi[rot, :]); Prot = np.zeros((d, d)); Prot[np.ix_(O, O)] = Prot_O
Prot_t = torch.tensor(Prot, device=DEV, dtype=dt)
Hb = H[:, :9]; Xn = X[:, 1:10]                       # state before steps 1..9, and the number at those steps
Rpart = (Hb @ Prot_t.T) @ wh.T; Tot = Hb @ wh.T; Npart = Tot - Rpart; Ipart = Xn[..., None] * wi
ms = lambda t: t.pow(2).mean((0, 1))
den = ms(Rpart) + ms(Npart) + ms(Ipart); rot_sh = ms(Rpart) / den; inp_sh = ms(Ipart) / den; cond = (den / ms(Z[:, 1:])).cpu().numpy()
A2 = torch.stack([m1p[:, 1:].reshape(-1), X[:, 1:].reshape(-1)], 1); pos = []
for n in range(d):
    y = Z[:, 1:, n].reshape(-1); cc = torch.linalg.solve(A2.T @ A2, A2.T @ y)
    p = cc[0] * A2[:, 0]; rest = y - p - cc[1] * A2[:, 1]
    pos.append(dict(coef=cc[0].item(), offset=(p.mean() / rest.std()).item(), pos_share=(p.pow(2).mean() / y.pow(2).mean()).item()))
# Q5
Xe = lists(200000, 5); acc = accuracy(W, Xe); lin = [accuracy(W, Xe, [n]) for n in range(d)]
neg = Zs.clamp(max=0).abs().mean((0, 1)) / rms                       # clipped amount per step, relative to rms
out = dict(name=name, d=d, acc=acc, dominant_share=share_dom, dominant_on=O.tolist(),
           modes=[[float(np.degrees(np.angle(l))), float(abs(l))] for l in lam if np.angle(l) >= -1e-9],
           neurons=[dict(j=n, **c[n], block=lab[n], near=near[n].item(), near_terms=near_t[n], canc=canc[n], rot=rot_sh[n].item(), inp=inp_sh[n].item(),
                         sumsq_over_z=float(cond[n]), **pos[n], lin_acc=lin[n], clip_mass=neg[n].item()) for n in range(d)])
os.makedirs("results", exist_ok=True); json.dump(out, open(f"results/{name.replace(' ', '_')}.json", "w"), indent=1)
print(f"{name}: {d} neurons, acc {100*acc:.2f}%, dominant pattern {100*share_dom:.1f}% of steps 4-8, modes " + " ".join(f"{a:.0f}:{s:.2f}" for a, s in out["modes"]))
for r in out["neurons"]:
    print(f"  {r['j']:2d} {r['block']:6s} on {100*r['on']:5.1f}% d2 {100*r['depth2']:5.1f}% {r['tag']:15s} near0 {100*r['near']:5.1f}%/{100*r['near_terms']:5.1f}% canc {r['canc']:.3f} rot {r['rot']:.2f} inp {r['inp']:.2f} "
          f"(sumsq/z {r['sumsq_over_z']:.1f}) pos {r['pos_share']:.2f} offset {r['offset']:+.2f} | linear {100*r['lin_acc']:.2f}% (cost {100*(acc-r['lin_acc']):.2f}) clip {r['clip_mass']:.3f}")
