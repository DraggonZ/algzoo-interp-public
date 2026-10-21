"""What does partial switching do, step by step? Adapted from gao/partial_function.py. Intervention: on steps where
some but not all of the four hiding neurons would be off, the off ones keep their negative pre-activation (no ReLU for
them at that step). Done only at: the record step itself (K = 0: a record that leaves some of the four on), one step
after a record (K = 1), later steps (K >= 2), or all. Controls on the same steps: (a) the same change with a random
sign (the off ones are set to +e or -e, e = their clipped amount); (b) four random other neurons unclipped instead.
200,000 lists (seed 2626, as gao/partial_function.py). Accuracy by kind of list, and the median score change at the
switching number's own position and at the largest's position on lists where that step's intervention fired."""
import numpy as np
from common import WI, WH, WO, C, lists, run, kinds

X = lists(2626, 200_000); B = len(X); y = X.argsort(1)[:, -2]; o = np.argsort(-X, 1)
pm = np.maximum.accumulate(np.c_[np.zeros(B), X], 1)[:, :-1]; rec = X > pm      # records (floor 0, as the store)
last_rec = np.full((B, 10), -1); lr = np.full(B, -1)
for t in range(10):
    lr = np.where(rec[:, t], t, lr); last_rec[:, t] = lr
Kstep = np.arange(10)[None] - last_rec                                            # 0 at a record, -1 never
K = kinds(X)
# extra kinds: the answer is the number right after the largest; the third sits right after a record that is later overtaken
lar, sec, thi = o[:, 0], o[:, 1], o[:, 2]
K["answer right after the largest"] = sec == lar + 1
K["answer two or more after the largest"] = sec > lar + 1
rng = np.random.default_rng(0)
SIGN = rng.choice([-1.0, 1.0], size=(B, 10, 4))


def intervene(when, mode="undo", group=None):
    fired = np.zeros((B, 10), bool)
    def hook(t, pre, hn, hprev):
        off = (pre[:, C] <= 0).sum(1); part = (off > 0) & (off < 4)
        k = Kstep[:, t]
        sel = part & {"K0": k == 0, "K1": k == 1, "K2+": k >= 2, "all": np.ones(B, bool)}[when]
        fired[:, t] = sel
        hn = hn.copy()
        if mode == "undo":
            g = C if group is None else group
            hn[np.ix_(sel, g)] = pre[np.ix_(sel, g)]
        elif mode == "sign":
            e = np.maximum(-pre[:, C], 0)
            # undo would be h = -e; the random-sign control is h = s e (s = +-1), the same size of change
            hn[:, C] = np.where(sel[:, None] & (e > 0), SIGN[:, t] * e, hn[:, C])
        return hn
    L, _, _ = run(X, hook=hook); return L, fired


L0 = run(X)[0]; p0 = L0.argmax(1)
def line(lab, L):
    p = L.argmax(1)
    return f"{lab:34s} " + " | ".join(f"{k.split(' (')[0][:30]} {np.mean(p[m] == y[m]):.3f}" for k, m in K.items())
print(line("network", L0))
others = [n for n in range(16) if n not in C]
for when in ("K0", "K1", "K2+", "all"):
    L1, fired = intervene(when); print(line(f"undo clipping at {when}", L1), flush=True)
    Ls, _ = intervene(when, "sign"); print(line(f"   random-sign control at {when}", Ls), flush=True)
    accs = []
    for s in range(3):
        g = list(rng.choice(others, 4, replace=False)); Lr, _ = intervene(when, "undo", g); accs.append(np.mean(Lr.argmax(1) == y))
    print(f"   four random other neurons unclipped at {when}: accuracy {min(accs):.3f}-{max(accs):.3f}")
    # score effects on lists where it fired once
    any_f = fired.any(1); tf = np.argmax(fired, 1)
    d = L1 - L0; ii = np.where(any_f)[0]
    print(f"   fired on {any_f.mean():.2f} of lists; median score change at the switching number's position "
          f"{np.median(d[ii, tf[ii]]):+.2f}, at the largest's position {np.median(d[ii, lar[ii]]):+.2f}, at the answer "
          f"{np.median(d[ii, y[ii]]):+.2f}, at other positions {np.median(np.delete(d[ii], 0, 1)):+.2f}; "
          f"wrong answers after: largest {np.mean((L1.argmax(1) == lar)[ii]):.3f} (was {np.mean((p0 == lar)[ii]):.3f}), "
          f"third {np.mean((L1.argmax(1) == thi)[ii]):.3f} (was {np.mean((p0 == thi)[ii]):.3f}), switching number {np.mean((L1.argmax(1) == tf)[ii]):.3f} (was {np.mean((p0 == tf)[ii]):.3f})", flush=True)
