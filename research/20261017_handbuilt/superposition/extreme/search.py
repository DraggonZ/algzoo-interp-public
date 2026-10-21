"""Most confidently wrong inputs. Among lists with clear gaps (second positive, largest >= 1.2 x second, second >= 1.2 x
third), the wrong margin = (best wrong score - second's score) / second's value. Random search over 8M lists, then the
top cases, then input-space gradient ascent from them under the same gap constraints (inputs only; weights untouched)."""
import sys; sys.path.insert(0, "../steer")
import numpy as np
from harness import *
def margin(S, X):
    o = X.argsort(1, descending=True); n = torch.arange(len(X), device=DEV); s = o[:, 1]
    ss = S[n, s]; Sw = S.clone(); Sw[n, s] = -1e9; best = Sw.max(1)
    return (best.values - ss) / X[n, s], best.indices, o
X = lists(8_000_000, seed=123); v = X.sort(1, descending=True).values
gap = (v[:, 1] > 0) & (v[:, 0] >= 1.2 * v[:, 1]) & (v[:, 1] >= 1.2 * v[:, 2]); X = X[gap]
m, w, o = margin(run(ARC, X), X)
print(f"{len(X):,} clear-gap lists; wrong on {100*(m > 0).double().mean().item():.2f}%; wrong margin quantiles 99/99.9/max of the wrong ones: {np.round(torch.quantile(m[m > 0], torch.tensor([0.5, 0.9, 0.99], device=DEV, dtype=dt)).cpu().numpy(), 3)}, max {m.max().item():.3f}")
top = m.argsort(descending=True)[:15]
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for i in top.tolist():
    x = X[i].cpu().numpy(); print(f"margin {m[i].item():6.3f} | largest at {o[i,0].item()}, second at {o[i,1].item()}, third at {o[i,2].item()}, names {w[i].item()} | largest/second {v[gap][i,0].item()/v[gap][i,1].item():5.2f} | {x}")
# kinds among wrong ones: what the wrong answer is
wr = m > 0; L, S_, T3 = o[:, 0], o[:, 1], o[:, 2]
kind = torch.where(w == L, 0, torch.where(w == T3, 1, 2))
print("wrong answers on clear-gap lists: the largest {:.0f}%, the third {:.0f}%, a smaller number {:.0f}%".format(*[100 * (kind[wr] == k).double().mean().item() for k in range(3)]))
