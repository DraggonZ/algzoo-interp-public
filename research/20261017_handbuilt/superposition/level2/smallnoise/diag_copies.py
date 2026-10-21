"""Why is v2 high on the noisy copies? For the copies with the largest v2 errors: truth by region (third > 0 / < 0) and
mistake kind on 2M sampled lists against v2's and the first version's breakdown. Sampling for diagnosis only."""
import numpy as np, torch
from lib import *
import estimate as E
q = np.loadtxt("across_v2_noisy.txt"); t = np.load("../../../hilton/hilton_noise.npz")["true"]; e = 100 * (q[:, 1] - t[q[:, 0].astype(int)])
print(f"v2 error vs true accuracy across the 200 copies: correlation {np.corrcoef(e, t[q[:,0].astype(int)])[0,1]:.2f}; "
      f"mean error for copies above 92% {e[t[q[:,0].astype(int)]>0.92].mean():+.2f}, below 88% {e[t[q[:,0].astype(int)]<0.88].mean():+.2f}")
C = noisy_copies(); g = torch.Generator(device=DEV).manual_seed(6)
for i in [int(q[j, 0]) for j in np.argsort(-e)[:3]]:
    W = C[i]; c = np.zeros((2, 5))
    X = torch.randn(2_000_000, 10, device=DEV, dtype=dt, generator=g); a = run(W, X).argmax(1); o = X.argsort(1, descending=True)[:, :3]
    neg = (X.gather(1, o)[:, 2] < 0).long()
    for s in range(2):
        m = neg == s; c[s] = [m.sum().item(), (m & (a == o[:, 1])).sum().item(), (m & (a == o[:, 0])).sum().item(), (m & (a == o[:, 2])).sum().item(), (m & ~((a == o[:, 0]) | (a == o[:, 1]) | (a == o[:, 2]))).sum().item()]
    print(f"copy {i}: truth {100*c[:,1].sum()/c[:,0].sum():.2f}%")
    for model in ["v2", "final"]:
        r = E.estimate(W, 16, 16, model, 4, detail=True, kind="split3-gl")
        print(f"  {model:5s} {100*r['acc']:.2f}%: " + " | ".join(f"{lab}: est {100*r['sign_acc'][s]:.2f}% (wrong {'/'.join(f'{100*x:.2f}' for x in r['sign_split'][s])}) truth {100*c[s,1]/c[s,0]:.2f}% (wrong {100*c[s,2]/c[s,0]:.2f}/{100*c[s,3]/c[s,0]:.2f}/{100*c[s,4]/c[s,0]:.2f})" for s, lab in enumerate(["third>0", "third<0"])))
