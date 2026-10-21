"""Noisy copies of ARC's network (every weight + N(0, s^2)), predicted by the propagation, scored on the shared truth
(../truth/noisy_truth.npz: 100 copies per s, accuracy and mistake shares on 2M lists each).
 (a) one prediction per s, Hilton's way: ARC's weights propagated with the added noise treated as zero-mean noise of
     variance s^2 per weight (gap as noise; fresh each step, not the same draw at every step);
 (b) per copy: the propagation run on that copy's own weights (no lists sampled), against its true accuracy.
Usage: run_noise.py G ncopies s1 [s2 ...]"""
import sys, time, numpy as np
from algzoo.core import example
from prop4 import Prop
G = int(sys.argv[1]); ncop = int(sys.argv[2]); S = [float(a) for a in sys.argv[3:]]
tr = np.load("/data/output/research/algzoo/20261017_handbuilt/superposition/level2/truth/noisy_truth.npz")
W0 = [np.asarray(w, float) for w in example()]
unflat = lambda v: [v[:16], v[16:272].reshape(16, 16), v[272:].reshape(10, 16)]
P = Prop(W0, G=G, lookahead="auto", chunk=2_000_000); a0 = P.run()
print(f"G {G}. ARC: predicted {100*a0:.2f}% (truth on the shared lists {100*tr['arc'][0]:.2f}%), mistakes naming the largest {100*P.share_L:.1f}% (true {100*tr['arc'][1]:.1f}%)", flush=True)
out = {}
for s in S:
    st = tr[f"stats_{s}"]; t0 = time.time()
    P = Prop(W0, G=G, var_dw=s ** 2, lookahead="auto", chunk=2_000_000); a = P.run()
    print(f"s {s}: (a) gap-as-noise prediction {100*a:.2f}% (mean of 100 copies {100*st[:,0].mean():.2f}%), predicted drop {100*(a0-a):.2f} vs true {100*(tr['arc'][0]-st[:,0].mean()):.2f}; "
          f"mistakes naming the largest {100*P.share_L:.1f}% (copies' mean {100*st[:,1].mean():.1f}%)  {time.time()-t0:.0f}s", flush=True)
    if ncop == 0: continue
    est = []; shL = []; t0 = time.time()
    for v in tr[f"dw_{s}"][:ncop]:
        P = Prop([w + d for w, d in zip(W0, unflat(v))], G=G, lookahead="auto", chunk=2_000_000); est.append(P.run()); shL.append(P.share_L)
    est = np.array(est); shL = np.array(shL); tt = st[:ncop, 0]; e = est - tt
    print(f"   (b) per copy ({ncop} copies): mean predicted {100*est.mean():.2f}% vs true {100*tt.mean():.2f}%; RMSE {100*np.sqrt((e**2).mean()):.2f} points, "
          f"bias {100*e.mean():+.2f}, correlation {np.corrcoef(est, tt)[0,1]:.3f} (spread of true copies: SD {100*tt.std():.2f}); "
          f"mistakes naming the largest: predicted mean {100*shL.mean():.1f}% vs true {100*st[:ncop,1].mean():.1f}%, correlation {np.corrcoef(shL, st[:ncop,1])[0,1]:.2f}  {time.time()-t0:.0f}s", flush=True)
    out[s] = np.c_[est, shL, st[:ncop]]
np.save(f"run_noise_G{G}.npy", out, allow_pickle=True)
