"""Development look: (1) how many triples hold half of the converged reduction's error (ref_triples_*.npz); (2) does
the loser of a tie stay the same when only the third number's position moves? At the tie nodes of cheap_est.py, for
each tied pair, the share of the 8 positions of the third number for which the same ordering loses; (3) on seed 208,
the share of wide bands (exact node width >= 0.3, exact_node_208.npz) whose largest number is last."""
import numpy as np, torch
import common as c, cheap_est as ce
log = c.Log("concentration.log")
T = c.T.cpu().numpy()
for name in ["ARC"] + [str(s) for s in c.SEEDS_DEV]:
    e = np.load(f"ref_triples_{name}.npz")["err"].sum(1); s = np.sort(e)[::-1]; cs = np.cumsum(s) / s.sum()
    log(f"{name:>4s}: triples holding half the error: {(cs < .5).sum() + 1} of 720")
W = c.nets()[0][1]
for tie, nd in (("23", ce.N23), ("12", ce.N12)):
    S, _ = c.scores(W, ce._X(c.T, nd)); D, _, _ = c.margin_from(S, c.T[:, 1]); los = (D < 0).cpu().numpy()
    agree = []; flips = 0
    for q in range(10):
        for r in range(q + 1, 10):
            # orderings where q is the true larger of the tied pair
            sel = [i for i, (p1, p2, p3) in enumerate(T.tolist()) if ((tie == "23" and (p2, p3) == (q, r)) or (tie == "12" and (p1, p2) == (q, r)))]
            v = los[sel]; agree.append(max(v.mean(), 1 - v.mean()))
    agree = np.array(agree)
    log(f"ARC tie {tie}: share of the third number's 8 positions with the majority loser, over the 45 tied pairs: median {np.median(agree):.2f}, "
        f"pairs where the loser never changes {(agree == 1).mean():.2f}")
ex = np.load("exact_node_208.npz")
for k, nm, col in (("A", "2nd-3rd tie", 0), ("B", "1st-2nd tie", 0)):
    wide = ex[k] >= 0.3
    log(f"208 {nm}: {wide.sum()} wide bands, of which largest number last (p1 = 9): {(T[wide, 0] == 9).sum()}")
