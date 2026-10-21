"""Where the best all-16 block account (block_full2.py: cancellers as clipped rotation, memory + keep gate on their own
weights, store/detector/late gate as formulas; 91%) disagrees with the network: by the network's own lead (top score minus
next), by what the account names instead (largest, third, other), and by the gap between the second and third numbers."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import block_full2 as bf
import simlib

X = bf.Xs; f = bf.plain(bf.G["store+detector"]); sims = {**bf.canc, **f, **bf.g9}
blk = bf.run_block(bf.mem + [5], {**bf.true, **sims})
L_net = simlib.run(X)[0]; L_acc = simlib.run(X, {**sims, **blk})[0]
a_net, a_acc = L_net.argmax(1), L_acc.argmax(1)
rank = X.argsort(1)[:, ::-1]; y = rank[:, 1]
print(f"network accuracy {np.mean(a_net == y):.3f}; account {np.mean(a_acc == y):.3f}; agreement {np.mean(a_net == a_acc):.3f}")
s = np.sort(L_net, 1); lead = s[:, -1] - s[:, -2]
q = np.quantile(lead, [0, .1, .25, .5, .75, 1])
print("disagreement by the network's lead (quantile bins):")
for lo, hi in zip(q[:-1], q[1:]):
    m = (lead >= lo) & (lead <= hi); print(f"  lead {lo:6.2f}-{hi:6.2f}: disagree {np.mean(a_net[m] != a_acc[m]):.3f}")
dis = a_net != a_acc
named = np.select([a_acc == rank[:, 0], a_acc == rank[:, 1], a_acc == rank[:, 2]], ["largest", "second", "third"], "other")
print("when they disagree, the account names:", {k: round(float(np.mean(named[dis] == k)), 3) for k in ("largest", "second", "third", "other")})
nn = np.select([a_net == rank[:, 0], a_net == rank[:, 1], a_net == rank[:, 2]], ["largest", "second", "third"], "other")
print("when they disagree, the network names:", {k: round(float(np.mean(nn[dis] == k)), 3) for k in ("largest", "second", "third", "other")})
x2, x3 = np.take_along_axis(X, rank[:, 1:2], 1)[:, 0], np.take_along_axis(X, rank[:, 2:3], 1)[:, 0]
gap = (x2 - x3); qg = np.quantile(gap, [0, .1, .25, .5, 1])
print("disagreement by the gap between second and third numbers:")
for lo, hi in zip(qg[:-1], qg[1:]):
    m = (gap >= lo) & (gap <= hi); print(f"  gap {lo:5.2f}-{hi:5.2f}: disagree {np.mean(dis[m]):.3f}")
# account names the largest: how long ago was the largest overtaken, or is it last?
pos1, pos2 = rank[:, 0], rank[:, 1]
m = dis & (named == "largest")
print(f"account names the largest ({m.mean():.3f} of lists): largest after second {np.mean(pos1[m] > pos2[m]):.2f} (all lists {np.mean(pos1 > pos2):.2f}); largest at position 9 {np.mean(pos1[m] == 9):.2f} (all {np.mean(pos1 == 9):.2f})")
m = dis & (named == "third")
print(f"account names the third ({m.mean():.3f} of lists): median gap {np.median(gap[m]):.2f} (all {np.median(gap):.2f})")
