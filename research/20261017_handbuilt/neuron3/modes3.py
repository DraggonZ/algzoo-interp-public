"""Neuron 3 inside the memory's modes (all-on memory block A). (a) neuron 3's share of each mode's eigenvector
(fraction of the mode's squared size in neuron 3); (b) neuron 3's final value split by mode family, variance shares;
(c) each mode family's final readout, and the part of it that passes through neuron 3's coordinate."""
from n3lib import *

ev, Vc = np.linalg.eig(A); ang = np.degrees(np.abs(np.angle(ev)))
fam = np.round(ang).astype(int); fams = sorted(set(fam))
print("modes (angle, |eigenvalue|):", [(int(a), round(abs(e), 3)) for a, e in zip(fam, ev)])
i3 = MEM.index(3)
for f in fams:
    cols = Vc[:, fam == f]; P_ = cols @ np.linalg.pinv(Vc)[fam == f]           # projector on the family
    w = (np.abs(cols) ** 2).sum(1); w /= w.sum()
    print(f"family {f:3d}°: share of its vector in each memory neuron (0,3,10..15):", np.round(w, 2))
X = lists(9, 200_000); S, P, H, JP, JH = jac(X); m = H[:, 9][:, MEM]
Vi = np.linalg.pinv(Vc); parts = {}
for f in fams:
    Pf = np.real(Vc[:, fam == f] @ Vi[fam == f]); parts[f] = m @ Pf.T
v = m[:, i3]
print("\nneuron 3's final value split by mode family: mean, and covariance share of its variance")
for f in fams: print(f"   {f:3d}°: mean {parts[f][:, i3].mean():6.2f}  share {np.cov(parts[f][:, i3], v)[0, 1] / v.var():5.2f}")
print("\nreadout of each family (std over lists, per position 0-9), and the part through neuron 3")
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for f in fams:
    full = parts[f] @ RM.T; via3 = np.outer(parts[f][:, i3], RM[:, i3])
    print(f"   {f:3d}° all  :", full.std(0)); print(f"   {f:3d}° via 3:", via3.std(0))
