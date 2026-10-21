"""Question: what are the memory's modes (rotation per step, shrink per step), which neurons hold each, and how does the
readout read each mode position by position?

The memory block A = W_hh[memory, memory] (neurons 0, 3, 10-15, all on). Modes: eigenvectors of A. For each mode j the
readout of position k is c_kj = W_oh[k, memory] v_j (v_j the right eigenvector). If the readout follows the dynamics
(W_oh[k+1] = W_oh[k] A, the shift test), then c_{k+1,j} = lambda_j c_kj: the phase advances by the mode's angle and
the size grows by 1/shrink... (it equals c_kj * lambda_j). We print the measured per-position step of each mode's
readout (ratio c_{k+1}/c_k as angle and size) against lambda_j.
Also with neuron 0 off, neuron 15 off, both off (the block with that row zeroed)."""
import numpy as np
from memlib import A, RM, MEM

np.set_printoptions(precision=2, suppress=True, linewidth=160)


def modes(M):
    lam, V = np.linalg.eig(M)
    order = np.lexsort((-np.abs(lam), np.abs(np.angle(lam))))
    return lam[order], V[:, order]


def describe(M, title):
    lam, V = modes(M); print(f"== {title}")
    for j, l in enumerate(lam):
        if np.angle(l) < -1e-9: continue
        v = V[:, j] / np.abs(V[:, j]).max()
        print(f"  mode angle {np.degrees(np.angle(l)):7.1f} deg  shrink {abs(l):.3f}  neurons |v|: " +
              " ".join(f"{n}:{abs(x):.2f}" for n, x in zip(MEM, v)))
    return lam, V


print("memory block A = W_hh[memory, memory], rows = to, columns = from, order", MEM)
for i, n in enumerate(MEM): print(f"  {n:2d}", A[i])
lam, V = describe(A, "memory block, all on")
print("\nshift test on the memory readout alone: cosine of W_oh[k+1, mem] with W_oh[k, mem] A, k = 0..8")
print("  ", np.array([np.dot(RM[k + 1], RM[k] @ A) / np.linalg.norm(RM[k + 1]) / np.linalg.norm(RM[k] @ A) for k in range(9)]))
print("  plain neighbours:", np.array([np.dot(RM[k + 1], RM[k]) / np.linalg.norm(RM[k + 1]) / np.linalg.norm(RM[k]) for k in range(9)]))
print("  norms of W_oh[k, mem]:", np.linalg.norm(RM, axis=1))

C = RM @ V   # 10 x 8 complex: readout of each mode at each position
print("\nreadout of each mode by position: size |c_k| and phase (deg); then the measured step c_{k+1}/c_k (angle, size) vs the mode's own")
for j, l in enumerate(lam):
    if np.angle(l) < -1e-9: continue
    c = C[:, j]; s = c / np.abs(C).max()
    ratio = c[1:] / c[:-1]
    print(f"  mode {np.degrees(np.angle(l)):6.1f} deg (shrink {abs(l):.2f}):")
    print("     size  ", " ".join(f"{abs(x):5.2f}" for x in s))
    print("     phase ", " ".join(f"{np.degrees(np.angle(x)):5.0f}" for x in c))
    print("     step angle", " ".join(f"{np.degrees(np.angle(x)):5.0f}" for x in ratio), f"| own {np.degrees(np.angle(l)):5.0f}")
    print("     step size ", " ".join(f"{abs(x):5.2f}" for x in ratio), f"| own {abs(l):5.2f}")

for off, title in (([0], "neuron 0 off"), ([7], "neuron 15 off"), ([0, 7], "neurons 0 and 15 off"), ([1], "neuron 3 off")):
    D = np.ones(8); D[off] = 0; print(); describe(np.diag(D) @ A, title)
