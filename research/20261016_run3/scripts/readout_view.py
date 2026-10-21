"""What does the output layer read? Its weights per neuron, and how much of the wheel plane it can see."""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
np.set_printoptions(linewidth=220, precision=2, suppress=True)
print("output weights (rows = position 0..9, columns = neuron 0..15):")
print("        " + " ".join(f"{i:6d}" for i in range(16)))
for j in range(10):
    print(f"pos {j}: " + " ".join(f"{v:6.2f}" for v in w_oh[j]))
print("column norms (how strongly each neuron is read):", np.round(np.linalg.norm(w_oh, axis=0), 2))
# share of each neuron direction inside the readout's row space
Q, _ = np.linalg.qr(w_oh.T)          # 16 x 10 orthonormal basis of the row space
vis = np.linalg.norm(Q.T @ np.eye(16), axis=0) ** 2
print("share of each single-neuron direction visible to the readout:", np.round(vis, 2))
# wheel plane from the single-input response
h = np.maximum(0, w_hi); H = [h]
for k in range(1, 41):
    h = np.maximum(0, w_hh @ h); H.append(h)
H = np.array(H)[1:]
mu = H.mean(0)
_, _, Vt = np.linalg.svd(H - mu, full_matrices=False)
for c in range(3):
    print(f"wheel component {c+1}: visible share {np.linalg.norm(Q.T @ Vt[c])**2:.2f}")
print("center: visible share", round(float(np.linalg.norm(Q.T @ mu) ** 2 / mu @ mu), 2))
# centered readout: scores relative to their mean over positions
Wc = w_oh - w_oh.mean(0)
Qc, _ = np.linalg.qr(Wc.T)
for c in range(3):
    print(f"wheel component {c+1}: visible share to centered readout (differences between positions) {np.linalg.norm(Qc[:, :9].T @ Vt[c])**2:.2f}")
