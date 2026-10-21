"""Export ARC's weights and a clock plane for the browser model (viz/model.json).
Clock plane: the lone-input response (+1 then zeros) of the ten neurons outside the
comparing group, ages 1-40; its top two directions (SVD)."""
import json

import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
ten = [0, 3, 5, 9, 10, 11, 12, 13, 14, 15]
h = np.maximum(0, w_hi); H = []
for _ in range(40):
    h = np.maximum(0, w_hh @ h); H.append(h)
H = np.array(H)[:, ten]
mu = H.mean(0)
U, S, Vt = np.linalg.svd(H - mu, full_matrices=False)
P = (H - mu) @ Vt[:2].T
print("share in top two directions:", ((S[:2] ** 2).sum() / (S ** 2).sum()).round(3))
print("turn per step (deg, ages 1-10):", np.round(np.degrees(np.diff(np.unwrap(np.arctan2(P[:10, 1], P[:10, 0])))), 0))
out = {"w_hi": w_hi.round(6).tolist(), "w_hh": w_hh.round(6).tolist(), "w_oh": w_oh.round(6).tolist(),
       "clock_neurons": ten, "clock_mean": mu.round(6).tolist(), "clock_axes": Vt[:2].round(6).tolist()}
json.dump(out, open("viz/model.json", "w"))
print("wrote viz/model.json")

# dial (v4): the twelve neurons outside 1, 6, 7, 8; plane from a lone number's trajectory (ages 1-9);
# spokes = each position's output weights on those neurons (minus the average over positions), projected.
R = [k for k in range(16) if k not in (1, 6, 7, 8)]
h = np.maximum(0, w_hi); T = []
for _ in range(9):
    h = np.maximum(0, w_hh @ h); T.append(h[R])
T = np.array(T); mu12 = T.mean(0)
_, S12, V12 = np.linalg.svd(T - mu12, full_matrices=False)
ax12 = V12[:2]
P12 = (T - mu12) @ ax12.T
if np.degrees(np.diff(np.unwrap(np.arctan2(P12[:, 1], P12[:, 0])))).mean() > 0:
    ax12 = ax12[[1, 0]]  # orient so the hand turns clockwise on screen
Wc = w_oh[:, R] - w_oh[:, R].mean(0)
spokes = (Wc @ ax12.T)
out = json.load(open("viz/model.json"))
out.update({"dial_neurons": R, "dial_mean": mu12.round(6).tolist(), "dial_axes": ax12.round(6).tolist(),
            "spokes": spokes.round(6).tolist(), "spoke_share": (np.linalg.norm(spokes, axis=1) / np.linalg.norm(Wc, axis=1)).round(3).tolist()})
json.dump(out, open("viz/model.json", "w"))
print("dial exported; spoke share in plane:", np.round(out["spoke_share"], 2))
