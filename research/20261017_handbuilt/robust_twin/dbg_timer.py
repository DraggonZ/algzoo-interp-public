import numpy as np
exec(open("v21_graded.py").read().split("H = simlib.run")[0])
H = simlib.run(X, params=P)[2]; ex = timer()
for n in ex:
    dif = np.abs(ex[n] - H[:, :, n]); b, t = np.unravel_index(dif.argmax(), dif.shape)
    print(NAMES[n], f"max diff {dif.max():.3f} at list {b} step {t}; share of entries off by >1e-6: {np.mean(dif > 1e-6):.4f}")
b, t = np.unravel_index(np.abs(ex[I['F']] - H[:, :, I['F']]).argmax(), (B, 10))
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(X[b]); print("rec", rec[b].astype(int)); print("net F", H[b, :, I['F']]); print("rule F", ex[I['F']][b])
print("net C0", H[b, :, I['C0']]); print("rule C0", ex[I['C0']][b])
