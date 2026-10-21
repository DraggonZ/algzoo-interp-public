"""Sampling-free estimate, version 2: the top three numbers exact, the other seven at their mean (normal truncated
below the third), plus a straight-line response of the scores to each of the seven around that path (the network's
own on/off pattern on the mean path, so cancellations stay exact). The leads are then normal with covariance
v_b * sum over background inputs of g g^T."""
import numpy as np, torch
import adf
DEV = "cuda"


def scores_and_jac(W, pos, val, bg="mean"):
    wi, wh, wo = W; n = pos.shape[0]; d = wi.shape[0]
    bmu, bvar = adf.background(val[:, 2])
    X = bmu[:, None].repeat(1, 10)
    for k in range(3): X[torch.arange(n), pos[:, k]] = val[:, k]
    isbg = torch.ones(n, 10, dtype=torch.bool, device=DEV)
    for k in range(3): isbg[torch.arange(n), pos[:, k]] = False
    h = torch.zeros(n, d, device=DEV, dtype=val.dtype); J = torch.zeros(n, 10, d, device=DEV, dtype=val.dtype)  # dh/dx_s
    for t in range(10):
        a = h @ wh.T + X[:, t:t + 1] * wi
        on = (a > 0).to(val.dtype)
        J = J @ wh.T
        J[:, t, :] += wi
        J = J * on[:, None, :]; h = a * on
    S = h @ wo.T; G = J @ wo.T                                                 # [n,10], [n,10 inputs,10 scores]
    G = G * isbg[:, :, None]
    return S, G, bvar


def estimate(W, pos, val, spread=True):
    S, G, bvar = scores_and_jac(W, pos, val)
    sc = bvar[:, None, None] * (G.transpose(1, 2) @ G) if spread else torch.zeros(S.shape[0], 10, 10, device=DEV, dtype=S.dtype)
    return adf.orthant(S, sc, pos[:, 1])[0]


if __name__ == "__main__":
    W = adf.net(None); z = np.load("check_conditional.npz")
    pos = torch.tensor(z["order"], device=DEV); val = torch.tensor(z["vals"], device=DEV); truth = z["truth"]
    for name, sp in [("mean path only", False), ("mean path + straight-line spread", True)]:
        v = estimate(W, pos, val, sp).cpu().numpy()
        print(f"{name:34s} mean {v.mean():.4f} (truth {truth.mean():.4f})  mean abs error {np.abs(v-truth).mean():.4f}  corr {np.corrcoef(v,truth)[0,1]:.3f}")
        if sp:
            bad = np.argsort(-np.abs(v - truth))[:6]
            for i in bad: print("   worst:", z["order"][i], z["vals"][i].round(2), "truth", truth[i].round(3), "est", v[i].round(3))
