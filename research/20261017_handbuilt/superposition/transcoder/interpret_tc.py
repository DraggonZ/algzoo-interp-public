"""Latent interpretability. On 40,000 random lists (all ten steps; ARC's real states as input), for each live latent:
activation frequency, the simple list feature whose |correlation| with the latent's activation is largest, and a
depth-2 regression tree on all features (fit on half the lists, R^2 on the other half). Features at step t (after
reading x_t): the current number; the step; a new largest arrives; the current number becomes the new second-largest;
the running largest, second and third; steps since the largest and since the second-largest arrived; current number
minus the previous largest / previous second; the current number above c x the previous second-largest
(c = 0.5, 1, 1.5, 2; only when that second is positive); the numbers at lags 1-3; largest minus second. Same analysis
on ARC's own 16 neurons as a reference. Also: a depth-2 classification tree for when the unit fires (balanced
accuracy, held out) and the feature best correlated with its size on the steps where it fires. Clearest latents = highest tree R^2 among those active on at least 2% of steps."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
from train_tc import pairs, states
from eval_tc import load, sd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
def features(X):
    n = len(X); NEG = -1e9; m1 = torch.full((n,), NEG, device=DEV, dtype=dt); m2 = m1.clone(); m3 = m1.clone()
    a1 = torch.zeros(n, device=DEV, dtype=dt); a2 = a1.clone(); F = []
    for t in range(10):
        x = X[:, t]; pm1, pm2 = m1.clone(), m2.clone()
        newL = x > m1; newS = (~newL) & (x > m2)
        m3 = torch.where(newL | newS, m2, torch.maximum(m3, x)); m2 = torch.where(newL, m1, torch.where(newS, x, m2)); m1 = torch.maximum(m1, x)
        a2 = torch.where(newL, a1, torch.where(newS, torch.full_like(a2, t), a2)); a1 = torch.where(newL, torch.full_like(a1, t), a1)
        c = lambda v: torch.where(v < -1e8, torch.zeros_like(v), v)
        lag = [X[:, t - j] if t >= j else torch.zeros(n, device=DEV, dtype=dt) for j in (1, 2, 3)]
        pos2 = pm2 > 0
        row = [x, torch.full_like(x, t), newL.double(), newS.double(), c(m1), c(m2), c(m3), t - a1, torch.where(m2 > -1e8, t - a2, torch.full_like(x, -1)),
               torch.where(pm1 > -1e8, x - pm1, torch.zeros_like(x)), torch.where(pm2 > -1e8, x - pm2, torch.zeros_like(x))] + \
              [(pos2 & (x > cc * pm2)).double() for cc in (0.5, 1.0, 1.5, 2.0)] + lag + [c(m1) - c(m2)]
        F.append(torch.stack(row, 1))
    return torch.stack(F, 1).reshape(-1, len(NAMES))
NAMES = ["current number", "step", "new largest arrives", "new second-largest arrives", "largest so far", "second-largest so far",
         "third-largest so far", "steps since largest", "steps since second-largest", "current minus previous largest",
         "current minus previous second", "current > 0.5 x previous second", "current > previous second", "current > 1.5 x previous second",
         "current > 2 x previous second", "number 1 step back", "number 2 steps back", "number 3 steps back", "largest minus second"]
def analyse(nm, Z, Fn, n_lists, show=10):
    half = n_lists // 2 * 10; Zc = Z.cpu().numpy(); out = []
    Fs = (Fn - Fn.mean(0)) / (Fn.std(0) + 1e-12)
    for i in range(Z.shape[1]):
        z = Z[:, i]; freq = (z > 0).double().mean().item()
        if freq == 0: continue
        zs = (z - z.mean()) / (z.std() + 1e-12); C = (Fs * zs[:, None]).mean(0).abs(); j = C.argmax().item()
        tr = DecisionTreeRegressor(max_depth=2, min_samples_leaf=200).fit(Fc[:half], Zc[:half, i]); p = tr.predict(Fc[half:])
        r2 = 1 - ((p - Zc[half:, i]) ** 2).sum() / ((Zc[half:, i] - Zc[half:, i].mean()) ** 2).sum()
        r2 = r2 if np.isfinite(r2) else np.nan
        on = (Zc[:, i] > 0); ct = DecisionTreeClassifier(max_depth=2, min_samples_leaf=200, class_weight="balanced").fit(Fc[:half], on[:half])
        pr = ct.predict(Fc[half:]); bal = 0.5 * (pr[on[half:]].mean() + (~pr[~on[half:]]).mean()) if 0 < on[half:].mean() < 1 else np.nan
        A = Fs[z > 0]; za = z[z > 0]; zz = (za - za.mean()) / (za.std() + 1e-12); A = (A - A.mean(0)) / (A.std(0) + 1e-12)
        Ca = (A * zz[:, None]).mean(0).abs().nan_to_num(); ja = Ca.argmax().item() if len(za) > 50 else 0
        out.append((i, freq, NAMES[j], C[j].item(), r2, tr, bal, ct, NAMES[ja], Ca[ja].item() if len(za) > 50 else np.nan))
    r2s = np.array([o[4] for o in out]); cs = np.array([o[3] for o in out]); bs = np.array([o[6] for o in out])
    print(f"\n== {nm}: {len(out)} live | best single-feature |corr| mean {cs.mean():.2f} (above 0.8: {(cs > 0.8).sum()}) | "
          f"depth-2 tree R2 mean {np.nanmean(r2s):.2f} (above 0.8: {(r2s > 0.8).sum()}, above 0.9: {(r2s > 0.9).sum()}) | "
          f"when it fires, depth-2 rule balanced accuracy mean {np.nanmean(bs):.2f} (above 0.95: {(bs > 0.95).sum()})", flush=True)
    for i, freq, fn, cc, r2, tr, bal, ct, fa, ca in out:
        print(f"  unit {i:3d}: active {100*freq:5.1f}% | best feature {fn} (|corr| {cc:.2f}) | tree R2 {r2:.2f} | firing rule bal. acc. {bal:.2f} | size when firing: {fa} (|corr| {ca:.2f})")
    print(f"  -- clearest {show} (tree R2, active >= 2%):")
    for i, freq, fn, cc, r2, tr, bal, ct, fa, ca in sorted([o for o in out if o[1] >= 0.02], key=lambda o: -np.nan_to_num(o[4]))[:show]:
        print(f"  unit {i} (active {100*freq:.0f}%, tree R2 {r2:.2f}, best feature {fn} {cc:.2f}; firing rule bal. acc. {bal:.2f}; size when firing: {fa} {ca:.2f}):")
        print("    activation tree:\n    " + export_text(tr, feature_names=NAMES, decimals=2).replace("\n", "\n    "))
        print("    firing rule:\n    " + export_text(ct, feature_names=NAMES, decimals=2, show_weights=False).replace("\n", "\n    "))
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(79); n = 40000; X = torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
    Fn = features(X); Fc = Fn.cpu().numpy()
    U, _ = pairs(X, sd); U = U.reshape(-1, 17)
    analyse("ARC's 16 neurons (reference)", states(X).reshape(-1, 16), Fn, n, show=5)
    for m, k in [(32, 4), (32, 8), (128, 8)]:
        with torch.no_grad(): Z = load(m, k).encode(U).to(dt)
        analyse(f"transcoder {m}x k{k}", Z, Fn, n)
