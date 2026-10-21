"""Per-neuron switching rules for ARC's memory (0, 3, 10-15) and hiding neurons (1, 6, 7, 8): how well small decision
trees predict each neuron's on/off at every step (steps 1-9) from list features: step; steps since the largest
arrived (age); steps since the previous record before it; the current number x / largest so far; x / second-largest so
far; whether x is positive; whether x is a record; the previous number / largest. Accuracy on held-out lists, against
always guessing the commoner state; depth 1, 2, 3, 5. Then: eigenvalues (angle per step, size) of the memory block's
linear map in its top on/off patterns."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, numpy as np, torch
from sklearn.tree import DecisionTreeClassifier, export_text
sys.path.insert(0, "../../family")
from common import *
from patterns import run_patterns, codes
MEM = [0, 3, 10, 11, 12, 13, 14, 15]; HID = [1, 6, 7, 8]
NAMES = ["step", "age", "age of previous record", "x / largest", "x / second", "x > 0", "x is a record", "previous x / largest"]
def features(X):
    n = len(X); m1 = torch.zeros(n, device=DEV, dtype=dt); m2 = torch.zeros(n, device=DEV, dtype=dt)
    Lp = torch.full((n,), -1, device=DEV, dtype=torch.long); Pp = Lp.clone(); F = []; xprev = torch.zeros(n, device=DEV, dtype=dt)
    for t in range(10):
        x = X[:, t]; rec = x > m1
        f = [torch.full((n,), float(t), device=DEV, dtype=dt), torch.where(Lp >= 0, t - Lp, -1).to(dt), torch.where(Pp >= 0, t - Pp, -1).to(dt),
             x / (m1 + 1e-3), x / (m2 + 1e-3), (x > 0).to(dt), rec.to(dt), xprev / (m1 + 1e-3)]
        F.append(torch.stack(f, 1))
        Pp = torch.where(rec, Lp, Pp); m2 = torch.where(rec, m1, torch.maximum(m2, x)); Lp = torch.where(rec, t, Lp); m1 = torch.maximum(m1, x); xprev = x
    return torch.stack(F, 1)
if __name__ == "__main__":
    X = torch.randn(300000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(102))
    P = run_patterns(ARC, X)[:, 1:]; F = features(X)[:, 1:].clamp(-50, 50)
    Fn = F.reshape(-1, F.shape[2]).cpu().numpy(); half = len(Fn) // 2
    print(f"{'neuron':>6s} {'on %':>6s} {'guess':>6s} {'depth1':>7s} {'depth2':>7s} {'depth3':>7s} {'depth5':>7s} | depth-2 rule")
    for j in MEM + HID:
        y = P[..., j].reshape(-1).cpu().numpy(); base = max(y[half:].mean(), 1 - y[half:].mean()); accs = []; rule = ""
        for d in (1, 2, 3, 5):
            tr = DecisionTreeClassifier(max_depth=d, min_samples_leaf=200).fit(Fn[:half], y[:half]); accs.append((tr.predict(Fn[half:]) == y[half:]).mean())
            if d == 2: rule = export_text(tr, feature_names=NAMES).replace("\n", " ; ").replace("|---", "").replace("|   ", "")
        print(f"{j:6d} {100*y.mean():5.1f}% {100*base:5.1f}% " + " ".join(f"{100*a:6.1f}%" for a in accs) + " | " + " ".join(rule.split())[:260])
    # memory block eigenvalues in the top patterns
    wh = ARC[1].cpu().numpy(); c = codes(P, MEM).flatten().cpu().numpy(); vals, cnt = np.unique(c, return_counts=True); order = np.argsort(-cnt)
    print("\nmemory block, eigenvalues in its top patterns (angle in degrees per step : size):")
    for code in vals[order][:8]:
        on = np.array([(code >> k) & 1 for k in range(8)], bool); B = wh[np.ix_(MEM, MEM)] * on[:, None]
        ev = np.linalg.eigvals(B); ev = ev[np.abs(ev) > 1e-6]; ev = ev[np.argsort(-np.abs(ev))]
        seen = [e for e in ev if e.imag >= -1e-12]
        print(f"  {100*cnt[vals == code][0]/len(c):5.1f}% off {[MEM[k] for k in range(8) if not on[k]]}: " + "  ".join(f"{np.degrees(np.angle(e)):6.1f}:{abs(e):.3f}" for e in seen))
