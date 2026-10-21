"""Cross-network check of the gate findings, on every trained network with a gate-like neuron.
Detection (one step after x_t, t = 1..8, 100,000 lists, seed 22):
  late gate: on after >= 90% of records and <= 5% of other numbers;
  keep gate: on after >= 90% of records, after >= 85% of non-records at or above the second-largest so far, and after
             <= 15% of positive non-records below half of it (looser than g17_across.py).
For each gate: (a) substitution by the rule family of gaterules_any.py (fitted on seed 22, scored on 50,000 lists,
seed 23); (b) decision only: on/off from a simple rule (keep gates: record or x_t >= c x second-largest, c = its 50%
point; late gates: x_t is a record), value from the network's own weights; (c) its own value scaled by 1.05."""
import numpy as np
from gatelib import lists, run, task_vars, acc_by_kind, other_models, WI, WH, WO
from gaterules_any import GateRules

Xf = lists(22, 100_000); X = lists(23, 50_000); Vf = task_vars(Xf); V = task_vars(X)
models = {"ARC": (WI, WH, WO)}; models.update(other_models())
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "L at 8"]
newf = Vf["new"][:, 1:9].astype(bool); xf = Xf[:, 1:9]; m2f = Vf["m2"][:, 1:9]
rec = np.zeros((len(X), 10), bool); rec[:, 1:] = V["new"][:, :9].astype(bool)
m2 = np.zeros((len(X), 10)); m2[:, 1:] = V["m2"][:, :9]
xs = np.zeros((len(X), 10)); xs[:, 1:] = X[:, :9]
for name, params in models.items():
    wi, wh, wo = params; wi = wi.reshape(-1); n = len(wi)
    _, Pf, Hf = run(Xf, params=params)
    on = Hf[:, 2:10, :] > 0
    rec_on = on[newf].mean(0); oth_on = on[~newf].mean(0)
    above = ~newf & (m2f > 0.2) & (xf >= m2f); below = ~newf & (m2f > 0.2) & (xf > 0) & (xf < 0.5 * m2f)
    late = [g for g in range(n) if rec_on[g] >= 0.9 and oth_on[g] <= 0.05]
    keepg = [g for g in range(n) if g not in late and rec_on[g] >= 0.9 and on[above][:, g].mean() >= 0.85 and on[below][:, g].mean() <= 0.15]
    if not late and not keepg:
        print(f"{name}: no gate found"); continue
    L0, P0, H0 = run(X, params=params); base = acc_by_kind(X, L0, KEYS)
    print(f"\n{name}: intact " + " | ".join(f"{k} {100 * base[k][0]:.1f}" for k in KEYS))
    gates = late + keepg
    GR = GateRules(params, gates).fit(Xf); VR = GR.values(X)
    for g in gates:
        kind = "late" if g in late else "keep"
        msg = f"  {kind} gate {g} (large-input neurons {GR.big}):"
        Lr = run(X, params=params, sub={g: VR[g]})[0]; r = acc_by_kind(X, Lr, KEYS)
        print(msg + "\n    rule substituted:   " + " | ".join(f"{k} {100 * r[k][0]:.1f}" for k in KEYS) +
              f"  (rule vs network on the intact run: rms {np.sqrt(((VR[g][:, 1:] - H0[:, 1:, g]) ** 2).mean()):.3f}, typical on-value {H0[:, 1:, g][H0[:, 1:, g] > 0].mean():.2f})")
        if kind == "keep":
            nr = ~newf & (m2f > 0.2) & (xf > 0); rr = (xf / np.where(m2f > 0, m2f, 1))[nr]; oo = on[nr][:, g]
            grid = np.linspace(0.2, 1.2, 101); sh = np.array([oo[(rr >= q - 0.025) & (rr < q + 0.025)].mean() for q in grid])
            c = grid[np.argmax(sh >= 0.5)]
            mk = rec | (xs >= c * m2) & (xs > 0)
            label = f"record or x >= {c:.2f} x second"
        else:
            mk = rec.copy(); label = "x_t a record"
        mk[:, 0] = H0[:, 0, g] > 0

        def hook(t, hp, pre, h, mk=mk, g=g):
            h[:, g] = pre[:, g] * mk[:, t]; return h
        r = acc_by_kind(X, run(X, params=params, hook=hook)[0], KEYS)
        print(f"    decision only ({label}; differs on {100 * (mk != (H0[:, :, g] > 0)).mean():.1f}% of steps): " + " | ".join(f"{k} {100 * r[k][0]:.1f}" for k in KEYS))

        def hook2(t, hp, pre, h, g=g):
            h[:, g] = np.maximum(pre[:, g], 0) * 1.05; return h
        r = acc_by_kind(X, run(X, params=params, hook=hook2)[0], KEYS)
        print("    own value x 1.05:   " + " | ".join(f"{k} {100 * r[k][0]:.1f}" for k in KEYS), flush=True)
