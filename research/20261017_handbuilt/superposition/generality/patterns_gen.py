"""Question 5: on/off pattern census and per-neuron switching rules for each target network (../patterns/patterns.py and
switch_rules.py, all 16 neurons, the network swapped). Census over 300,000 random lists, steps 1-9: distinct patterns of
all 16 neurons and of the hiding group, share of steps covered by the top 1, 3, 10, 30. Rules: decision trees of depth
1, 2, 5 on the list features of switch_rules.py (step, age of the largest, age of the previous record, x / largest so far,
x / second so far, x > 0, x is a record, previous x / largest), trained on half the lists, accuracy on the other half
against always guessing the commoner state. A neuron 'follows a one-line rule' when depth 2 reaches 95% and beats the
guess by at least 3 points; 'nearly constant' when the guess alone reaches 97%."""
from common_gen import *
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from switch_rules import features, NAMES
def census(P, idx):
    w = 2 ** torch.arange(len(idx), device=DEV); c = (P[..., idx].long() * w).sum(-1).flatten().cpu().numpy()
    cnt = Counter(c.tolist()); top = [v for _, v in cnt.most_common()]; tot = len(c)
    return len(cnt), [100 * sum(top[:k]) / tot for k in (1, 3, 10, 30)]
if __name__ == "__main__":
    X = torch.randn(300000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(102))
    F = features(X)[:, 1:].clamp(-50, 50); Fn = F.reshape(-1, F.shape[2]).cpu().numpy(); half = len(Fn) // 2
    summ = []
    for nm, W in TARGETS.items():
        P = onoff(W, X)[:, 1:]; hid, _, _ = hiding_group(W)
        na, ca = census(P, list(range(16))); nh, ch = census(P, hid)
        print(f"\n{nm}: all 16 neurons {na} patterns, top 1/3/10/30 cover " + " ".join(f"{x:.1f}%" for x in ca) + f"; hiding group {hid}: {nh} patterns, top 1/3 cover {ch[0]:.1f}% {ch[1]:.1f}%")
        print(f"  {'neuron':>6s} {'on %':>6s} {'guess':>6s} {'depth1':>7s} {'depth2':>7s} {'depth5':>7s} | depth-1 split")
        rows = []
        for j in range(16):
            y = P[..., j].reshape(-1).cpu().numpy(); base = max(y[half:].mean(), 1 - y[half:].mean()); accs = []
            for d in (1, 2, 5):
                tr = DecisionTreeClassifier(max_depth=d, min_samples_leaf=200, random_state=0).fit(Fn[:half], y[:half]); accs.append((tr.predict(Fn[half:]) == y[half:]).mean())
                if d == 1: split = f"{NAMES[tr.tree_.feature[0]]} <= {tr.tree_.threshold[0]:.2f}" if tr.tree_.node_count > 1 else "-"
            tag = "nearly constant" if base >= 0.97 else ("one-line rule" if accs[1] >= 0.95 and accs[1] - base >= 0.03 else ("no depth-2 rule" if accs[1] - base < 0.01 else "partial"))
            rows.append((j, base, *accs, tag, j in hid))
            print(f"  {j:6d} {100*y.mean():5.1f}% {100*base:5.1f}% " + " ".join(f"{100*a:6.1f}%" for a in accs) + f" | {split:28s} {tag}{' (hiding)' if j in hid else ''}")
        R = np.array([[r[1], r[2], r[3], r[4]] for r in rows]); tags = Counter(r[5] for r in rows)
        hacc = np.mean([r[3] for r in rows if r[6]])
        summ.append((nm, na, ca[3], nh, ch[1], 100 * R[:, 2].mean(), 100 * (R[:, 2] - R[:, 0]).mean(), tags["one-line rule"], tags["nearly constant"], tags["partial"], tags["no depth-2 rule"], 100 * hacc))
    print(f"\n{'network':10s} {'patterns':>8s} {'top30':>6s} {'hiding pats':>11s} {'top3':>6s} | {'depth-2 mean':>12s} {'gain':>5s} | one-line rule, nearly constant, partial, none | hiding neurons' depth-2 mean")
    for s in summ: print(f"{s[0]:10s} {s[1]:8d} {s[2]:5.1f}% {s[3]:11d} {s[4]:5.1f}% | {s[5]:11.1f}% {s[6]:5.1f} | {s[7]:2d} {s[8]:2d} {s[9]:2d} {s[10]:2d} | {s[11]:.1f}%")
