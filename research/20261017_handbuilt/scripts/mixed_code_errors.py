"""Are second-third close calls worse when the two numbers are stored in different codes? Codes: a number that arrived
before the largest is revealed by the switch-off; one that arrived after is kept by neuron 5. Classes: both before,
both after, mixed. ARC's second-third error rate by class and relative gap, on 8 million random lists."""
import torch as th

from algzoo.core import example, forward

g = th.Generator(device="cuda").manual_seed(11)
X = th.randn(8_000_000, 10, device="cuda", generator=g, dtype=th.float64)
ans = forward(example(), X).argmax(-1); o = X.argsort(-1, descending=True); v = X.gather(1, o[:, :3])
ok = v[:, 2] > 0                                             # three positive numbers
gap = (v[:, 1] - v[:, 2]) / v[:, 1]
b2, b3 = o[:, 1] < o[:, 0], o[:, 2] < o[:, 0]
cls = {"both before largest": b2 & b3, "both after largest": ~b2 & ~b3, "mixed: second before, third after": b2 & ~b3, "mixed: second after, third before": ~b2 & b3}
B = ans == o[:, 2]
print("second-third mistake rate (answer = third) by class and relative gap; three positive numbers")
for lo, hi in ((0, .01), (.01, .03), (.03, .1), (.1, .3)):
    sel = ok & (gap >= lo) & (gap < hi)
    print(f"gap [{lo}, {hi}): " + " | ".join(f"{k} {100*(B & sel & c).double().sum().item()/max(1,(sel & c).double().sum().item()):5.1f}% (n={int((sel & c).sum().item())})" for k, c in cls.items()))

print("\nwithin each class: does it matter which of the two arrived first? (second-third mistake rate)")
first3 = o[:, 2] < o[:, 1]                                   # the third-largest arrived before the second-largest
for name in ("both before largest", "both after largest"):
    c = cls[name]
    for lo, hi in ((0, .01), (.01, .03), (.03, .1)):
        sel = ok & c & (gap >= lo) & (gap < hi)
        r = lambda m: 100 * (B & sel & m).double().sum().item() / max(1, (sel & m).double().sum().item())
        print(f"{name}, gap [{lo}, {hi}): third arrived first {r(first3):5.1f}% | second arrived first {r(~first3):5.1f}%")
