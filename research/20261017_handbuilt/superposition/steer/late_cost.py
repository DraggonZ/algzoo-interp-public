"""Where does the 1% last-step turn-down lose accuracy? Split answer changes by the role of the last number (largest,
second, third, smaller) and by whether the change fixes or breaks."""
from harness import *
def run_last(W, X, b):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0] * ((1 - b) if t == 9 else 1))
    return h @ W[2].T
X = lists(4_000_000, seed=21); o = X.argsort(1, descending=True); rank9 = (o == 9).int().argmax(1)
a0 = run(ARC, X).argmax(1); a1 = run_last(ARC, X, 0.01).argmax(1); s = o[:, 1]
for r, name in ((0, "largest"), (1, "second"), (2, "third")):
    m = rank9 == r
    fix = ((a0 != s) & (a1 == s) & m).double().sum().item() / len(X); brk = ((a0 == s) & (a1 != s) & m).double().sum().item() / len(X)
    print(f"last number is the {name:8s}: fixed {100*fix:.3f}, broken {100*brk:.3f} points")
m = rank9 >= 3; fix = ((a0 != s) & (a1 == s) & m).double().sum().item() / len(X); brk = ((a0 == s) & (a1 != s) & m).double().sum().item() / len(X)
print(f"last number is smaller  : fixed {100*fix:.3f}, broken {100*brk:.3f} points")
brk_m = (a0 == s) & (a1 != s)
print("broken lists: new answer is the last position", f"{100*((a1 == 9) & brk_m).double().sum().item()/max(brk_m.sum().item(),1):.0f}%; second's position at step 9 among broken {100*((s == 9) & brk_m).double().sum().item()/max(brk_m.sum().item(),1):.0f}%")
