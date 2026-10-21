"""Why do earlier close competitors hurt more (compose.log)? Guess: the earliest close competitor was the largest so far
when it arrived, so it was hidden and later revealed, and a revealed vote comes out stronger. Test by intervention with
an eleventh number put first (outside the readout window, but it sets the store): a value between the competitor and the
second makes the competitor arrive as an ordinary kept number, not a record; a control value below the competitor
leaves it a record. Prediction (written first): the first raises accuracy toward the 'later competitor' level (98.5%),
the control does not."""
import sys; sys.path.insert(0, "../steer")
from harness import *
from compose import make
torch.manual_seed(0)
X = make(False, 0.97); o = X.argsort(1, descending=True); s = o[:, 1]; n = torch.arange(len(X), device=DEV)
m = X[:, :5].max(1).values; xs = X[n, s]
def run_pre(X, x0):
    h = torch.relu(torch.zeros(len(X), 16, device=DEV, dtype=dt) @ ARC[1].T + x0[:, None] * ARC[0])
    for t in range(10): h = torch.relu(h @ ARC[1].T + X[:, t:t + 1] * ARC[0])
    return (h @ ARC[2].T).argmax(1)
a = run(ARC, X).argmax(1); wr = a != s
print(f"top two in the last five, first five below 0.97 x second: right {100*(~wr).double().mean().item():.2f}%; wrong answers naming the first half's largest {100*(a[wr] == X[:, :5].argmax(1)[wr]).double().mean().item():.0f}%")
for lab, x0 in (("first number between the competitor and the second", (m + xs) / 2), ("control: first number at half the competitor", 0.5 * m.clamp(min=0)), ("control: first number 0", torch.zeros_like(m))):
    print(f"{lab:52s}: right {100*(run_pre(X, x0) == s).double().mean().item():.2f}%")
L = o[:, 0]; first = a < 5
print(f"wrong answers: the largest {100*(a[wr] == L[wr]).double().mean().item():.0f}%, a first-half number {100*first[wr].double().mean().item():.0f}%, another last-half number {100*(~first[wr] & (a[wr] != L[wr])).double().mean().item():.0f}%")
g = (X[n, L] - xs) / xs
print(f"right when largest/second - 1 < 3%: {100*(~wr[g < 0.03]).double().mean().item():.1f}% (n {int((g < 0.03).sum()):,}); otherwise {100*(~wr[g >= 0.03]).double().mean().item():.1f}%")
