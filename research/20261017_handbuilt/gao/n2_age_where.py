"""Which lists need neuron 2's timing signal? Scrub neuron 2 alone under the store claim without m1's place (m1 moved,
m2 kept at its place) and with it; classify the lists whose answer moves. Kinds: 'overtaken second' = the second-largest
was a record before the largest arrived; 'neighbour' = the third-largest sits right after the second."""
import numpy as np
import scrub
from simlib import lists, run
X = lists(202, 100_000); L = run(X)[0]; y = X.argsort(1)
sec, lar, thi = y[:, -2], y[:, -1], y[:, -3]
rec = np.array([[X[b, i] > X[b, :i].max(initial=0) for i in range(10)] for b in range(len(X))])
over = rec[np.arange(len(X)), sec] & (sec < lar); nb = over & (thi == sec + 1)
res = {}
for label, opts in (("m1 moved", ("m2",)), ("m1 kept", ("m1pos", "m2"))):
    sub = {k: v for k, v in scrub.scrubbed_values(X, ["store+detector"], opts={"store+detector": opts}).items() if k == 2}
    Ls = run(X, sub)[0]; mv = Ls.argmax(1) != L.argmax(1); res[label] = mv
    print(f"{label}: answers moved {mv.mean():.3f}; accuracy {(Ls.argmax(1) == sec).mean():.3f}")
mv = res["m1 moved"] & ~res["m1 kept"]
print(f"lists moved only when m1's place changes: {mv.mean():.3f}")
for name, k in (("all lists", np.ones(len(X), bool)), ("overtaken second", over), ("  with third right after", nb), ("largest at last step", lar == 9)):
    print(f"   {name:26s}: share of lists {k.mean():.3f}, share of these moved {mv[k].mean():.3f}")
print("   position of the largest among moved lists:", np.bincount(lar[mv], minlength=10) / mv.sum())
print("   position of the largest among all lists:  ", np.bincount(lar, minlength=10) / len(X))
