"""Search for the simplest inputs that break account properties, with hypothesis (random search plus shrinking toward
simple values). Each property was expected to hold from the account; the shrunk counterexample shows where it fails.
A increasing positive list -> names 8. B decreasing positive list -> names 1. C clear gaps (largest >= 1.2 x second,
second >= 1.2 x third, second > 0) -> names the second. D two spikes a > b > 0 on a background no larger than b/2 ->
names b's position."""
import sys; sys.path.insert(0, "../steer")
import torch
from hypothesis import given, settings, strategies as st, HealthCheck, Verbosity
from hypothesis.errors import Found
from harness import ARC, DEV, dt
W = [p.cpu() for p in ARC]
def answer(x):
    h = torch.zeros(16, dtype=dt)
    for v in x: h = torch.relu(W[1] @ h + v * W[0])
    return int((W[2] @ h).argmax())
fl = st.floats(min_value=-5, max_value=5, allow_nan=False)
pos = st.floats(min_value=0.01, max_value=5, allow_nan=False)
def search(name, strat, prop):
    found = []
    @settings(max_examples=20000, deadline=None, suppress_health_check=list(HealthCheck), database=None)
    @given(strat)
    def t(x):
        if not prop(x):
            found.append(x); raise AssertionError
    try: t(); print(f"{name}: no counterexample in 20,000 tries", flush=True)
    except AssertionError:
        x = found[-1]
        if isinstance(x, tuple): xx, j = mk(x); print(f"{name}: simplest counterexample {[round(v, 3) for v in xx]} (second at {j}) -> names {answer(xx)}", flush=True)
        else: print(f"{name}: simplest counterexample {[round(v, 3) for v in x]} -> names {answer(x)}", flush=True)
inc = st.lists(pos, min_size=10, max_size=10).filter(lambda x: all(a < b for a, b in zip(x, x[1:])))
inc = st.lists(pos, min_size=10, max_size=10).map(lambda x: sorted(x)).filter(lambda x: len(set(x)) == 10)
if __name__ == "__main__": pass
if "A" in sys.argv: search("A increasing", inc, lambda x: answer(x) == 8)
if "B" in sys.argv: search("B decreasing", inc.map(lambda x: x[::-1]), lambda x: answer(x) == 1)
def gaps(x):
    s = sorted(x, reverse=True); return s[1] > 0 and s[0] >= 1.2 * s[1] and s[1] >= 1.2 * s[2]
if "C" in sys.argv: search("C clear gaps", st.lists(fl, min_size=10, max_size=10).filter(gaps), lambda x: answer(x) == sorted(range(10), key=lambda i: -x[i])[1])
spk = st.tuples(st.lists(st.floats(min_value=-5, max_value=1, allow_nan=False), min_size=10, max_size=10), st.integers(0, 9), st.integers(0, 9), st.floats(1.01, 3))
def mk(t):
    bg, i, j, ratio = t; x = [v * 0.5 for v in bg]; x[i] = 2.0 * ratio; x[j] = 2.0; return x, j
search("D two spikes", spk.filter(lambda t: t[1] != t[2]).map(mk), lambda xj: answer(xj[0]) == xj[1]) if False else None
def propD(t):
    x, j = mk(t); return answer(x) == j
if "D" in sys.argv: search("D two spikes", spk.filter(lambda t: t[1] != t[2]), propD)
