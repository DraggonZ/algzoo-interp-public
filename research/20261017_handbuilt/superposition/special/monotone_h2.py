"""(Version 2: all values at least 0.05 apart, so no ties.) Simplest violations of monotonicity, found by hypothesis. M1: the network names the second; raising the second by 5%,
staying at most 0.9 of the largest (outside the comparators' resolution zone), makes it wrong. M2: lowering the third by
5% makes a right answer wrong. Also: M1 violation rate on random lists split by second/largest."""
import sys
from hypothesis import given, settings, strategies as st, HealthCheck
from hypo import answer
import torch
fl = st.floats(min_value=-3, max_value=3, allow_nan=False)
def top(x): return sorted(range(10), key=lambda i: -x[i])
def m1(x):
    o = top(x); s, l = o[1], o[0]
    if not (x[s] > 0 and x[s] * 1.05 <= 0.9 * x[l] and answer(x) == s): return True
    y = list(x); y[s] *= 1.05; return answer(y) == s
def m2(x):
    o = top(x); s, t = o[1], o[2]
    if not (x[t] > 0 and answer(x) == s): return True
    y = list(x); y[t] *= 0.95; return answer(y) == s
for name, prop in (("M1 second up 5% (second <= 0.9 largest)", m1), ("M2 third down 5%", m2)):
    found = []
    @settings(max_examples=30000, deadline=None, suppress_health_check=list(HealthCheck), database=None)
    @given(st.lists(fl, min_size=10, max_size=10).filter(lambda x: min(abs(a - b) for i, a in enumerate(x) for b in x[i + 1:]) > 0.05))
    def t(x):
        if not prop(x): found.append(x); raise AssertionError
    try: t(); print(f"{name}: none in 30,000")
    except AssertionError:
        x = found[-1]; o = top(x); print(f"{name}: simplest {[round(v, 3) for v in x]} | largest at {o[0]}, second at {o[1]}, third at {o[2]} | names {answer(x)}, after the change names {answer([v * (1.05 if (i == o[1] and 'M1' in name) else 0.95 if (i == o[2] and 'M2' in name) else 1) for i, v in enumerate(x)])}", flush=True)
