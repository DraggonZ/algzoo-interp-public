"""Exact sampling-free accuracies agree with Monte Carlo."""
import numpy as np
import pytest
import torch as th

from algzoo.core import accuracy, example, forward, load
from algzoo.exact_s2 import exact_accuracy
from algzoo.exact_z2 import HI, LO, angle_mass, exact_z2

cuda = pytest.mark.skipif(not th.cuda.is_available(), reason="needs CUDA")


def test_top_two_density_integrates_to_one():
    assert abs(90 * angle_mass(LO, HI) - 1) < 1e-10


@cuda
def test_exact_length3_matches_sampling():
    params = load("2nd_argmax_4_3_1073741824_bo5.pth")
    exact = exact_accuracy(*params)[0]
    mc, n = accuracy(params, 2**26, seed=3)
    assert abs(exact - mc) < 4 * np.sqrt(mc * (1 - mc) / n)


@cuda
@th.no_grad()
def test_exact_z2_matches_sampling():
    params = example()
    exact = exact_z2(params)[0]
    x = th.randn(2**24, 10, device="cuda", generator=th.Generator("cuda").manual_seed(5))
    v, idx = x.topk(2, dim=-1)
    x2 = th.zeros_like(x).scatter_(1, idx, v)
    mc = (forward(params, x2).argmax(-1) == idx[:, 1]).double().mean().item()
    assert abs(exact - mc) < 4 * np.sqrt(mc * (1 - mc) / 2**24)


def test_gm_rule_integrates_monomials():
    import math

    from algzoo.gm_rule import gm_rule

    X, w = gm_rule(4, 2)
    for a in [(0, 0, 0, 0), (2, 1, 0, 0), (1, 1, 1, 2), (5, 0, 0, 0)]:
        exact = math.prod(math.factorial(k) for k in a) / math.factorial(4 + sum(a))
        assert abs((w * np.prod(X[:, :4] ** np.array(a), axis=1)).sum() - exact) < 1e-14


def test_exact_nd_equals_length3_method():
    from algzoo.exact_nd import exact_accuracy as exact_n

    params = load("2nd_argmax_16_3_1073741824_bo5.pth")
    acc, stats = exact_n(params)
    assert stats["regions"] == 926
    assert abs(acc - exact_accuracy(*params)[0]) < 1e-6   # centroid rule on narrow pieces: ~1e-7


def test_exact_nd_length4_matches_monte_carlo():
    # reference: two independent 2^32-sample runs (seeds 7000 and 8000), 0.98745991 and 0.98745941, SE 1.7e-6 each
    from algzoo.exact_nd import exact_accuracy as exact_n

    acc, stats = exact_n(load("2nd_argmax_8_4_1073741824_bo5.pth"))
    assert stats["regions"] == 4078
    assert abs(acc - 0.98745966) < 4 * 1.2e-6
