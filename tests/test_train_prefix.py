"""The stacked trainer keeps models independent; the prefix tree gives the same scores as plain runs."""
import numpy as np
import torch as th

from algzoo.core import example
from algzoo.prefix import cost_in_runs, tree_scores
from algzoo.train import init, train


def test_stacked_models_train_as_if_alone():
    g = th.Generator().manual_seed(1)
    W = init(2, 4, 3, generator=g)
    xs_all = th.randn(20, 2, 8, 3, generator=g)
    both = train(2, d=4, n=3, n_train=20 * 8, batch=8, device="cpu", W=[w.detach().clone().requires_grad_() for w in W], xs=lambda s: xs_all[s])
    for i in range(2):
        Wi = [w[i:i + 1].detach().clone().requires_grad_() for w in W]
        alone = train(1, d=4, n=3, n_train=20 * 8, batch=8, device="cpu", W=Wi, xs=lambda s, i=i: xs_all[s, i:i + 1])[0]
        for a, b in zip(both[i], alone):
            assert np.allclose(a, b, atol=1e-6)


def test_prefix_tree_matches_plain_runs():
    params = example()
    rng = np.random.default_rng(0)
    X = np.full((500, 10), -0.46)
    for row in X:
        pos = rng.choice(10, 3, replace=False)
        row[pos] = rng.choice([0.2, 0.9, 1.4], 3)
    w_hi, w_hh, w_oh = params
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(0, h @ w_hh.T + X[:, t, None] * w_hi)
    S, steps, distinct = tree_scores(params, X)
    assert np.allclose(S, h @ w_oh.T)
    assert steps < 10 * distinct
    assert cost_in_runs(params, steps, distinct) < len(X)
