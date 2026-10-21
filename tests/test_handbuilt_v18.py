import numpy as np

from algzoo.handbuilt_v18 import build_v18, scores


def _list(vals, bg=-0.5):
    x = np.full(10, bg)
    for k, v in vals.items():
        x[k] = v
    return x[None]


def test_lone_record_is_hidden():
    net = build_v18()
    for r in range(10):
        for v in (0.3, 1.0, 2.5):
            np.testing.assert_allclose(scores(net, _list({r: v}))[0][:9], 0.0, atol=1e-9)


def test_overtaken_record_shows_a_quarter_of_its_value():
    net = build_v18()
    for r in range(9):
        for L in range(r + 1, 10):
            for v, u in ((1.0, 1.01), (0.4, 1.5), (2.0, 1.2)):
                s = scores(net, _list({r: v, L: v * u}))[0]
                assert abs(s[r] - 0.25 * v) < 1e-9
                assert s.argmax() == r


def test_accuracy_on_random_lists():
    X = np.random.default_rng(30).standard_normal((100_000, 10))
    acc = (scores(build_v18(), X).argmax(1) == X.argsort(1)[:, -2]).mean()
    assert 0.95 < acc < 0.965
