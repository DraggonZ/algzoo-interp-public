import numpy as np

from algzoo.handbuilt import DETECT_GAIN, Net, add_running_max, run, running_max_readout


def _part1(x):
    net = Net()
    add_running_max(net)
    return net, run(net.params, x)[..., 1:, :]


def test_running_max_is_exact():
    x = np.random.default_rng(0).standard_normal((100_000, 10))
    net, h = _part1(x)
    want = np.maximum.accumulate(np.maximum(x, 0.0), axis=-1)
    np.testing.assert_allclose(running_max_readout(net, h), want, atol=1e-12)


def test_detector_fires_only_on_new_largest_in_proportion_to_excess():
    x = np.random.default_rng(1).standard_normal((100_000, 10))
    net, h = _part1(x)
    prev = np.concatenate([np.zeros((len(x), 1)), np.maximum.accumulate(np.maximum(x, 0.0), axis=-1)[:, :-1]], 1)
    want = DETECT_GAIN * np.maximum(x - prev, 0.0)
    np.testing.assert_allclose(h[..., net.idx["detect"]], want, atol=1e-12)


def test_hand_picked():
    net, h = _part1(np.array([-1, -2, 0.5, 0.3, 1.2, 1.2, -3, 2.0, 1.99, 0.1]))
    np.testing.assert_allclose(h[:, net.idx["detect"]], DETECT_GAIN * np.array([0, 0, .5, 0, .7, 0, 0, .8, 0, 0]), atol=1e-9)
    np.testing.assert_allclose(running_max_readout(net, h), [0, 0, .5, .5, 1.2, 1.2, 1.2, 2, 2, 2], atol=1e-12)
    # an all-negative input leaves both neurons silent
    _, h = _part1(-np.ones(10))
    assert not h.any()


def _second(x):
    z = np.zeros(x.shape[:-1] + (2,))
    return np.stack([np.sort(np.concatenate([z, x[..., :t + 1]], -1), -1)[..., -2] for t in range(x.shape[-1])], -1)


def test_second_max_is_exact_and_gates_switch_within_band():
    from algzoo.handbuilt import add_second_max, gates, second_max_readout
    x = np.random.default_rng(2).standard_normal((50_000, 10))
    net = Net()
    add_running_max(net)
    add_second_max(net)
    h = run(net.params, x)[:, 1:]
    m2 = _second(x)
    np.testing.assert_allclose(second_max_readout(net, h), m2, atol=1e-12)
    g1, g2 = gates(net, h)
    pm1 = np.concatenate([np.zeros((len(x), 1)), np.maximum.accumulate(np.maximum(x, 0), 1)[:, :-1]], 1)
    pm2 = np.concatenate([np.zeros((len(x), 1)), m2[:, :-1]], 1)
    rec, sec = x > pm1, (x > pm2) & (x <= pm1)
    clear1 = (x - pm1 > 0.01 * x) | (x <= pm1)
    clear2 = clear1 & ((x - pm2 > 0.01 * x) | (x <= pm2))
    np.testing.assert_allclose(g1[clear1], np.where(rec, x, 0)[clear1], atol=1e-9)
    np.testing.assert_allclose(g2[clear2], np.where(sec, x, 0)[clear2], atol=1e-9)


def test_lone_input_casts_no_vote_at_any_age():
    from algzoo.handbuilt import build_v2
    P = build_v2().params
    for p in range(10):
        for v in (0.3, 1.0, 2.5):
            x = np.zeros(10)
            x[p] = v
            s = run(P, x)[-1] @ P[2].T
            assert np.ptp(s) < 1e-9


def test_two_spike_all_pairs():
    from algzoo.handbuilt import build_v2
    P = build_v2().params
    for r in (0.3, 0.7, 0.95, 0.99):
        for p1 in range(10):
            for p2 in range(10):
                if p1 != p2:
                    x = np.zeros(10)
                    x[p1], x[p2] = 1.0, r
                    assert (run(P, x)[-1] @ P[2].T).argmax() == p2


def _v3_scores(x, **kw):
    from algzoo.handbuilt import build_v3
    net = build_v3(**kw)
    h = run(net.params, np.atleast_2d(x))[:, -1]
    return h @ net.w_oh.T


def _list(**vals):
    x = np.full(10, -0.5)
    for k, v in vals.items():
        x[int(k[1:])] = v
    return x


def test_v3_hides_the_largest_and_reveals_it_as_a_spike():
    np.testing.assert_allclose(_v3_scores(_list(p3=1.0))[0], 0, atol=1e-6)
    s = _v3_scores(_list(p3=1.0, p6=1.5))[0]
    np.testing.assert_allclose(s, np.eye(10)[3], atol=1e-6)


def test_v3_clears_older_votes_at_a_reveal_and_keeps_later_ones():
    s = _v3_scores(_list(p1=0.8, p3=1.0, p5=0.7, p6=1.5, p8=0.9))[0]
    np.testing.assert_allclose(s, 1.0 * np.eye(10)[3] + 0.9 * np.eye(10)[8], atol=1e-6)


def test_v3_accuracy():
    x = np.random.default_rng(3).standard_normal((100_000, 10))
    acc = (_v3_scores(x).argmax(1) == x.argsort(1)[:, -2]).mean()
    assert acc > 0.985


def _v4_scores(x):
    from algzoo.handbuilt import build_v4
    net = build_v4()
    assert net.w_hi.shape == (16,)
    h = run(net.params, np.atleast_2d(x))[:, -1]
    return h @ net.w_oh.T


def test_v4_has_16_neurons_and_votes_like_v3():
    # v4 reads the actual last input, so negative numbers leave small negative votes; check the positive ones
    s = _v4_scores(_list(p3=1.0))[0]
    assert s.max() < 1e-6
    s = _v4_scores(_list(p3=1.0, p6=1.5))[0]
    assert s.argmax() == 3 and abs(s[3] - 1.0) < 1e-6 and np.delete(s, 3).max() < 1e-6
    s = _v4_scores(_list(p1=0.8, p3=1.0, p5=0.7, p6=1.5, p8=0.9))[0]
    np.testing.assert_allclose(s[[1, 3, 5, 8]], [0, 1.0, 0, 0.9], atol=1e-6)
    # the last input votes on position 9, read from the clock's dip
    s = _v4_scores(_list(p3=1.0, p6=1.5, p9=1.2))[0]
    assert s.argmax() == 9 and abs(s[9] - 1.2) < 1e-6 and abs(s[3] - 1.0) < 1e-6


def test_v4_matches_v3():
    x = np.random.default_rng(4).standard_normal((100_000, 10))
    two = (x > 0).sum(1) >= 2  # with at most one positive number every score ties near zero
    a, b = _v3_scores(x[two]).argmax(1), _v4_scores(x[two]).argmax(1)
    assert (a == b).mean() > 0.999
    assert (_v4_scores(x).argmax(1) == x.argsort(1)[:, -2]).mean() > 0.985


def test_v5_matches_v4_with_the_trained_networks_input_layout():
    from algzoo.handbuilt import build_v4, build_v5
    v4, v5 = build_v4(), build_v5()
    # one large positive input weight (the detector), four large negative ones (cancelling neurons)
    assert (v5.w_hi > 100).sum() == 1 and (v5.w_hi < -100).sum() == 4 and (v5.w_hi != 0).sum() == 5
    x = np.random.default_rng(5).standard_normal((100_000, 10))
    two = (x > 0).sum(1) >= 2
    s4 = run(v4.params, x)[:, -1] @ v4.w_oh.T
    s5 = run(v5.params, x)[:, -1] @ v5.w_oh.T
    assert (s4.argmax(1)[two] == s5.argmax(1)[two]).mean() > 0.998
    assert (s5.argmax(1) == x.argsort(1)[:, -2]).mean() > 0.985


def test_v5_with_a_rebuilding_clock_matches_the_gated_clock():
    from algzoo.handbuilt import build_v5
    a, b = build_v5(), build_v5(clock="ringup")
    x = np.random.default_rng(6).standard_normal((100_000, 10))
    two = (x > 0).sum(1) >= 2
    pa = (run(a.params, x)[:, -1] @ a.w_oh.T).argmax(1)
    pb = (run(b.params, x)[:, -1] @ b.w_oh.T).argmax(1)
    assert (pa[two] == pb[two]).mean() > 0.998
    assert (pb == x.argsort(1)[:, -2]).mean() > 0.985
