"""r9a analysis candidate (19 neurons, wider-track form; NOT a 16-neuron candidate): r6a_gap with the hiding wave
relayed between the clocks and three relay neurons (rules in algzoo.handbuilt_v9a). The relays sit in the hiding
slots 1, 6, 7; the clocks are helpers in no slot. MAP: handbuilt_v9a.relay_map (RMS scale per row).
Constants read off ARC: 4, as r6a_gap (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the
runner-up) + 16 loudness scales (MAP). No new constant."""
from algzoo.handbuilt_v9a import build_v9a, relay_map

_net, _P, _C = build_v9a(n_relay=3)
MAP, LAYOUT = relay_map(_net, _P)


def weights():
    return _P
