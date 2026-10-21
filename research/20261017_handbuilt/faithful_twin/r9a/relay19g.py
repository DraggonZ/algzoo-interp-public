"""r9a analysis candidate (19 neurons, wider-track form; NOT a 16-neuron candidate): relay19.py with neuron 3 (delay
cell mem6) reading its gap from the hiding slots, the relays' level minus the runner-up comparator, instead of the
clocks' level (the same value). Rules in algzoo.handbuilt_v9a. MAP: handbuilt_v9a.relay_map (RMS scale per row).
Constants read off ARC: 4, as r6a_gap (memory imprint 1, keep share 0.25, memory fade 0.99, keep cut 0.75 of the
runner-up) + 16 loudness scales (MAP). No new constant."""
from algzoo.handbuilt_v9a import build_v9a, relay_map

_net, _P, _C = build_v9a(n_relay=3, gap_from_relays=True)
MAP, LAYOUT = relay_map(_net, _P)


def weights():
    return _P
