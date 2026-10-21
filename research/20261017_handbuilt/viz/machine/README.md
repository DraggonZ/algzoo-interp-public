# The Second-Largest Machine page, version 3 (2026-10-21)

An interactive page that runs ARC's network live in the browser. Files:
- `machine_v3_template.html`: the page, with `__WEIGHTS__` replaced by `page_data.json` at build time.
- `modes_data.py`: writes `page_data.json` (ARC's weights, per-neuron display scales from `arc_weights.json`, the memory block's three rotating planes and each memory neuron's mark in them).
- `check.py`: headless Chromium check (run with `/data/output/tools/playwright-env/bin/python`, from a folder holding the built page `second_largest_machine_v3.html`): browser scores against Python (max difference 3e-13), the exact split of the state by source position (error 1e-14), switch-off statistics, page width at 1280 and 400 pixels, screenshots.
- `proto.py`, `proto2.py` and logs: the checks behind the memory dials.

## What changed from version 2

- Controls and the input tape are docked at the top of the screen, so stepping does not need scrolling.
- New panel "Inside the memory": three dials, one per rotating plane of the memory block (38°, 84°, 127° per step, all memory neurons on). Each arrow is one number's share of the memory neurons, from the exact split h_t = Σ_p (∂h_t/∂x_p) x_p (no biases), projected on the plane. Stepping animates the turn.
- Memory group text updated (clock reading, switching needed: 44% without it, compensation result).

## Observations behind the dials (`proto.log`, `proto2.log`)

- Per-step turn of a kept number's share, steps p+3 to 9, 2,000 lists: median 39° in the 38° plane (middle half 33-51°), 76° in the 84° plane (52-86°), 110° in the 127° plane (12-129°). The slow and middle planes turn visibly; the fast one less reliably.
- At the end of the list the second-largest's arrow does **not** point at its readout spoke in each plane (median deviation 55-76°). So the page does not claim that the readout reads votes off the dials directly; votes are shared with other groups (as `parameters/memory_register/README.md` found).
- Clock hand: on lists where position r is the largest and stays so, the angle between the largest's share in the slow plane and each memory neuron's mark (the direction that pushes that neuron up most) passes through 180°:
  - neuron 15 at age 3 (178°), neuron 12 at age 4 (160°), neuron 11 at age 5 (167°), neuron 10 at age 6 (168°);
  - these match the switch-off ages measured before (`memory/README.md` §1.4: 15 at ages 3-4, 12 sometimes at 4, 11 at 5, 10 at 6);
  - exception: neuron 3 is also opposite at age 3 (167°) but is rarely off then (0-4%); neurons 0 and 14 are near their opposite only at ages 0 and 8, and neuron 0's switch-off at age 2 comes from outside the memory (`parameters/memory_register/README.md` §5b).
- Reading (interpretation): for four of the eight memory neurons, the slow plane's phase of the largest's share alone predicts the switch-off age. This is a one-plane version of step 5b's finding that the dip is a sum over five modes, with the slow mode carrying most of it. Not tested by intervention.
