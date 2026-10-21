# A faithful twin, judged by ARC's nudge tests (started 2026-10-19)

Status: in progress (2026-10-19). Best twin on the nudge tests so far: the 16-neuron twin below (shifts right in sign 12 of 12 at 1.1 times ARC's size, rates 14 points off, 93.0% accurate). A 17-neuron validation build with ARC's runner-up mechanism reproduces ARC's overall near-tie blur but not yet the nudge tests better. Accuracy bar set at about 92% (2026-10-19): ARC's last points come from many small trained compensations no rule explains. Goal (2026-10-19): a hand-built 16-neuron network that responds to ARC's nudge tests as ARC does; ARC's mistakes reproduced is a secondary measure; Hilton's test is out of reach. Every weight still comes from a rule; constants may be read off ARC.

## Measures

- `battery.py`: ARC's steering tests (steer/RESULTS.md) as one fixed battery of 37 rates on fixed lists: memory and store nudges on near ties, keep-gate nudges on last-step contenders, the late-number bias, hiding nudges, extra and appended numbers, and a sensitivity map (one group at one step x1.02, answers changed, 40 cells). Score: mean gap to ARC's rates in points.
- `shifts.py`: the same nudges measured as the shift they cause in the decision's lead, in units of the second's own vote (median over lists). Rates mix the shift with how blurred a network's near ties are (a hand-built network decides near ties perfectly, so its rates jump between 0 and 100); shifts compare the mechanisms directly. Score: same sign as ARC, and median size ratio.

## Where the twins stand

| network | accuracy | shifts: same sign as ARC | shifts: size / ARC's (median) | rates: mean gap |
|---|---|---|---|---|
| v18 (sent to Hilton) | 95.9% | 6 of 12 | 10.2 | 31.5 |
| v19 | 94.2% | 6 of 12 | 11.4 | 30.1 |
| v23 from v19 (helper) | 94.7% | 12 of 12 | 7.5 | 25.2 |
| v23 | 95.8% | 12 of 12 | 2.1 | 21.7 |
| twin: v23 with ARC's constants (below) | 93.0% | 12 of 12 | 1.1 | 14.0 |
| same, ARC's gains (10) | 87.0% | 12 of 12 | 1.0 | 12.0 |

v18 and v19 move the wrong way under memory and store nudges: their memory holds level plus vote, ARC's holds a vote next to a larger negative imprint.

## The twin's constants, each read off ARC or set by a rule (`algzoo.handbuilt_v23.build_v23(U_mem=1, share=0.25, rho=0.99, level_exact=False)`)

- Memory imprint U = 1 vote per unit of the largest. Read off ARC's memory nudge (its lead shift implies an imprint of about one vote); it is also the smallest imprint that keeps a slot on for every number below the largest, so each slot holds the number's shortfall below the largest. Slots then switch off where the hiding wave's far lobes are positive: 6-7 positions after the largest and 8-9 before it (`clip_where.py`). ARC's memory neurons also switch off at fixed ages after a new largest (2, 5, 6 steps); whether that is the same thing is untested.
- Keep-gate share 0.25 of a vote: read off ARC's keep-gate nudge (5% moves the last-step contender's lead by about 0.012 votes, a quarter of v23's 0.5 share). Note: first-order bookkeeping on ARC (`steer/detector_share.py`) gave a median share of 0.44-0.53; the nudge reading is used because the battery tests nudges.
- Memory fading 0.99 per step with the level added to every slot alike at each jump (ARC's level mode shrinks 0.99 per step and is fed by the store along one direction). Gives ARC's late-number bias without any tuning: a step-9 contender wins 57.8% of near ties where it is truly second half the time (ARC 57.5%; v23 49%).

Remaining differences: the memory nudge when the third arrived first (5-6 times ARC's shift), the store nudge (a third to two-thirds of ARC's), the hiding nudge (1.8 times: v23's hiding neurons alone hide the largest, ARC shares it with memory and store), and above all near-tie blur: on near ties ARC names the third about a fifth of the time in every arrangement, the twin almost never in some arrangements and 30-40% in others. ARC-scale gains (10) blur near ties a little but cost 6 points of accuracy.

Files: `battery.py`, `shifts.py` (+`.log`), `variants*.py`, `shifts2.py`, `shifts3.py`, `ideal_runnerup.py`, `leak_shape.py`, `clip_where.py`, `check_v23f.py`.

Findings on the way:
- ARC's leak in decisions (`leak_shape.log`): per unit, the largest moves the second-over-third lead by a median 0.05 (90th percentile 1.5), positive smaller numbers by 0.018 (0.28), negative numbers by 0.005; numbers before the second leak about four times more than numbers after it.
- A keep gate relative to the runner-up so far (ARC's rule, idealized, `ideal_runnerup.log`) blurs near ties far too much on its own (third wins 39%, accuracy 82%) because later numbers are discounted; with uncompensated fading 0.94 the two partly offset (accuracy 91%, third wins 40%). The comparator form a runner-up neuron gives for free is exactly fair (no blur). Not built.

## Where ARC's near-tie blur comes from (`arc_leak_source.py`, `ideal_runnerup*.py`)

- In ARC, the number that was runner-up when a contender arrived moves the second-over-third lead by a median |0.07-0.10| per unit; other positive smaller numbers by 0.011 (`arc_leak_source.log`). Prediction (written first) right in size, not in sign: the median sign is near zero (-0.026 and +0.007), so the effect switches. ARC's keep gate compares each number with the runner-up so far (account item 2).
- Idealized (`ideal_runnerup3.log`): votes alpha x + beta relu(x - theta m2), m2 the runner-up before x, beta 0.25 (twin's keep share), theta 0.3 (ARC's measured leak 0.073 over 0.25), every vote fading 0.99 per step since arrival (ARC's level mode, uncompensated): third wins 21.0% of near ties (ARC 20.8%), tree 40% of the possible gain (ARC 31%), accuracy 97.9%, 125 maps. Both constants read off ARC; the account's 'later votes discounted by the runner-up, earlier votes fade, the two roughly offset' made concrete.

## v24, a 17-neuron validation build (`algzoo.handbuilt_v24`, `v24_eval*.py`)

v23's twin plus one neuron that keeps the runner-up so far exactly (relu(g2 (m2 - x)); then m2 = x + it/g2 - jump in every case), a keep gate at the number's own step relative to the runner-up before it, and votes read as they have faded. Not a 16-neuron network: in ARC the runner-up shares hiding neuron 8 with part of the hiding wave; in v23 the flip neuron cannot hold both exactly (one value, two unknowns), which is presumably part of ARC's blur.

| build | accuracy | third wins at near ties | tree's share of gain | lead maps | late bias | rates gap | shifts: size / ARC's |
|---|---|---|---|---|---|---|---|
| ARC | 95.3% | 20.8% | 31% | 7,955 | 57.5% | | |
| twin16 | 93.0% | 8.8% | 90% | 87 | 57.8% | 14.0 | 1.1 |
| v24, last number direct only | 89.4% | 26.9% | 64% | 120 | 96.4% | 17.2 | 1.5 |
| v24, last number gated too | 91.5% | 21.1% | 45% | 127 | 44.3% | 15.8 | 1.5 |

(v24 rows: imprint 1.5 and fading by age; imprint 1 lost 2 points to slots switching off where the wave's far lobes are positive.) Reading: the runner-up mechanism gives ARC's overall near-tie blur and a tree nearly as uninformative as ARC's, confirming it as a main source; but the battery does not improve, because where the blur falls differs (ARC blurs 'second early' near ties to 72%, v24 to 87%; the last-step contender is either far too strong or slightly weak). Choices so far were made one at a time with a reason each; a sweep over these switches against the battery would be fitting and was not done.

## Fitting the runner-up into 16 neurons (2026-10-19)

- Why the exact runner-up needs a 17th neuron in v23's design: a quantity carried from step to step needs a neuron whose self-weight keeps it (the store keeps the largest with +1; the flip neuron flips its part with -1). Quantities rebuilt every step from others (the comparators' levels, the two routes, the detector) can share neurons; the runner-up cannot be rebuilt from anything else, and every neuron that keeps something is already used. The flip neuron cannot also keep the runner-up: one value, two unknowns.
- Easy trades (`v24 shape="no_flip"`): dropping the flip neuron for the runner-up leaves a wave with only the rotating pair; 62% (symmetric) or 53% (no spill on the next position). Dropping the buffer neuron brings back ARC's nine-positions-in-eight-slots leak (11-25 points earlier). Both too costly.
- ARC's neuron 8 (`arc_n8.log`): its pre-activation is about -12.3 x + 1.75 (largest) + 11.1 (runner-up) (R2 0.86; with the hiding neurons' previous values 0.998): a comparator against a blend, mostly the runner-up, carried approximately through its own and the other hiding neurons' values.
- ARC's cheap runner-up: its store lags a step, so a comparator reading the store without the detector compares a number with the largest two steps back, which right after a new largest is the previous largest, the runner-up at that moment (neurons 6 and 8 do this, bridge/README.md). Built into v24 as `runner="store_lag"` (16 neurons, keep gate at the number's own step against theta x that): 90.0% accurate, third wins 28.7% of near ties, 138 maps, tree 79% of the gain; nudge rates 17.2 points off ARC (v23 with ARC's settings 14.0; exact 17-neuron runner-up 15.8). Where it differs: with the largest first, both later contenders are discounted by the largest alike, so fading makes the earlier one lose (wins 22-24% where ARC's earlier contender wins 72%). ARC's own store of the runner-up must hold it beyond the step after a record.

Reading: every form of the runner-up tried so far brings ARC's overall near-tie blur but places it differently from ARC, and the nudge rates get worse, not better. The best twin on the nudge tests is still v23 with ARC's settings. What would decide it is how ARC's neuron 8 keeps the runner-up over later steps (its account says it stores a late second weakly).

## Calibrating the nudge tests (`calib/`, 2026-10-19)

The scores above had no reference for 'close'. `calib/calib.py` runs the battery and the lead shifts on ARC with small relative weight noise (the same mechanism, slightly perturbed) and on the other trained 16-neuron networks (same recipe, different solutions), with neuron groups found from activity (`calib/groups_auto.py`; it reproduces ARC's named groups exactly). 18 of 38 siblings have all four groups; the other 20 have no neuron the rule recognises as a keep gate and are left out, so the reference is the siblings most like ARC in layout. `calib/summarize.py` collapses the tests into nine separate questions (memory, store, keep and hiding nudges; how well near ties follow the true order; late bias; hiding neurons alone; shift register; sensitivity map) and divides each network's distance to ARC by the median sibling's distance (1 = as far as a typical other trained network).

| network | accuracy | median distance over the nine questions | questions farther than a typical sibling |
|---|---|---|---|
| ARC with noise 0.001-0.003 | 93-95% | 0.01-0.03 | none |
| trained siblings | 88-96% | 1.00 by construction (10-90%: 0.54-1.26; closest 0.34) | |
| v18 (sent to Hilton) | 95.9% | 1.26 | memory 34x, store 23x, keep 4x, near-tie order 4.6x, late bias 1.3x |
| v23 | 95.8% | 1.37 | memory 2.8x, near-tie order 4.6x, hiding alone 3.4x, keep, hiding nudge, store, late bias 1.2-1.5x |
| v23 with ARC's settings | 93.0% | 0.69 | near-tie order 2.5x, hiding alone 2.0x, hiding nudge 1.5x |
| v24, exact runner-up (17 neurons) | 91.5% | 1.10 | hiding alone 6.7x, late bias 1.9x, shift register 1.5x, hiding nudge 1.4x, keep 1.1x |

Reading: v23 with ARC's settings is closer to ARC on these questions than a typical other trained network (0.69 against 1; about the 20th percentile of siblings), but not as close as ARC with noise. Its remaining differences are the three above: near ties decided too sharply, hiding neurons doing all the hiding alone, and the hiding nudge 1.8 times too strong. The runner-up build fixes near-tie order (0.54) but moves late bias and the hiding questions away. Siblings disagree with ARC even in sign on the memory nudge (median +0.055 against ARC's -0.031), so several of ARC's nudge responses are specific to ARC's solution, not to the task or the recipe. Note: the old 37-rate score counted one difference (near-tie sharpness) many times; this one counts it once.

## Is part of ARC's hiding kept in memory? (`reveal_full.py`, 2026-10-19)

Prediction (written first): if memory and store share the hiding, an overtaken largest is revealed only partly, so the lead's coefficient on it falls and the record it overtook gets a positive coefficient. Wrong. ARC reveals fully: the second's own coefficient is 60.1 when it was an overtaken largest against 61.8 when it never was one, and the record it overtook gets -17.0 (negative, -0.28 per unit of the second's own); v23 with ARC's settings 1.01 against 1.00, and -0.13. So ARC's hiding neurons do carry all of the hiding that a new largest removes. The smaller hiding nudge in ARC must then come from how long a change to the hiding neurons lasts: ARC's hiding wave shrinks about 6% per step and is re-driven every step, mostly by the store (robust_twin/DESIGN.md), so a nudge at step 5 partly heals before the readout; v23's clock keeps a nudge forever. Re-driving v23's clock needs a second source of the largest's jump and age; v23 has none (ARC's store carries the largest's age as a slow wave, gao/n2_age_where.log).

Persistence test (`hide_persist.log`): a 10% turn-down of ARC's hiding neurons at step 5 leaves only 2.1% one step later, then 1.7, 1.4, 1.3%; in v23 the clock keeps the full 10% to the end. So ARC's hiding neurons are rebuilt almost entirely every step from other neurons (store, memory, input), not carried by their own rotation; the persistent part of the hiding wave lives elsewhere, presumably in the memory ('one rotating system'). Consequence for a twin: matching the hiding nudge means hiding neurons that are recomputed each step from memory and store. In v23 the memory slots hold votes and wave lobes together, so a hiding signal rebuilt from them would pick up votes, which is what ARC's hiding neurons are known to hold (account item 8: a share of the votes cast since the largest arrived, erased at the next new largest). Not built yet.

## twin16 against the post's checks (2026-10-19, main session)

- Matched into ARC's slots by exact relabelling and rescaling (`match_twin.py`, `.log` -> `twin16_matched.npz`; the linear-route neuron takes slot 9, ARC's late gate, since the twin has none). Group patching (`../../hilton/faithful/faithful_twin16.log`): store and detector 85.4%, slot 9 (linear route) 76.8%, keep gate 70.3% (v19: 41.7%), memory 24.6% (v19: 21.3%), hiding 15.6% (v19: 8.7%). Gap as noise 10.0% (chance). So it responds to nudges like ARC but its hiding and memory still hold different values.
- Weight noise (`../noise_control/run_twin16.py`, `.log`; relative noise as for v19/v21): at 0.0001 already 8-11% accurate (chance), 350 maps; like v19 it cannot be a noise control. Clean: 87 maps, tree 99.1% vs guess 91.2%, third wins 8.8%.
- Exact ties: on 1.3% of random lists the twin's top two scores are equal to rounding error (only 14% of these have a negative second), so argmax decides by float rounding and index order. The relabelled copy agrees on 99.4% of lists and differs only on such ties; accuracy 93.0% unrelabelled, 92.6% relabelled (4M lists each). Untraced.
- Post switched to twin16 (decision of 2026-10-19). Figure 2 from `paper/figures/twin16_matched.npz` (roles added); Figure 4 point 87 maps, 10% unexplained; Figure 5 hand-built bar 6-17% (`../../hilton/topdown/mistakes_twin16.log`: twin16 repeats 16.9% of ARC's mistakes, v19 7.2%, v21 6.2%); redraw-seven flips 1.2% (`../context/controls_handbuilt.log`).
