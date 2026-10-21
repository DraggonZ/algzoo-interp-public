# A robust twin: design notes

Status (2026-10-18): **v21 meets both bars.** 28 neurons (approved), every weight from a written rule. Accuracy 95.9% on 1,000,000 lists (trained network 95.3%); under 1% noise on every neuron at every step 89.4% (three seeds; trained 90.5%); 0.1% 95.1% (trained 95.1%); 3% 78.0% (trained 73.4%). v20 (22 neurons): 91.4% and 82.5%. v18 at trained scale: 92.1% and 56.7%.

## Goal and success criteria

A hand-built 16-neuron network on the trained network's layout, every weight from a written rule (no optimization of any kind), that is about as accurate as the trained network and about as robust. Provisional bar: accuracy of at least 93%, and at least 85% under noise of 1% of each neuron's typical value on every neuron at every step (the trained network: 95.3% and 90%). v18 at trained scale: 92.1% and 57%.

## What makes the current twins fragile (measured)

- **A structural tie.** v18 hides only a record's jump d over the previous largest p; the record's slot shows (1 − c) p openly. When the second-largest is that overtaken p, the largest's position ties with it, and only a tie-breaker of 1% of a vote decides. On 37% of lists the runner-up is the largest's position, by a median lead of 0.07 (`look_scores.py`). The trained network hides most of a record's value: the largest's position scores a median 10.5 against 26.6 for the second.
- **Sensitive timer neurons.** Per-neuron first-order sensitivity to 1% noise, relative to the lead (`sensitivity.log`): v18's flip 2.1 and clocks 1.0-1.3; the trained network's cancelling neurons 0.10-0.19.
- **A small wave on a large level.** In the hiding group, |wave| / |level| has median 0.24 in the trained network and 0.039 in v18 at trained scale, which partly switches on 17% and 1.3% of live steps (`wave_level.log`). Noise scales with a neuron's whole value, the signal is the wave, so the ratio sets robustness. A larger ratio means the level no longer keeps the neurons positive, so they clip: that is the trained network's partial switching.
- Not the lever: raising the vote gain against the level, because v18's wave and readout scale with it (`vote_gain_probe.log`).

## Step 1: v19, hiding more of the record (done)

Builder `algzoo.handbuilt_v19.build_v19(kappa=...)`; kappa = 0 is v18 exactly (checked). The keep gate writes (1 − c)(p − kappa d) for a record, and the wave carries (1 + kappa) d (clock, flip, the memory's backward half, the flag's missing peak), so a revealed record's vote is still exactly (1 − c) x. The gate gets a level kappa (1 − c) m1 so the negative part is not clipped, subtracted where the gate is read. Checked on constructed lists (`v19_simple.py`).

Results (`v19_check.log`, `sens_v19.log`), trained scale, kappa 0.5 against 0: the tie is gone (where the largest is the runner-up, median lead 1.5 instead of 0.07); sensitivities halve (flip 0.9, clocks 0.6, store 0.17); noise 1e-3 85.5% (76.9%), 1e-2 61.3% (56.7%); but accuracy 87.6% (92.1%). Cost: only the wave's peak is compensated, so the spill onto neighbouring positions grows by 1 + kappa (`v19_where.py`). At v18's gains: 94.3% against 96.0%.

So step 1 removes one cause and halves the timer's sensitivity, but the twin stays far from the bar: the wave is still small next to the level.

## Step 2 findings so far

- **Late wipe is ruled out.** Wiping the timer one step after a record, through the detector, would let its level stay small without clipping. But when the largest arrives at the last step (10% of lists), the previous largest is the answer and is still hidden: nothing reveals it, and the linear readout cannot undo hiding conditionally. So the timer must be wiped through the input at the record's own step, and its level then depends on the gap m1 − x.
- **Clipping from ordinary numbers costs little; failed wipes cost a lot.** v19 at trained scale: removing the timer's clipping on non-record steps gains 0.3 points; an idealized perfect wipe at records gains 3.6 (kappa 0) to 4.5 (kappa 0.5) points (`clip_cost.log`). A wipe fails when the new record's jump is small against the old record's wave.
- **The flip neuron cannot be spared.** Without it (beta 0) v18 loses about 4 points; with kappa > 0 the unsharpened spill onto the next position exceeds the revealed record's own vote, and accuracy collapses (`no_flip.log`). So a carrier neuron for a late cleanup is not available in 16 neurons.
- **The trained network survives tiny jumps (corrected 2026-10-18: it does not wipe them fully; it has more margin).** On lists whose second-largest is an overtaken record, by relative lead of the largest: jump under 2%, trained 82% against v18 36% and v19 30%; 2-5%, 95% against 51% and 42%; 5-10%, 98% against 78% and 64% (`small_jump.log`). At record steps with a jump under 1% its four cancelling neurons fall from a summed 37 to 0.3, with pre-activations already negative (`trained_wipe.log`). Correction (see `../bridge/README.md`, corrections): all four are off on only 4.5% of records with jumps under 1%, and the last neuron switches off at a median 1.01-1.14 times the largest, so the wipe is not complete; what differs from v18 is the margin on those lists (median 9.8 against 0.1), not the wipe.

- **A lower wipe threshold trades one failure for another** (v19 `delta`: the clock is wiped when x > (1 − delta) m1; `delta_probe.log`). Trained scale, kappa 0: delta 0.05 lifts small-lead overtaken records from 44% to 79% but drops lists whose second is a kept number within 10% below the largest, after it, from 97% to 86%; net best 93.1% at delta 0.02 (from 92.0%), 1% noise 58% (57%). The twin's clock holds its own phase, so an early wipe destroys the hiding for good.
- **The trained network's cancellers regain their size after an early silencing, but not their pattern at once.** When a kept number within 5% below the largest silences them (summed 1 against 88 without it), they are back one step later at about the right size, 13° off their undisturbed direction, then 8° and 6° (`trained_recover.log`). Corrected 2026-10-18 (`../bridge/recover_redrive_wave.log`): that angle is dominated by the level the four share. With each state's mean over the four removed, their turning pattern is 72°, 66° and 44° off one, two and three steps later; so the size comes back at once and the pattern only slowly. Without the store's timing signal it stays 90° off for three steps.
- **What re-drives them.** Their wave comes 81% from themselves and 19% from the store and detector, and the store's push points along the wave (`trained_drive.log`). The trained store also tracks where the largest is (`gao/n2_age_where.log`; an extra found in all 22 networks). So the trained hiding wave is a damped rotation (shrinking 6% per step) continually re-driven by the store, not a free-running rotation started once by a record as in the twins. Errors, clipping and early silencing then heal within a few steps.

- **Hiding by subtraction alone needs a sharp pattern** (`hide_shape.log`, idealized scoring with no network: every positive number votes, the largest's vote is subtracted as m1 x shape around its position, and the subtraction vanishes when it is overtaken). v18's wave shape gives 33%, a 45-degree pair alone 34%, a sharp pattern 99%; three rotation pairs plus the flip 95% but need seven neurons instead of four.
- **The trained network cancels a broad wave with a broad wave** (`trained_shape.log`, final-score parts by offset from the largest, per unit of m1, centred): cancelling neurons −10.8, −21.5, −14.0, +3.4, +18.3, +16.9 at offsets −1 to +4; the rest +15.1, +31.9, +19.5, −0.5, −12.2, −10.2; total +2.0, +9.1, +3.1, +3.3, +5.0, +3.3. So its timer in the rest of the network is itself a broad cosine around the largest, as v18's copies are, and the largest's position is left at about a third of a vote. Structurally the two designs agree; the differences are the four listed above (how much of the record is hidden, the wipe threshold and recovery, the wave-to-level ratio, the erasure).

## Design consequences (current view)

1. Wipe through the input, with a threshold below the largest, so every record wipes (forced by last-step records).
2. Make the hiding a damped rotation re-driven every step by a signal that knows the record's age, so early wipes and noise heal. In 16 neurons the candidate driver is the store, as in the trained network; the twin's store would then carry more than the largest so far, and every place that reads it would need compensating.
3. Keep the wave large against the level (the trained ratio is about a quarter); clipping is then routine, and the rules must hold through it.

Each of these departs from v18's exact-linear bookkeeping, where the memory's copies match a clock that never fades. A twin built on these lines would be a new design, not a v18 variant.

## Step 2: a timer that works through clipping (next)

The level must come down until the wave is about a quarter of it, which makes the timer clip on roughly a sixth of steps, as in the trained network. The rules must then say what a clipped timer means. What the trained network's clipping does (measured, `gao/README.md`): it routes the vote of the number right after a record into the hiding neurons, where the overtaking erases it; it holds later kept votes down until an overtaking; and it leaves a history in the hiding neurons' direction. Open design questions:

1. Can the spill be kept small while the wave grows? A sharper wave needs more rotation components or an erasure like the trained network's; the erasure would come with clipping.
2. Can a clipped timer be described by a rule simple enough to write weights from? Candidate: treat the clip as storing min(level, wave) per neuron and make the readout of the newest slot read that loss as the arriving number's vote.
3. Wipe at a record: the timer is wiped through its input (level g (m1 − x)), which is why its level falls for every positive number. A late wipe through the detector would keep the level up but breaks records at the last step.

Files: `look_scores.py`, `sensitivity.py`, `wave_level.py`, `vote_gain_probe.py`, `v19_check.py`, `v19_where.py`, `v19_simple.py`, `sens_v19.py` and their logs; builder in `/data/output/src/algzoo/src/algzoo/handbuilt_v19.py`.

## v20 (built)

Builder `algzoo.handbuilt_v20.build_v20`; rules in its docstring. Ideas: the timer's level is LP m1 with no input term, so it only has to cover the wave (LP = 1.1 K); it is wiped one step after a record through the detector and re-started two steps after through a carrier of the jump; a reveal group (the timer's pre-activations plus GQ (x − m1)) is silent except at records and hands back the hiding when the largest arrives at the last step; the keep gate writes every positive number's full vote at once, so a record's own slot needs no compensation and its reveal is exact; nine memory slots. Checked on constructed lists (`v20_simple.py`).

Results (`v20_noise.log`, 50,000 lists): clean 91.4%; noise 1e-3 90.4%, 1e-2 82.0%, 3e-2 66.5%; timer noise at the last step only 88.9% (v19 75.3%, trained 94.4%). Most sensitive neurons now: store 0.16, newest memory slots 0.11-0.14, timer 0.07-0.09 (trained cancellers 0.10-0.19). Fixes on the way: the memory level set to the rule's minimum, 0.85 K A0 (from 1.2); the flag's gain lowered from 10,000 to GQ, because cancelling a gain of 10,000 against the detector at the readout made 0.1% noise flip position 8 on 6.7% of lists (`v20_ties.py`).

Remaining: the spill of revealed records onto neighbours (0.3 of the jump two positions away; the trained network erases the worst of it), and the store, an integrator whose noise random-walks into the memory levels.

Noise on one group at a time (`v20_groups.log`, 1% on every step, lev 1.0, lam 12.15): memory 86.0%, store 86.9%, timer 87.7%, detector 90.5%, keep gate 90.4%, carrier and flag 90.6%, reveal 91.1% (clean 91.4%). The memory and the store cost most, and both through the memory's level: it exists only so that the negative lobes of the timer's compensation are not cut off (it is set by the deepest lobe, 0.805 of the wave), noise scales with it, and the store, an integrator whose noise random-walks, supplies it to each new slot. The compensation is the spill by another name. So the next gain needs a sharper hiding pattern (smaller negative lobes, less spill, a lower memory level), at a cost of about six more neurons (a 90-degree rotation pair in the timer and in the reveal group).

## v21 (built; meets both bars)

Builder `algzoo.handbuilt_v21.build_v21`. v20 plus a 90-degree rotation pair in the timer (E0-E2) and in the reveal group (QE0-QE2): 28 neurons. The hiding pattern is (2 cos 45d + 2 cos 90d + cos 180d) / 5, the sharpest the timer's three frequencies allow (a delta minus the constant and 135-degree parts): peak 1, lobes +0.08, −0.20, −0.48, +0.20 at d = 1-4. So the spill one place from a revealed record falls from 0.14 to 0.08 of its wave, and the memory level from 0.81 to 0.49 K A0. Constants: A0 10, kappa 0.5 (a judgement call: the hidden largest trails an overtaken record by half its jump), a = s = 10, wipe weight W = 1e4 on the detector (the timer is inactive when it acts), reveal and flag gain 10 K, position 9's veto 1,000.

Results (`v21_simple.py`, `v21_noise.log`, `v21_confirm.log`): accuracy 95.87% (1,000,000 lists); noise 1e-3 95.1%, 1e-2 89.3-89.5%, 3e-2 78.0%; timer noise at the last step only 94.5% (trained 94.4%). Most sensitive neurons: store 0.11, newest memory slots 0.05-0.09, keep gate 0.05, timer 0.03 (trained: cancelling neurons 0.10-0.19).

What made the difference, in order: hiding more of each record (v19's kappa) so the largest's position no longer ties with an overtaken record; a timer whose level has no input term (late wipe, carrier, reveal group), so its level only covers the wave; a same-step keep gate so a record's slot needs no compensation; no large gains cancelled against each other anywhere (the flag); and a sharper hiding pattern, which cut both the spill and the memory's level.

Not copied from the trained network: v21 does not clip (its timer never partly switches), does not erase, and uses 28 neurons with separate jobs. It shows that a hand-built network can be as accurate and as robust as the trained one; it does not show how the trained network gets both in 16.
