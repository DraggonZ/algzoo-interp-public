# Findings index: ARC's 16-neuron second-largest network

This is an index of what has been measured, with a link to the note that holds each result. Write your own plan first. Then check it against the facts below, to avoid rerunning what was measured. The interpretations section is optional and comes last on purpose.

Compiled 2026-10-21 from the folder READMEs and NOTE.md. Links are relative to this folder. Numbers are rounded; the linked note has exact values and sample sizes.

**The network.** ARC's AlgZoo network has 16 ReLU neurons, no biases and 432 weights. It reads 10 standard-normal numbers, one per step, and names the position of the second-largest. It is right on 95.3% of lists. Update: h_0 = 0, h_{t+1} = ReLU(W_hh h_t + W_hi x_t), scores = W_oh h_10. Groups: store 2, new-largest detector 4, keep gate 5, late gate 9, hiding neurons 1, 6, 7, 8 (called "cancellers" in older notes), memory 0, 3, 10-15.

**Hand-built versions** (every weight from a written rule): v2 (one lasting vote); v4, v5 (exact, a different algorithm); v7 (ARC's roles neuron by neuron); v18, "the twin" (ARC's layout, sent to Hilton); v19 (v18 hiding more of each record, kappa 0.5); v20, v21 (22 and 28 neurons, built for robustness); v22a (50 neurons); v23 (ARC's corrected mechanism, 16 neurons); v24 (v23 plus an exact runner-up, 17 neurons).

## 1. Facts

### The network and its groups
- 32 of the 432 numbers are exact free choices: 16 neuron scales and a 16-number shift common to all readout rows. The KL curvature has exactly 32 zero directions, and the other 400 span 16 factors of ten. [parameters/README.md Exp 17](parameters/README.md)
- Silencing the hiding neurons after the largest arrives makes its position win on 70% of lists (any four other neurons: 5%). Restoring them at an overtaking leaves the network right on 16% (others: 92%). [NOTE.md account item 4](NOTE.md)
- Among all 495 other sets of four neurons, silencing gives a median of 6% and at most 24%. [NOTE.md "Two claims"](NOTE.md)
- With a fixed on/off pattern, each step is exactly linear. All 16 neurons use about 1,200 patterns, the memory 104, the hiding group 16. Two million lists use about 930,000 distinct on/off sequences. [superposition/README.md "switching linear system"](superposition/README.md), [level2/deduce](superposition/level2/deduce/README.md)
- Roles stated for whole groups predict 28 weight signs, 27 correctly. Those 28 weights (6.5%) carry 53% of the sensitivity to 1% nudges. [parameters/README.md Exp 7](parameters/README.md)
- Scaling all inputs never changes the answer. Shifting all inputs by −0.3 drops accuracy to 92%, so there is no shift symmetry. [parameters/README.md Exp 10](parameters/README.md)
- Ignoring negative inputs entirely costs KL 0.036 and 0.1 point. Shrinking the whole input vector by two-thirds for negative numbers costs KL 0.012. [parameters/spd/README.md](parameters/spd/README.md)

### Store and detector
- The store holds about 17.6 × the largest, one step behind, and fades about 3% per step. Removing the fade costs 10 points; firing the detector at the fresh store costs about 4. [hilton/README.md store_rungs](hilton/README.md)
- The store's profile by age rises for four steps, then falls; the memory and hiding neurons shape it. A 93-number store rule agrees with ARC on 96.5% of lists. [hilton/topdown/README.md](hilton/topdown/README.md)
- The store also carries when the largest arrived (a slow wave, up to about a sixth of its value). Hiding and memory neurons use it. Moving the largest's place costs accuracy in all 22 trained networks. [gao/README.md](gao/README.md)
- "Detector row = −0.55 × store row" costs 1.4 points (KL 0.066), over the budget. [parameters/README.md Exp 9](parameters/README.md)
- The store's self-weight is the stiffest single weight (15% of curvature). Five weights hold half of it. [parameters/README.md Exp 18](parameters/README.md)
- A kept vote enters directly and through the detector one step later, about half each. A last-step number has only the direct route: it wins 58% of near ties where it is truly second 50%. A 1% turn-down of the last input removes this bias at 0.04 point of accuracy. [superposition/steer/README.md](superposition/steer/README.md)

### Gates
- The keep gate is on after 99.8% of records. For other numbers its 50% point is 0.69-0.78 of the second-largest so far. The late gate is on after 97.5% of records and after 0.6% of other numbers. [gates/README.md §1](gates/README.md)
- A gate sees the arriving number only through exact hinges on the detector and the four hiding neurons. [gates/README.md §2](gates/README.md)
- Forcing the keep gate's on/off to "a record, or ≥ c × second-largest" leaves 95.3% for any c from 0.65 to 0.8. Its value must be right to about 5% (0.7-0.9 points). [gates/README.md §3](gates/README.md)
- Rules: the late gate fits in 7 numbers (−0.6 point); the keep gate's value needs about 550 (−0.6 to −0.7). [gates/README.md §3](gates/README.md)
- Keep gate silenced: 21%. After a kept number its write gives the vote and re-hides the part of the largest that number uncovered. Silencing only that write drops those lists from 95% to 0.1%. [gates/README.md §4](gates/README.md)
- Forcing the keep gate off below the exact second-largest raises accuracy from 95.4% to 95.9%. The network does not have that exact quantity. [gates/README.md §4](gates/README.md)
- Without the late gate's push on hiding neurons 1 and 6, lists whose answer is an overtaken record fall from 95.5% to 49%. [gates/README.md §4](gates/README.md)
- Zeroing the late gate one step after a near-tie overtaking leaves 8% right instead of 94% (size-matched random control 88%). [hiding/README.md §3](hiding/README.md)
- The late gate's push reaches neurons 1 and 6 two steps after a record. Fixing this delay recovered the 9→6 weight (−0.63 against ARC's −0.64). [parameters/README.md Exp 22](parameters/README.md)
- Notes disagree on gates elsewhere. One count finds late gates in two other networks (one doing ARC's job) and a keep gate in one. Another finds record-triggered neurons in four. [gates/README.md §6](gates/README.md), [hiding/README.md §3](hiding/README.md)

### Hiding neurons
- Each hiding neuron's input is c × (reference − arriving number), with c about 11-13. Neuron 7's reference is the largest. The others' references are mixes of the previous and current largest, set by age. [hiding/README.md §1](hiding/README.md)
- One step after a record, neurons 6 and 8 compare the arriving number with the previous largest; 7 and 1 compare it with the new largest. [bridge/README.md](bridge/README.md)
- A clipped neuron's excess passes to the others a step later in ratios the weights fix (6→1 at 1.04). So the references follow a running second-largest. [hiding/README.md §1](hiding/README.md)
- A written rule with 648 fitted numbers replaces the four at 94.3% (network 95.2%). Near-tie overtakings get 84% (network 89%). [hiding/README.md §1](hiding/README.md)
- The four's own block keeps one rotating pair, 43° per step, shrinking 6%. The arriving number's push falls in directions that die within a step. [hiding/README.md §1](hiding/README.md)
- They are rebuilt every step: a 10% nudge leaves 2% one step later. [superposition/twin/README.md](superposition/twin/README.md)
- Some but not all of the four are off on 14% of steps. A positive number below the largest causes this on 63% of steps one step after a record. [gao/README.md partial_trigger](gao/README.md)
- Undoing that partial clipping costs 24 points. Random-sign changes of the same size cost as much at the steps after a record. [hiding/README.md §2](hiding/README.md)
- Blindness: every neuron except the gates reads the hiding neurons at right angles to their input weights, so the arriving number cancels. Imposed on 13 rows it costs KL 0.019 (0.54 point). A random direction costs 69-86 points. [parameters/README.md Exp 9-11, 19](parameters/README.md)
- ARC's store weights into the four are −0.048 to −0.051 times their input weights, within 3%. No rule uses this yet. [parameters/hiding_rules/LEDGER.md round 5](parameters/hiding_rules/LEDGER.md)
- Formulas for the four miss the true pattern by 4.7° in direction (55.9%). Turning the true pattern by 1° gives 87%; the target is about 0.5°. [gao/README.md direction_vs_size](gao/README.md)
- Their own previous values bring a formula to 0.3°. The missing history is their earlier partial switching. [gao/README.md "missing history"](gao/README.md)
- Under Hilton's noise (each weight + N(0, 0.0025²)), the 64 weights into the hiding neurons carry 3.6 of the 4.9 lost points. [hilton/README.md](hilton/README.md)

### Memory
- With all 8 on, the memory block turns at 0° (×1.02), ±38° (×0.82), ±84° (×1.00), ±127° (×0.86) and 180° (×0.92) per step. [memory/README.md §1.1](memory/README.md)
- Without any memory switching, accuracy is 44%. [memory/README.md §1.4](memory/README.md)
- Neuron 0 is off exactly two steps after every record. That rule in place of its ReLU, with the other neurons unchanged, gives 93.6% (real 95.4%). [gao/README.md n0_rule](gao/README.md)
- Neurons 11 and 10 switch off when the largest is 5 and 6 steps old (85-97%). The slow 38° mode does most of the turning that sets these ages. [parameters/memory_register/README.md §2c, §5b](parameters/memory_register/README.md)
- Neuron 15 is off when the summed waves of the numbers so far fall below zero. That rule costs 0.3 point; no switching costs 7. [memory/README.md §2](memory/README.md)
- Switching rules fitted per step and age give 93.6% but hold about 8,300 numbers. A plain step-by-age table gives 87%. [memory/README.md §1.5](memory/README.md) **Superseded** (2026-10-21): a compact rule on the list with about 50 fitted numbers gives 94.0% against 95.4% on fresh lists (about 80 numbers: within 1.0 point). Neurons 0, 11, 10 off at fixed ages of the largest, 12 always on, 13 and 14 off only at the first positive number, 3 by a score of the largest's and second-largest's ages, 15 by summed waves (the only one needing sizes, and most of the loss). [switching/README.md](switching/README.md)
- Neuron 3 is a memory coordinate (about two-thirds of the 127° rotation) that also carries the gap between the largest and the second-largest (hiding 1 minus hiding 8 ≈ age factor × gap), which makes the hiding of the largest grow with that gap. Replacing it by the gap formula costs 2.8 points; by another list's gap, 25. Not a partial ninth slot. [neuron3/README.md](neuron3/README.md)
- A kept number's vote reaches the memory two steps after it is read, through the keep gate. A dropped number writes 1-2% as much. [parameters/memory_register/README.md §2](parameters/memory_register/README.md)
- On average a vote's shape depends on its age, not its position (cosine about 0.9). Its phase advances by the mode angles (median deviation 9-12°). [parameters/memory_register/README.md §2b](parameters/memory_register/README.md)
- On one list the phase spreads 54-79°. All memory neurons stay on from step 4 to 6 on only 1.3% of lists. [parameters/memory_register/README.md §2b](parameters/memory_register/README.md)
- At steps 4-6 the state predicts each vote's final profile only to 50-80% relative error. Knowing the largest's age barely helps. [parameters/memory_register/README.md §2d](parameters/memory_register/README.md)
- The readout for position k+1 is the row for k pushed one step through the dynamics (cosine 0.97). The relation fails into positions 8 and 9. [memory/README.md §1.3](memory/README.md)
- Used as a weight rule, that readout relation costs 30-37 points. [parameters/README.md Exp 4](parameters/README.md)
- Votes are not fair: own-position sensitivity varies ±25% and later votes read stronger. Imposing equal votes costs 35-55 points. **Supersedes** "fair within about 10%" ([NOTE.md items 5, 7](NOTE.md)). [parameters/README.md Exp 12](parameters/README.md)
- Kept votes fade about 20% per step; revealed records fade about 2%. Slope over scatter stays at 1.7-2.4 everywhere. [parameters/README.md Exp 15](parameters/README.md)
- Snapping the angles to multiples of 42° costs KL 0.97 (84%). Moving each angle 1° costs 0.069; 0.25° costs 0.004. **Supersedes** the 1° precision wall of [superposition/README.md "precision wall"](superposition/README.md) (2026-10-18). [parameters/memory_register/README.md §3](parameters/memory_register/README.md)
- Refitting the readout and the memory's incoming rows (232 weights, each moved 1-15%) brings the 42° snap to KL 0.006 (output fit) or 0.023 (states fit). The fast pairs need no precision: snapped alone, KL 0.002. **Supersedes** "a quarter of a degree" for all angles (step 3, 2026-10-20). [parameters/memory_register/README.md §5](parameters/memory_register/README.md)
- The hiding neurons' 32 weights into the memory do most of that compensation (KL 0.012 with the readout). [parameters/memory_register/README.md §5c](parameters/memory_register/README.md)
- Replacing the 38° mode by its mean at the readout leaves 38%, the largest loss of any mode. [memory/README.md §1.3](memory/README.md)
- With 8 dimensions, a vote model on a regular 45° grid confuses positions 0 and 8 (77%); ARC's uneven angles keep votes nearly sharp out to six positions and push the confusion to the farthest ones (86.5%). [bridge/README.md ideal_angles](bridge/README.md)
- An overtaken record barely reaches later positions on average (−9 to 0 per unit, against +44 at its own). Single later positions still get up to +28. [memory/README.md §3](memory/README.md)
- Neurons 3 and 15 are small differences of large terms (input 4-5% of the summed term size). Neuron 3 helps positions 1, 5 and 6. [bridge/README.md](bridge/README.md), [memory/README.md §2](memory/README.md)
- Six memory neurons are 86-97% linear in v18's memory. Random rotating memories fed the same writes reach 81-96%. [bridge/README.md memory_control](bridge/README.md)

### The erasure
- The number right after a record keeps its vote in the hiding neurons (40 per unit), so an overtaking wipes it. Putting it back makes the network name that number on 51% of such lists (normally 2%; random share 14%). [gao/README.md unerase, vote_in_cancellers](gao/README.md)
- About 10 of 22 networks erase a neighbour, wherever their revealed wave spills (rank correlation −0.68 at one step after, −0.93 at two). [gao/README.md erasure_across](gao/README.md)
- v18 lacks the erasure: 83% against 97% on those lists. [gao/README.md recheck_r5](gao/README.md)

### Close calls and mistakes
- Mistakes: 61% name the third, 15% the largest, 18% are lists with at most one positive number, 6% the fourth or lower. [NOTE.md item 6](NOTE.md)
- Mistakes differ by network: median correlation 0.02 across 22 networks, and 0.5% weight noise reshuffles them. [NOTE.md item 6](NOTE.md)
- Other trained networks share a median 30% of ARC's mistakes; an imprecise perfect solver 21%; hand-built networks 5-8%. [hilton/topdown/README.md](hilton/topdown/README.md), [superposition/verify](superposition/verify/README.md)
- ARC repeats 62% of its own mistakes when the seven smaller numbers are set to their conditional mean, and about 56% when they are redrawn. [superposition/verify/README.md](superposition/verify/README.md)
- The third wins 44% of exact ties, 25% at 1-2% gaps, 11% at 3-5%. For largest against second the zone is about 1-2% wide. [superposition/width_band/README.md](superposition/width_band/README.md)
- Per neuron, the pushes sum to about 30 times the decision on typical lists and 100-300 times at near ties. Per group the ratio is 1.5-2 typical and about 10 at near ties. **Supersedes** "a twentieth" ([family/README.md](family/README.md)). [superposition/README.md corrections](superposition/README.md)
- The winner's lead is a median 2.2% of the pushes on it. In hand-built networks it is near 0 (v18 about 4e-6). [gao/README.md margin_vs_push](gao/README.md)
- To cover half of all lists, ARC needs about 8,000 distinct second-over-third lead maps. Hand-built networks need 46-48. All 47 trained networks checked are dense. [superposition/siblings](superposition/siblings/README.md), [level2/deduce](superposition/level2/deduce/README.md)
- A tree on positions and two ratios predicts near-tie decisions at 86% (base 79%), but needs about 25,000 numbers. Up to 400 numbers stay at base. [superposition/context/README.md](superposition/context/README.md)
- A median of 11 (neuron, step) units could each flip a near tie with a 5% nudge. No single neuron's switching makes the steep pieces. [superposition/context/README.md](superposition/context/README.md)
- The runner-up at the moment a contender arrives moves the second-over-third lead 7-9 times more than other smaller numbers. [superposition/twin/README.md](superposition/twin/README.md), [maps](superposition/maps/README.md)
- One extra step with input 0 makes ARC name the position before the second on 94.5% of lists. [superposition/steer/README.md](superposition/steer/README.md)
- Slot 0's weakness runs through the 126° mode: removing 20% of it fixes those lists but costs 2.3 points elsewhere. [superposition/extreme/README.md](superposition/extreme/README.md)
- Ten thousand more training steps lift ARC to 96.0-96.4%, almost entirely outside its 20 most sensitive directions. Close calls stay dense. [superposition/finetune/README.md](superposition/finetune/README.md)
- ARC's readout is 12% away from the cross-entropy optimum for its own states. The optimum gains 0.34 point (rejected as an explanation: it optimises against the task). [parameters/README.md Exp 6](parameters/README.md)

### Precision and sensitive weight directions
- Only 28 weights can be zeroed together within KL 0.004; 56 are negligible alone. [parameters/README.md Exp 19](parameters/README.md)
- Rounding every weight on a 1% log grid costs 1.1 points. The 12-neuron memory-and-hiding block needs rank 11 of 12. [parameters/README.md Exp 8](parameters/README.md)
- A rule with random error per weight still needs this many weights copied from ARC: 105 at 1% error, 251 at 5%, 345 at 25%. These counts were checked in the network. [parameters/README.md Exp 18](parameters/README.md)
- Near ARC, 5-20 weight directions carry its accuracy and 50-100 its specific mistakes. From genuinely clean weights, about 100 of 432 must be restored. [superposition/README.md "weight directions"](superposition/README.md)
- ARC tolerates about 1% noise on its neurons (90.5% at 1%). Hand-built twins are 1,000-100,000 times less tolerant. [gao/README.md noise_tolerance](gao/README.md)
- Each memory angle 1° off costs KL 0.069; each shrink factor 1% off costs 0.030. [parameters/memory_register/README.md §3](parameters/memory_register/README.md)

### Other trained networks
- The readout-shift relation holds in 17 of 21 other networks (cosine 0.8-0.98). [NOTE.md item 7](NOTE.md)
- With an 85% cutoff, all 22 networks have neurons a new largest silences. Only 4 of 21 pass both wipe tests. [NOTE.md](NOTE.md)
- Networks that forget small numbers are more accurate (rank correlation 0.74); never-forgetters average 90.8%. [gao/README.md forget_across](gao/README.md)
- Accuracy tracks lead over pushes (0.75). Networks that switch more cancel less (0.80). [gao/README.md robust_across, switch_vs_cancel](gao/README.md)
- ARC's fully understood small networks barely cancel (lead over pushes 1.0 and 0.10, against 0.022 here). Cancellation rises with list length and falls with width. [gao/README.md cancel_grid](gao/README.md)
- 32-neuron networks tolerate final-state errors far better, but their neurons are no cleaner. Replacing the state at every step fails at every width. [superposition/README.md "Width series"](superposition/README.md)
- In 16-neuron networks the stiffest directions sit on weights into the hiding neurons (ARC: 70% of top-5 weight, against 15% of all weights). [superposition/README.md "Stiff and sloppy"](superposition/README.md)
- Which neuron does which job, the hiding group's size (3-6) and the memory's timing differ by network. [superposition/generality/README.md](superposition/generality/README.md)

### Hand-built networks
- v2 reaches 98.9% and v5 98.97%, exact, in 16 neurons, with a different algorithm from ARC's. [NOTE.md "Hand-built networks"](NOTE.md)
- v7 reaches 88% and fails on a record at the last step (11%). Exact sharp-switch designs with ARC's roles needed 17-18 neurons (not a lower bound). [NOTE.md](NOTE.md)
- v18 reaches 96.0%. At ARC's gain scale it gets 92.1%, and its input weights then match ARC's at cosine 0.99. [bridge/README.md](bridge/README.md)
- v18's loss at that scale comes from a structural tie (it hides only a record's jump), not from partial switching. **Supersedes** the partial-switching reading in the same note. [bridge/README.md corrections](bridge/README.md)
- v21 (28 neurons) gets 95.9%, and 89.4% under 1% neuron noise (ARC 90.5%). [robust_twin/DESIGN.md](robust_twin/DESIGN.md)
- v23 (16 neurons) gets 95.8% with 47 lead maps. Looping its memory back costs 11-25 points. [superposition/v23/README.md](superposition/v23/README.md)
- On ARC's nudge tests, v23 with constants read off ARC (93.0%) sits at 0.69 of a typical trained sibling's distance; v18 at 1.26. [superposition/twin/README.md](superposition/twin/README.md)
- v22a (50 neurons) hides the largest down to the runner-up and halves that part of the gap to ARC. [superposition/hide_runnerup/README.md](superposition/hide_runnerup/README.md)
- Patching v18's groups into ARC leaves 85% (store and detector), 75% (late gate), 49% (keep gate), 26% (memory) and 16-19% (hiding). [hilton/README.md faithful](hilton/README.md)
- A robust hand-built network noised down to ARC's accuracy keeps about 55 lead maps. [superposition/noise_control/README.md](superposition/noise_control/README.md)
- Training v21 further lifts it to 98.3% in 3,000 steps, and its close calls stop being predictable from the top three numbers. [superposition/train_from_handbuilt/README.md](superposition/train_from_handbuilt/README.md)

### Accuracy estimates (Hilton's challenge)
- Exact close-call integration along 720 lines misses by 0.65 points on 22 networks and 1.02 on 16 later ones; sampling at that cost misses by 0.1. [NOTE.md "Accuracy estimate"](NOTE.md)
- Top three exact and the rest at their mean, on a grid: ARC 95.32% against 95.319%; 33 networks RMSE 0.52. It costs about 10 million runs. **Supersedes** run 3's 0.94 ([../20261016_m16_10/](../20261016_m16_10/)). [hilton/README.md propagate](hilton/README.md)
- Tie-line quadrature at equal operations: held-out networks 0.40 against plain sampling 0.52, but quasi-random whole lists reach 0.45. [superposition/level2/cheap/README.md](superposition/level2/cheap/README.md)
- Tie-band widths by linear solves: 0.82 held out at about 1,700 runs, against sampling's 0.63. [hilton/cheap/README.md](hilton/cheap/README.md)
- A mixture-of-Gaussians forward propagation gives 95.7% to 95.0% at grids of 20 to 80, still drifting, at about 1e12 operations. A single Gaussian gives 39%. [superposition/level2/switching/README.md](superposition/level2/switching/README.md)
- With v18 as idealized weights, "gaps as noise" predicts 10%, which is chance. **Retracted** as a faithfulness measure. [hilton/README.md](hilton/README.md)
- Mean accuracy under weight noise is predicted to first order within 0.1-0.7 points, and each copy within 0.25. This uses gradients, not the account. [hilton/README.md](hilton/README.md)
- One exact linear map per arrangement estimates ARC 1.5 points low and other networks 4.3 low. [superposition/level2/deduce/README.md](superposition/level2/deduce/README.md)
- Modelling the seven smaller numbers as Gaussian gets RMSE 0.12 on 32 networks, at 171-445 million runs each. [superposition/level2/smallnoise/README.md](superposition/level2/smallnoise/README.md)
- The account's 35 structural constants computed from noisy weights predict copies' accuracy only at correlation 0.52. [superposition/README.md level2/noise](superposition/README.md)

### Surprise accounting and rules (bits)
- A lookup table of ARC's weights within KL 0.02 takes 2,369 bits, about 6 per direction. Each halving of the budget adds about 200 bits. [parameters/README.md Exp 20](parameters/README.md)
- Net bits: negligible set −94; blindness as a soft prior +50 (exact −668); signs about +10. [parameters/README.md Exp 20](parameters/README.md)
- Knowing the 64 hiding weights exactly would save 469 bits. [parameters/README.md Exp 23](parameters/README.md)
- Weights derived from the hiding rule, never fitted to ARC's weights, match ARC's rows at cosine 0.99-1.00. [parameters/README.md Exp 22, 25](parameters/README.md)
- The hiding rule went from −94 to +97 bits in five rounds of automated search (26 constants). **Supersedes** +56 after three rounds ([parameters/WRITEUP_levels.md](parameters/WRITEUP_levels.md)). [parameters/hiding_rules/LEDGER.md summary](parameters/hiding_rules/LEDGER.md)
- Mechanisms that paid: waves from mutual feedback; one shared store reference; a record's aftermath through the two gates (+62); restore links fixed by input-weight ratios (+28). [parameters/hiding_rules/LEDGER.md](parameters/hiding_rules/LEDGER.md)
- About 10 of the best rule's 26 constants restate ARC weights. The write-up says "about 17 of 30" for the round-3 rule, while the ledger's count for it is about 12. [parameters/hiding_rules/LEDGER.md](parameters/hiding_rules/LEDGER.md), [WRITEUP_levels.md](parameters/WRITEUP_levels.md)
- Round 4 read the previous-number term as a stand-in for the memory. **Superseded** in round 5: it returns through the detector and neuron 7. [parameters/hiding_rules/LEDGER.md round 5](parameters/hiding_rules/LEDGER.md)
- Restating the memory block by its eigenvalues and basis costs 314 bits net. An angle rule can save at most about 13. [parameters/memory_register/README.md §3, §5d](parameters/memory_register/README.md)
- A "compensated memory" score of +352 bits is an artefact: the rows read ARC's pre-activations back (leak check +821). [parameters/memory_register/README.md §5d](parameters/memory_register/README.md)
- Readout loop (2026-10-21, three rounds): readout rows predicted by the one-step push of a neighbouring row through the dynamics reach +21 bits (from uncovered rows), +56 (rows solved jointly between anchor rows) and +82.5 (a chain, each row from ARC's row below; chains allowed by an overnight ruling, pending confirmation). One fitted constant; leak checks pass. The memory block's own 64 weights are still unexplained. With the rule's rows in place the network gets 45%: the gain is in describing ARC's rows, not in a working replacement. [parameters/memory_register/LEDGER.md](parameters/memory_register/LEDGER.md)
- ARC's readout column sums are −1.1 to +0.6, against columns of size about 3-14. [parameters/memory_register/LEDGER.md round 3](parameters/memory_register/LEDGER.md)
- In the memory's slow plane, the largest's share points away from neurons 15, 12, 11 and 10 at ages 3, 4, 5 and 6, their switch-off ages; neuron 3 is an exception. Observation only. [viz/machine/README.md](viz/machine/README.md)
- Overnight loop, final (2026-10-21): about +172 bits against the fair yardstick (one typical size per group-to-group block, readout offset ignored; 1,916-bit list), audited and noise-corrected. Mechanisms: readout fair votes (most), hiding neurons as comparators reusing the detector's reference (about +28), the detector's row tied to the store's by the reset condition (about +5), a memory delay line. The old hiding rule's +97 is −125 against the fair yardstick. **Supersedes** the readout loop line above and the hiding-rule line. [parameters/AUTORESEARCH_NIGHT_20261021.md](parameters/AUTORESEARCH_NIGHT_20261021.md)
- Noise control: on 40 copies of ARC with weights shuffled within blocks, every merged rule loses bits (at most +7 before paying for its form, against +96 to +397 on ARC). Search charges: round 8's is ample, round 9's should be about 5 bits, round 15's about 8; round 14's choice does not beat noise. [parameters/noise_control/README.md](parameters/noise_control/README.md)
- Training: ARC's model predates ARC's standard recipe; the 37 same-recipe networks use AdamW with weight decay. Readout column sums are not conserved by that training, and final weights barely remember their initial values. [parameters/training/README.md](parameters/training/README.md)
- Universal across 37 same-recipe networks: the readout fair-votes relation (31 of 37; the first structure to form in training), a detector, neurons a record switches off, memory neurons switching at fixed ages, the third as the commonest wrong answer. Same form, particular values: memory angles and fades, switch-off ages (ARC's 5 and 6 are rare), store fade, hiding group size. Mostly in the most accurate networks: blindness (all of the best nine, 17 of 32 overall), the detector-store tie (4 of the best nine, 5 of 35). ARC sits at the tidy end of most measures. [parameters/universal/README.md](parameters/universal/README.md)

### Methods tested
- Simulating all 16 neurons from the account's formulas leaves 51%. v18 passes from its exact rules (96.0%, KL 0) but gives 48% at the account's level of detail. [gao/README.md](gao/README.md)
- On fragile twins, the right idea with one gain 2% off scores 58%, below a misreading at 69%. On a sturdier twin, slight errors score at or above misreadings. [gao/README.md twin_graded, robust_graded](gao/README.md)
- The shuffled-error control separated accounts from misreadings on ARC but misclassified on the twin. [gao/README.md shuffled_control](gao/README.md)
- The output-level resampling test of forgetting passes on ARC, on v18 (which keeps everything) and on v7. A nudge inside separates them: ratios 0.02, 1.0 and 0.000. [gao/README.md twin_forget](gao/README.md)
- Features found by resampling beat a matched count chosen without it by 8-11 points. KL-fitted formulas then reach 87% for the hiding group. [gao/README.md sim_rich](gao/README.md)
- One backward pass predicts the cost of a 2% error per neuron (rank 0.99). It misses group substitutions by a median 13 points. [gao/README.md first order](gao/README.md)
- Sparse autoencoders reconstruct ARC's states to 0.4% unexplained variance. Spliced back they leave 33-61% (last step only) or 22-33% (every step), and they match hand-built roles no better than the neurons do. [superposition/README.md sae](superposition/README.md)
- A dense transcoder at 0.24% unexplained variance still leaves 40% and 70%. So precision, not sparsity, sets the failure. [superposition/transcoder/README.md](superposition/transcoder/README.md)
- In v21, whose roles are known, an autoencoder recovers 8 of 26 roles, because the roles are dense. On a sparse toy the same recipe recovers 26 of 28. [superposition/planted/README.md](superposition/planted/README.md)
- Alignment search with "value at position s" variables reaches 17-25% interchange accuracy (random 10-15%). [superposition/README.md das](superposition/README.md)
- SPD from a random start uses about 30 of 48 recurrent pieces per step at KL 0.022. Its runs match at a median 0.4, and it needs about five times as many numbers as the weights. [parameters/spd/README.md](parameters/spd/README.md)
- SPD started from the neurons keeps them and gets lower loss on every term (KL 0.007, 86 uses against 101). It finds the late gate's timing and no multi-neuron mechanism. [parameters/spd/README.md](parameters/spd/README.md)
- At the step a bigger number arrives, an interchange test passes by construction, because the four hiding neurons are zero in both runs. [methods/README.md](methods/README.md)
- On v2, a complete additive vote model reproduces only 3-14% of v2's mistakes. [methods/README.md](methods/README.md)
- The naive memory nudge had the wrong sign (72% to 55%, predicted 87%). The corrected account's balanced nudge moved decisions the predicted way in 16 of 16 settings. [superposition/steer/README.md](superposition/steer/README.md)
- Of 16 later written predictions at the mechanism level, 9 were right, 4 partly and 3 wrong. [parameters/WRITEUP_levels.md](parameters/WRITEUP_levels.md)

## 2. Dead ends

| Branch | Why it closed | Fundamental or contingent |
|---|---|---|
| Superposed votes on a rotating clock, taken literally | 24-88% ([NOTE.md](NOTE.md)) | Contingent on that design |
| Simple vote models (per-input splits, fading plus fixed discount, broad old-largest votes) | 60-85%; ±50 pushes cancel to a few units ([NOTE.md](NOTE.md)) | Fundamental for models this coarse |
| Exact accuracy region by region | About 530,000 on/off patterns per million lists ([NOTE.md](NOTE.md)) | Fundamental: the count is the obstacle |
| Deductive per-arrangement maps | About 9,000 maps for half the lists, as many cases as sampling needs lists ([level2/deduce](superposition/level2/deduce/README.md)) | Fundamental for case-by-case arguments |
| First order for large group errors (trapezoid, all-ten-scores) | Misses by 10-15 points ([gao/README.md](gao/README.md)) | Fundamental to linearization |
| Single Gaussian over all 16 neurons | 39%: blurs the switches ([hilton/README.md](hilton/README.md)) | Fundamental for one Gaussian; a mixture works but costs about 1e12 operations (contingent) |
| Stratified sampling from close-call features | 2.2-2.6× fewer runs, then levels off ([gao/README.md](gao/README.md)) | Fundamental at that feature level |
| Blurring estimate bands by the small numbers' first-order effect | Worse (1.74 vs 0.78) ([NOTE.md](NOTE.md)) | Fundamental to linearizing at the mean |
| "Gaps as noise" with v18 as idealized weights | Chance (10%): the gaps are as large as the weights ([hilton/README.md](hilton/README.md)) | Contingent on idealized weights near ARC's |
| Memory switching from formula signs | Gains about 1 point; 8,300 numbers for −1.9 ([memory/README.md](memory/README.md)) | Contingent: needs about 8,300 fitted numbers |
| Keep-gate value rule | About 550 numbers ([gates/README.md](gates/README.md)) | Contingent |
| Memory rotation as a rule for its weights | An angle rule saves ≤13 bits; the 56 basis numbers hold the bits ([memory_register §5](parameters/memory_register/README.md)) | Contingent on finding a basis rule |
| Readout-shift and fair-vote rules for readout weights | 30-55 points; about 1% precision needed ([parameters Exp 4, 12](parameters/README.md)) | Contingent on rule precision |
| Low-rank or rounded weights | Rank 11 of 12 still costs 1.3 points ([parameters Exp 8](parameters/README.md)) | Fundamental: no generic compression |
| Zeroing negligible weights as a bit saving | −94 bits: naming them costs more ([parameters Exp 20](parameters/README.md)) | Fundamental under this accounting |
| Optimal readout as an explanation | Optimises against the task (decision) ([parameters Exp 6](parameters/README.md)) | Contingent on the standard |
| Partial switching as a calibration of kept votes | Undoing it breaks answers broadly ([gao/README.md](gao/README.md)) | Fundamental: wrong hypothesis |
| Clipping only needs to be seen consistently (memory fed unclipped values) | Random signs cost as much ([bridge/README.md](bridge/README.md)) | Fundamental: retracted reading |
| Sparse autoencoders and transcoders as replacements | Splicing fails; a dense transcoder fails too ([superposition](superposition/README.md)) | Fundamental at this precision in a recurrent net |
| Alignment search with input-value variables | 17-25% interchange accuracy ([superposition das](superposition/README.md)) | Contingent on the choice of variables |
| SPD from random starts | Dense, not reproducible ([spd](parameters/spd/README.md)) | Contingent on starts and settings tried (four) |
| v18 with a recirculating memory | −28 points on overtaking lists ([bridge/README.md](bridge/README.md)) | Contingent on v18's tall hidden wave |
| 2-D clock erasure in v18; over-hiding in the readout | Never beat the base; 66.7% ([NOTE.md](NOTE.md), [gao/README.md](gao/README.md)) | Contingent on design |
| Graded 16-neuron variants of v7 | All below v7's 88% ([NOTE.md](NOTE.md)) | Contingent |
| Crowding and redundancy explanations of the forgetters' advantage | Error gap present at 2-3 positives; robust networks cancel less ([gao/README.md](gao/README.md)) | Fundamental: rejected hypotheses |
| Accuracy estimate from the compact switching rule (2026-10-21): one on/off pattern per arrangement of the top three, exact integral | ARC 93.8% (true 95.32%) at 720 runs, sampling misses 0.8; 32 sisters RMSE 5.6 against sampling's 1.0. The pattern is wrong on about 1 list in 10, errors cancel in accuracy but not per arrangement. Running-age switching: 2.0 RMSE but about 100,000 linear pieces ([estimate/README.md](estimate/README.md)) | Fundamental for one pattern per arrangement; repeats the per-arrangement dead end |
| Hiding-rule search loop | Stopped after round 5 at +97 bits ([LEDGER](parameters/hiding_rules/LEDGER.md)) | Contingent: stopped, not exhausted |

## 3. Open questions

- What sets the memory's 56 basis numbers, which decide which neuron switches at which age?
- How does ARC's memory recirculate without the overtaken largest's wave coming back?
- What is neuron 3's job? What fully decides neuron 15's switching?
- Why is ARC's readout about 10% larger than the optimum for its own states?
- What is the 400th, nearly flat curvature direction?
- What do partial clips two or more steps after a record compute, and why do they halve kept votes?
- How does the late gate's push on neurons 1 and 6 become a score difference, and why must it land on 1 and 6?
- Why does the keep gate cut at about 0.75 of the second-largest rather than nearer 1?
- Which roughly 36 hiding-rule constants have a mechanism behind them? Why is neuron 8's row still twice the budget?
- Does the store-to-hiding relation (−0.05 × input weight) give a cleaner hiding rule?
- How does the trained network get wide margins and robustness together with small levels?
- Is any description of near-tie decisions shorter than the 432 weights?
- Can a mechanistic accuracy estimate beat sampling at equal compute?
- Why do few networks have single-neuron gates?
- Do missing biases force some memory neurons to hover near zero?

## 4. Interpretations (optional reading)

Each item is a reading, not a measurement. The facts it rests on are linked.

- *Interpretation.* ARC is clean in function, mostly clean in mechanism, and not clean in its values. Rests on sign prediction, the 2,369-bit table and the +97-bit rule. [WRITEUP_levels.md](parameters/WRITEUP_levels.md), [LEDGER](parameters/hiding_rules/LEDGER.md)
- *Interpretation.* The unit of explanation is the joint cancellation of groups and its leftover, not each group alone. Rests on the push-to-decision ratios and the failure of per-group short rules. [family/README.md](family/README.md), [superposition/README.md](superposition/README.md)
- *Interpretation.* The hiding neurons are comparators whose references records reset and their own clipping updates, so the group holds a running second-largest. Rests on the reference tables and the 6→1 pass-on. [hiding/README.md](hiding/README.md)
- *Interpretation.* The memory is mainly a clock of the largest's age, with kept votes riding on it in the same rotating code. Rests on age-locked switching and the decoder errors. [parameters/memory_register/README.md](parameters/memory_register/README.md)
- *Interpretation.* Both gates belong to the hiding system: the late gate tops up hiding after a record, and the keep gate re-hides what a kept number uncovers. [gates/README.md](gates/README.md)
- *Interpretation.* The erasure fixes each network's own spill, placed where the spill falls. It is a correlation across networks, not tested by intervention outside two of them. [gao/README.md](gao/README.md)
- *Interpretation.* Dense close calls are not forced by the mechanism. They fit one compromise: nine positions in eight memory dimensions, with the leftover spread everywhere. [superposition/v23/README.md](superposition/v23/README.md)
- *Interpretation.* On this task a simulation score measures how exact a description is more than whether it is right. [gao/README.md](gao/README.md)
- *Interpretation.* Heavy cancellation is how a network computes beyond its comfortable capacity. Untested; rests on cancellation across lengths and widths. [gao/README.md cancel_grid](gao/README.md)
- *Interpretation.* Training does not reward simple close calls; small context-dependent corrections raise accuracy. [superposition/train_from_handbuilt/README.md](superposition/train_from_handbuilt/README.md), [finetune](superposition/finetune/README.md)

## 5. Where things live

| Folder | What it holds | Dates |
|---|---|---|
| `../20261016_m16_10/`, `../20261016_run2/`, `../20261016_run3/` | Early account (clock, sliding votes), exact two-spike accuracy, run-3 estimator and techniques | 2026-10-16/17 |
| `NOTE.md`, `README.md` | Running note with the current account; run history of hand-built versions | 2026-10-17 to 19 |
| `methods/`, `lit_review/` | Methods checked on v2; prior work | 2026-10-17 |
| `gao/` | Simulation and resampling tests, cancellation, erasure, robustness, first order | 2026-10-17/18 |
| `bridge/`, `robust_twin/` | v18 matched to ARC; v19-v21 robustness designs | 2026-10-18 |
| `memory/`, `hiding/`, `gates/` | Per-group rules and interventions | 2026-10-18 |
| `hilton/` (`cheap/`, `topdown/`, `propagate/`, `faithful/`) | Faithfulness, noise prediction, sampling-free estimates, block-by-block tests | 2026-10-18/19 |
| `family/` | Near-tie leftovers, mistakes shared with other networks | 2026-10-18 |
| `superposition/` (many subfolders) | Dictionaries, switching patterns, steering, close calls, width, v22-v24, Level 2 estimates | 2026-10-18/19 |
| `parameters/` | Weight census, curvature, bits, blindness, hiding rules | 2026-10-19/20 |
| `parameters/hiding_rules/` | Search loop ledger and frozen scorer | 2026-10-20 |
| `parameters/spd/` | Parameter decomposition | 2026-10-20 |
| `parameters/memory_register/` | Memory decoding, angle precision, compensation | 2026-10-20/21 |
| `paper/` | Post drafts (do not edit) | 2026-10-17 to 20 |
