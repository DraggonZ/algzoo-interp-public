# The memory of ARC's network as a small linear machine (2026-10-18)

## Summary

The memory is neurons 0, 3 and 10-15. While none switches, it is exactly linear: each step it turns its contents and adds what other neurons write. It turns at five speeds: a slowly growing level (0° per step, times 1.02), three rotating pairs (38°, 84°, 127° per step, times 0.82, 1.00, 0.86) and a sign flip (180°, 0.92).

Most writes are read, at the end, about two positions before the step they were made at. The keep gate writes a sharp vote there. The hiding neurons and the detector write broad two-signed patterns, mostly in the slower rotations. The readout of positions 0-7 follows the turning; position 9 reads mainly the level.

Without switching, accuracy falls from about 95% to 44%. Neuron 0 is off exactly two steps after every new largest (19 points). Other neurons switch off where the current largest's turning wave crosses zero in them: 15 at ages 3-4 (steps since the largest arrived), 12 at 4, 11 at 5, 10 at 6. There other kept numbers decide.

Substitution: the linear machine, switched by rules written in the list's numbers, gives 93.6% on 200,000 fresh lists, 1.9 points below the network. It loses about 4 points when the last new largest comes at step 8 or 9, and under 1 on ordinary lists. But the rules hold about 8,300 fitted numbers. A compact on/off table by step and age gives 87%.

Neurons 3 and 15 each subtract two large, nearly cancelling inputs: hiding neurons 1 and 8 for neuron 3, memory neurons 11 and 10 for neuron 15. Neuron 15 adds about 22 per unit to a revealed record's score at positions 1-2. Neuron 3 adds 6-10 per unit to votes at positions 1 and 6. Neuron 15 is off when the waves of the numbers so far sum below zero. A rule from that sum costs 0.3 points; no switching costs 7.

An overtaken record barely reaches later positions. Per unit of its value, scores after the overtaking change by −9 to 0 on average, against +44 to +48 at its own position. The wave written while it was the largest is small and negative there; no later route dominates. Single positions still get up to +28 (position 8). Position 9 falls because the hiding neurons read the larger largest (or the jump shrinks). Zeroing the late gate's writes after the overtaking moves these numbers by at most 3: it is not the route. A hidden wave in the memory is about one vote tall (41 votes in the hand-built twin v18), so leftovers cost little. All 22 trained networks show it on average when the jump is fixed.

Most surprising: part of a revealed vote is written after the overtaking. For a record overtaken three steps after it arrived, hiding neurons 6 and 8 write about 60% of its vote two steps after the overtaking.

Open: a compact switching rule; the late gate's job at overtakings; why gaps change how votes are assembled.

## Words used here

- *Position* k: the k-th number of the list (0-9); its score is the network's output k. *Step* t: the update after reading number t.
- *Record* or *new largest*: a number above every earlier number and above 0. *Overtaking* at q: a record that beats an earlier positive largest at r. *Age*: steps since the current largest arrived (−1 while no number has been positive).
- *Kept* number: the keep gate (neuron 5) is on one step after it. *Hidden*: the hiding (cancelling) neurons 1, 6, 7, 8 cancel the record's score at its position while it is the largest. *Revealed vote*: the score a record keeps at its position once overtaken.
- *Per unit*: the change of a final score when one number grows by one (finite difference of 1e-4 or 1e-3, mean over lists unless "median" is said). Two nudges for an overtaken record: *record alone* (x_r grows, so the new record's jump shrinks) and *joint* (x_r and x_q grow together, so the jump is fixed and the largest grows), as in `../bridge/tail_cancel2.py`.
- *Writer*: a non-memory neuron (or the input) whose activity at step t−1 enters the memory at step t through W_hh[memory, writer]. *Exact split* (`decomp.py`): within the real run's on/off pattern the network is linear, so the final memory readout is exactly the sum of each write carried to the end through the real on/off pattern (checked to 1e-8 or better in every log that uses it).
- *Closed* substitution: the other neurons run on their own weights and read the machine; the machine reads them. *Hybrid*: the machine reads the real run's other neurons; the other neurons run on their own weights and read the machine. *Open*: the machine reads the real run's other neurons; scores are the machine's readout plus the real other neurons'.

## 1. The machine

### 1.1 Modes (`modes.py`, `modes.log`)

Observation. Eigenvalues of the 8x8 memory block (all on): 0° at 1.025, ±38.3° at 0.815, ±83.5° at 0.996, ±126.6° at 0.858, 180° at 0.918. This confirms the brief's numbers. The level mode lives mostly in neurons 0, 10, 11, 12 and 14. The 127° mode lives mostly in neuron 3 (and 15), the 180° mode in 13, 3 and 15.

Observation. With neuron 0 off, the block has no slowly fading level any more (largest real mode 0.22) and its rotations shrink faster (36° at 0.74, 88° at 0.91, 180° at 0.78). With neuron 15 off, the rotations become 61° and 115° at about 0.70.

### 1.2 What writes into the memory, and where it lands (`mode_writes.py`, `mode_writes.log`; `impulse.py`, `impulse.log`; `landing_real.py`, `landing_real.log`)

Observation (weights only: memory linear and all on, no feedback). For most writers, a unit of activity written at step t is read at the end mainly at position t−2, for every write step from 3 to 9:

| Writer | Main peak (per unit of its activity) | Main opposite lobe | Modes carrying most of it (shares) |
|---|---|---|---|
| keep gate 5 | +3.0 to +4.0 at t−2 | −0.8 to −1.8, three to six back | 84° 0.43, 38° 0.24, 127° 0.13 |
| late gate 9 | +5.1 to +6.6 at t−2 | −4.5 to −5.0 at t−5 | 38° 0.43, 84° 0.38 |
| hiding 8 | +2.7 to +4.0 at t−2 | −1.5 to −1.9 at t−3 | 84° 0.59, 127° 0.21 |
| hiding 7 | +2.4 to +3.3 at t−2 | −2.5 to −3.7 at t−5 | 38° 0.54, 84° 0.29 |
| detector 4 | +2.1 to +3.6 at t−2 (and +4.4 at position 9) | −2.7 to −4.7 at t−5 or t−6 | 38° 0.42, 84° 0.31, 180° 0.31 |
| hiding 6 | −3.6 to −5.1 at t−2 | +3.0 to +4.4 at t−5 | 38° 0.45, 84° 0.38 |
| hiding 1 | −2.2 to −3.3 at t−2 | +1.3 to +1.8 at t−3 | 84° 0.55, 127° 0.23 |
| store 2 | +1.5 to +2.0 at t−4 | −1.0 to −1.3 at t−3 | 180° 0.42, 127° 0.32, 84° 0.22 |
| input | +2.1 to +2.5 at t−1 | −2.1 to −2.7 at t−4 | 38° 0.63, 127° 0.19 |

The largest value at positions after t (positions up to 7) is at most 1.7 per unit, against peaks of 2-6.6.

Observation (real runs, exact split, `impulse.log`). What drives the writers, per unit of the number:
- The detector fires at the record's step (10.2 per unit of the jump).
- The keep gate fires one step after a kept number (10.1-12.6 per unit) and one step after a record (12.2).
- The late gate fires one step after a record (1.5).
- The store holds about 17.6 per unit of the largest, fading to 9.9 seven steps after the record.
- Hiding 7 holds about 13.4 per unit of a record for three steps from the next step on, fading to 7.4. Hiding 6 holds 6-13 from two steps on. Hiding 8 grows late (11.7 and 17.0 at steps 8 and 9 for a record at 2).

A record that stays the largest is written mostly by the hiding neurons, step after step: their memory writes are ±50-80 per unit at single positions and cancel among themselves. At the end, the memory part of a record at 2 is −1.7, 63.3, 58.8, 33.9, −7.2, −52.6, −67.8, −42.8, −35.3, 36.4 at positions 0-9. The other neurons' readout is nearly its negative, so the total stays within about ±17. A kept number at 5 gets its vote mainly from the keep gate's write two steps later (+27.9 at position 5; total vote 27.5).

Observation (`landing_real.log`). On real lists, the writes made at steps 5-7 are read 5 to 33 times more at or before their own step than after it (summed absolute size; for example 95.1 against 18.4 at step 5). Writes at steps 1-3 are read about as much after as before (32.1 after against 20.2 before at step 1). These are absolute sizes, which mostly cancel against the hiding neurons, so this does not say how a single vote spills.

### 1.3 How the readout reads the modes (`modes.log`, `mode_content.log`, `vote_by_neuron.log`)

Observation. The memory's own readout follows its turning for positions 0-7. The cosine of W_oh[k+1, memory] with W_oh[k, memory] times the block is 0.97, 0.99, 0.99, 0.99, 0.88, 0.99 and 0.96 for k = 0-6, against −0.19 to 0.67 for plain neighbours. It fails into positions 8 and 9 (0.41 and 0.31). Each rotating mode's readout phase steps by roughly its own angle from position to position (38° mode: 46-52° per position over positions 0-5; 84°: 77-87° over 0-4; 127°: 124-136° over 0-4). The 38° mode's readout size grows towards early positions (1.00 at position 0, 0.26 at position 8), which offsets its fading. The level mode is read mainly at position 9 (relative size 0.46, against 0.04-0.13 at positions 0-8).

Observation. Replacing one mode family's part of the final memory state by its mean (readout only) leaves this accuracy (network 95.3% on these lists):
- 38°: 38.2%, the largest loss;
- 84°: 54.7%;
- 127°: 60.7%, with the largest losses at positions 5 and 6 (58 points each);
- 0°: 78.0%, with 32 points lost at position 9;
- 180°: 79.5%, with the largest losses at positions 0-4 (18-32 points).

Observation. At the end of a list, a vote at each position is read mainly through one or two memory neurons. For kept numbers: neuron 10 at positions 1-3, 11 at 4, 12 at 5-6 and 0 at 7. For revealed records: neuron 14 at 0, 15 at 1-2, 10 at 2-3 and 11 at 4 (`vote_by_neuron.log`). Most of these readout weights are negative (for example −4.56 for neuron 11 at position 4), so a vote shows up as a dip in that neuron. The exception is neuron 15, whose weights are positive at positions 0-2.

### 1.4 Switching (`switching.log`, `offrate_age.log`, `delay_line.log`, `rule_cost.log`)

Observation. Removing the ReLU from all eight memory neurons gives 44.4% (network 95.4% on the same 100,000 lists). Removing it only from neuron 0 costs 19.1 points, almost all at age 2 (20.4 points). Removing it from neuron 15 costs 7.3 points, spread over ages 0-5. For neurons 3 and 10-14 the cost is 0.0-1.0 points each, but 5.8 together. So the brief's "only neurons 0 and 15 need to switch" understates the others' joint share.

Observation (`offrate_age.log`, `delay_line.log`). Off-rates by step and age:
- Neuron 0 is off on 100% of age-2 steps, at every step.
- Neuron 10 is off at age 6 (94-96%), neuron 11 at age 5 (85-97%), neuron 12 at age 4 (14-43%).
- Neuron 14 is off at age 0 early in the list (65-100% at steps 0-2).
- Neurons 3 and 15 are mixed over ages 0-6.

A record that stays the largest drives each of these neurons through a minimum near zero at those ages. Per unit of the record, the change at that age is 0.0 for neuron 12 at age 4, −0.0 for 11 at age 5, −0.1 for 10 at age 6, and 0.2 and 0.1 for 15 at ages 3-4. At their peak ages the same neurons change by +18 to +32.

Interpretation. Each memory neuron switches off where the current largest's turning wave crosses zero in it. Kept numbers and earlier records dip the memory neurons, so near a zero crossing they decide the sign. Neuron 0 is the exception: at age 2 its pre-activation is −3.96 on average (`rule_cost.log`), pushed down by the keep and late gates (`../gao/n0_age2.log`), so it is always off.

### 1.5 Substitution (`machine_sub.log`, `sign_rules*.log`, `rule_leave_one.log`, `wave_rule.log`, `rule_size.log`)

The machine computes m_t = D_t (A m_{t-1} + W_hh[memory, others] o_{t-1} + w x_t) with the trained weights. D_t is the on/off pattern set by a rule. This is the modal machine of 1.1 written in neuron coordinates, with switching as resets of single coordinates. With each neuron's own sign as the rule it reproduces the network exactly (95.4%, agreement 100%). So the substitution measures how well the rules predict the switching. Unless stated, the test uses 100,000 lists (seed 11) and the rules are fitted on other lists.

| Rules for on/off | Fitted numbers | Closed | Open |
|---|---|---|---|
| none (all on, linear) | 0 | 44.4% | 50.0% |
| neuron 0 off at age 2, rest on | 0 | 80.9% | 76.5% |
| table: each neuron off or on by (step, age), from another sample | one on/off value per neuron and cell | 87.0% | 85.3% |
| sign of a linear fit per (step, age) cell on the list's numbers (`sign_rules2.log`) | | 92.8% | 93.0% |
| same plus the turning waves of the previous largest and the second-largest (`sign_rules3.log`) | 8,340 in 176 neuron-cells; 344 cells fixed on or off (`rule_size.log`) | 93.5% | 93.6% |
| same rules on 200,000 fresh lists (seed 99; `sign_rules3_fresh.log`) | | 93.6% (−1.9) | 93.7% (−1.8) |

The hybrid connection gives 93.6% on the fresh lists.

By kind of list (share of lists in brackets), closed connection, fresh lists:

| Kind | Change |
|---|---|
| at most one positive number (1%) | +0.8 |
| new largest at step 9 (10%) | −4.2 |
| new largest at step 8 (10%) | −3.6 |
| two new largests in a row earlier (26%) | −2.5 |
| ordinary (54%) | −0.9 |
| second-largest after the largest (50%) | −0.9 |
| second-largest an overtaken record (50%) | −2.9 |

Observation (`rule_cost.log`, `sign_rules.log`, `rule_leave_one.log`). One neuron on a fitted rule, the rest on their own sign, costs 0.0-1.2 points. All eight on rules cost 2.5-3.3 points with the same fits. Putting any single neuron back on its own sign recovers at most 0.7 points (neuron 10). So the loss comes from many small sign errors that add up. The network's own switching corrects one wrong neuron, but not many at once.

Observation (`wave_rule.log`). A rule with the same shape for every step does worse. It writes each pre-activation as a sum of per-number waves, one shape per kind of number (62 numbers per neuron). It matches 75% (neuron 3) to 98% (neuron 10) of pre-activation variance and agrees with the true on/off on 86-96% of steps. Alone it is fine for neurons 13, 14, 15 and 3 (−0.0, −0.4, −0.3, −0.8), poor for 0, 10 and 11 (−2.7, −2.2, −2.0), and 88.2% for all eight.

Judgement call. I counted the fitted-rule result as "under 2 points". It is, but only just, and only with thousands of fitted numbers. It shows that the switching can be predicted from the list. It is not a compact switching rule.

## 2. Neurons 3 and 15 (`n3_n15_role.log`, `mode_content.log`, `vote_by_neuron.log`, `n15_switch.log`, `wave_rule.log`)

Observation: their small difference.
- Neuron 3's two largest input terms come from hiding neuron 1 (weight +1.20, median size 10.6) and hiding neuron 8 (−1.07, 8.6). They correlate at −0.91, so they mostly cancel; the pre-activation's median size is 2.98.
- Neuron 15's two largest come from memory neuron 10 (−1.09, 12.2) and 11 (+1.28, 9.2). They correlate at −0.86; the pre-activation's median size is 3.18.
- For comparison, neuron 10's two largest terms correlate at +0.35, and its pre-activation (10.7) is larger than either term.

Interpretation. Neuron 3 reads what hiding neurons 1 and 8 do not share. Neuron 15 reads the difference between two neighbouring memory neurons (10 and 11), which carry the 38° rotation together (block weight from 11 to 10: 1.10, `modes.log`).

Observation: what they do for the scores.
- Neuron 3's readout weights are +1.69 at position 1, +1.70 at 6 and −0.91 at 5. Replacing its final value by its mean costs 39, 29 and 25 points on lists whose second-largest is at positions 1, 5 and 6, and 4-18 elsewhere.
- Its readout adds 5.7 per unit to a kept vote at position 1 (the vote is 18.8), 9.9 at position 6 (vote 33.7), and 8.8 to a revealed vote at position 1 (vote 29.5).
- Neuron 15's readout weights are +1.01, +2.30 and +2.62 at positions 0-2, and −0.7 to −1.1 elsewhere. Mean replacement costs 22-52 points at positions 0-4.
- Neuron 15's readout adds 21.9 and 22.5 per unit to a revealed record's score at positions 1 and 2 (totals 29.5 and 28.3).

Observation: neuron 15's switching.
- On lists with only one positive number, 15's pre-activation per unit of that number is +2.0 to +4.3 at ages 1-2. It is near zero at ages 3-4 (−1.07 to +0.80), and +4.4 to +19.4 at ages 5-9. On these lists 15 is off on up to 53% of steps at age 3 and 78% at age 4, and on at most 12% at ages 1-2 and 5-9.
- In ordinary lists, the per-number wave fit gives the other kinds of number mostly negative waves in 15 at ages 1-6: kept numbers −1.1 to −6.2 per unit (except +2.2 at age 4), overtaken records −0.6 to −4.3.

Rule (interpretation, tested). Neuron 15 is off when the summed waves of the numbers so far are below zero. This happens mostly when the current largest is 3-4 steps old, or when kept numbers and earlier records 1-6 steps old outweigh a young largest. The sum-of-waves rule agrees with 15's true on/off on 87.8% of steps (always on: 72.8%). When 15 alone follows it, the cost is 0.3 points (no switching: 7.3; a compact fit on eight named quantities: 4.0, with agreement 76.1%).

Not shown: why the network computes these particular differences, or a test that changes the difference alone.

## 3. How an overtaken record's wave is kept off later positions (`tail_route.log`, `tail_routes_all.log`, `tail_pos8.log`, `rewrite_gap.log`, `tail_interv.log`, `late_gate_role.log`, `hidden_size.log`)

Observation. Exact split over all 36 placements r < q ≤ 8 (up to 20,000 lists each), means over placements, per unit of the old record:

| Route | at r | q+1 | q+2 | q+3 | q+4 | q+5 | position 9 |
|---|---|---|---|---|---|---|---|
| total, joint nudge | 48.3 | −8.5 | −7.1 | −4.4 | −6.2 | −5.9 | −17.6 |
| total, record alone | 44.4 | −0.3 | −5.5 | −4.4 | −6.9 | −6.6 | −20.8 |
| wave written while the record was the largest (joint) | 31.9 | −9.6 | −12.7 | −5.4 | −5.8 | −6.3 | +26.1 |
| store's writes after q (joint) | 9.8 | −7.8 | −9.2 | −9.3 | −14.4 | −8.3 | −0.1 |
| other memory writes after q (joint) | 2.9 | +20.9 | +12.4 | −8.3 | −14.6 | −18.4 | +1.2 |
| other neurons' own readout (joint) | 5.7 | −14.1 | +1.1 | +17.8 | +27.7 | +27.8 | −52.6 |
| detector's write at q+1 (record alone) | 6.8 | +3.0 | +8.5 | −1.0 | +5.4 | +1.1 | −34.6 |

Position by position:
- **Positions after q, up to 7.** The wave written while the record was the largest is small and negative there. Later writes and the other neurons' readout add amounts of both signs and largely cancel. No single route dominates.
- **Position 8.** For overtakings at 6 or 7 shortly after the record, single placements reach +28.0 per unit (record at 5, overtaken at 7, record alone) and +24.7 (joint). There the wave written while hidden is +9.5 to +15.8 (records at 4-6, overtaken at 7). Neuron 8's direct readout adds +29 to +43, partly cancelled by the other neurons' readouts (`tail_pos8.log`). Neuron 8 holds the second-largest, so position 8 sees part of its value.
- **Position 9.** The level mode carries +26 per unit of the old record to position 9. It is outweighed by the hiding neurons' readout of the larger largest (−54.7 together, −47.0 of it through neuron 7; joint nudge; `pos9_split.log`), or by the detector's smaller jump (−34.6, record alone). So position 9's drop compares the last number with the largest. It is not the old record's wave.

Observation: the late gate (the brief's requested test, `tail_interv.log`). Zeroing the late gate's weights into the memory at steps q+1 and q+2 of every overtaking:
- moves the old record's per-unit profile by at most 3 at positions after the overtaking and at most 8 anywhere (for example 49.9 to 55.7 at r, record at 1 overtaken at 4, joint; −91.1 to −98.9 at position 0, record at 4 overtaken at 7, joint);
- costs 5.9 points of accuracy (95.3% to 89.4%; lists whose second-largest is an overtaken record 96.3% to 86.8%; other lists 94.4% to 92.0%);
- compares with a random-sign change of the same weights at the same steps (88.3%) and the same removal at q+3 and q+4 (93.1%).

Without it, fewer mistakes name the largest (20% to 14%) and more name a fourth or lower number (17% to 21%). Its writes change the new record's own per-unit profile by up to 9 (`late_gate_role.log`). So the late gate matters at overtakings, but not for the tail; its job there is open.

Observation: another route, the rewrite (`rewrite_gap.log`, `tail_interv.log`). How the revealed vote at r is assembled depends on the gap q − r (joint nudge, mean over placements):

| Gap | Vote at r | From the wave written while hidden | From hiding 6 and 8's writes two steps after the overtaking |
|---|---|---|---|
| 1 | 22.9 | −26.0 | −16.0 |
| 2 | 38.6 | 25.0 | 7.5 |
| 3 | 49.4 | 6.4 | 30.0 (61% of the vote) |
| 4 | 62.2 | 44.1 | 20.9 |
| 5-7 | 61-65 | 74-114 | −22 to 1 |

Zeroing hiding 6 and 8's writes at q+2 removes most of the vote at gap 3: 45.9 to 3.2 (record at 1, overtaken at 4), 45.0 to 6.5 (2, 5), 43.5 to 4.9 (4, 7). At gap 5 it does not (50.4 to 39.9 for 2, 7; record alone). The accuracy cost of that zeroing (73.2%) is matched by random signs (74.6%), so only the profile change is specific. One step after a record, neurons 6 and 8 compare the arriving number with the previous largest (`../bridge/n6_n8_threshold.log`), so they carry the old record's value. Hiding 6's write at steps 5-9 has its positive lobe at position t−5 (section 1.2). Written at t = q+2, that is position q−3: the old record's position exactly when the gap is 3.

Observation (`hidden_size.log`). A record that stays the largest has a memory wave whose largest value is 67.8-69.3 per unit. The vote it reveals when overtaken three steps later is 46.5-61.0, so the ratio is 1.1-1.5. In v18 (gains 10,000) the ratio is 41 (10.25 against 0.25).

Interpretation. The trained network does not stop the old wave exactly. Its hidden wave is about one vote tall. What the wave written while hidden leaves at later positions is a small, mostly negative fraction of a vote, and margins absorb it. v18's recirculating variant fails because its hidden level is 41 votes tall, so any leftover is decisive.

## 4. Other trained networks (`across_tail.log`, `across_vote.log`)

Observation. Role-free checks on all 22 trained networks (400,000 lists each):
- For an overtaken record (joint nudge), the mean per-unit effect at positions after the overtaking is −2.1 to −26.9 in every network (ARC −4.9). The worst single later position is 24.7 in ARC (its revealed vote 49.3) and at most 19.6 elsewhere (new111, vote 26.9).
- With the record-alone nudge, four networks (zoo2, new103, new106, new115) have a positive mean later effect (+4.2 to +7.9).
- A kept vote's largest change at later positions is 4.6 per unit in ARC (vote 26.0) and −12.9 to +11.6 elsewhere (votes 11.9-27.8).

So "the overtaken record's value is kept off later positions on average" holds across networks (joint nudge). Single later positions can still get a large share of the vote: 19.6 against 26.9 in new111. The route-level findings (modes, writers' landing positions, the rewrite, the switching cascade) were checked on ARC only, because they need neuron roles.

## Observations, interpretations, proposals

Observations (measured): everything marked so above.

Interpretations (not tested directly):
- Switching sits at the current largest's zero crossings (1.4).
- Neuron 15 is the difference of two neighbours on a rotation (2).
- A one-vote-tall hidden wave is why recirculation is affordable (3).
- The rewrite at gap 3 is a matter of where hiding 6's write lands rather than a dedicated mechanism (3).

Proposals (not done):
- A hand-built memory whose hidden wave is about one vote tall, recirculating rather than a delay line. This would test the last interpretation directly; it fits the "robust twin" idea in the NOTE's backlog.
- A compact switching rule built from the zero-crossing description: the current largest's wave in each neuron plus the dips of kept numbers, with a few shared shapes.
- An intervention that changes only neuron 15's difference (scale the 10 and 11 inputs together) to see what it is for.

## What is unknown

- A switching rule with few numbers that loses under 2 points. The best compact rules lose 8 points (age table) or more.
- What the late gate's writes do at overtakings: they cost 6 points but barely touch the old record's profile.
- Why the network computes neuron 3's particular difference (hiding 1 minus hiding 8).
- Why the revealed vote is assembled by different routes for gaps 1, 3 and 5+, and whether the gap-3 rewrite matters for accuracy beyond what random-sign controls show.
- Whether the route-level findings hold in other networks.

## Notes against the brief's starting points

- Eigenvalues: confirmed (1.1).
- "Only 0 and 15 need to switch": the others are worth 5.8 points together (1.4).
- Shift relation (cosine 0.97): for the memory's own readout it holds into positions 1-7 and fails into 8 and 9 (1.3).
- Tail numbers: the brief's (+19, position 9 −17 to −44) are medians from `../bridge/tail_cancel2.log`. Mine are means and reach +28 at single later positions. Position 9 ranges from −42.7 to +8.6 by placement.

## Scripts and logs

All run as `PYTHONPATH=/data/output/src/algzoo/src /opt/venv/bin/python <script>` from this folder, numpy on 2 threads.

| Script | Log | Shows |
|---|---|---|
| `memlib.py` | | weights, run with hooks, task quantities, kinds of list, scores |
| `decomp.py` | | exact split of a nudge into memory writes by writer and step |
| `modes.py` | `modes.log` | memory block weights, eigenvalues, mode make-up, memory readout shift test, per-mode readout by position |
| `mode_writes.py` | `mode_writes.log` | each writer's unit write by mode, where it lands, mode shares |
| `impulse.py` | `impulse.log` | writers' time course and per-writer final scores for a hidden record and a kept vote |
| `landing_real.py` | `landing_real.log` | on real lists, how much of each step's writes is read before and after that step |
| `mode_content.py` | `mode_content.log` | modes per neuron; terms of neurons 3 and 15; readout ablation per mode family |
| `vote_by_neuron.py` | `vote_by_neuron.log` | which neurons read a vote at each position |
| `switching.py` | `switching.log` | off-rates by age class; cost of removing switching per neuron and age class |
| `offrate_age.py` | `offrate_age.log` | off-rate per neuron by step and age |
| `delay_line.py` | `delay_line.log` | per-unit change of each memory neuron by age, for kept numbers and records (the zero crossings) |
| `machine_sub.py` | `machine_sub.log` | substitution ladder: linear, neuron 0 rule, age table, with 3 and 15 on their own sign; closed, hybrid, open |
| `rule_cost.py` | `rule_cost.log` | clipped amounts; cost of the age table one neuron at a time |
| `sign_rules.py` | `sign_rules.log` | fitted-sign rules per cell, single neurons and combinations |
| `sign_rules2.py` | `sign_rules2.log` | richer features, fixed cells: 92.8% |
| `rule_leave_one.py` | `rule_leave_one.log` | which neuron's rule costs most in combination |
| `sign_rules3.py` | `sign_rules3.log` | wave features: 93.5% closed, 93.6% open |
| `sign_rules3_fresh.py` | `sign_rules3_fresh.log` | the same rules on 200,000 fresh lists: 93.6% (−1.9) |
| `rule_size.py` | `rule_size.log` | number of fitted numbers in those rules |
| `wave_rule.py` | `wave_rule.log` | per-number wave sum as a switching rule; kernels of 15, 3, 0 |
| `n3_n15_role.py` | `n3_n15_role.log` | readout weights; readout-only and full mean ablation by position, all memory neurons |
| `n15_switch.py` | `n15_switch.log` | neuron 15 on lone records; compact eight-quantity rule |
| `tail_route.py` | `tail_route.log` | exact split of the old record's profile for four placements, both nudges |
| `tail_routes_all.py` | `tail_routes_all.log` | the same over all placements, by distance from q |
| `tail_pos8.py` | `tail_pos8.log` | position 8 for overtakings at 6 and 7, by route and by reading neuron |
| `pos9_split.py` | `pos9_split.log` | position 9: the other neurons' readout by neuron |
| `rewrite_gap.py` | `rewrite_gap.log` | revealed vote by gap: written while hidden against rewritten by hiding 6 and 8 |
| `tail_interv.py` | `tail_interv.log` | zeroing writers' weights into memory after overtakings, with random-sign and other-step controls |
| `late_gate_role.py` | `late_gate_role.log` | mistakes and new-record profile without the late gate's writes |
| `hidden_size.py` | `hidden_size.log` | hidden wave size against revealed vote, trained and v18 |
| `across_tail.py` | `across_tail.log` | tail check on 22 networks |
| `across_vote.py` | `across_vote.log` | forward spill of kept votes on 22 networks |
