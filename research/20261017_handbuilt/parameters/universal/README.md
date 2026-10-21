# Which of ARC's quantities are universal, and which are particular? (`parameters/universal/`, started 2026-10-21)

Handoff: an internal working note, not included. Status: done (steps 1-4; the optional step 5 was not done, see the end).

## Question

ARC's network is one training run. Some of what we measured in it should recur in every network trained the same way; those quantities deserve an explanation from the task and the recipe. Others are particular to this run, like a material's constants; those can only be measured. This study sorts ARC's quantities into the two kinds, plus a middle kind: the same form everywhere, with values that vary.

## Material

ARC's network, and 37 networks with 16 neurons and 10 steps trained with the public recipe: ARC's zoo seeds 0-4 (`models/zoo/`, with initial weights and 11 checkpoints each) and our seeds 100-115 and 200-215 (`../../../20261016_run3/models/`). ARC's own network was trained with an earlier, unpublished recipe (cross-entropy, small weight decay), so it is not strictly a member of the same family. That is itself a reason to check whether it is typical.

## Plan (written before checking FINDINGS.md)

1. `nets.py`: loads the 38 networks and runs them on random lists. One scale per neuron is free (rescaling a neuron and dividing its outgoing weights changes nothing). Every quantity that compares weights is computed in one fixed choice of scales: each neuron is rescaled so its activity has standard deviation 1 over random lists (all steps). Quantities that are ratios within one neuron's row and column need no such choice.
2. `roles.py`: roles by behaviour, on 200,000 random lists. A "record" is a positive number larger than every earlier one, from step 1 on.
   - store: the neuron whose activity best follows the largest number seen before the current step (correlation; a clear store has at least 0.9);
   - new-largest detector: on at least 90% of record steps and at most 10% of other steps (first written as 5%; changed after the first run because two networks sat at 6%, inside the same cluster). A "loose detector" is on at 90% of records and on 10-35% of other steps;
   - hiding neurons: off on at least 85% of record steps (the handoff's cutoff) and on at least 65% of other steps;
   - late-like gate: one step after a record, on at least 90% of the time; otherwise on at most 5%;
   - keep-like gate: one step after a record, on at least 90%; otherwise on 5-60% of steps;
   - memory: the rest.
   Calibration on ARC (before any other network): these criteria give store 2 (correlation 0.98), detector 4, hiding 1, 6, 7, 8, late gate 9, keep gate 5, which are the groups named in earlier notes.
3. `quantities.py`: one row per network, one column per quantity (list below), ARC's value against the 37, and where ARC ranks.
4. `checkpoints.py`: the universal relations across the zoo seeds' checkpoints.
5. Optional: the overnight rules' mechanisms on a few sister networks.

### Quantities (all unchanged by rescaling a neuron and by neuron order)

Task level, on 2^20 random lists: accuracy; among mistakes, the share naming the third, the largest, the fourth or lower, and lists with at most one positive number; on near ties (third within 5% of the second, second positive): accuracy, and how often the third wins.

Mechanism level:
- readout shift relation: readout row k+1 against row k pushed one step through the dynamics in the most common on/off pattern at steps 4-8 (cosine of centred rows, median over k = 0..7, after fitting one scale per k);
- memory block: eigenvalues of the memory neurons' block of recurrent weights, all on (angles and shrink factors; eigenvalues do not depend on neuron scales);
- switch-off ages: memory neurons that are off at one age of the largest (off at least 85% of the time at that age, at most 30% at other ages);
- blindness: for each row outside the gates, |Σ_j W[i,j] w_j| / Σ_j |W[i,j] w_j| over the hiding neurons j (w_j their input weights). Near 0 means the arriving number cancels in what row i reads from the hiding neurons. The null is the same ratio with random signs on the terms;
- hiding comparators: each hiding neuron's reference (the input at which it would switch, from its other inputs, in units of the arriving number), fitted as a mix of the largest and second-largest so far; and its gain, |input weight| / activity standard deviation;
- overnight weight relations: (a) detector row = −c × store row (cosine) and c against store self-weight / (detector → store weight); (b) the hiding neuron whose incoming weights best match the detector's, with sign (ARC's neuron 7 is about −1); (c) each hiding neuron's store weight / input weight, divided by the detector's same ratio (ARC about 1); (d) delay line: the largest share of one other memory neuron in a memory row (squared weights); (e) readout column sums: |column mean| / column standard deviation, root mean square over columns.

## Predictions (written before any network other than ARC was run)

Step 1, roles:
- P1. Every network has a clear store (correlation ≥ 0.9) and at least one detector. Confidence moderate: the less accurate zoo networks (88-93%) may be messier.
- P2. Every network has a hiding group of 3-6 neurons (earlier: all 22 checked have neurons a record silences).
- P3. A late-like gate in fewer than half the networks, a keep-like gate in fewer than half; ARC is unusual in having both.
- P4. ARC's role counts are typical except for the gates.

Step 2, quantities:
- P5. Accuracy: ARC in the top quarter (it is 95.3%; zoo seeds are about 88-95%).
- P6. Mistake kinds: the third is the most common wrong answer in every network (50-70% of mistakes).
- P7. Near ties: the third wins 15-30% of near ties in every network.
- P8. Readout shift relation: median cosine ≥ 0.8 in about four-fifths of the networks (earlier: 17 of 21).
- P9. Memory spectrum: rotating pairs with shrink factors 0.8-1.0 in every network; the angles differ (same form, particular values).
- P10. Switch-off ages: age-clipped memory neurons in most networks, at different ages (same form, particular values).
- P11. Blindness: median ratio well below its random-sign null in most networks (universal).
- P12. Hiding references are mixes of the largest and second-largest in every network (fit R² ≥ 0.8); the mix varies (same form).
- P13. Detector row = −c × store row: cosine ≥ 0.9 and c within 20% of store self-weight / (detector → store weight) in most networks. Reason: "a record overwrites the store" is a job every network must do.
- P14. Neuron 7's mirror (a hiding row ≈ −1 × the detector's): in fewer than half (particular).
- P15. Hiding store ratio equal to the detector's (ratio 0.7-1.3) for most hiding neurons in most networks (universal), since all compare with the largest held in the store.
- P16. Delay line (one memory row with ≥ 80% share from one other memory neuron): in a minority (particular).
- P17. Small readout column sums (ratio below 0.3, where random rows give about 0.3-0.4): in most networks; ARC typical.

Step 4, timing:
- P18. The store, the detector and the overwrite relation form early (by 2^22-2^24 training lists); the readout shift relation forms last.

## Results

### The answer

Sorted by how many of the 37 same-recipe networks share each of ARC's quantities. "Best nine" means the nine most accurate of the 37 (94.5-96.1%; ARC has 95.3%, fifth among them). That split was chosen after seeing that tidiness tracks accuracy.

| ARC's quantity | of the 37 | of the best nine | Kind |
|---|---|---|---|
| The third is the commonest wrong answer (ARC 61% of mistakes) | 37 (50-73%) | 9 | universal |
| Readout row k+1 is row k pushed one step through the dynamics ("fair votes") | 31 with cosine ≥ 0.8 (median 0.96) | 7 | universal |
| Readout column sums smaller than for random rows | 37 | 9 | universal in form; ARC's value particular (see below) |
| A new-largest detector (strict or loose) | 35 | 9 | universal |
| Neurons that a record switches off, with negative input weights | 35 | 9 | universal (nearly by definition) |
| A memory neuron that switches off at one age of the largest | 37 | 9 | same form; ARC's ages (2, 5, 6) particular: most networks use ages 1-2, only 5 use age 5 or more |
| Memory block of rotating pairs that fade slowly | 29 | 8 | same form; ARC's angles (38°, 84°, 127°, 180°) particular. Slowest angle runs 4-57° across networks |
| Store fade (store self-weight 0.97) | 6 at ≥ 0.9 | 2 | same form, particular value (0.02-0.97) |
| Hiding group size (4) | 25 have 3-6 | 9 | same form, particular value (0-6; commonest 3) |
| Blindness: rows read the hiding neurons so the arriving number cancels | 17 of 32 below 0.1 (random 0.5) | 9 of 9 | shared by accurate networks; ARC's near-exactness (0.004) seen in 4 |
| Hiding neurons read the store at the detector's ratio (overnight relation) | 21 of 33 (17 for every hiding neuron) | 6 | common |
| Detector row = −c × store row, with c from "a record wipes the store" (overnight relation) | 5 of 35 (12 have the row relation without the right c) | 4 | particular overall, but common among the best |
| Delay line: one memory neuron reads almost only another, and switches off one age later (overnight relation) | 11 at share ≥ 0.8; 4 at ≥ 0.95 | 5; 3 | particular overall; same form in the best networks (ages 1→2, 2→3; ARC 5→6) |
| A hiding neuron mirrors the detector (ARC's neuron 7; overnight relation) | 10 of 33 | 2 | particular |
| One neuron that clearly holds the largest | 18 | 4 | particular (others spread it over several neurons) |
| A late gate; a keep gate; both | 4; 9; 1 | 1; 1; 0 | particular |
| ARC's memory has 8 neurons and 3 rotating pairs | none smaller; typical 11 and 4 | | particular |
| Readout column sums at 3% of column spread | none of the 37 below 9% | | particular, probably ARC's different recipe |

What it means for which weights deserve an explanation:
- **The readout deserves its explanation.** The fair-votes relation, which carries most of the overnight rule's 170 bits, holds in about five of six same-recipe networks and is already in place after a million training lists. It is a property of the task and the recipe, not of ARC's run.
- **The hiding neurons, the store and the detector deserve an explanation of their form, not their numbers.** Every network has switch-off neurons and a detector. How tightly their rows are tied together varies a lot. The tight ties ARC has (exact blindness, detector as minus the store, neuron 7 as a mirror) appear mostly in the most accurate networks. So they look like what good training reaches, not like laws every network obeys.
- **The memory's exact angles, fades and switch-off ages are material constants.** Every network has rotating, slowly fading memory and age-clipped neurons, but the values differ from network to network. Rules that copy ARC's angles or ages describe this run only.
- **The gates are ARC's own.** Only one of the 37 has both a late gate and a keep gate.
- **ARC is not a typical member of the family.** It sits at the tidy end on almost every structural measure: the smallest memory, one of the clearest stores, near-exact blindness, the smallest readout column sums. Some of this goes with its accuracy (it beats about 90% of the 37). Its readout column sums, at least three times smaller than in any of the 37, are more likely an effect of its earlier, different training recipe.

Overnight relations, in one line each: fair votes, universal; hiding neurons at the detector's store ratio, common (about three in five); detector as minus the store with the overwrite constant, rare (5 of 35) but present in four of the best nine; delay line, rare in its exact form, recurring in the best networks with different ages; neuron 7's mirror, rare; small readout column sums, universal in form, but ARC's size particular.

### Step 1: roles (`roles.py`, `roles.log`, `roles.json`)

Counts per network are in `roles.log`. Over the 37: a clear one-neuron store in 18 (correlation 0.74-0.99); a strict detector in 29, a loose one only in 6, none in 2; hiding groups of 0, 1, 2, 3, 4, 6 neurons in 2, 3, 7, 16, 8, 1 networks; a late-like gate in 4; a keep-like gate in 9; both in 1 (seed 205); memory 8-12 neurons (median 11). ARC: clear store, one detector, 4 hiding neurons, both gates, 8 memory neurons. ARC's memory is the smallest of all 38 (tied with seed 209).

A sixth of the networks have a "loose detector" that fires on records and on about a quarter of other steps. That kind of neuron was not described before.

### Step 2: quantities (`quantities.py`, `quantities.log`, `quantities.csv`, `quantities_detail.json`, `summary.log`, `counts.py`, `counts.log`)

`summary.log` gives, for every quantity, ARC's value, the 37's median and 10-90% range, and ARC's percentile. `counts.log` gives the counts in the table above and each quantity's rank correlation with accuracy. ARC's own values reproduce earlier notes: mistakes 61% third, 15% largest, 18% few positives, 6% fourth or lower; memory angles 0°, 38°, 84°, 127°, 180°; neuron 0 off at age 2, 11 at 5, 10 at 6; delay pair 11 → 10 (share 0.98); hiding store ratios 0.98-1.06 of the detector's; neuron 7's cosine with the detector −0.997.

Accuracy tracks several structural quantities across the 37 (rank correlations): blindness −0.71 (more accurate, more blind), memory size −0.65, hiding group size +0.51, share of hiding neurons at the detector's store ratio +0.58. ARC's near-tie figures and its high share of "few positives" mistakes follow its accuracy (correlations 0.86-0.92), so they are not separate facts about ARC.

Two quantities turned out uninformative: the hiding gain (|input weight| / activity spread) is about 1.0 in every network, because a hiding neuron's activity is mostly its input. And the share of hiding neurons with a negative input weight is 100% everywhere, which nearly follows from their definition.

### Step 4: when they form (`checkpoints.py`, `checkpoints.log`, `checkpoints.csv`)

Followed on the same neurons (the final network's roles) through the 12 saved points of the five zoo seeds:
- The readout-shift relation forms first. Its cosine is near 0 at initialisation, and 0.7-0.85 after 2^20 lists, before any detector or hiding neuron exists. It is close to its final value by 2^22-2^23 (seed 2 later loses some of it, ending at 0.68).
- Readout column sums shrink steadily, from the random level (0.26-0.41) to 0.08-0.12.
- The detector and the store appear at about 2^23 lists (8 million), together with a jump in accuracy from about 45% to 60-70% (seed 3 gets only a loose detector). In the seed where the detector row becomes tightly minus the store row (seed 0, cosine −0.96), that forms between 2^23 and 2^24. Its constant c does not reach the overwrite value in any zoo seed.
- Hiding neurons, and the blindness that comes with them, appear late: at 2^23-2^25 in seeds 0, 1, 3 and 4 (seed 0's second one at 2^27), and at 2^27 in seed 2. In seed 2, blindness drops from 0.80 to 0.12 at the same checkpoint where accuracy rises from 83% to 90%.
- Relations that end loose stay loose. Seed 1's detector row never gets closer than cosine −0.72 to minus its store row, and seed 4's never closer than −0.31.

### Predictions, kept with outcomes

- P1 (clear store and a detector everywhere): **failed.** A clear one-neuron store in only 18 of 37, a detector in 35. The rest spread the largest over several neurons (read from all neurons, it is still about 98% recoverable).
- P2 (hiding group of 3-6 everywhere): **failed** in 12 of 37, which have 0-2.
- P3 (gates in fewer than half; ARC unusual with both): held (4, 9, 1).
- P4 (ARC's role counts typical except gates): **failed.** ARC's memory is the smallest.
- P5 (ARC in the top quarter for accuracy): held (89th percentile).
- P6 (third the commonest wrong answer everywhere): held (37 of 37).
- P7 (third wins 15-30% of near ties everywhere): **failed** in 14. The less accurate networks go above 30%.
- P8 (readout shift ≥ 0.8 in about four-fifths): held (31 of 37).
- P9 (rotating, slowly fading memory everywhere; angles differ): mostly held (29 of 37 with largest modulus 0.8-1.2; angles differ).
- P10 (age-clipped neurons in most, at different ages): held (37 of 37).
- P11 (blindness in most): **failed as stated.** Only 17 of 32 have it, but all of the best nine do.
- P12 (hiding references are mixes of the largest and second-largest, R² ≥ 0.8): **failed,** including in ARC (R² 0.69). Earlier notes already said the references depend on the largest's age; a fixed mix is too simple.
- P13 (detector = −c × store with the overwrite c in most): **failed.** 5 of 35; 4 of the best nine.
- P14 (mirror in fewer than half): held (10 of 33).
- P15 (hiding store ratio equal to the detector's in most): held, barely (21 of 33).
- P16 (delay line in a minority): held (11 of 37).
- P17 (small column sums in most; ARC typical): first half held (37 of 37 below the random level); **ARC is not typical:** its value is below all 37.
- P18 (store, detector and overwrite relation early; readout shift last): **failed** for the readout shift, which forms first. Store and detector do form early (about 2^23 lists).

### Judgement calls

- The detector cutoff was widened from 5% to 10% of other steps after the first run, because two networks sat at 6%, inside the same cluster.
- All weight comparisons use one fixed choice of neuron scales (activity spread 1). A different choice would change the cosines somewhat, but not ratios such as the store ratio, the overwrite constant or blindness.
- The relations are traced in each network's best detector and clearest store. When a network spreads its store over several neurons, the detector-versus-store test is not a fair test of the overwrite idea there. That may make relation (a) look rarer than it is.
- The "best nine" split was chosen after seeing the correlation with accuracy.

### Not done

Step 5 (scoring the overnight rules on sister networks) was not done. The rules name ARC's neurons and both gates, and only one sister has both gates, so each rule would have to be rewritten per network. Each network would also need its own curvature computation. The counts above already show which mechanisms the sisters share.
