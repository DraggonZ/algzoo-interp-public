# Do the sister networks have a neuron like ARC's neuron 3? (`neuron3/sisters/`, started 2026-10-21)

## Question

In ARC's network, memory neuron 3 reads two hiding neurons (1 at +1.20, 8 at −1.07) whose terms mostly cancel. What is left is close to g(step, age of the largest) × (largest so far − second-largest so far). Flattening that difference to its per-(step, age) mean costs 17 points of accuracy. Do the 37 sister networks (same public recipe, `../../parameters/universal/`) have such a neuron?

Material: the 38 networks of `../../parameters/universal/nets.py`, roles from its `roles.json` (hiding and memory neurons by behaviour). Hiding references from its `quantities_detail.json`.

## Method (written before running)

Script `sisters.py`, log `sisters.log`, per-neuron table `sisters.csv`.

1. **Terms.** For each memory neuron i, the terms entering its pre-activation at steps 3-9 are W_hh[i,j]·h_j(previous step) for every neuron j, plus the input term w_i·x_t. These are unchanged by the free neuron scales (up to one common factor), so no gauge is needed.
2. **Strict flag ("neuron-3-like").** Its two largest terms by median |size| both come from hiding neurons, have opposite signs, correlate below −0.7, and the median |pre-activation| is under a third of the larger term's median |size|.
3. **Loose flag** (my wording of "large relative to the pre-activation"; a judgement call). Some pair of hiding neurons enters with opposite weight signs, each of the two terms has median |size| at least half the median |pre-activation|, and their sum (the weighted difference) has a standard deviation at least half that of the pre-activation.
4. **Gap fit.** For every pair of hiding neurons and every memory neuron, the weighted sum W[i,j]h_j + W[i,k]h_k (after step t) is fitted per (step t, age of the largest) cell to g·(largest − second-largest so far) + c. Steps 2-9; cells with at least 2,000 lists and a positive largest; cells where the quantity barely varies (variance under 1% of its overall variance, for example age 0 where the hiding neurons are off) are left out. Reported: median R² over cells. The same fit for the pair's unweighted difference h_j − h_k with each neuron scaled to activity standard deviation 1 (my reading of "raw difference"). And each hiding neuron's reference mix of the largest and second-largest, from `quantities_detail.json`.
5. **Readout and substitution.** Peakedness of each memory neuron's readout column (largest |weight| / median |weight|) and the positions above twice the median. Closed substitution on 200,000 fresh lists (means fitted on 200,000 other lists): run step by step; at each step t ≥ 1 add (cell mean − actual) of the hiding difference entering neuron i to its pre-activation, the cell being (t, age of the largest at t−1). For flagged neurons, the two-term difference; also the gap formula in place of the mean. For comparison, for every memory neuron, the same with its single largest hiding term, and with its top opposite-sign hiding pair. A random non-flagged memory neuron (fixed seed) is picked per network for the headline comparison.

Accuracy is only ever measured, never fitted to.

## Predictions (written before running anything)

- **Q1. ARC** comes out with neuron 3 strictly flagged, and no other ARC memory neuron strictly flagged.
- **Q2. Strict neuron-3-like neurons in about 6 of the 37** (range 3-12), mostly among the most accurate networks (those with blindness and 3+ hiding neurons). Reason: blindness (rows that read hiding neurons so the arriving number cancels) is in 17 of 32 networks, and a cancelling pair is a special case of it. Networks with 0-1 hiding neurons (5 of them) cannot have one.
- **Q3. Loose flag in about half** (15-20 of 37).
- **Q4. Gap formula.** For flagged pairs, median R² 0.6-0.9, similar to ARC's. For opposite-sign pairs in general, lower (0.3-0.6), because the cancellation of the largest is what makes the remainder a gap.
- **Q5. Hiding references** mix the largest and the second-largest in every network with hiding neurons, with most weight on the largest for some neurons and on the second for others (as in ARC: one at about 0.85 × largest, one at about 1.05 × second).
- **Q6. Readout.** Flagged neurons have peaked readouts (peakedness ≥ 3) more often than other memory neurons, but not always at ARC's positions (1 and 6).
- **Q7. Substitution.** Flattening a flagged neuron's hiding difference costs at least 5 points in most flagged networks. The same for a random non-flagged memory neuron's largest hiding term costs under 2 points in most networks. The gap formula recovers most of the flagged loss (cost under a third of the flattening cost).


## Results

Files: `sisters.py` → `sisters.log`, `sisters_all.csv` (one row per memory neuron), `pairs_all.csv` (every hiding pair × memory neuron), `nets_all.json`; `summarize.py` → `summary.log`; `table.py` → `table.md`; `refs.py` → `refs.log` (hiding references refitted). All on 200,000 random lists for the measurements and fits, 200,000 other lists for accuracy.

### Findings in plain words

- **ARC's neuron 3 is close to unique.** By the strict test, one of the 37 sisters has such a neuron (seed 115, neuron 8). ARC comes out with neuron 3 and nothing else, as it should.
- **The one strict match is not a gap neuron.** Its cancelling pair carries the arriving number, not the gap between the largest and the second-largest. The gap formula fits it poorly (median R² 0.42), and putting the gap formula in place of the pair recovers little (flattening costs 55 points, the gap formula still costs 46).
- **Two sisters have a functional match that misses the strict cut.** Seed 108's neuron 5 and seed 111's neuron 14 read two cancelling hiding neurons whose difference follows the gap well (R² 0.83 and 0.86, better than ARC's 0.65). The gap formula recovers about 70% of what flattening costs (12 → 3.5 points; 32 → 10). They fail only because their pre-activation is a bit larger: 0.37 and 0.47 of the larger term, against the cutoff of a third. In seed 111 the pair is like ARC's: one hiding neuron compares with the largest, the other mostly with the second-largest.
- **Why the gap usually does not appear.** Any hiding neuron's activity is roughly (its reference − the arriving number). Scaled to equal spread, the difference of any two hiding neurons follows g·gap + c well (median R² 0.70 in the sisters, 0.80 in ARC), because the arriving number cancels. But a memory neuron reads the pair with its own weights, which rarely cancel the arriving number. So the weighted pair follows the gap poorly (median R² 0.19 for opposite-sign pairs). Adding the arriving number to the fit lifts it to 0.78. What makes ARC's neuron 3 special is the balance of its two weights, not the hiding neurons themselves.
- **Flattening costs are not special.** Flattening one hiding term of any memory neuron costs a lot (median 30 points), since that term carries the largest and the arriving number. Flattening the top cancelling pair of a random memory neuron costs a median of 13 points. In ARC itself, neuron 3's 17 points is second of eight (neuron 0's pair costs 21). What is special about neuron 3 is that the gap formula recovers most of it (17 → 4.4). Even in ARC it is not alone in this: for neurons 10 and 11 the gap formula recovers about 90% of smaller costs (4 and 8 points). In the sisters it recovers a median of a quarter. The better the gap fit, the more it recovers (rank correlation 0.76).
- **Peaked readouts are common.** Memory neurons' largest readout weight is a median 3.7 times their median weight. ARC's neuron 3 (2.8, at positions 1 and 6) is less peaked than a typical memory neuron by this measure. The sister candidates peak at single positions: 7 (seed 115), 3 (seed 108), 7 (seed 111).
- **Hiding references are mixes, mostly of the largest.** 91 of 103 sister hiding neurons fit as a mix (median R² 0.71); the median share on the largest is 0.78. Only 7, in 6 networks, lean on the second-largest as ARC's neuron 8 does. The other 12 (in 5 networks: zoo2, s103, s111, s201, s205) have references that depend strongly on the age of the largest, so a fixed mix fails (R² 0.07). For example, zoo2's neuron 0 is forced on one step after a record, compares with about 0.2·largest + 0.3·second at age 1 and with 1.3·largest at age 2. The fits in `quantities.csv` fail the same way for these 12.

### Counts

| | ARC | sisters (of 37) |
|---|---|---|
| Networks with at least two hiding neurons (a pair is possible) | yes | 32 |
| Memory neurons whose two largest inputs are hiding terms | 4 of 8 (neurons 0, 3, 11, 14) | 29 of 392, in 14 networks |
| ... of opposite sign, correlating below −0.7 | 4 | 20 |
| ... and pre-activation under a third of the larger term (strict) | 1 (neuron 3) | 1 (seed 115, neuron 8) |
| ... under half (near misses) | 1 | 4, in 3 networks (seeds 108, 111, 115) |
| Loose flag | 3 of 8 | 20 of 392, in 9 networks |
| Gap formula, weighted top pair, median R² | 0.55 (neuron 3: 0.65) | 0.31 |
| Same with the arriving number added | 0.78 | 0.78 |
| Gap formula recovers ≥ 2/3 of the flattening cost (neurons whose flattening costs ≥ 3 points) | 3 of 8 (neurons 3, 10, 11; neuron 14 just under) | 28 of 263 |

The loose flag turned out less selective than intended: it marks three ARC neurons (0, 3, 15). Neuron 0 reads hiding 6 against 7, with a pair difference that is not gap-like (R² 0.44).

### Strict and near-miss neurons

| Neuron | Accuracy | Pair | Corr | Pre ÷ larger term | Gap R² (with arriving number) | Readout peak | Flatten pair → gap formula (points) |
|---|---|---|---|---|---|---|---|
| ARC neuron 3 (strict) | 95.3 | hiding 1, 8 | −0.91 | 0.28 | 0.65 (0.72) | 2.8 at 1, 6 | 17.2 → 4.4 |
| s115 neuron 8 (strict) | 95.3 | 5, 7 | −0.85 | 0.26 | 0.42 (0.82) | 4.9 at 7 | 55.3 → 46.4 |
| s115 neuron 4 | 95.3 | 5, 7 | −0.85 | 0.42 | 0.46 (0.84) | 3.7 at 8 | 57.5 → 51.9 |
| s108 neuron 5 | 93.8 | 15, 1 | −0.90 | 0.37 | 0.83 (0.84) | 5.7 at 3 | 12.2 → 3.5 |
| s111 neuron 14 | 96.0 | 9, 12 | −0.83 | 0.47 | 0.86 (0.87) | 3.6 at 7 | 32.4 → 10.3 |

Comparison, a random non-flagged memory neuron per network (35 sisters): flattening its largest hiding term costs a median of 30 points (10-90%: 13-54); flattening its top opposite pair, 13 points (4-40). In ARC (neuron 13): 21.5 and 6.7.

Note on ARC's gap-formula cost: 4.4 points here against 2.8 in `../sub.log`. Here the gap uses the true largest and second-largest; `../sub.py` pads the seen numbers with zeros, so a negative largest or second counts as 0. That is a different formula, not a different network.

### Per network

"Closest candidate": the memory neuron whose two largest inputs are opposite hiding terms, with the smallest pre-activation relative to the larger term. Columns in brackets: hiding pair; correlation of the two terms; median |pre-activation| ÷ larger term's median size; gap R² of the weighted pair; readout peakedness and positions above twice the median |readout|; cost of flattening the pair → cost with the gap formula (points of accuracy). "Top pairs" columns: over all memory neurons of the network, each neuron's top opposite-sign hiding pair.

| Network | Accuracy | Hiding | Memory | Strict | Loose | Closest candidate (pair; corr; pre ÷ larger term; gap R²; readout peak at; flatten → gap formula) | Top pairs: median gap R² | Top pairs: median flatten cost | Random other neuron: largest term / top pair |
|---|---|---|---|---|---|---|---|---|---|
| ARC | 95.3 | 4 | 8 | 1 | 3 | n3 (1, 8; -0.91; 0.28; 0.65; 2.8 at 1,6; 17.2 → 4.4) | 0.55 | 12.8 | 21.5 / 6.7 |
| zoo0 | 91.1 | 2 | 12 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.11 | 21.9 | 41.1 / – |
| zoo1 | 88.1 | 2 | 12 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.58 | 21.3 | 64.0 / 59.7 |
| zoo2 | 94.8 | 3 | 10 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.36 | 5.3 | 8.4 / 2.9 |
| zoo3 | 93.2 | 3 | 11 | 0 | 1 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.11 | 3.6 | 16.8 / 5.7 |
| zoo4 | 89.0 | 1 | 12 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | – | – | 51.7 / – |
| s100 | 94.9 | 4 | 10 | 0 | 4 | n10 (4, 13; -0.97; 0.75; 0.49; 3.6 at 3,4,5; 8.6 → 4.2) | 0.34 | 18.8 | 49.9 / 20.2 |
| s101 | 91.4 | 3 | 11 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.53 | 23.4 | 25.4 / 16.2 |
| s102 | 91.3 | 3 | 11 | 0 | 1 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.12 | 21.8 | 28.0 / 8.5 |
| s103 | 95.4 | 3 | 9 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.35 | 8.0 | 7.5 / 4.0 |
| s104 | 91.3 | 4 | 10 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.22 | 13.2 | 43.4 / 18.3 |
| s105 | 90.0 | 1 | 12 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | – | – | 15.1 / – |
| s106 | 94.9 | 4 | 9 | 0 | 0 | n10 (8, 7; -0.81; 1.51; 0.85; 2.9 at 1,8; 25.9 → 7.8) | 0.51 | 16.1 | 30.5 / 7.7 |
| s107 | 92.2 | 3 | 11 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.16 | 25.1 | 58.0 / 52.0 |
| s108 | 93.8 | 3 | 10 | 0 | 3 | n5 (15, 1; -0.90; 0.37; 0.83; 5.7 at 3; 12.2 → 3.5) | 0.13 | 23.1 | 55.4 / 25.0 |
| s109 | 94.1 | 2 | 11 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.12 | 30.0 | 57.4 / 43.8 |
| s110 | 91.3 | 2 | 10 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.12 | 20.2 | 17.2 / 9.8 |
| s111 | 96.0 | 4 | 9 | 0 | 3 | n14 (9, 12; -0.83; 0.47; 0.86; 3.6 at 7; 32.4 → 10.3) | 0.81 | 13.4 | 18.4 / 6.3 |
| s112 | 91.5 | 1 | 12 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | – | – | 9.7 / – |
| s113 | 90.5 | 4 | 10 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.30 | 10.9 | 22.8 / 3.8 |
| s114 | 89.8 | 2 | 11 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.15 | 23.3 | 29.8 / 18.9 |
| s115 | 95.3 | 3 | 10 | 1 | 4 | n8 (5, 7; -0.85; 0.26; 0.42; 4.9 at 7; 55.3 → 46.4) | 0.66 | 23.1 | 23.9 / 13.6 |
| s200 | 93.3 | 3 | 11 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.15 | 14.1 | 42.5 / – |
| s201 | 94.4 | 4 | 10 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.30 | 8.5 | 15.8 / 5.6 |
| s202 | 89.7 | 3 | 11 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.18 | 9.6 | 45.5 / 36.1 |
| s203 | 90.0 | 4 | 10 | 0 | 2 | n13 (9, 10; -0.88; 0.87; 0.09; 6.0 at 6; 25.3 → 22.4) | 0.22 | 11.2 | 52.5 / 39.2 |
| s204 | 89.0 | 2 | 12 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.29 | 21.8 | 33.7 / – |
| s205 | 93.6 | 3 | 9 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.34 | 9.7 | 24.6 / 15.9 |
| s206 | 91.3 | 3 | 11 | 0 | 1 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.17 | 12.2 | 37.4 / 12.2 |
| s207 | 91.2 | 2 | 12 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.10 | 18.2 | 21.8 / 10.8 |
| s208 | 84.2 | 0 | 11 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | – | – | – |
| s209 | 95.9 | 6 | 8 | 0 | 1 | n14 (7, 3; -0.91; 0.92; 0.30; 32.6 at 1,2,6,8; 42.5 → 39.7) | 0.20 | 15.0 | 11.4 / 2.6 |
| s210 | 96.2 | 4 | 10 | 0 | 0 | n10 (7, 9; -0.90; 0.82; 0.66; 2.2 at 9; 7.7 → 3.4) | 0.53 | 10.7 | 27.9 / 11.5 |
| s211 | 93.5 | 3 | 11 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.29 | 26.0 | 38.6 / 17.0 |
| s212 | 84.7 | 3 | 11 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.11 | 34.1 | 20.6 / 1.3 |
| s213 | 94.0 | 0 | 12 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | – | – | – |
| s214 | 94.9 | 3 | 10 | 0 | 0 | n11 (9, 6; -0.86; 1.18; 0.36; 2.1 at 1,8; 28.5 → 23.5) | 0.62 | 9.1 | 52.0 / 28.5 |
| s215 | 92.6 | 3 | 10 | 0 | 0 | none (no memory neuron whose two largest inputs are opposite hiding terms) | 0.54 | 9.7 | 30.4 / 12.8 |

Hiding references per network: `refs.log` (trimmed refits) and `summary.log` (from `quantities_detail.json`).

### Predictions, kept with outcomes

- **Q1** (ARC: neuron 3 strict, nothing else): **held.**
- **Q2** (strict in about 6 of 37, range 3-12): **failed.** 1 of 37 (seed 115), and that one is not a gap neuron. With a cutoff of a half instead of a third, 3 networks. The reason: cancelling pairs are common (20 neurons in 14 networks have two anticorrelated hiding terms on top), but the cancellation is rarely tight.
- **Q3** (loose in 15-20): **failed.** 9 of 37. My loose test requires each term to be at least half the pre-activation, which is stricter than I thought.
- **Q4** (flagged pairs R² 0.6-0.9; opposite pairs in general 0.3-0.6): **failed for the one flagged sister** (0.42). For opposite pairs in general it is lower still (0.19). The near misses at seeds 108 and 111 reach 0.83-0.86. The reason for low values is the arriving number, which the memory neuron's weights do not cancel.
- **Q5** (references mix largest and second everywhere; one leaning on each, as in ARC): **partly held.** 91 of 103 fit as mixes, mostly on the largest. A second-leaning hiding neuron exists in only 6 networks. 12 neurons in 5 networks have age-dependent references that no fixed mix fits.
- **Q6** (flagged neurons more peaked than others): **failed as a distinction.** Peaked readouts are common among memory neurons (median 3.7). ARC's neuron 3 (2.8) is below that median.
- **Q7** (flattening a flagged pair ≥ 5 points; a random other neuron's largest term < 2; gap formula recovers most): first part held (55 points at seed 115). **Second part failed badly**: a random neuron's largest hiding term costs about 30 points. That comparison was ill-chosen: one hiding term alone carries the largest and the arriving number, so flattening it is drastic. Third part **failed** for seed 115 (46 of 55 points remain), held for the near misses at seeds 108 and 111 (about 70% recovered).

### Judgement calls

- The loose criterion is my wording of "large relative to the pre-activation"; it flags three ARC neurons, so it is not specific.
- The "raw difference" of a pair was read as the difference of the two activities, each scaled to standard deviation 1. Unweighted raw activities would depend on the free neuron scales.
- Cells where the pair barely varies (variance under 1% of its overall variance, mostly age 0) were left out of the median R².
- Substitution cells are (step, age of the largest at the previous step), with "no positive number yet" as its own cell. Means and gap fits come from one set of lists, accuracy from another.
- The strict cutoff of a third separates ARC's neuron 3 (0.28) from the sisters' near misses (0.37, 0.47) by a small margin. Read the near misses as part of the answer.
