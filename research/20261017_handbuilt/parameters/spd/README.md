# Parameter decomposition (SPD) of ARC's second-largest network (started 2026-10-20)

Question: if we run Stochastic Parameter Decomposition (SPD) on ARC's 16-neuron network, what pieces does it find, how
faithful are they, and do they match our hand-found account (neuron groups, blindness, hiding feedback, late-gate timing,
memory rotation, store-detector loop)?

## What SPD does (Bushnaq, Braun, Sharkey 2025, arXiv 2506.20790; code github.com/goodfire-ai/spd, branch spd-paper)

- Each weight matrix is written as a sum of C rank-one pieces ("subcomponents"), each an outer product u v^T.
- Each piece has its own small network (1 → 16 GELU → 1, then a hard sigmoid clipped to 0-1) that reads the piece's
  input activation v·a and outputs a "causal importance" between 0 and 1 for this input.
- Training: random masks m = importance + (1 − importance) × uniform(0, 1). The network run with each piece scaled by its
  mask must reproduce the original output (all matrices masked together, and one matrix at a time). The pieces must sum to
  the original weights. A penalty, β × Σ importance^p, pushes importances to zero.
- A piece is "active" on an input when its importance is above a threshold (their code's statistics use 0.1).
- The paper gives no bit count. Its predecessor (APD, arXiv 2501.14926) motivates the method by minimum description
  length: few pieces active per input, each simple (low rank).

## Setup and design choices

- Code: a compact reimplementation (`scripts/spd_rnn.py`, about 200 lines of PyTorch), not their framework. Reasons:
  - their framework swaps `nn.Linear` modules in a feed-forward model; ARC's network reuses one recurrent matrix at ten
    steps, which would need custom hooks there anyway;
  - the whole method fits in a short file whose every line can be checked against their `losses.py` and
    `component_utils.py` (cloned into `sources/spd_repo`, not committed);
  - no new packages are needed (torch from the shared environment).
- Matrices decomposed: input weights (16×1), recurrent (16×16), readout (10×16). Pieces: 8, 48 and 20 (about three times
  each matrix's rank, plus room for the readout's free shift). Judgement call.
- Recurrence: the input and recurrent pieces get an importance per list and per step, from that step's input activation in
  ARC's own forward pass (the analogue of SPD's per-token masks in language models). The recurrent matrix is used at steps
  1-9 only, since the state starts at zero. The readout gets one importance per list.
- Output divergence: KL(ARC's output ‖ masked network's output), our standard's measure.
- Loss weights: faithfulness 1 (sum of squared weight errors / 432), both reconstruction terms 1, importance penalty β
  tuned in a short pilot, p = 1 (their small-model setting). Adam, cosine learning rate from 1e-3, batch 2048. (The pilot
  below changed the faithfulness weight to 100 and the learning rate to 3e-3.)
- Evaluation on 2^20 fresh lists: KL and accuracy with (a) only active pieces (mask 1 if importance > 0.1, else 0),
  (b) masks equal to the importances, (c) all pieces.

## Predictions (written before the pilot and the main run)

1. Faithfulness with only active pieces: KL between 0.02 and 0.1 (worse than our budget), accuracy 90-95%. SPD is
   trained on random masks, not on the 0/1 masks used for evaluation.
2. The recurrent matrix will be dense in use: 15-30 live pieces, most active at almost every step (the memory rotates
   every step). A few sparse pieces, active only around records, will write to the gates and the detector.
3. The readout splits into about one piece per position, and the readout's free common shift becomes a piece that is
   never active (a pure symmetry artefact).
4. The input weights stay one piece, active at every step, because a number arrives at every step.
5. Mechanisms: a store-detector piece (reads the detector, writes the store) active at records: likely. A late-gate piece
   (reads neuron 9, writes hiding 1 and 6), active two steps after a record: likely. A hiding-feedback piece (reads and
   writes within hiding 1, 6, 7, 8): 50/50. Blindness will not appear as a piece; at most as a piece that reads the
   fresh-number direction and writes only to the gates. Memory modes will not appear as separate pieces.
6. Neuron rescaling: pieces from a re-scaled copy of ARC, scaled back, will match the original pieces about as well as a
   second random seed does (the losses are unchanged by rescaling except through the faithfulness weighting).

## Pilot: learning rate, faithfulness weight and β (5,000 steps each, `results/pilot_*.log`, `results/pilot2_*.log`)

- With SPD's default weights (faithfulness 1, learning rate 1e-3), the pieces never added up to ARC's weights in 5,000
  steps (mean squared weight error 0.2-0.4). The network with all pieces switched on was far from ARC (KL 2.0 and more).
- ARC's weights are large (the hiding neurons' and detector's input weights are 10-13) and its output is very sensitive to them (our lookup table needs about
  6 bits per direction). So the faithfulness term needs more weight here: with faithfulness 100 and learning rate 3e-3 the
  squared error fell to under 1e-5 and the all-pieces network reached KL 0.010. This is a departure from their settings.
- At β = 1e-3 almost every piece was active on every list and step (about 350 recurrent piece-uses per list, out of
  48 × 9 = 432 possible). At β = 1e-2 with unconverged weights, pieces were sparse but the network was useless (KL 5).

Main runs: faithfulness 100, learning rate 3e-3, 20,000 steps, β = 1e-3, 3e-3 and 1e-2.

Revised prediction before the main runs (prediction 2 already looks too optimistic about sparsity):
2'. At the β where the active-only network stays under KL 0.1, the recurrent matrix will use more than half of its
    possible piece-steps. SPD will not find a sparse description of this network.

## Results

All evaluations on 2^20 fresh random lists (ARC: 95.31% right on them). "Active" means importance above 0.1; the
active-only network sets every other piece's mask to 0. Piece-uses are summed over steps; the most a list can use is
8 × 10 input, 48 × 9 recurrent and 20 readout. Logs: `results/<run>.log`; piece-by-piece descriptions:
`results/<run>_analysis.log`.

### Faithfulness and sparsity

| run | start | β | KL, active pieces only | accuracy, active only | KL, all pieces | recurrent piece-uses per list | distinct recurrent pieces per list | input piece-uses | readout pieces per list |
|---|---|---|---|---|---|---|---|---|---|
| main_b1e-3 | random | 1e-3 | 0.022 | 94.8% | 0.004 | 276 | 39 | 48 | 19.7 |
| seed1_b1e-3 | random, other seed | 1e-3 | 0.024 | 94.8% | 0.008 | 300 | 42 | 53 | 19.8 |
| gauge1_b1e-3 | random, ARC re-scaled | 1e-3 | 0.019 | 95.0% | 0.005 | 263 | 36 | 62 | 20.0 |
| long_b2e-3 | random, 40k steps | 2e-3 | 0.065 | 94.2% | 0.017 | 150 | 25 | 19 | 16.1 |
| main_b3e-3 | random | 3e-3 | 0.115 | 92.5% | 0.055 | 150 | 26 | 26 | 18.8 |
| main_b1e-2 | random | 1e-2 | 0.366 | 87.0% | 0.875 | 58 | 17 | 12 | 14.6 |
| neuronwarm_b1e-3 | neuron basis, gates warmed up | 1e-3 | **0.007** | 95.2% | 0.002 | **86** | 16 | 10 | 13.1 |
| neuronwarm_b3e-3 | neuron basis, gates warmed up | 3e-3 | 0.029 | 94.8% | 0.008 | 80 | 16 | 9.3 | 12.4 |
| (baseline) neuron basis, no training | — | — | 0 (exact) | 95.3% | 0 | 101 | 16 | 10 | 12.4 |
| (baseline) singular value decomposition | — | — | 0 (exact) | 95.3% | 0 | 144 | 16 | 10 | 10 |

"Neuron basis" (`scripts/baseline.py`, `results/baseline.log`): piece j of the recurrent matrix and of the readout is
column j (everything neuron j sends). It is needed only when neuron j is on, so it is "active" exactly then. On average
8 neurons are on at step 1 and 11-12 at later steps. The input weights are one piece, used at every step.

- Prediction 1 held for the random start at β = 1e-3: KL 0.022 with active pieces only, just over our budget of 0.02,
  and 94.8% right. Masks set to the importances themselves (rather than 0/1) are worse (KL 0.056), because many
  importances sit between 0 and 1.
- Prediction 2 and 2' held. From a random start SPD is dense. At the β that keeps KL near our budget, each list uses
  about 30 of the 48 recurrent pieces at each step: more rank-one pieces than a 16 × 16 matrix needs to be written exactly.
  Larger β buys sparsity only by losing faithfulness, and the summed pieces then drift from ARC's weights (KL with all
  pieces 0.05 at β = 3e-3, 0.9 at β = 1e-2).
- **The main finding: SPD's random-start solutions are worse, on SPD's own terms, than the network's neurons.** The plain
  neuron basis is exactly faithful and uses 101 recurrent piece-uses per list, against 276 for SPD at KL 0.022. Started
  from the neuron basis (with the importance networks trained alone for 3,000 steps first), SPD keeps the neurons: the
  recurrent pieces still read one neuron each (cosine 0.96-1.00 with a single neuron). It only learns that some of a
  neuron's on-steps do not matter, cutting 101 uses to 86 at KL 0.007. Its training loss is lower than the random
  start's on every term (importance 105 against 318, reconstruction 0.004 against 0.018). So the random start is stuck in
  a much worse local optimum, and the best SPD solution we found is "the neurons, used when they matter".
- The first neuron-start attempt, without warming up the importance networks, failed (`neuroninit_b1e-3`, KL 0.74): random
  importance networks at the start give masks that wreck the output (KL 33), and the large early gradients scrambled the
  pieces.

### What the pieces are (random start, `main_b1e-3_analysis.log`)

- **Input weights: split by the number's sign (prediction 4 failed).** All input pieces share one input (the number), so
  their importance can depend only on the number's size and sign. SPD made two pieces that are on at every step and six
  that are on only when the number is positive. For negative numbers the remaining input vector is about a third of ARC's
  (hiding neurons about −3.4 instead of −10.6 to −13.2; detector +1.4 instead of +10.2), `main_b1e-3_input_split.log`.
  - A direct check in ARC itself (`scripts/neg_input_test.py`, `results/neg_input_test.log`, 2^20 lists): giving negative
    numbers SPD's vector costs KL 0.007; a third of ARC's input weights costs 0.012; ignoring negative numbers entirely
    costs only 0.036 and 0.1 point of accuracy.
  - Why, in our account's terms: the second-largest of ten numbers is negative on only about 1% of lists, so negative
    numbers rarely matter. Scaling only the hiding neurons' entries by a third is much worse (KL 0.075) than scaling the
    whole vector (0.012). That fits blindness: the next step cancels the fresh number only if the whole input vector keeps
    its proportions. This is a real fact about ARC that our account had not stated.
  - The six positive-only pieces have nearly the same importance pattern; one piece would do. That redundancy is an
    optimisation artefact.
- **Recurrent pieces: mixtures, not our groups.** Most pieces read and write several groups at once (for example "writes
  late gate 0.74; reads memory 0.70, store 0.14"). Of the 45 live pieces, 30 are used at 6 or more of the 9 steps of an average list, and 16 at
  7.5 or more. The pieces whose removal hurts most all write mainly to the late gate (neuron 9), which must stay silent
  except after records, so any error in its input is costly.
  - Late-gate timing: one piece writes only to the late gate, reads mainly the detector, and is active when the largest
    arrived one step earlier (0.74) and almost never otherwise (0.02-0.11). That is "the late gate fires one step after a
    record". The push from the late gate onto hiding neurons 1 and 6 did not appear as its own piece (prediction 5).
  - Blindness: the two pieces that read the fresh-number direction most (cosine 0.55 and 0.72) write almost only to the late
    gate (0.92 and 0.85 of their output). Consistent with "only the gates read the arriving number" (prediction 5 held
    in its weak form). It is not new: it must hold in sum, since the pieces add up to ARC's matrix.
  - No clean store-detector piece, no hiding-feedback piece (reading and writing only hiding neurons), and no piece that is
    one memory rotation mode (one piece is mostly the 43° mode, reading 0.66 and writing 0.76 of its weight there).
- **Readout: 20 dense pieces (prediction 3 failed).** Every piece is active on nearly every list; none is one position;
  none points along the readout's free common shift (cosine with the all-ones direction at most 0.2). Twenty rank-one
  pieces are used for a matrix of rank 10.

### What the pieces are (neuron start, `neuronwarm_b1e-3_analysis.log`)

Each piece is one neuron's outgoing weights, so the interesting output is when each is needed:
- detector: needed when the largest arrived one step earlier (0.75), almost never otherwise (0-0.03);
- late gate: needed one step later (0.87), never after (0-0.02), so its push lands two steps after a record, as in our
  account;
- keep gate: needed one step after a record (0.89), otherwise about a quarter of the time (numbers near the second-largest);
- hiding and memory neurons: needed at most steps.
These are the firing patterns our account already describes; SPD adds nothing beyond them here.

### Symmetries and stability (prediction 6 failed, in an informative way)

Pieces were matched between runs by the cosine of their rank-one matrices (readout pieces after removing the common
shift), one-to-one. Two sets of random rank-one pieces match at a median of 0.17.
- Another random seed: recurrent pieces match at a median of about 0.4, readout about 0.5. The decomposition is not
  reproducible; it depends on where training starts.
- ARC re-scaled by random neuron scales and given a random readout shift (same start seed), scaled back: median about
  0.8. Closer than another seed, but the shared random start explains part of that.
- β = 3e-3 against β = 1e-3, same seed: median about 0.65. Again the shared start dominates.
- So the 32 free symmetries are not the main source of arbitrariness; the random start is. The neuron-start pieces are
  immune to the neuron rescalings by construction (rescaling maps a neuron's column to itself), and matched each other
  across β at 0.95-1.00.

### Description length

SPD gives no bit count. Its implicit measure (from APD) is how many pieces, of what rank, are active per input. Our
2,369 bits is something else: the bits needed to state ARC's 432 weights precisely enough to stay within KL 0.02, against
a Gaussian prior per block. The two can only be set side by side in counts of numbers:

| description | numbers to state | active per list |
|---|---|---|
| ARC's weights (our lookup table: 2,369 bits at KL 0.02) | 432 | all 432 |
| SPD, random start, β = 1e-3 | 8 × 17 + 45 × 32 + 20 × 26 = 2,096, plus 49 per piece for the importance networks | about 39 recurrent, 8 input, 20 readout pieces: about 1,900 numbers |
| SPD, neuron start, β = 1e-3 | 432 (the neuron columns) plus importance networks | 16 recurrent, 1 input, 13 readout pieces: about 870 numbers counting each piece fully |

- The random-start decomposition needs about five times as many numbers as the weights it explains, and its pieces must be
  stated at least as precisely (its summed pieces already miss ARC by KL 0.004). It cannot save bits against our lookup
  table; it would cost several thousand more.
- The neuron-start decomposition restates the weights unchanged and adds "when is each neuron needed". At best it saves
  nothing on the weights. Its importance networks carry information (which neuron-uses are idle), but our accounting has
  no slot for it, since our behaviour is "match ARC's output", not "explain which weights matter per input".
- Against our explanations' savings (about 100-150 bits net out of 2,400 so far): SPD offers no comparable saving here.

## What this says about our account

- SPD did not find any structure our account lacks, apart from one small fact: negative numbers hardly matter, and the
  input weights for them can be shrunk by a third, or dropped, at small cost (KL 0.012 and 0.036). That fact is consistent
  with blindness, which needs the input vector's proportions kept.
- On this network, neurons are a better unit than SPD's learned pieces. The ReLU already switches whole columns off, and
  the role of each neuron (gates, detector, store, hiding, memory) is visible in when its column is needed. This supports
  our choice to describe ARC neuron by neuron.
- The mechanisms that live across neurons (memory rotation modes, hiding neurons reading each other, the store-detector
  recursion) did not appear as pieces. Rank-one pieces with per-step on/off switches cannot express "a rotation in a
  plane, used every step" more compactly than the matrix itself.
- Judgement calls that could have changed this: the per-step importance design (a per-list design would be even
  denser), C = 48 recurrent pieces, the faithfulness weight of 100, and 20,000 steps. A wider search over starts might find
  a random-start solution as good as the neuron start; we found none in four random starts.

## What to try next

- Start SPD from other structured bases (our hand-built 16-neuron network's roles, or the rotation modes) and see whether
  it keeps them, as it kept the neurons.
- Replace the per-step on/off importance with one per list and position of the record, which matches how our account
  describes timing, and see whether the memory modes then become pieces.
