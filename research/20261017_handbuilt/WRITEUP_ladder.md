# Explaining ARC's 16-neuron network, order by order (draft, 2026-10-21)

## Summary

ARC's AlgZoo network reads ten random numbers, one per step, and names the position of the second-largest. It has 16 neurons and 432 weights, and it is right on 95.3% of lists. This note states what we understand about it as a ladder of approximations, the way physics states an effective theory. Each rung says what it explains, how it was tested, and how much error it leaves.

In short:
- **What the network computes, and the algorithm it uses, are understood.** Every number big enough to matter casts a vote for its own position. Four neurons hide the vote of the current largest, and a bigger number switches them off, so the old largest's vote shows just as it becomes the second-largest. Switching parts off confirms each group's job.
- **How each group carries out its job is understood,** with short rules for most of it. The memory's switching, which the network needs (44% without it), follows a rule on the list with about 50 numbers. Every one of the 16 neurons now has a known job.
- **Why the weights have their values is understood for a small part, about 9%.** The explained part comes from a few exact reuses. One is universal: it appears in nearly every network trained the same way. Others appear mainly in the most accurate networks. The memory's values are particular to this training run.
- **Which way a near tie falls is not understood,** and may have no short explanation.

## The ladder

| Rung | What it says | Tested by | Error left |
|---|---|---|---|
| 0. Task | Name the second-largest | — | ARC itself is wrong on 4.7% of lists |
| 1. Algorithm | Votes, hiding, reveal; five groups with one job each | Switching groups off; a 16-neuron network built by hand from the outline | The hand-built network reaches 96%, but it is a different network: copying its groups into ARC keeps ARC working only for the store (85%) |
| 2. Mechanism per group | Rules for each group's activity and switching | Putting the rule in place of the group | Memory switching: 1.4 points with about 50 numbers. Hiding neurons: 1 point with 648 numbers. All 16 neurons' activities from short formulas at once: 51%, because each must be right to a few percent |
| 3. Values | Rules that fix the weights themselves | Surprise accounting; noise control; 37 sister networks | About 9% of the weights' description saved; the rest must still be listed |
| 4. Near ties | Which way a close call falls | Redrawing the smaller numbers; rules on the three largest | No short rule found; every input moves the decision a little |

## Rung 1: the algorithm

Five groups do the work. The **store** holds the largest number so far, and a **detector** fires when a new largest arrives. A **keep gate** lets a number vote only if it is at least about three-quarters of the second-largest so far. The **hiding neurons** cancel the vote of the current largest. The **late gate** keeps a new largest hidden for the step after it arrives. The **memory** holds the votes and rotates them each step, so each lands on its own position at the readout.

Evidence: silencing the hiding neurons once, right after the largest arrives, makes the network name the largest on about 70% of lists; silencing any other four neurons gives at most 24%. Switching off the keep gate drops accuracy to 21%. Switching off the late gate drops lists whose answer is an overtaken largest from 95% to 49%. Nudges predicted from the account moved close decisions the predicted way in 16 of 16 settings, after the first version of the account was corrected.

What this rung does not give: the exact values. A network built by hand from this outline works, but it stores things differently, so it is a separate solution of the same algorithm, not a cleaned-up copy of ARC.

## Rung 2: how each group works

Each group has a rule that reproduces its activity:
- the hiding neurons compare the arriving number with a reference built from the largest and previous largest; the written rule leaves 94.4% (network 95.4%) with 648 fitted numbers;
- the memory rotates at 0°, 38°, 84°, 127° and 180° per step. Without any switching the network gets 44%. Its switching is mostly a clock of the largest's age: neuron 0 is off two steps after the largest arrives, 11 five steps after, 10 six steps after; 12 can stay on; 13 and 14 are off only at the first positive number; neuron 3 follows a score of the largest's and second-largest's ages; neuron 15 is off when the turning votes of the numbers so far sum below zero, the only one that needs sizes. This rule, with about 50 fitted numbers, leaves 94.0% against the network's 95.4% on fresh lists (`switching/README.md`; the earlier rule needed about 8,300 numbers). It fails most on lists whose last new largest comes at step 9 or very early;
- neuron 3 is one of the memory's coordinates (about two-thirds of the 127° rotation) and also carries the gap between the largest and the second-largest, read as hiding neuron 1 minus hiding neuron 8. That gap makes the hiding of the largest grow with how far it stands above the second; without it, lists with a big gap fail by naming the largest. Replacing neuron 3's input by the gap formula costs 2.8 points, by another list's gap 25 (`neuron3/README.md`);
- the late gate fits a 7-number rule (−0.6 point); the keep gate's value needs about 550.

Rules that reproduce accuracy can still be wrong on single lists. The switching rule's on/off pattern is wrong on about 1 list in 10; those errors largely cancel in overall accuracy, but not list by list (see rung 4).

For the neurons' activities themselves, the error is not in the idea but in precision. A typical near-tie decision is a small remainder of large terms: the groups' contributions are about ten times the final margin, and they cancel. So each group's rule must be right to a few percent on each list, and short rules are not.

## Rung 3: why the weights have their values

The measure is surprise accounting: is "the rule plus corrections" a shorter description of ARC's 432 weights than a plain list? The plain list expects each block of weights between two groups to have its own typical size, and ignores the one readout direction that changes no answer. It takes about 1,916 bits. The best rule saves about 172 bits, about 9%, after an independent audit and a noise control. The rule names these mechanisms:

| Mechanism | Bits saved | Kind |
|---|---|---|
| **Fair votes in the readout.** Each position's readout row is the previous row pushed one step through the dynamics, because a number one position later is read one step younger. | Most of the total | **Universal:** 31 of 37 sister networks; the first structure to form in training |
| **Comparators reusing one reference.** Each hiding neuron computes its input weight times (arriving number − reference) and reuses the detector's reference. One hiding neuron's whole row is the detector's row mirrored; all four read the store at the detector's ratio. | About 28 | **Tied to accurate solutions:** the store ratio in 21 of 33 sisters; blindness (the arriving number cancels in what other neurons read) tracks accuracy across the 37 (rank correlation 0.71) |
| **The detector's row tied to the store's.** The multiple follows from "a new record resets the store exactly": 0.560 predicted from two other weights, 0.552 fitted. | About 5 | Rare (5 of 35), but in four of the nine most accurate |
| **A delay line in the memory.** One memory neuron is another one step later. | A few | Rare in exact form; recurs in the best networks at different ages |

Checks: each mechanism saves far more on ARC than on 40 copies of ARC with weights shuffled within blocks (the rules save at most 7 bits on any copy before paying for their form, against 96-397 on ARC). One fresh agent rebuilt the rule from its written description alone and matched the score. An audit removed about 27 bits of overcredit; the noise control corrected the charges for choices made by score.

What the rule leaves: the memory's 64 weights, apart from the delay line; the gates' large weights, which look like margins set by training; the first readout row. The sister networks suggest why the memory resists. Its form is universal (rotating modes, neurons switching off at fixed ages of the largest), but its values differ from network to network. ARC's switch-off ages of 5 and 6 steps are rare; most sisters use 1-2. These look like material constants of this training run: a theory can measure them, not derive them.

ARC's own model was trained with an earlier, unpublished recipe, so it is not strictly a member of the sister family. It sits at the tidy end of almost every structural measure.

## Rung 4: near ties

On lists where the third-largest is within 5% of the second, about one list in ten, ARC names the third 21% of the time; these lists hold nearly half its mistakes. Which way such a decision falls depends on every input a little: redrawing the seven smaller numbers flips 17% of them. A rule on the three largest numbers needs about 25,000 parameters to beat the base rate by seven points. Hand-built networks settle their near ties by a short rule; every trained network checked does not.

The same limit stops a mechanistic accuracy estimate, ARC's own challenge. The latest attempt fixed the on/off pattern for each arrangement of the three largest numbers from the switching rule and integrated the rest exactly. It estimated 93.8% against the true 95.32% at 720 runs, where random sampling at that cost misses by about 0.8 points; over 32 sister networks it missed by 5.6 points against sampling's 1.0 (`estimate/README.md`). The best earlier estimate, built on the widths of close-call bands, misses by about 0.7-0.8 points at about 1,750 runs, roughly level with sampling. Describing the network arrangement by arrangement takes about as many cases as sampling takes lists.

## What this means

- **Function and algorithm:** explained, and tested by intervention.
- **Mechanism:** explained, mostly by short rules: the memory's switching in about 50 numbers, and a known job for every neuron. Exact activity rules still cost many numbers, because the decision is a small remainder of large, cancelling terms.
- **Values:** about 9% explained. The explained part is a few exact reuses. The biggest is universal and forced by the task. The others are what accurate solutions converge to. The memory's values look particular to this run.
- **Close calls:** not explained, and possibly not shortly explainable. This is where the network's specific mistakes live, and why no estimate from the explanation has beaten sampling.

Judgement calls behind the numbers (decisions of 2026-10-21): rows may be described in a fixed order, each given earlier rows; the plain list uses one typical size per block and ignores the readout's shared offset; patterns found by looking at the weights pay for the search. Details: `parameters/AUTORESEARCH_NIGHT_20261021.md`, `parameters/noise_control/README.md`, `parameters/universal/README.md`, `parameters/training/README.md`, and `FINDINGS.md`.
