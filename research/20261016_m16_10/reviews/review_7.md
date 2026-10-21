# Review 7: the rewritten report summary and the remaining plan (01:15–01:45 UTC, 2026-10-17)

The results are mostly right, but the Summary does not yet explain the mechanism to a newcomer, and six claims overstate the logs. Spend the next two hours on the write-up, then close three small goal-1 loose ends. More estimator work would not change the challenge verdict, so stopping research early after that is reasonable.

I checked numbers against the run-3 logs and ran nothing on any model.

## 1. Summary clarity

**What is missing.** The Summary never says what the clock is for: its position records how long ago a number arrived. The core mechanism, two cancelling copies of the clock and a new largest wiping one, is only in the body. The split "six compare, ten hold the clock" even suggests the comparing neurons are not part of it. Sliding needs a worked example, and the three goals need a status.

**Suggested neuron bullets** (every fact is in the body or logs):

- **Six neurons compare values.** They hold the largest and second-largest numbers so far. ARC and two commenters described them first. Run 2 derived their exact formulas from the weights.
- **A hidden clock records when each number arrived.** A positive number sets the neurons' activity rotating, about one turn every 8 or 9 steps in ARC's model. How far it has turned says how long ago the number arrived. Seven of the other ten neurons each light up at a different point of the turn. Two fire one step after any input. One (neuron 3) stays weak and has no clear role.
- **The output cannot see the clock, because two copies cancel.** The clock runs through the six comparing neurons and through the other ten, and their effects on the scores are equal and opposite. This matters because the largest number is never the answer, so it must leave no vote.
- **A bigger number reveals the old largest's clock.** It wipes four comparing neurons. Their copy vanishes, the other copy shows, and it votes for where the old largest, now second, arrived.
- **Votes slide like a conveyor belt.** A second-largest arriving after the largest votes for the newest position, 9. Each later step moves every vote back one position. If it arrived at position 6, its vote moves to 8, 7 and then 6 as the last three numbers arrive.

**Sentence-level fixes.**

| Where | Problem | Suggested fix |
|---|---|---|
| Task paragraph | omits "at every budget", which later paragraphs rely on | add "at every amount of compute" |
| "30 to 160 degrees per step" | meaningless unit for a newcomer | "from one turn in 13 steps to almost half a turn per step" |
| "driven directly by the input ... place votes less cleanly" | undefined and vague | "with many large input weights ... match the simple picture less often when the small inputs are zero" |
| "each sliver's exact position is its own number" | unclear | "each sliver sits in a slightly different place for each arrangement of the three largest" |
| "spends its effort where inputs are common" | not the reason | "random inputs land in each sliver as often as real ones, so counting needs no map of the slivers" |
| "squared error is 2.3 times sampling's" | abstract | "typical error 0.9 points, against 0.6 for sampling at the same compute. Sampling would match it with less than half the compute." |
| Techniques paragraph | semicolon chain; "a quarter of answers flip" reads as ARC's model | bullets; "in models that find the median, a quarter flipped" |

**Order.** After the task, add a three-line status: goal 1 mostly met, goal 2 met within this family of models, goal 3 open but this run's own bar of three times sampling met. Move "short sequences" next to the best estimate, with Techniques after. In the body, merge the stub "The other ten neurons" into the run-3 section, renamed "How the other ten neurons work".

## 2. Correctness

**Overclaims.**

1. **"Every neuron now has a known job"** (Summary, note). Neuron 3 is only "small", and neuron 9's role is not stated.
2. **"All 22 have the hidden clock", right after "wrote predictions down first".** The written-down wheel test passed in 15 of 16 new models, and the planned turn-rate measure (ages 1–20) read zero for two. The all-22 result uses ages 1–9, a window chosen afterwards (step 88). Label it. Hiding holds in 21 of 22 (seed 113: 0.35). The realistic-background test was written down first, but proposed after seeds 0 and 4 failed at zero.
3. **The output is "nearly blind" to the weights' main rotation "in all 22".** Visibility is at most a tenth of a random plane's in 15 models, and 0.29–0.58 in seven. Comparing two logs, I also find seven models (seeds 0, 1, 3, 4, 102, 105, 108) where that rotation's speed differs from the activity clock's by 30–130 degrees per step. Seed 1 reads 28 in the weights and 157 in activity. There the weight test may describe a different rotation. Say "far less visible than chance in all 22, under a tenth in 15", and run check (c) below.
4. **"A neuron that responds exactly like ARC's neuron 4."** Neuron 4's probe profile is one spike at arrival, so any neuron that fires on arrival and then stays silent scores 1.00. The new-largest behaviour was checked on full inputs only in ARC's model.
5. **"Its two steps add about equally"** (note, limits). The reduction is off by 0.52 points and the band step by 0.78, so the band step's squared error is more than twice as large. In the figure, the ideal three-input route sits at 0.54 points for about 300 runs, where sampling is at 1.5. At this budget the band step, not the floor, is what loses. That deserves a Summary sentence.
6. **"A general proof ... cannot exist."** Too strong. Suggested: "No proof of impossibility is within reach. For one fixed model, a method that already knows the answer exists, so impossibility only makes sense across many models, and proving it there needs lower bounds complexity theory cannot yet give."

**Mismatched or unlabelled numbers.**

- "Run 2's best was about 60 times worse" was measured on noisy copies, where the new estimate is about 3–7 times. Compare like with like.
- The six-model "tie" (1.17) counts 6 operations per integration point. Under the 30-operation rule behind 2.3 it is about 1.5 (review 6's arithmetic, no log). Technique 11 mixes the two rules.
- The floor's crossover is "roughly 2,000" runs in the body and limits, "a few thousand" in the Summary. At 0.54 points it is about 2,500.
- "No short rule predicts" should be "no short rule we tried". Tables of up to 180 numbers explain a third to two-thirds on held-out placements.
- With three inputs kept, seven of 22 reduction errors are negative. With four to six, nearly all are positive (ARC's model: +0.01 at three, +0.16 at four). So the three-input fit is partly cancellation. Say so under limit 6.

**Older sections.**

- "Four neurons act as a clock" (7, 6, 1, 8): say they carry the input-reachable copy of the same clock.
- "Small numbers matter too" (top three explain 63% of per-input variation) sits just before "The three largest inputs set the accuracy". Add: per input they matter, on average their effects cancel.
- The old "93.5% of mistakes have a gap below 0.1" uses a looser gap than the new table, where 18% have only one positive number. Drop or relabel it.
- Small fixes: the estimator uses a fixed −0.46, not each input's expected value. "Words used below" still defines trained models as six. "The tiny model below" (line 107) is above. Two-spike accuracy is 97.7% here and 97.8% in techniques (log: 0.9776). "Steps without a new input" means steps where the new number is zero. A blank line is missing before "## Prior work".

**Claims table.** It stops at row R34 (step 73). The headline 2.3 has no row, nor do steps 79–91.

## 3. Remaining directions

Goal 3 is at its honest end: a better number at one budget would not change the verdict. Most remaining value is in the write-up.

1. **Fix the write-up** (about 2 hours, no compute), including the note, limits, techniques and claims rows from R35. This is the deliverable, so no kill rule.
2. **Close three goal-1 loose ends** (about 45 minutes, a few GPU minutes). (a) Zero neurons 3, 5 and 9 one at a time in ARC's model. Kill: under 0.1 points lost means "no measurable role". (b) Run the existing neuron-4 full-input check on the 15 matched neurons. Kill: if most fail, write "fires only on arrival". (c) In the seven mismatched models, measure the angle between the weights' rotation plane and the activity plane. Kill: if large in most, say the weights confirm the hiding only where the speeds match.
3. **Optional transformer pilot for goal 2** (at most 1 hour). Every "carried over" result is from RNNs. Smallest test: in ARC's local longest-cycle transformer, measure how much of the first layer's output the unembedding cannot see but second-layer heads read. Kill: if no clean definition of the hidden part emerges in 30 minutes, record that and stop.

Stopping after directions 1 and 2 is a fine outcome.

Not recommended: more estimator work. Time-shift sharing would move 2.3 to about 1.2–1.6 at one budget, needs a fresh batch of seeds, and still loses beyond a few thousand runs. Further attempts at a near-tie bias formula or copies' bad ages have failed several times each.

## Recipes page

Only "small gains need a second seed or fold" applies. The 16 fresh models were that check, and they shrank the gain.
