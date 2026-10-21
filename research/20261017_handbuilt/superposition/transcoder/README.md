# Transcoders for ARC's recurrent cell (2026-10-18)

A transcoder replaces ARC's cell: it reads the previous state and the current number, passes them through a wide layer where only k units may be active (TopK), and predicts the next state. The question was whether this variant does better than the sparse autoencoders in `../sae/`, which reconstruct ARC's states well but break ARC when spliced in. It does not. Transcoders predict the next state about as well as the autoencoders reconstruct it, break ARC just as badly, and their units match the hand-built roles no better than ARC's own 16 neurons do. A control shows why: even a dense transcoder that could copy ARC's cell exactly, and random noise of the same size, break ARC as much. The limit is the precision ARC needs, not sparsity.

## Results

Prediction of the next state (held out, `train_tc.log`). Unexplained variance is 8-11% with two active units, 2.6-4.4% with four and 0.7-1.4% with eight, across widths 16 to 128. That is a little worse than the autoencoders at the same settings (0.4-0.5% at k = 8), which is expected since a transcoder must compute a step, not copy a state. Wide transcoders leave many units unused: 63 of 128 never fire at k = 2, and 34 at k = 8.

Faithfulness (`eval_tc.log`, grid accuracy from `family/score.py`; ARC itself is 95.3%). With the transcoder at every step, its own output fed back, accuracy is 12-39%. With the transcoder at the last step only, it is 15-58%. The best is 128 wide with k = 8: 39% every step, 58% last step, reproducing about 40% of ARC's mistakes. The autoencoders gave 22-33% and 33-61%. Clipping the transcoder's output at zero, as ARC's ReLU does, changes this by only 2-3 points, in either direction.

How much error ARC tolerates (`finetune_tc.log`, `dense_check.log`). Adding random noise to ARC's states, sized to leave 0.7% of their variance unexplained, drops grid accuracy to 46% (every step) and 70% (last step). Noise at 0.1% still gives 70% and 88%. A dense transcoder of width 16 with all 16 units allowed, which can represent ARC's cell exactly, trained to 0.24% unexplained variance, reaches only 40% (every step) and 70% (last step) on random lists. So sparsity is not what breaks the splice. Any fit that is not almost exact fails, because ARC's decisions are about a twentieth of the parts that produce them.

The cheap fix: training on the downstream output (`finetune_tc.log`). Fine-tuning with ARC's cell replaced at every step, with a loss on the states along the spliced run plus a loss on ARC's final scores, raises every-step grid accuracy of the 128-wide k = 8 transcoder from 39% to 74% (score weight 1) or 79% (score weight 10). The 32-wide k = 8 one goes from 30% to 62%. But this is not a more faithful copy of ARC's cell. Single-step unexplained variance rises from 0.7% to 6-10%, last-step accuracy falls from 58% to 47-51%, and it still reproduces only about 40% of ARC's mistakes. The transcoder has become a different, partly self-sufficient solver that ARC's output layer can read. At k = 4 the fine-tune barely helps (25% to 35%).

Units and simple list features (`interpret_tc.log`). For each unit, a depth-2 regression tree on simple list features predicts its activation (held-out R², higher is clearer). ARC's own neurons score a mean of 0.64. Transcoder units score lower: 0.41 for 32 wide with k = 8, 0.32 for k = 4, and 0.25 for 128 wide with k = 8. When each unit fires is easier to predict: a depth-2 rule gets a balanced accuracy of 0.85-0.89 for both transcoder units and neurons. Many transcoder units are rare: 10 of 32 and 41 of 91 live units fire on under 1% of steps.

The ten clearest units, from the 32-wide k = 8 transcoder unless noted, in plain words:

1. How far the current number falls below the largest so far; silent when a new largest arrives (active 71% of steps). ARC's neuron 7 does the same job more cleanly.
2. A positive new largest has just arrived; its size is the gap between the new largest and the second-largest (active 24%).
3. A near copy of item 2 (active 22%). The dictionary splits one role across two units.
4. The previous number, when it is positive; the size grows with it (active 35%).
5. The previous number, when it was the largest so far (the largest arrived one step ago); the size is that number (active 27%).
6. The number two steps back, when positive and the largest arrived recently (active 20%).
7. The largest arrived five or more steps ago; the size grows with that age and with the gap between largest and second (active 31%).
8. The largest arrived four to eight steps ago, or recently with a current number above about 0.5 (active 47%).
9. Early in the list, while there is no positive second-largest yet and the current number is close to or above the largest (active 30%).
10. From the 128-wide transcoder: both the largest and the second-largest so far are small; the size falls as the largest grows (active 54%).

These are the same kinds of quantity ARC's neurons carry: the largest, the gap to it, recent numbers and the age of the largest. None is a cleaner single feature than ARC's best neurons.

Hand-built roles (`roles_tc.log`, the fair comparison of `../sae/eval_roles_fair.py`). After removing the parts every unit shares (largest so far, current number, step), each hand-built role is matched to its best-correlated unit. ARC's neurons give a mean best correlation of 0.86 for the v19 roles, 0.71 for v21 and 0.73 for v7z. Transcoders of every width and k give 0.83-0.88, 0.68-0.73 and 0.60-0.75. The 32-wide k = 8 transcoder is at or slightly above the neurons (0.88, 0.73, 0.75). The differences are within what the widths and k values vary by, so this is no real gain.

## Method

`train_tc.py`: input is the previous state divided by each neuron's standard deviation over all steps, plus the current number (17 numbers). The target is the next state, scaled the same way. The encoder is a ReLU then TopK, and the decoder is linear with unit-norm columns and a bias. There is no dead-unit revival. It uses Adam for 6,000 steps of 8,192 pairs, with a learning rate of 0.001 dropping to 0.0003 at step 4,000. Data is 800,000 random lists, all ten steps, with the last million pairs held out. This is the `../sae/train_sae.py` recipe with the input and target changed.

`eval_tc.py`: splices the transcoder in place of ARC's cell at every step or the last step only, and scores with `score.report`. `finetune_tc.py`: the downstream fine-tune (3,000 steps of 4,096 fresh lists, ten steps unrolled, loss on scaled states along the spliced run plus a weight times the divergence from ARC's softmax scores), and the noise reference. `dense_check.py`: the same recipe with k = 12 or 16 at widths 16 and 32. `interpret_tc.py`: 40,000 random lists. It uses 19 list features (current number, step, new largest, new second-largest, running top three, steps since the largest and the second-largest, current minus previous largest or second, current above 0.5, 1, 1.5 or 2 times the previous second, numbers 1-3 steps back, largest minus second), depth-2 trees fitted on half the lists and scored on the other half, with ARC's neurons as a reference. `roles_tc.py`: the fair role comparison, with autoencoder rows recomputed as a check (they match `../sae/eval_roles_fair.log`).

## Caveats

Each setting was trained once, with one seed. The fine-tune tried one setting per model (one or two score weights, 3,000 steps). A longer or differently weighted fine-tune might push every-step accuracy higher, but the rising single-step error suggests it would do so by departing from ARC's cell. The decoder output is linear, so states fed back can be slightly negative; clipping them changed little. Grid accuracy uses the scorer's grid lists; random-list accuracies (in `finetune_tc.log` and `dense_check.log`) are close to it. Unit descriptions come from correlations and shallow trees, not from interventions. The ten plain-word descriptions were written by reading those trees, and some are approximate.
