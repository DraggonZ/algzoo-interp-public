# Is ARC unlucky? Close-call density across sibling networks (2026-10-18)

Status: done (`siblings.py`, `siblings.log`). Every trained network is dense; ARC is typical.

Measures, per network: accuracy; share of near ties (third within 5% of the second) where it names the third; second-over-third lead maps at 3% precision covering half of 100,000 lists; and how much of the possible gain over always-guessing a tree on the top three (positions and two ratios) reaches on near ties.

| networks | accuracy | names the third at near ties | lead maps covering half | tree's share of the possible gain |
|---|---|---|---|---|
| ARC | 95.4% | 20.8% | 7,955 | 31% |
| 38 other trained 16-neuron networks | 84-96% | 16-36% | 1,388-36,023 | 19-39% |
| trained 20-neuron network | 97.9% | 13.3% | 1,494 | 26% |
| 7 trained 32-neuron networks | 99.0-99.4% | 2-5% | 5,962-33,258 | 3-20% |
| hand-built v19 (16) and v21 (28) | 94.2%, 95.9% | 7.2%, 7.3% | 46, 48 | 91%, 84% |

Prediction (written first): all trained 16-neuron networks dense, right; the 32-neuron networks cleaner, wrong: they make far fewer near-tie mistakes but what remains is even less predictable from the top three (3-20% of the gain). Among the 16-neuron networks the tree's gain does not rise with accuracy (correlation −0.29).

Reading: dense close calls are what this training recipe produces, at every width and accuracy tried; ARC is not an unlucky draw. The hand-built networks show the task does not require it. Width buys fewer close-call mistakes, not simpler ones.
