# Morph and average (2026-10-18)

Status: done. Borrowed from Obermeyer et al., 'An ECG biomarker for sudden cardiac death discovered with deep learning' (Nature, July 2026, doi 10.1038/s41586-026-10674-6): they follow a risk predictor's gradient through a VAE's latent space to morph real low-risk heartbeats into higher-risk ones, then look for a shape a person can name. The VAE keeps morphs realistic; here every list of ten numbers is a valid input, so the morphs run on the inputs directly and no generative model is needed.

## Method (`morph.py`, `profiles.py`)

Near ties the network gets right (third within 5% of the second, second above 0.3): move only the seven smaller numbers, in unit-length gradient steps of 0.01 x the second, toward 'the network names the third', keeping the top three fixed and the smaller numbers below the third, until the answer flips (at most 300 steps).

## Results

- Only 37.5% of right near ties can be flipped this way; the median morph that flips one moves the smaller numbers by 0.38 x the second in total length. The context rarely suffices to flip a decision.
- Averaged by position relative to the second, the morph oscillates: raise the numbers one to four steps before the second (+0.02 to +0.06), lower those two to four steps after it (−0.10 to −0.19), raise the one seven after, lower eight and nine after. Relative to the third: lower its neighbours (−0.12 just before, −0.09 just after). The alternation over about five to eight steps looks like the rotation modes' code, but this is an interpretation.
- Within each arrangement of the three largest, the mean morph is stable: two random halves of an arrangement's lists give mean morphs with cosine 0.97 (median over 661 arrangements). Single morphs agree only weakly (mean cosine 0.26); an arrangement's mean explains 39% of single morphs' variance.
- The stable directions are not compact: shared profiles by offset from the second, third and largest (57 numbers) explain 25% of the arrangement means' variance on held-out arrangements (15% for the second's profile alone).

## Reading

The morphs do find a describable object: for each arrangement, a reliable direction in which the context pushes the decision toward the third. But each of the 720 arrangements has its own direction (about 5,000 numbers in all), and shared position profiles capture a quarter of them. This matches the 25,000-number rule and the unstructured leak coefficients (`../context/`): the lean is reliable but not compressible. A generative model would not change this here, since the inputs have no manifold to keep morphs on.
