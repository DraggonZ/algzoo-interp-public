"""Idealized: keep gate relative to the runner-up at arrival with the twin's keep share (beta 0.25, alpha 0.75, theta
0.75 as ARC's account), votes fading by rho per step of age without compensation (ARC's account item 5: later votes are
discounted by the runner-up, earlier votes fade, the two roughly offset). Prediction (written first): some rho near 0.95
gives accuracy near ARC's (95%) with the third winning about a fifth of near ties and a tree near ARC's 31%."""
from ideal_runnerup import measure, score_fn
if __name__ == "__main__":
    for rho in (1.0, 0.98, 0.96, 0.95, 0.94, 0.92):
        measure(f"keep 0.25 vs runner-up, fading {rho}", score_fn("runner-up", alpha=0.75, beta=0.25, rho=rho))
