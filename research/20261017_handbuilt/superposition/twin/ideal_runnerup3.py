"""Same as ideal_runnerup2 with the discount size read off ARC: in ARC the runner-up at the second's arrival moves the
lead by a median |0.073| per unit (arc_leak_source.log); with keep share 0.25 that is theta about 0.3. Prediction
(written first): third wins near ARC's 21%, accuracy above 92%, tree gain lower than v23's."""
from ideal_runnerup import measure, score_fn
if __name__ == "__main__":
    for theta in (0.3, 0.4):
        for rho in (1.0, 0.99, 0.98, 0.97):
            measure(f"theta {theta}, fading {rho}", score_fn("runner-up", alpha=0.75, beta=0.25, rho=rho, theta=theta))
