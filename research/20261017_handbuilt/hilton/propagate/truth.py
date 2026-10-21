"""True accuracy of a network on many random lists (GPU), for scoring the sampling-free estimates."""
import torch
def true_acc(W, n=20_000_000, seed=0, chunk=2_000_000):
    wi, wh, wo = W; g = torch.Generator(device="cuda").manual_seed(seed); c = 0
    for i in range(0, n, chunk):
        X = torch.randn(chunk, 10, device="cuda", dtype=torch.float32, generator=g)
        h = torch.zeros(chunk, wi.shape[0], device="cuda")
        for t in range(10): h = torch.relu(h @ wh.float().T + X[:, t:t + 1] * wi.float())
        c += ((h @ wo.float().T).argmax(1) == X.argsort(1)[:, -2]).sum().item()
    return c / n
if __name__ == "__main__":
    import adf
    a = true_acc(adf.net(None)); print(f"ARC network, 20M lists: {100*a:.3f}% (standard error {100*(a*(1-a)/2e7)**.5:.3f})")
