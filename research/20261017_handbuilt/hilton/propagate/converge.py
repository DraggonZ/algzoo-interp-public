import full_mean, adf
W = adf.net(None)
for nu, na in [(56, 96), (56, 128), (80, 128), (56, 192)]:
    acc, mass = full_mean.estimate(W, nu, na); print(f"grid {nu} x {na} x {na}: {100*acc:.3f}%  mass {mass:.5f}", flush=True)
