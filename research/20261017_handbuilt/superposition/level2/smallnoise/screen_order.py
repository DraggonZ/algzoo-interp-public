"""Is the order-specific error of the final estimate from the cheap screen? Same grid (split3-gl n=16), screen off."""
import numpy as np, estimate as E
from lib import *
W = load_net("ARC"); names = ['L,2,3', 'L,3,2', '2,L,3', '3,L,2', '2,3,L', '3,2,L']
for scr in [1e-3, -1]:
    E.SCREEN = scr; r = E.estimate(W, 16, 16, 'final', 4, detail=True, kind='split3-gl')
    print(f"screen {scr}: accuracy {100*r['acc']:.3f}%  runs {r['runs']/1e6:.0f}M")
    for i in range(6): print(f"  {names[i]} acc {100*r['order_acc'][i]:.2f}  wrong naming largest/third/another (points) {np.round(100*r['order_split'][i],2)}")
