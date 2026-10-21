"""First look: memory block, on/off statistics per step, zero continuation."""
from collections import Counter
import numpy as np
from algzoo.core import example
wi, wh, wo = example()
np.set_printoptions(precision=2, suppress=True, linewidth=220)
MEM=[0,3,10,11,12,13,14,15]; HID=[1,6,7,8]
print("input weights", wi)
print("memory block\n", wh[np.ix_(MEM,MEM)])
print("readout (centred) on memory\n", (wo-wo.mean(0))[:,MEM])
X=np.random.default_rng(1).standard_normal((200000,10)); h=np.zeros((len(X),16)); hs=[h]
for t in range(10):
    h=np.maximum(h@wh.T+X[:,t:t+1]*wi,0); hs.append(h)
for t in range(1,11):
    on=(hs[t]>0)
    print(t, "frac on per neuron", np.round(on.mean(0),2))
    c=Counter(map(lambda r:r.tobytes(), on[:,MEM])).most_common(3)
    print("   memory patterns", [(np.where(np.frombuffer(k,bool))[0].tolist() and [MEM[i] for i in np.where(~np.frombuffer(k,bool))[0]], round(v/len(X),3)) for k,v in c])
