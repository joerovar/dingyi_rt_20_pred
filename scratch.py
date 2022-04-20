import numpy as np

nrs = np.random.randint(0, 4,size=(4, 2))
print(nrs)
idx = np.random.randint(0, 4*4, size=2)
print(idx)
a = np.zeros((4, 4*4))
a[:, idx] = nrs
print(a)

