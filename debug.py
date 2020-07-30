a = 3

b = [a for _ in range(3)]

print(b)

import numpy as np

a = [1,0]
b = [1,1]
c = [0,1]

d = np.vstack([a,b,c])
print(d)