import numpy as np
n_samples=100000
x = np.random.normal(0, 1, n_samples)
y = np.random.normal(np.cos(x), 0.5, n_samples)
print(round(np.corrcoef(x,y)[0,1],4))

import matplotlib.pyplot as plt
plt.scatter(x,y,c='black',s=1)
plt.xlabel('X')
plt.ylabel('Y')
