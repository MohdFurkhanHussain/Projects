import matplotlib.pyplot as plt

import numpy as np

for i in range(50):
    plt.plot(np.random.rand(100), linewidth=1)

plt.title("Too much data can be confusing")
plt.grid(True)
plt.tight_layout()
plt.show()
