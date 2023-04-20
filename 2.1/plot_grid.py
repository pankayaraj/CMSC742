import matplotlib.pyplot as plt
import numpy as np
import torch


Grid = [[0 for i in range(10)] for j in range(10)]
Grid[0][0] = 1
Grid[9][9] = 2

fig, ax = plt.subplots()
ax.imshow(Grid)
plt.savefig("2.1/figures/Grid.png")