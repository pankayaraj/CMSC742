import torch





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

grid = torch.load("2.1/results/oir/0/occu")
# Create a 2D array
data = np.array(grid)

data = data/np.sum(data)

# Define x and y coordinates
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])

# Create a meshgrid from x and y
X, Y = np.meshgrid(x, y)

# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



# Plot the surface with a colormap
surf = ax.plot_surface(X, Y, data, cmap='RdYlBu_r')

# Add colorbar and set labels
fig.colorbar(surf)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Occupancy Measure')
ax.set_zlim(0,0.05)

ax.set_title("esp_decay = 0.99 + OIR", fontsize = 30)
# Show the plot
plt.savefig("2.1/figures/oir/esp_oir_0.png")
