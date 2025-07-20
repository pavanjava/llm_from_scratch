import matplotlib.pyplot as plt
import numpy as np
import torch

inputs = torch.tensor(
    [
        [0.8, 0.6, 0.1],        # I
        [0.7, 0.7, 0.1],        # Love
        [0.2, 0.8, 0.6],        # Coding
        [0.1, 0.7, 0.7],        # GenAI
        [0.5055, 0.6921, 0.3243], # context vector for 'I'
        [0.4950, 0.6941, 0.3336], # context vector for 'Love'
        [0.4068, 0.7084, 0.4140], # context vector for 'Coding'
        [0.3911, 0.7105, 0.4284] # context vector for 'GenAI'
    ]
)
# Create 3D plot with vectors from origin to each point, using different colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a list of colors for the vectors
colors = ['r', 'g', 'b', 'c', 'm', 'y']

# Corresponding words
words = ['I', 'Love', 'Coding', 'GenAI', 'I-ContextVector', 'Love-ContextVector',
         'Coding-ContextVector', 'GenAI-ContextVector']

# Extract x, y, z coordinates
x_coords = inputs[:, 0].numpy()
y_coords = inputs[:, 1].numpy()
z_coords = inputs[:, 2].numpy()

# Plot each vector with a different color and annotate with the corresponding word
for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
    # Draw vector from origin to the point (x, y, z) with specified color and smaller arrow length ratio
    ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    ax.text(x, y, z, word, fontsize=10, color=color)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot limits to keep arrows within the plot boundaries
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

plt.title('3D Plot of Word Embeddings with Colored Vectors')
plt.show()