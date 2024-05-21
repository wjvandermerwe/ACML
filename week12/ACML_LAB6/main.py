import numpy as np
import matplotlib.pyplot as plt
import torch

def create_grid_world(size):
    grid = torch.zeros(size, size)
    return grid

def display_grid_world_table(grid):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(grid.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.imshow(grid, cmap='gray_r', origin='upper')

    plt.title('Grid World with Empty Squares')
    plt.show()



goal = (9,9)
size = 10


position = (0,0)





grid = create_grid_world(size)
display_grid_world_table(grid)
