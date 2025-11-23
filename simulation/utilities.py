import numpy as np

def in_bounds(pos, grid_size):
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size

def distance(p1, p2):
    """euclidean distance"""
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def initialize_positions(grid_size, num_uavs, num_users, seed=42):
    """Generates random initial positions for UAVs and users within the grid."""
    np.random.seed(seed)
    uav_positions = np.random.randint(0, grid_size, size=(num_uavs, 2))
    user_positions = np.random.randint(0, grid_size, size=(num_users, 2))
    return uav_positions, user_positions
