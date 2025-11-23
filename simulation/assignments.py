import numpy as np
from simulation.utilities import distance

# def assign_users_to_uavs(uav_positions, user_positions, coverage_radius, snr_threshold):
#     """Assigns each user to the closest UAV if within coverage radius."""
#     # print("Assigning users to UAVs")
#     # print(f"UAV positions: {uav_positions}")
#     uav_user_assignment = [[] for _ in range(len(uav_positions))]
#     for m, user in enumerate(user_positions):
#         dists = [distance(user, uav_positions[i]) for i in range(len(uav_positions))]
#         # print(f"Distances from user {m} to UAVs: {dists}")
#         min_dist = min(dists)
#         best_uav = np.argmin(dists)
#         snr = 1.0 / (1.0 + min_dist)  # Simplified SNR calculation
#         if snr > snr_threshold and min_dist < coverage_radius:
#             uav_user_assignment[best_uav].append(m)
#             # print(f"User {m} assigned to UAV {best_uav}")
#     # print(f"User assignments: {uav_user_assignment}")
#     return uav_user_assignment

def assign_users_to_uavs(uav_positions, user_positions, coverage_radius, snr_threshold):
    """
    Assign each user to the closest UAV if within coverage radius.
    Each user can contribute to the utility of only one UAV.

    Args:
        uav_positions (np.ndarray): Array of UAV positions.
        user_positions (np.ndarray): Array of user positions.
        coverage_radius (float): Coverage radius for UAVs.
        snr_threshold (float): Signal-to-noise ratio threshold for user coverage.

    Returns:
        list: A list of lists where each sublist contains the indices of users assigned to a UAV.
    """
    # Initialize the list of users assigned to each UAV
    uav_user_assignment = [[] for _ in range(len(uav_positions))]
    
    # Track which users have already been assigned
    assigned_users = set()

    for user_idx, user in enumerate(user_positions):
        if user_idx in assigned_users:
            continue  # Skip if the user has already been assigned

        # Find the closest UAV within the coverage radius
        closest_uav = None
        min_dist = float('inf')

        for uav_idx, uav in enumerate(uav_positions):
            dist = np.sqrt((user[0] - uav[0])**2 + (user[1] - uav[1])**2)
            snr = 1.0 / (1.0 + dist)  # SNR is inversely proportional to distance

            # Check if the UAV satisfies the coverage conditions
            if dist < coverage_radius  and dist < min_dist: # and snr > snr_threshold
                closest_uav = uav_idx
                min_dist = dist

        # Assign the user to the closest UAV if found
        if closest_uav is not None:
            uav_user_assignment[closest_uav].append(user_idx)
            assigned_users.add(user_idx)  # Mark this user as assigned

    return uav_user_assignment
