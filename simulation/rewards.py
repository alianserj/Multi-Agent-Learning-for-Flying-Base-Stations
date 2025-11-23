import numpy as np
from simulation.utilities import distance

def compute_global_utility(assignments):
    """Computes the global utility as the total number of unique users being served."""
    unique_users = set()
    for assigned_users in assignments:
        unique_users.update(assigned_users)
    global_utility = len(unique_users)
    return global_utility

def compute_marginal_contribution(uav_positions, assignments, coverage_radius, user_positions):
    """Computes the marginal contribution utility for each UAV and the users contributing to it."""
    global_utility = compute_global_utility(assignments)
    # print("assignments", assignments)
    # print(f"Global utility with all UAVs: {global_utility}")
    marginal_contributions = np.zeros(len(uav_positions))
    contributing_users = [[] for _ in range(len(uav_positions))]
    
    for i in range(len(uav_positions)):
        # print(f"\nEvaluating marginal contribution for UAV {i}")
        # Simulate the scenario where UAV i is not present
        simulated_assignments = [users.copy() for users in assignments]
        temp_uav_positions = uav_positions.copy()
        temp_uav_positions[i] = [1e4, 1e4]  # Set to a large value instead of infinity
        # print(f"Simulated UAV positions without UAV {i}:\n{temp_uav_positions}")
        
        # Reassign users that were assigned to UAV i
        unsatisfied_users = []
        for user in assignments[i]:
            closest_uav = None
            min_dist = float('inf')
            for j, uav in enumerate(temp_uav_positions):
                dist = distance(user_positions[user], uav)
                if dist < coverage_radius and dist < min_dist:
                    closest_uav = j
                    min_dist = dist
            if closest_uav is None:
                unsatisfied_users.append(user)
                # print(f"User {user} is unsatisfied")
            else:
                simulated_assignments[closest_uav].append(user)
                # print(f"User {user} reassigned to UAV {closest_uav}")
        
        # Remove unsatisfied users from the assignments
        for user in unsatisfied_users:
            for uav_users in simulated_assignments:
                if user in uav_users:
                    uav_users.remove(user)
        
        # print(f"Simulated assignments without UAV {i}:\n{simulated_assignments}")
        
        simulated_global_utility = compute_global_utility(simulated_assignments)
        # print(f"Simulated global utility without UAV {i}: {simulated_global_utility}")
        marginal_contributions[i] = global_utility - simulated_global_utility
        # print(f"Marginal contribution for UAV {i}: {marginal_contributions[i]}")

        # Identify contributing users
        for user in assignments[i]:
            if closest_uav is None or user not in simulated_assignments[closest_uav]:
                contributing_users[i].append(user)
        # print(f"Contributing users for UAV {i}: {contributing_users[i]}")
    
    return marginal_contributions, contributing_users
