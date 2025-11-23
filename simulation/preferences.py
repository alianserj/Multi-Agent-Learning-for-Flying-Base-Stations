import numpy as np
import matplotlib.pyplot as plt
from simulation.parameters import *
from simulation.assignments import assign_users_to_uavs
from simulation.rewards import compute_marginal_contribution, compute_global_utility
from simulation.utilities import in_bounds, distance

def visualize_deviation(uav_positions, user_positions, uav_idx, new_x, new_y, assignments, action_idx, mcu_increase, global_utility, simulated_global_utility):
    """
    Visualize the map if one UAV deviated to new_x, new_y versus if it did not.

    Args:
        uav_positions (np.ndarray): Array of UAV positions (x, y).
        user_positions (np.ndarray): Array of user positions (x, y).
        uav_idx (int): Index of the UAV that is deviating.
        new_x (float): New x-coordinate of the deviating UAV.
        new_y (float): New y-coordinate of the deviating UAV.
        assignments (list of lists): List of lists where each sublist contains the indices of users assigned to a UAV.
        action_idx (int): Index of the action being taken.
        mcu_increase (float): Increase in marginal contribution utility.
        global_utility (float): Global utility with all UAVs.
        simulated_global_utility (float): Global utility without the deviating UAV.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Original positions
    ax[0].set_title(f"Original Positions (UAV {uav_idx}, Action {action_idx})")
    ax[0].scatter(user_positions[:, 0], user_positions[:, 1], c='blue', label='Users', alpha=0.6)
    ax[0].scatter(uav_positions[:, 0], uav_positions[:, 1], c='red', label='UAVs', marker='x')
    for i, pos in enumerate(user_positions):
        ax[0].text(pos[0], pos[1], f"{i} ({pos[0]:.1f}, {pos[1]:.1f})", fontsize=8, ha='right')
    for i, pos in enumerate(uav_positions):
        ax[0].text(pos[0], pos[1], f"{i} ({pos[0]:.1f}, {pos[1]:.1f})", fontsize=8, ha='left')
    for uav, users in enumerate(assignments):
        for user in range(len(user_positions)):
            uav_x, uav_y = uav_positions[uav]
            user_x, user_y = user_positions[user]
            dist = distance((uav_x, uav_y), (user_x, user_y))
            if dist < COVERAGE_RADIUS:
                ax[0].plot([uav_x, user_x], [uav_y, user_y], 'gray', linestyle='dotted', linewidth=0.7)
    ax[0].legend(loc='upper right')
    ax[0].grid(True)

    # Deviated positions
    ax[1].set_title(f"Deviated Positions (MCU Increase: {mcu_increase:.2f}, Global Utility: {global_utility}, Without UAV {uav_idx}: {simulated_global_utility})")
    deviated_positions = uav_positions.copy()
    deviated_positions[uav_idx] = [new_x, new_y]
    deviated_assignments = assign_users_to_uavs(deviated_positions, user_positions, COVERAGE_RADIUS, SNR_THRESHOLD)
    ax[1].scatter(user_positions[:, 0], user_positions[:, 1], c='blue', label='Users', alpha=0.6)
    ax[1].scatter(deviated_positions[:, 0], deviated_positions[:, 1], c='red', label='UAVs', marker='x')
    for i, pos in enumerate(user_positions):
        ax[1].text(pos[0], pos[1], f"{i} ({pos[0]:.1f}, {pos[1]:.1f})", fontsize=8, ha='right')
    for i, pos in enumerate(deviated_positions):
        ax[1].text(pos[0], pos[1], f"{i} ({pos[0]:.1f}, {pos[1]:.1f})", fontsize=8, ha='left')
    for uav, users in enumerate(deviated_assignments):
        for user in range(len(user_positions)):
            uav_x, uav_y = deviated_positions[uav]
            user_x, user_y = user_positions[user]
            dist = distance((uav_x, uav_y), (user_x, user_y))
            if dist < COVERAGE_RADIUS:
                ax[1].plot([uav_x, user_x], [uav_y, user_y], 'gray', linestyle='dotted', linewidth=0.7)
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

def update_preferences(action_preferences, uav_positions, user_positions, assignments, learning_rate, T=1):
    """
    Update UAV action preferences by simulating rewards for all possible actions,
    considering the marginal utility of the actions over the next T actions.

    Args:
        action_preferences (np.ndarray): Array of action preferences for each UAV.
        uav_positions (np.ndarray): Current positions of UAVs.
        user_positions (np.ndarray): Current positions of users.
        assignments (list of lists): List of lists where each sublist contains the indices of users assigned to a UAV.
        learning_rate (float): Learning rate for preference updates.
        T (int): Number of actions to look forward. Default is 1.

    Returns:
        np.ndarray: Updated action preferences.
    """
    # print(f"Updating action preferences based on simulated marginal contributions over {T} actions")
    # print(f"Initial UAV positions:\n{uav_positions}")
    # print(f"Initial user positions:\n{user_positions}")
    # print(f"Initial assignments:\n{assignments}")
    # print(f"Initial action preferences:\n{action_preferences}")

    for i in range(len(action_preferences)):
        # print(f"\nEvaluating UAV {i}")
        for action_idx, (dx, dy) in enumerate(ACTIONS):
            # print(f"  Action {action_idx}: Move by ({dx}, {dy})")
            total_marginal_contribution = 0
            simulated_positions = uav_positions.copy()
            simulated_assignments = None  # Initialize to None
            for t in range(T):
                # Simulate new position if UAV i takes this action
                new_x = simulated_positions[i, 0] + 10*dx
                new_y = simulated_positions[i, 1] + 10*dy
                # print(f"    Step {t}: Simulated new position for UAV {i}: ({new_x}, {new_y})")

                if in_bounds((new_x, new_y), GRID_SIZE):
                    simulated_positions[i] = [new_x, new_y]
                    # print(f"    Step {t}: New positions:\n{simulated_positions}")

                    # Assign users to UAVs based on simulated positions
                    simulated_assignments = assign_users_to_uavs(simulated_positions, user_positions, COVERAGE_RADIUS, SNR_THRESHOLD)
                    # print(f"    Step {t}: Simulated assignments:\n{simulated_assignments}")

                    # Compute simulated marginal contributions for the new position
                    simulated_marginal_contributions, _ = compute_marginal_contribution(simulated_positions, simulated_assignments, COVERAGE_RADIUS, user_positions)
                    # print(f"    Step {t}: Simulated marginal contributions:\n{simulated_marginal_contributions}")

                    total_marginal_contribution += simulated_marginal_contributions[i]
                    # print(f"    Step {t}: Total marginal contribution so far: {total_marginal_contribution}")
                else:
                    # print(f"    Step {t}: Out of bounds, skipping.")
                    break

            # Compute original MCU
            original_mcu, _ = compute_marginal_contribution(uav_positions, assignments, COVERAGE_RADIUS, user_positions)
            # print(f"  Original MCU for UAV {i}: {original_mcu[i]}")

            # Compute global utility and simulated global utility
            global_utility = compute_global_utility(assignments)
            simulated_assignments = [users.copy() for users in assignments]

            temp_uav_positions = uav_positions.copy()
            temp_uav_positions[i] = [1e4, 1e4] 

            unsatisfied_users = []
            for user in assignments[i]:
                closest_uav = None
                min_dist = float('inf')
                for j, uav in enumerate(temp_uav_positions):
                    dist = distance(user_positions[user], uav)
                    if dist < COVERAGE_RADIUS and dist < min_dist:
                        closest_uav = j
                        min_dist = dist
                if closest_uav is None:
                    unsatisfied_users.append(user)
                else:
                    simulated_assignments[closest_uav].append(user)
            
            # Remove unsatisfied users from the assignments
            for user in unsatisfied_users:
                for uav_users in simulated_assignments:
                    if user in uav_users:
                        uav_users.remove(user)

            # print(f"  Simulated assignments without UAV {i}:\n{simulated_assignments}")
            simulated_global_utility = compute_global_utility(simulated_assignments)
            # print(f"  Global utility: {global_utility}, Simulated global utility without UAV {i}: {simulated_global_utility}")

            # Visualize the deviation
            mcu_increase = total_marginal_contribution / T

            # Update preferences for the action based on the increase in MCU
            mcu_difference = mcu_increase - original_mcu[i]
            # print(f"  MCU before deviation: {original_mcu[i]}, MCU after deviation: {mcu_increase}")
            # print(f"  Difference in MCU: {mcu_difference}")
            action_preferences[i, action_idx] += learning_rate * mcu_difference
            # print(f"  Updated preference for UAV {i}, Action {action_idx}: {action_preferences[i, action_idx]}")
            # if simulated_assignments is not None:
                # visualize_deviation(uav_positions.copy(), user_positions, i, new_x, new_y, simulated_assignments, action_idx, mcu_difference, global_utility, simulated_global_utility)

    # print(f"Updated action preferences:\n{action_preferences}")
    return action_preferences

def choose_actions(action_preferences, beta):
    """
    Choose actions for UAVs based on softmax probabilities.

    Args:
        action_preferences (np.ndarray): Array of action preferences for each UAV.
        beta (float): Inverse temperature parameter for softmax.

    Returns:
        list: List of chosen actions for each UAV.
    """
    # Subtract the max value for numerical stability
    max_preferences = np.max(action_preferences, axis=1, keepdims=True)
    stabilized_preferences = action_preferences - max_preferences

    # Compute softmax probabilities
    exp_preferences = np.exp(beta * stabilized_preferences)
    action_probabilities = exp_preferences / np.sum(exp_preferences, axis=1, keepdims=True)

    # Choose actions based on the computed probabilities
    chosen_actions = [np.random.choice(len(action_preferences[i]), p=action_probabilities[i]) for i in range(len(action_preferences))]
    return chosen_actions, action_probabilities