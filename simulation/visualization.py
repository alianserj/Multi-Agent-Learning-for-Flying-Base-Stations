import matplotlib.pyplot as plt
import numpy as np
from simulation.parameters import *
from simulation.utilities import in_bounds, distance

def visualize_progress(uav_positions, user_positions, iteration, assignments, rewards, contributing_users,global_utility, display_coordinates=False, display_contributing_users=False, display_numbers_only=True):
    """
    Visualize the UAV positions, user positions, and the marginal contributions.

    Parameters:
        uav_positions (np.ndarray): Array of UAV positions (x, y).
        user_positions (np.ndarray): Array of user positions (x, y).
        iteration (int): Current iteration number.
        assignments (list of lists): List of lists where each sublist contains the indices of users assigned to a UAV.
        rewards (np.ndarray): Array of rewards for each UAV.
        contributing_users (list of lists): List of lists where each sublist contains the indices of users contributing to the marginal contribution of a UAV.
        display_coordinates (bool): Toggle to display coordinates of UAVs and users. Default is False.
        display_contributing_users (bool): Toggle to display contributing users table. Default is False.
        display_numbers_only (bool): Toggle to display only UAV and user numbers without coordinates. Default is True.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Original positions
    ax[0].set_title(f"UAV and User Positions - Iteration {iteration}")
    ax[0].scatter(user_positions[:, 0], user_positions[:, 1], c='blue', label='Users', alpha=0.6)
    ax[0].scatter(uav_positions[:, 0], uav_positions[:, 1], c='red', label='UAVs', marker='x')
    if display_coordinates:
        for i, pos in enumerate(user_positions):
            ax[0].text(pos[0], pos[1], f"{i} ({pos[0]:.1f}, {pos[1]:.1f})", fontsize=8, ha='right')
        for i, pos in enumerate(uav_positions):
            ax[0].text(pos[0], pos[1], f"{i} ({pos[0]:.1f}, {pos[1]:.1f})", fontsize=8, ha='left')
    elif display_numbers_only:
        for i, pos in enumerate(user_positions):
            ax[0].text(pos[0], pos[1], f"{i}", fontsize=8, ha='right')
        for i, pos in enumerate(uav_positions):
            ax[0].text(pos[0], pos[1], f"{i}", fontsize=8, ha='left')
    for uav, users in enumerate(assignments):
        for user in range(len(user_positions)):
            uav_x, uav_y = uav_positions[uav]
            user_x, user_y = user_positions[user]
            dist = distance((uav_x, uav_y), (user_x, user_y))
            if dist < COVERAGE_RADIUS:
                ax[0].plot([uav_x, user_x], [uav_y, user_y], 'gray', linestyle='dotted', linewidth=0.7)
    ax[0].legend(loc='upper right')
    ax[0].grid(True)

    # Marginal contributions and rewards
    x = np.arange(len(uav_positions))
    width = 0.4

    ax[1].set_title("Marginal Contributions and Rewards")
    ax[1].bar(x - width / 2, rewards, width, label="Reward", alpha=0.7)
    ax[1].bar(x + width / 2, [len(contributing_users[i]) for i in range(len(uav_positions))], width, label="Marginal Contribution", alpha=0.7)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"UAV{i}" for i in range(len(uav_positions))])
    ax[1].set_xlabel("UAVs")
    ax[1].set_ylabel("Value")
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    # Global utility
    global_utility = global_utility
    ax[1].text(0.5, 0.9, f"Global Utility: {global_utility:.2f}", transform=ax[1].transAxes, fontsize=12, ha='center')

    plt.tight_layout()
    plt.show()

    # Contributing users table
    if display_contributing_users:
        fig, table_ax = plt.subplots(figsize=(8, 4))
        table_ax.set_title("Contributing Users")
        table_ax.axis('off')
        table_data = [["UAV", "Users"]]
        for i, users in enumerate(contributing_users):
            table_data.append([f"UAV {i}", ", ".join(map(str, users))])
        table = table_ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.7])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

def visualize_small_sample(uav_positions, user_positions):
    print("Visualizing a small sample of users and UAVs")
    plt.figure(figsize=(10, 10))
    plt.scatter(user_positions[:10, 0], user_positions[:10, 1], c='blue', label='Users', alpha=0.6)
    plt.scatter(uav_positions[:3, 0], uav_positions[:3, 1], c='red', label='UAVs', marker='x')
    for i, pos in enumerate(user_positions[:10]):
        plt.text(pos[0], pos[1], str(i), fontsize=8, ha='right')
    for i, pos in enumerate(uav_positions[:3]):
        plt.text(pos[0], pos[1], str(i), fontsize=8, ha='left')
    plt.title('Small Sample of UAV and User Positions')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid()
    plt.show()
