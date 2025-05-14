import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    """
    Compute the moving average over a window size w.
    :param x: Input 1D array of values
    :param w: Window size for smoothing
    :return: Smoothed array
    """
    return np.convolve(x, np.ones(w), 'valid') / w

# === Moving average window size ===
window = 2000  # Adjust this value to control smoothing level

# === Load reward log file ===
rewards = np.loadtxt("avg_episode_rewards.txt")

# === Apply moving average for smoothing ===
smoothed = moving_average(rewards, window)

# === Plot the raw and smoothed reward curve ===
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Raw Reward", alpha=0.3)
plt.plot(range(len(smoothed)), smoothed, label=f"Smoothed (w={window})", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Final Reward")
plt.title("Episode Reward Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"reward_curve_w{window}.png")
plt.show()