from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor

from RL import DroneRL

# === Training configuration ===
num_envs = 10000                       # Number of parallel drone environments
each_env_steps = 400000             # Steps per environment
total_timesteps = num_envs * each_env_steps  # Total training steps across all environments

# === Select computation device ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device.upper())
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

# === Initialize vectorized drone environment ===
env_raw = DroneRL(num_envs=num_envs, dt=0.1)  # Base environment with parallel support
env = VecMonitor(env_raw)                    # Monitor wrapper for episode statistics

# === Create Soft Actor-Critic (SAC) model ===
model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=2000000,
    batch_size=512,
    tau=0.005,
    gamma=0.99,
    train_freq=2,
    gradient_steps=5,
    ent_coef="auto",
    target_entropy="auto",
    learning_starts=10000,
    policy_kwargs=dict(net_arch=[512, 512, 256]),
    verbose=1,
    device="cuda"
)

# === Training loop with tqdm progress bar ===
frame_counter = 0
refresh_interval = 100  # 每 100 步刷新一次 tqdm

with tqdm(total=total_timesteps, desc="Training Progress", unit="step") as pbar:

    def callback(locals_, globals_):
        """
        Progress bar callback to sync tqdm with training frames.
        """
        global frame_counter
        frame_counter += 1

        # 仅每 N 步刷新 tqdm 条，避免刷新过于频繁导致卡顿
        if frame_counter % refresh_interval == 0:
            pbar.n = min(frame_counter * num_envs, total_timesteps)
            pbar.refresh()

        return True

    print(f"Starting training: {num_envs} envs, {each_env_steps} steps each, total {total_timesteps} steps")
    model.learn(total_timesteps=total_timesteps, callback=callback)


# === Save trained model ===
model.save("sac_parallel_model.zip")
print("Training complete. Model saved as sac_parallel_model.zip")

# === Save final rewards for each episode ===
with open("avg_episode_rewards.txt", "w") as f:
    for r in env_raw.finished_episode_rewards:
        f.write(f"{r}\n")

print(f"Saved {len(env_raw.finished_episode_rewards)} episode rewards to avg_episode_rewards.txt")