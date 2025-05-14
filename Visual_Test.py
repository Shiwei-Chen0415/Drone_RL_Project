from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

from stable_baselines3 import SAC
from RL import DroneRL
import numpy as np
import torch
import time

# === Visualization configuration ===
num_envs = 4  # Number of parallel environments to visualize
display_interval = 0.01  # Frame delay in seconds for viewer refresh
print(f"Launching {num_envs} visualization environments...")

# === Create environment and enable viewer ===
env_sim = DroneRL(num_envs=num_envs)
env_sim.reset()
env_sim.env.visualize = True
env_sim.env.viewer = env_sim.env.gym.create_viewer(env_sim.env.sim, gymapi.CameraProperties())

# === Setup camera to focus on environment 0 ===
cam_pos = gymapi.Vec3(25.0, 10.0, 8.0)      # Camera position
cam_target = gymapi.Vec3(25.0, 2.5, 2.5)    # Target look-at point
env_sim.env.gym.viewer_camera_look_at(env_sim.env.viewer, None, cam_pos, cam_target)

# === Load pre-trained SAC model ===
model_path = "sac_parallel_model.zip"
model = SAC.load(model_path, env=env_sim)
print(f"Model loaded: {model_path}")

# === Begin parallel test rollout ===
obs = env_sim.reset()
step_count = 0
print("Starting test. Press ESC or close the window to exit.")

while not env_sim.env.is_viewer_closed():
    # Predict actions for all environments
    actions, _ = model.predict(obs, deterministic=True)

    # Step environments with actions
    obs, rewards, dones, _ = env_sim.step(actions)

    step_count += 1
    print(f"[Step {step_count}] Avg Reward: {np.mean(rewards):.2f} | Active Envs: {np.sum(~np.array(dones))}")

    # Pause for visualization
    time.sleep(display_interval)
