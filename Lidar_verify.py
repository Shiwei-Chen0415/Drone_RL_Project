from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from Env import DroneEnv
from DroneSetting import Drone

# === Current LiDAR yaw angle (in radians) ===
yaw_angle_rad = 0.0

def set_lidar_yaw_deg(angle_deg: float):
    global yaw_angle_rad
    yaw_angle_rad = math.radians(angle_deg)
    env.yaw_angle_tensor[0] = yaw_angle_rad
    print(f"[Lidar] Set yaw angle to {angle_deg:.1f}° ({yaw_angle_rad:.2f} rad)")

# === Enable rotation test nodes ===
ENABLE_ROTATION_NODES = True

# Initialize environment
env = DroneEnv(visualize=True, num_envs=1)
drone = Drone(env, env_index=0, dt=0.1)

# Waypoint settings
waypoints = [
    [1.0, 2.5, 3.0], [2.0, 2.5, 3.0], [3.0, 2.5, 1.0], [6.0, 3.0, 1.25], [7.0, 3.0, 1.25],
    [8.0, 2.5, 1.25], [9.5, 1.25, 1.25], [11.0, 1.25, 2.5], [13.0, 2.0, 3.75],
    [15.5, 2.0, 3.75], [17.0, 2.0, 2.5], [19.0, 3.75, 2.5], [21.0, 3.75, 2.5],
    [23.0, 3.0, 3.75], [26.0, 3.0, 3.75], [28.0, 3.0, 2.5], [30.5, 1.25, 2.5],
    [33.0, 1.25, 2.5], [36.0, 1.0, 2.5], [38.0, 2.5, 2.5], [40.0, 4.0, 2.5],
    [42.0, 3.0, 2.5], [44.0, 2.0, 2.5], [46.0, 2.5, 2.5], [48.0, 2.5, 2.5],
]
waypoints = [torch.tensor(p, device="cuda") for p in waypoints]

rotation_nodes = {
    tuple(waypoints[0].tolist()): 0.0,
    tuple(waypoints[1].tolist()): 45.0,
    tuple(waypoints[2].tolist()): 0.0,
    tuple(waypoints[3].tolist()): 45.0,
    tuple(waypoints[4].tolist()): 0.0,
}

# Control parameters
SPEED = 1.0
STEP_DT = 0.1
MAX_STEP = SPEED * STEP_DT
lidar_log = []

# Start position
drone.reset(waypoints[0].tolist())

# Main loop: fly along waypoints
for i in range(len(waypoints) - 1):
    start = waypoints[i]
    end = waypoints[i + 1]
    direction = end - start
    distance = torch.norm(direction).item()
    dir_unit = direction / (distance + 1e-6)
    current_pos = start.clone()
    steps_needed = int(distance / MAX_STEP)

    for _ in range(steps_needed):
        env.step_simulation()
        current_pos += dir_unit * MAX_STEP
        current_pos = torch.clamp(current_pos, env.boundary_min_tensor[0], env.boundary_max_tensor[0])
        env.root_tensor[drone.state_manager.drone_index, 0:3] = current_pos
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_tensor))

    # At waypoint: record LiDAR
    env.root_tensor[drone.state_manager.drone_index, 0:3] = end
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_tensor))
    pos_np = end.detach().cpu().numpy()
    lidar = env.get_lidar_distance_all(env.yaw_angle_tensor)[0].detach().cpu().numpy()

    print(f"Waypoint {i + 1} | Pos {np.round(pos_np, 2)} | Lidar: {np.round(lidar, 2)}")
    lidar_log.append(("normal", i + 1, pos_np, lidar.copy()))

    # If current node requires LiDAR rotation
    key = tuple(np.round(pos_np, 2))
    if ENABLE_ROTATION_NODES and key in rotation_nodes:
        print(f"[Before Rotation] Pos {np.round(pos_np, 2)} | Lidar: {np.round(lidar, 2)}")
        lidar_log.append(("rotate_before", i + 1, pos_np, lidar.copy()))

        set_lidar_yaw_deg(rotation_nodes[key])

        lidar = env.get_lidar_distance_all(env.yaw_angle_tensor)[0].detach().cpu().numpy()
        print(f"[After Rotation] Pos {np.round(pos_np, 2)} | Lidar: {np.round(lidar, 2)}")
        lidar_log.append(("rotate_after", i + 1, pos_np, lidar.copy()))

# === Save flight and LiDAR logs (with formatted floats) ===
save_dir = "lidar_plots"
os.makedirs(save_dir, exist_ok=True)

log_file = os.path.join(save_dir, "lidar_logs.txt")
with open(log_file, "w") as f:
    for tag, step, pos_np, lidar in lidar_log:
        pos_str = ", ".join([f"{v:.1f}" for v in np.round(pos_np, 2)])

        # === Lidar values with direction labels ===
        labels = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        lidar_items = [f"{labels[j]}:{lidar[j]:.2f}" for j in range(6)]
        lidar_str = ", ".join(lidar_items)

        f.write(f"{tag.upper()} - Waypoint {step} | Pos [{pos_str}] | Lidar: [{lidar_str}]\n")

print(f"Flight and LiDAR logs saved to {log_file}")

# === Prepare data for plotting ===
normal_steps, normal_lidars = [], []
rotate_steps, rotate_lidars = [], []

for tag, step, pos, lidar in lidar_log:
    if tag == "normal":
        normal_steps.append(step)
        normal_lidars.append(lidar)
    elif tag == "rotate_after":
        rotate_steps.append(step)
        rotate_lidars.append(lidar)

normal_lidars = np.array(normal_lidars)
rotate_lidars = np.array(rotate_lidars)

labels = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
normal_colors = ['red', 'green', 'red', 'green', 'red', 'green']
rotate_color = 'orange'

# === Plot and save LiDAR distance for each direction ===
for i in range(6):
    plt.figure()
    plt.plot(normal_steps, normal_lidars[:, i], label=f"Normal {labels[i]}", color=normal_colors[i])
    plt.scatter(rotate_steps, rotate_lidars[:, i], label=f"Rotated {labels[i]}", color=rotate_color, marker='x')
    plt.title(f"Direction {labels[i]} Lidar_Distance")
    plt.xlabel("Waypoint Index")
    plt.ylabel("Distance (m)")
    plt.grid(True)
    plt.legend()

    save_path = os.path.join(save_dir, f"lidar_direction_{labels[i]}.png")
    plt.savefig(save_path)
    plt.close()

print(f"All LiDAR plots saved in folder: {save_dir}/")

# Keep viewer open until ESC pressed
print("Flight complete. Press ESC in the Isaac Viewer window to exit.")
while not env.is_viewer_closed():
    env.step_simulation()
