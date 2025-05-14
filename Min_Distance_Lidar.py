from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch
import numpy as np
import os
from Env import DroneEnv

def main():
    num_envs = 4
    save_dir = "lidar_logs"
    os.makedirs(save_dir, exist_ok=True)

    env = DroneEnv(visualize=True, num_envs=num_envs)
    env.gym.refresh_actor_root_state_tensor(env.sim)

    # Custom positions for each drone
    custom_positions = [
        [1, 1, 1],
        [5, 2, 2],
        [10, 3, 3],
        [15, 4, 4]
    ]
    assert len(custom_positions) == num_envs

    # Update positions and velocities
    root_tensor = env.root_tensor
    drone_indices = env.drone_actor_indices
    for i in range(num_envs):
        pos = torch.tensor(custom_positions[i], dtype=torch.float32, device="cuda")
        vel = torch.zeros(3, dtype=torch.float32, device="cuda")
        root_tensor[drone_indices[i], 0:3] = pos
        root_tensor[drone_indices[i], 7:10] = vel

    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(root_tensor))

    # Run one simulation step
    env.step_simulation()
    env.gym.refresh_actor_root_state_tensor(env.sim)

    # Get min_distance matrix from Lidar (shape: [num_envs, 6])
    lidar_all = env.get_lidar_distance_all().detach().cpu().numpy()

    # Print as raw matrix
    print("\nRaw min_distance matrix (shape: [4 environments, 6 directions]):")
    print(lidar_all)

    # Save as .txt and .npy
    txt_path = os.path.join(save_dir, "min_distance_matrix.txt")
    npy_path = os.path.join(save_dir, "min_distance.npy")
    np.savetxt(txt_path, lidar_all, fmt="%.3f")
    np.save(npy_path, lidar_all)

    print(f"\n Saved matrix to:\n - {txt_path}\n - {npy_path}")

    # Reload and verify
    reloaded = np.load(npy_path)
    print("\n Reloaded matrix from .npy file:")
    print(reloaded)

if __name__ == "__main__":
    main()
