from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from Env import DroneEnv
from DroneSetting import Drone
import torch
import os

class DroneRL(VecEnv):
    """
    A vectorized reinforcement learning environment for drone navigation.
    Utilizes Isaac Gym for parallel GPU-accelerated physics simulation.
    """

    def __init__(self, num_envs=16, dt=0.1):
        self.num_envs = num_envs
        self.dt = dt

        # Initialize simulation and individual drones
        self.env = DroneEnv(visualize=False, num_envs=num_envs)
        self.drones = [Drone(self.env, env_index=i, dt=self.dt, env_rl_ref=self) for i in range(num_envs)]

        # Start and goal positions
        self.init_position = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32, device="cuda")
        self.base_goal = torch.tensor([48.0, 2.5, 2.5], dtype=torch.float32, device="cuda")
        self.goal_position = self.base_goal.repeat(self.num_envs, 1).clone()
        # Step counters
        self.max_steps = 1000
        self.current_steps = torch.zeros(num_envs, dtype=torch.int32, device="cuda")
        self.global_step = 0
        self.yaw_tensor = torch.zeros(self.num_envs, dtype=torch.float32, device="cuda")

        # Observation: pos(3) + vel(3) + lidar(6) + goal vector(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Direct tensor access to drone states
        self.root_states = self.env.root_tensor
        self.drone_indices = torch.tensor(self.env.drone_actor_indices, device="cuda", dtype=torch.long)

        super().__init__(num_envs, self.observation_space, self.action_space)

        # Create directory for logging episode rewards
        self.log_dir = "episode_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.finished_episode_rewards = []

        self.reset()

    def reset(self):
        """
        Reset all environments to the initial state and return the first observation.
        """
        for drone in self.drones:
            drone.reset(self.init_position.tolist())

        # Construct initial observations
        obs_batch = []
        yaw_angles = self.yaw_tensor.clone()
        lidar_batch = self.env.get_lidar_distance_all(yaw_angles)

        for i, drone in enumerate(self.drones):
            pos = drone.get_position()
            vel = drone.get_velocity()
            lidar = lidar_batch[i]
            goal_vec = self.goal_position[i] - drone.get_position()
            yaw = drone.get_yaw()
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)
            obs = torch.cat([pos, vel, lidar, goal_vec, cos_yaw.unsqueeze(0), sin_yaw.unsqueeze(0)])
            obs_batch.append(obs)

        return np.array([obs.detach().cpu().numpy() for obs in obs_batch], dtype=np.float32)

    def apply_batch_forces(self, actions_tensor):
        """
        Apply force actions to all drones using batched tensor operations.

        :param actions_tensor: Tensor of shape (num_envs, 3) with action vectors in [-1, 1]
        """
        root_states = self.env.root_tensor
        drone_indices = self.env.drone_actor_indices

        positions = root_states[drone_indices, 0:3]
        velocities = root_states[drone_indices, 7:10]

        MAX_THRUST = 1.0
        MAX_SPEED = 5.0
        # Update yaw via each Drone's update_yaw method
        self.yaw_tensor += actions_tensor[:, 3] * 0.1
        self.yaw_tensor = (self.yaw_tensor + np.pi) % (2 * np.pi) - np.pi
        # Clamp actions and update velocity
        forces = torch.clamp(actions_tensor[:, :3], -MAX_THRUST, MAX_THRUST)
        new_velocities = velocities + forces * self.dt
        new_velocities = torch.clamp(new_velocities, -MAX_SPEED, MAX_SPEED)

        # Update positions based on new velocity
        new_positions = positions + new_velocities * self.dt
        new_positions = torch.clamp(
            new_positions,
            min=self.env.boundary_min_tensor,
            max=self.env.boundary_max_tensor
        )

        # Write updates to simulation
        root_states[drone_indices, 0:3] = new_positions
        root_states[drone_indices, 7:10] = new_velocities
        self.env.gym.set_actor_root_state_tensor(self.env.sim, gymtorch.unwrap_tensor(root_states))

    def step_async(self, actions):
        """
        Store actions for async stepping.
        """
        self._actions = actions

    def step_wait(self):
        """
        Perform one environment step with vectorized drone dynamics and reward computation.
        """
        actions_tensor = torch.tensor(self._actions, dtype=torch.float32, device="cuda")

        # Get yaw_angles from each Drone instance
        yaw_angles = self.yaw_tensor.clone()

        # Pass yaw_angles to get_lidar_distance_all
        lidar_distances = self.env.get_lidar_distance_all(yaw_angles)

        # Apply forces and simulate step
        self.apply_batch_forces(actions_tensor)
        self.env.step_simulation()
        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)

        positions = self.root_states[self.drone_indices, 0:3]
        velocities = self.root_states[self.drone_indices, 7:10]
        lidar_tensor = lidar_distances.detach()

        # Compute rewards and termination
        reward, done_tensor, terminated, truncated = self.compute_reward(
            positions, velocities, lidar_tensor, self.current_steps
        )

        # Reset environments where done
        reset_indices = torch.nonzero(done_tensor).squeeze(-1)
        self.reset_batch(reset_indices, terminated, truncated, reward)
        self.current_steps[~done_tensor] += 1

        # Construct new observation
        local_pos = positions
        goal_vec = self.goal_position - positions
        cos_yaw = torch.cos(yaw_angles)
        sin_yaw = torch.sin(yaw_angles)
        obs_tensor = torch.cat([local_pos, velocities, lidar_tensor, goal_vec,
                                cos_yaw.unsqueeze(1), sin_yaw.unsqueeze(1)], dim=1)

        obs_batch = obs_tensor.detach().cpu().numpy().astype(np.float32)
        reward_batch = reward.detach().cpu().numpy().astype(np.float32)
        done_batch = done_tensor.detach().cpu().numpy().astype(bool)
        info_batch = [{} for _ in range(self.num_envs)]

        # Log final rewards of finished episodes
        for i in range(self.num_envs):
            if done_batch[i]:
                ep_reward = reward[i].item()
                self.finished_episode_rewards.append(ep_reward)
                log_path = os.path.join(self.log_dir, f"env_{i:03d}.txt")
                with open(log_path, "a") as f:
                    f.write(f"{ep_reward}\n")

        return obs_batch, reward_batch, done_batch, info_batch

    def reset_batch(self, reset_indices: torch.Tensor, terminated: torch.Tensor,
                    truncated: torch.Tensor, reward: torch.Tensor):
        """
        Reset selected environments by indices, with randomized goals.

        :param reset_indices: Tensor of environment indices to reset
        """
        if reset_indices.numel() == 0:
            return

        root_states = self.env.root_tensor
        drone_indices = torch.tensor(self.env.drone_actor_indices, device="cuda", dtype=torch.long)
        reset_drone_indices = drone_indices[reset_indices]

        reset_pos = self.init_position.expand(len(reset_indices), 3)
        reset_vel = torch.zeros(len(reset_indices), 3, device="cuda")

        root_states[reset_drone_indices, 0:3] = reset_pos
        root_states[reset_drone_indices, 7:10] = reset_vel
        self.env.gym.set_actor_root_state_tensor(self.env.sim, gymtorch.unwrap_tensor(root_states))

        self.current_steps[reset_indices] = 0
        for i in reset_indices.tolist():
            goal_noise = torch.randn(3, device="cuda") * 1.0 # Add some noise to robust the Model
            self.goal_position[i] = self.base_goal + goal_noise

    def compute_reward(self, positions, velocities, lidar_tensor, current_steps):
        """
        Calculate per-environment rewards and termination flags.

        :return: reward, done, terminated, truncated
        """
        goal = self.goal_position
        dist = torch.norm(positions - goal, dim=1)
        speed = torch.norm(velocities, dim=1)
        min_lidar = torch.min(lidar_tensor, dim=1).values

        terminated = dist < 1.0
        truncated = current_steps >= self.max_steps
        done = terminated | truncated

        # Reward components
        goal_dir = torch.nn.functional.normalize(goal - positions, dim=1)
        vel_dir = torch.nn.functional.normalize(velocities + 1e-6, dim=1)
        direction_reward = torch.clamp(torch.sum(goal_dir * vel_dir, dim=1), min=0.0) * 10.0

        distance_reward = torch.clamp(100.0 - 2 * dist, min=0.0)
        arrival_bonus = torch.where(terminated, torch.tensor(100.0, device=positions.device), 0.0)

        danger_factor = torch.sigmoid((1.0 - min_lidar) * 4.0)
        danger_penalty = danger_factor * (5 + speed ** 2)

        crash_mask = (min_lidar < 0.25).float()
        crash_penalty = crash_mask * (15 + speed ** 2)

        distance_factor = torch.clamp(dist / 50.0, 0.0, 1.0)
        timeout_penalty = torch.where(truncated, distance_factor * 50.0, 0.0)

        reward = direction_reward + distance_reward + arrival_bonus
        reward -= danger_penalty + crash_penalty + timeout_penalty

        return reward, done, terminated, truncated

    def close(self):
        """
        Close the Isaac Gym viewer.
        """
        if hasattr(self.env, "viewer") and self.env.viewer is not None:
            self.env.gym.destroy_viewer(self.env.viewer)

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.num_envs)]
