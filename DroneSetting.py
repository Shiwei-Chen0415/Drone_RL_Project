from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch
import math
import numpy as np

from Env import DroneEnv

# Drone class managing dynamics and basic physics interactions
class Drone:
    def __init__(self, env, env_index, dt=0.1, env_rl_ref=None):
        """
        Initialize a drone in a parallel environment.
        :param env: Reference to the shared drone environment
        :param env_index: Index of this drone's environment
        :param dt: Simulation time step
        """
        self.env = env
        self.env_index = env_index
        self.dt = dt
        self.env_rl_ref = env_rl_ref
        # Handle access to position, velocity, boundaries
        self.state_manager = DroneStateManager(env, env_index)

        # Drone control constraints
        self.MAX_THRUST = 1.0     # Max force (acceleration in m/s^2)
        self.MAX_SPEED = 5.0   # Max velocity (m/s)

    def get_yaw(self):
        return self.env_rl_ref.yaw_tensor[self.env_index]

    def get_position(self):
        """
        Return the drone's current position.
        :return: Tensor of shape (3,) representing (x, y, z)
        """
        return self.state_manager.get_position()

    def get_velocity(self):
        """
        Return the drone's current velocity.
        :return: Tensor of shape (3,) representing (vx, vy, vz)
        """
        return self.state_manager.get_velocity()

    def reset(self, position):
        """
        Reset drone position and velocity in simulation.
        :param position: List or tensor of target reset position (x, y, z)
        """
        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        root_states = self.state_manager.root_states
        drone_idx = self.state_manager.drone_index
        device = root_states.device

        global_position = torch.as_tensor(position, dtype=torch.float32, device=device)
        zero_vel = torch.zeros(3, dtype=torch.float32, device=device)

        root_states[drone_idx, 0:3] = global_position
        root_states[drone_idx, 7:10] = zero_vel
        # Reset yaw angle to 0.0
        self.yaw = 0.0  # Reset yaw to the initial direction
        self.env.gym.set_actor_root_state_tensor(self.env.sim, gymtorch.unwrap_tensor(root_states))

    def handle_collision(self, collision_axes):
        """
        Simulate bounce on collision by reversing velocity on specified axes.
        :param collision_axes: List of axis indices (e.g. [0, 2] for X and Z)
        """
        restitution_coefficient = 0.1

        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        root_states = self.state_manager.root_states
        drone_idx = self.state_manager.drone_index

        velocity = root_states[drone_idx, 7:10].clone()
        for axis in collision_axes:
            velocity[axis] *= -restitution_coefficient

        root_states[drone_idx, 7:10] = velocity
        self.env.gym.set_actor_root_state_tensor(self.env.sim, gymtorch.unwrap_tensor(root_states))


# State manager class for accessing and managing per-drone state data
class DroneStateManager:
    def __init__(self, env, env_index):
        """
        Initialize state manager for a specific drone environment.
        :param env: Reference to the parallel drone environment
        :param env_index: Index of the drone within the vectorized environment
        """
        self.env = env
        self.env_index = env_index

        self.root_state_tensor = env.gym.acquire_actor_root_state_tensor(env.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.drone_index = env.drone_actor_indices[env_index]

        self.boundary_min_tensor = env.boundary_min_tensor[env_index]
        self.boundary_max_tensor = env.boundary_max_tensor[env_index]

        self.fixed_wall_positions = env.fixed_wall_positions
        self.fixed_box_positions = env.fixed_box_positions

    def get_position(self):
        """
        Return drone's current position (x, y, z)
        """
        return self.root_states[self.drone_index, 0:3].detach()

    def get_velocity(self):
        """
        Return drone's current velocity (vx, vy, vz)
        """
        return self.root_states[self.drone_index, 7:10].detach()