from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch
import math
import numpy as np

class DroneEnv:
    def __init__(self, visualize=False, num_envs=1):
        # Initialize Isaac Gym API handle
        self.gym = gymapi.acquire_gym()

        # Simulation parameter configuration
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z  # Set Z-axis as the up direction
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)  # Zero gravity for microgravity or space simulation

        self.sim_params.use_gpu_pipeline = True   # Use PhysX engine and GPU pipeline
        self.sim_params.physx.use_gpu = True
        self.sim_params.substeps = 2  # Sub-steps per simulation step
        self.sim_params.physx.num_threads = 12  # PhysX CPU thread count
        self.sim_params.physx.solver_type = 0  # Use PGS solver
        self.sim_params.dt = 0.1  # Fixed simulation time step
        print("Isaac Gym GPU Pipeline:", self.sim_params.use_gpu_pipeline)
        print("Isaac Gym PhysX GPU:", self.sim_params.physx.use_gpu)

        # Create simulation instance
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)

        # Viewer setup if visualization is enabled
        self.visualize = visualize
        self.viewer = None

        if self.visualize:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            cam_pos = gymapi.Vec3(25.0, 10.0, 8.0)
            cam_target = gymapi.Vec3(25.0, 2.5, 2.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Add ground plane (visual reference)
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0
        plane_params.static_friction = 0
        plane_params.dynamic_friction = 0
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

        # Create obstacle asset configuration
        box_opts = gymapi.AssetOptions()
        box_opts.fix_base_link = True  # Static obstacle
        box_opts.disable_gravity = True
        box_opts.thickness = 0.001  # Ensure collider generation
        self.box_asset = self.gym.create_box(self.sim, 1.0, 1.0, 1.0, box_opts)
        self.wall_horizontal = self.gym.create_box(self.sim, 1.0, 5.0, 2.5, box_opts)
        self.wall_vertical = self.gym.create_box(self.sim, 1.0, 2.5, 5.0, box_opts)

        # Drone asset with collision, dynamic
        drone_opts = gymapi.AssetOptions()
        drone_opts.disable_gravity = True
        drone_opts.thickness = 0.001
        self.drone_asset = self.gym.create_sphere(self.sim, 0.25, drone_opts)

        # Parallel environment setup
        self.num_envs = num_envs
        self.envs = []
        self.drone_handles = []
        self.drone_actor_indices = []
        #Special for Lidar verify
        self.yaw_angle_tensor = torch.zeros(self.num_envs, dtype=torch.float32, device="cuda")

        # Environment space boundaries
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(50.0, 5.0, 5.0)
        envs_per_row = int(np.sqrt(num_envs))

        # Obstacle local coordinates
        self.fixed_wall_positions = [
            (5, 2.5, 3.75, "horizontal"),
            (10, 3.75, 2.5, "vertical"),
            (15, 2.5, 1.25, "horizontal"),
            (20, 1.25, 2.5, "vertical"),
            (25, 2.5, 3.75, "horizontal"),
            (30, 3.75, 2.5, "vertical"),
            (35, 2.5, 1.25, "horizontal"),
        ]

        self.fixed_box_positions = [
            (37, 1.0, 3.5),
            (39, 2.0, 1.5),
            (42, 4.0, 2.0),
            (44, 3.0, 4.0),
        ]

        # Create multiple environments
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            self.envs.append(env)

            # Add drone
            drone_pose = gymapi.Transform()
            drone_pose.p = gymapi.Vec3(1.0, 3.0, 2.0)  # 无需加 offset

            drone_handle = self.gym.create_actor(env, self.drone_asset, drone_pose, f"drone_{i}", i, 0)
            self.drone_handles.append(drone_handle)
            drone_index = self.gym.get_actor_index(env, drone_handle, gymapi.DOMAIN_SIM)
            self.drone_actor_indices.append(drone_index)
            drone_color = gymapi.Vec3(0.0, 1.0, 0.0)
            self.gym.set_rigid_body_color(env, drone_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, drone_color)

            # Add walls
            for j, pos in enumerate(self.fixed_wall_positions):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(
                    pos[0],
                    pos[1],
                    pos[2]
                )
                asset = self.wall_horizontal if pos[3] == "horizontal" else self.wall_vertical
                handle = self.gym.create_actor(env, asset, pose, f"wall_{i}_{j}", i, 0)
                self.gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(1.0, 0.0, 0.0))

            # Add boxes
            for j, pos in enumerate(self.fixed_box_positions):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(
                    pos[0],
                    pos[1],
                    pos[2]
                )
                handle = self.gym.create_actor(env, self.box_asset, pose, f"box_{i}_{j}", i, 0)
                self.gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(0.0, 0.0, 1.0))

        # Prepare GPU tensor interface
        self.gym.prepare_sim(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))

        # Define boundaries per environment
        box_size = torch.tensor([50.0, 5.0, 5.0], device="cuda")
        self.boundary_min_tensor = torch.zeros((self.num_envs, 3), device="cuda")
        self.boundary_max_tensor = box_size.expand(self.num_envs, 3).clone()

    def step_simulation(self):
        """
        Step simulation and render viewer if enabled
        """
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        if self.visualize and self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim)

    def is_viewer_closed(self):
        """
        Check if the viewer window has been closed
        """
        if self.viewer is not None:
            return self.gym.query_viewer_has_closed(self.viewer)
        return False

    def get_lidar_distance_all(self, yaw_angles=None):
        """
        Compute LiDAR distances for all drones across environments.
        :param yaw_angles: Tensor(shape=(num_envs,)), yaw angles (in radians) of each drone.
                           Defaults to a zero vector if not provided.
        """
        def lidar_with_aabb(positions, directions, obstacle_centers, obstacle_sizes, max_range=10.0):
            """
            Compute the shortest LiDAR distance between rays and all obstacles (AABB boxes).
            Uses the slab method for ray-box intersection.
            positions: drone positions (E, 3)
            directions: LiDAR ray directions (E, 6, 3)
            obstacle_centers: center positions of each obstacle (E, obs_num, 3)
            obstacle_sizes: sizes of each obstacle (E, obs_num, 3)
            max_range: maximum LiDAR detection range
            """
            num_envs, num_obs, _ = obstacle_centers.shape
            positions = positions.unsqueeze(1).unsqueeze(2)  # (E, 1, 1, 3) for broadcasting
            directions = directions.view(num_envs, 6, 1, 3)  # (E, 6, 1, 3)

            # Compute min and max corners of AABBs
            obs_min = obstacle_centers - obstacle_sizes / 2
            obs_max = obstacle_centers + obstacle_sizes / 2
            obs_min = obs_min.unsqueeze(1)
            obs_max = obs_max.unsqueeze(1)

            # Avoid division by zero by clamping direction components to 1e-6
            dir_eps = directions.clamp(min=1e-6)

            # Slab method: compute intersections with each axis-aligned plane
            t1 = (obs_min - positions) / dir_eps
            t2 = (obs_max - positions) / dir_eps

            tmin = torch.min(t1, t2)
            tmax = torch.max(t1, t2)

            # Compute t_near and t_far for each ray
            t_close = tmin.max(dim=3).values
            t_far = tmax.min(dim=3).values

            # Validate intersections: t_far > t_close and both positive
            valid = (t_far >= t_close) & (t_far >= 0.0) & (t_close >= 0.0)

            # Set invalid intersections to max_range
            t_close = torch.where(valid, t_close, torch.full_like(t_close, max_range))

            # For all obstacles, take the minimum distance
            min_dist, _ = t_close.min(dim=2)
            return min_dist

        def compute_boundary_lidar(positions, directions, boundary_min, boundary_max, max_range=10.0):
            """
            Compute LiDAR distances between rays and environment boundaries.
            Uses ray-plane intersection equations, without modeling physical walls.
            positions: drone positions (E, 3)
            directions: LiDAR ray directions (E, 6, 3)
            boundary_min: minimum corner of boundary box (E, 3)
            boundary_max: maximum corner of boundary box (E, 3)
            max_range: maximum LiDAR detection range
            """
            E = positions.shape[0]
            lidar_boundary = torch.full((E, 6), max_range, dtype=torch.float32, device="cuda")

            # Iterate over each direction (front/back/left/right/up/down)
            for i in range(6):
                dir_vec = directions[:, i, :]  # Direction vector (E, 3)

                t_list = []
                for axis in range(3):  # For each x/y/z axis, compute intersections with min/max planes
                    for boundary in [boundary_min[:, axis], boundary_max[:, axis]]:
                        denom = dir_vec[:, axis]  # Direction component along axis
                        mask = torch.abs(denom) > 1e-6  # Only compute if not parallel
                        t = torch.full((E,), float('inf'), dtype=torch.float32, device="cuda")
                        t_valid = (boundary - positions[:, axis])[mask] / denom[mask]
                        t[mask] = t_valid
                        t_list.append(t)

                t_stack = torch.stack(t_list, dim=1)  # (E, 6*2), six planes' intersection t values
                t_stack[t_stack <= 0] = float('inf')  # Keep only forward-facing intersections
                t_min, _ = t_stack.min(dim=1)  # Closest intersection per env
                lidar_boundary[:, i] = torch.minimum(t_min, torch.full_like(t_min, max_range))

            return lidar_boundary

        # === Main function body ===
        MAX_RANGE = 10.0
        num_envs = self.num_envs
        positions = self.root_tensor[self.drone_actor_indices, 0:3]  # Current drone positions (E, 3)

        # === Compute ray directions based on yaw angles ===
        if yaw_angles is None:
            yaw_angles = torch.zeros(num_envs, device="cuda")

        cos_yaw = torch.cos(yaw_angles)
        sin_yaw = torch.sin(yaw_angles)

        # 6 basic directions: front, back, left, right, up, down
        dir_front = torch.stack([cos_yaw, sin_yaw, torch.zeros_like(cos_yaw)], dim=1)
        dir_back = torch.stack([-cos_yaw, -sin_yaw, torch.zeros_like(cos_yaw)], dim=1)
        dir_left = torch.stack([-sin_yaw, cos_yaw, torch.zeros_like(cos_yaw)], dim=1)
        dir_right = torch.stack([sin_yaw, -cos_yaw, torch.zeros_like(cos_yaw)], dim=1)
        dir_up = torch.tensor([0, 0, 1], dtype=torch.float32, device="cuda").repeat(num_envs, 1)
        dir_down = torch.tensor([0, 0, -1], dtype=torch.float32, device="cuda").repeat(num_envs, 1)

        # Final direction tensor (E, 6, 3)
        directions = torch.stack([
            dir_front, dir_back, dir_left, dir_right, dir_up, dir_down
        ], dim=1)

        # === Boundary check (ray vs boundary planes) ===
        boundary_min = self.boundary_min_tensor
        boundary_max = self.boundary_max_tensor
        lidar_boundary = compute_boundary_lidar(positions, directions, boundary_min, boundary_max, max_range=MAX_RANGE)

        # === Obstacle check (ray vs AABB) ===
        if not hasattr(self, "obstacle_tensor"):
            # Initialize obstacle info only once
            obstacle_list = self.fixed_wall_positions + self.fixed_box_positions
            obs_np = np.array([obs[:3] for obs in obstacle_list], dtype=np.float32)
            obs_sizes = []
            for obs in obstacle_list:
                if len(obs) == 4 and obs[3] == "horizontal":
                    obs_sizes.append([1.0, 5.0, 2.5])
                elif len(obs) == 4 and obs[3] == "vertical":
                    obs_sizes.append([1.0, 2.5, 5.0])
                else:
                    obs_sizes.append([1.0, 1.0, 1.0])
            self.obstacle_tensor = torch.tensor(obs_np, dtype=torch.float32, device="cuda")
            self.obstacle_size_tensor = torch.tensor(obs_sizes, dtype=torch.float32, device="cuda")

        # Adjust obstacle positions to global positions for each environment
        obs_tensor = self.obstacle_tensor.unsqueeze(0).expand(num_envs, -1, -1)
        obs_size_tensor = self.obstacle_size_tensor.unsqueeze(0).expand(num_envs, -1, -1)

        # Compute shortest distance in obstacle directions
        lidar_obs = lidar_with_aabb(positions, directions, obs_tensor, obs_size_tensor, max_range=MAX_RANGE)

        # === Combine boundary and obstacle checks, take the minimum as final LiDAR reading ===
        lidar_real = torch.minimum(lidar_obs, lidar_boundary)
        return lidar_real

if __name__ == "__main__":
    env = DroneEnv(visualize=True, num_envs=4)
    print("Parallel environments loaded. Press 'ESC' to close viewer...")

    first_print = True

    while not env.is_viewer_closed():
        env.step_simulation()
        env.gym.refresh_actor_root_state_tensor(env.sim)
        lidar_all = env.get_lidar_distance_all().detach().cpu().numpy()

        if first_print:
            print("Current drone Lidar distances (Front, Back, Left, Right, Up, Down):")
            print("  +X Front:", lidar_all[0][0])
            print("  -X Back :", lidar_all[0][1])
            print("  +Y Left :", lidar_all[0][2])
            print("  -Y Right:", lidar_all[0][3])
            print("  +Z Up   :", lidar_all[0][4])
            print("  -Z Down :", lidar_all[0][5])
            first_print = False
