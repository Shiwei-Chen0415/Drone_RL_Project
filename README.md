# Drone Navigation using LiDAR and SAC in Isaac Gym

**Project Overview**

This project implements a GPU-accelerated reinforcement learning system for drone navigation using NVIDIA Isaac Gym Preview 4 platform.
The drone learns to reach its goal while avoiding obstacles using 6-direction multi-deck LiDAR sensing and the Soft Actor-Critic (SAC) algorithm.

## Features
1. **GPU-accelerated parallel simulation** using Isaac Gym (up to 10,000 drone environments).

2. **6 Directions LiDAR sensor module** with 6-directional raycasting and slab-based AABB collision detection, featuring direction-adjustable scanning adaptable to drone heading or user control.

3. **Soft Actor-Critic (SAC)** for robust, sample-efficient policy learning.

4. **Well design reward** shaping including direction alignment, collision penalties, and timeouts.

5. **Dynamic visualization** with real-time trajectory and reward plotting.

6. **Waypoint-based LiDAR** verification with yaw-angle adjustments.

## Project Structure

1. **Env.py**: Core environment setup with drone and obstacle assets, LiDAR logic and implement.

2. **DroneSetting.py**: Drone dynamics and collision response.

3. **RL.py**: Reinforcement learning environment wrapper (VecEnv) for Stable-Baselines3.

4. **Main_Train.py**: Main training entrypoint using SAC and vectorized rollout.

5. **Lidar_verify.py**: Runs the drone through predefined waypoints to verify LiDAR accuracy.

6. **Visual_Test.py**: Visualizes pre-trained models in Isaac Gym viewer.

7. **Draw.py**: Reward smoothing and plotting after training.
## LiDAR Verification
To test LiDAR accuracy during waypoint navigation:
```bash
python Lidar_verify.py
```
This will:

Sends a drone along 20+ preset 3D waypoints

Logs LiDAR readings at each step

Rotates yaw at certain nodes and compares sensor changes

Saves plots under **lidar_plots**/ and logs to **lidar_logs.txt**


## Training Instructions
```bash
python Main_Train.py
```
This will:

Launch 10,000 parallel drone environments.

Train a SAC agent to reach the goal while avoiding collisions.

Save the trained model to **sac_parallel_model.zip.**

Record reward logs to avg_episode_rewards.txt.

## Reward Visualization
After training:
```bash

python Draw.py
```
This will:

Smooths episode rewards using moving average

Plots both raw and smoothed curves

Save figures

## Dependencies

Python = 3.8

Isaac Gym (Preview 4)

PyTorch 2.0.1

CUDA 11.8

Stable-Baselines3

NumPy, Matplotlib, TQDM

Make sure Isaac Gym is correctly installed and CUDA drivers are compatible.

## License
This project is for academic or research use only. Commercial use requires permission.

