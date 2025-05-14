import numpy as np

class PotentialFieldPlanner:
    def __init__(self, goal_position, obstacles):
        # 初始化目标点和障碍物信息
        self.goal_position = np.array(goal_position)  # 设置目标点位置
        self.obstacles = [np.array(o) for o in obstacles]  # 将障碍物位置列表转换为numpy数组
        self.safe_distance = 0.25  # 碰撞判定的安全距离，距离小于此视为碰撞
        self.avoid_distance = 1.0  # 避障距离，当无人机距离障碍物小于此值时开始产生排斥力

    def get_force(self, position, velocity):
        # 计算无人机当前位置到目标点的方向向量
        direction_to_goal = self.goal_position - position  # 从当前位置指向目标的向量
        distance_to_goal = np.linalg.norm(direction_to_goal)  # 计算到目标的距离

        # 如果距离目标点非零，对方向向量进行归一化处理
        if distance_to_goal > 0:
            direction_to_goal /= distance_to_goal

        # 吸引力计算：目标越近，吸引力越小，避免速度过冲目标
        # 吸引力大小与距离目标的距离成正比，但最多不超过0.5
        goal_force = direction_to_goal * min(0.5, distance_to_goal * 0.1)

        # 初始化排斥力向量，用于累加各个障碍物产生的排斥力
        avoidance_force = np.zeros(3)

        # 遍历所有障碍物，计算每个障碍物对无人机产生的排斥力
        for obs in self.obstacles:
            direction_to_obs = position - obs  # 当前位置指向障碍物的方向向量
            distance_to_obs = np.linalg.norm(direction_to_obs)  # 到障碍物的距离

            # 如果距离障碍物小于避障距离，则计算排斥力
            if distance_to_obs < self.avoid_distance:
                if distance_to_obs < 1e-6:  # 防止除以零，即障碍物和无人机完全重合的特殊情况
                    continue

                direction_to_obs /= distance_to_obs  # 方向向量归一化

                # 排斥力大小：距离越近，力越大，反比于距离平方
                repulsion_strength = 1.0 / (distance_to_obs ** 2) - 1.0 / (self.avoid_distance ** 2)
                repulsion_strength = max(repulsion_strength, 0)  # 确保排斥力非负

                # 计算排斥力，方向远离障碍物
                avoidance_force += direction_to_obs * repulsion_strength

        # 计算总合力：目标点吸引力 + 障碍物排斥力
        total_force = goal_force + avoidance_force

        # 限制合力的大小，避免力过大导致动作不稳定
        max_force = 1.0
        total_force = np.clip(total_force, -max_force, max_force)

        # 返回最终的合力向量，作为无人机的动力输入
        return total_force
