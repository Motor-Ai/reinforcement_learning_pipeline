
class Reward:
    """Base class for computing rewards in a CARLA environment."""
    def __init__(self, scene_duration: float, sim_time: float):
        self.scene_duration = scene_duration # = episode length
        self.sim_time = sim_time
        self.time_penalty = -1.0
        self.goal_threshold = 0.5 # in meters

    def __call__(self, distance_to_goal, prev_distance, collision: bool) ->  float:
        """
        Compute the reward based on:
        - A small penalty for being far from the goal.
        - A lower reward for getting closer to the goal.
        - A smaller step penalty.
        - A reduced bonus for reaching the goal.

        Args:
            distance_to_goal (float): The distance to the goal.

        Returns:
            reward (float): The computed reward.
        """
        reward = 0.0

        # Reduced progress-based reward
        if prev_distance is not None:
            reward += 1.0 * (prev_distance - distance_to_goal)

        # Check if goal is reached. Implicitly rewarded by terminating the episode.
        goal_reached = distance_to_goal < self.goal_threshold
        if goal_reached:
            reward += 50.0

        # Collision adds the remaining time penalty, since the episode will terminate
        if collision:
            reward += -50.0
            reward += self.time_penalty * (self.scene_duration - self.sim_time)

        return reward
