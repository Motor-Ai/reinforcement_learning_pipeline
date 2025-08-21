import carla
import numpy as np

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

        # Reduced progress-based reward
        if prev_distance is None:
            progress_reward = 0.0
        else:
            progress_reward = 1.0 * (prev_distance - distance_to_goal)

        # Check if goal is reached
        goal_reached = distance_to_goal < self.goal_threshold

        # Total reward
        reward = self.time_penalty + progress_reward + 50.0 * goal_reached - 50.0 * collision #TODO: add collision time penalty?

        # preprocess the reward
        # self.preprocessor.set_reward_range(-400, 200)
        # reward = self.preprocessor.preprocess_reward(reward)
        return reward
