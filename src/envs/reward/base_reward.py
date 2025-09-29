from typing import List

import carla


class Reward:
    """Base class for computing rewards in a CARLA environment."""
    def __init__(self, scene_duration: float, step_frequecny: float):
        self.episode_length = int(scene_duration / step_frequecny)
        self.time_penalty = -1.0
        self.goal_threshold = 0.5 # in meters

    def __call__(
        self,
        distance_to_goal: float,
        prev_distance: float,
        collision: bool,
        timestep: int,
        lane_invasions: List[carla.LaneInvasionEvent]
    ) -> float:
        """
        Compute the reward based on:
        - Progress towards the goal (+1 per meter)
        - Reaching the goal (+50)
        - Collision (-50)
        - Time penalty (-1 per step until goal is reached)
        - Lane invasion (-25 per illegal lane invasion)

        Args:
            distance_to_goal (float): Current distance to the goal.
            prev_distance (float): Previous distance to the goal.
            collision (bool): Whether a collision has occurred.
            timestep (int): Current timestep in the episode.
            lane_invasions (list of lane invasion events): one event for each lane invasion.
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
            reward -= 50.0
            reward += self.time_penalty * (self.episode_length - timestep)

        # Add penalty for illegal lane invasion.
        for lane_invasion in lane_invasions:
            for crossed_lane_marking in lane_invasion.crossed_lane_markings:
                if crossed_lane_marking.lane_change == carla.LaneChange.NONE:
                    reward -= 25

        return reward
