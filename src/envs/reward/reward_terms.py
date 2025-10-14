from __future__ import annotations

from src.envs.reward.reward_term_base import RewardTerm

from typing import TYPE_CHECKING
import carla

if TYPE_CHECKING:
    import gymnasium as gym


class TimePenalty(RewardTerm):
    """
    A class applying a penalty at each time step.
    """

    def measure(self) -> float:
        """
        Apply a penalty at each time step. Add a penalty for all future steps if the episode
        ends prematurely due to a collision.
        @return: the reward
        """
        # Ensure that the car does not learn to crash to avoid small penalty at each time step,
        if self._env.collision_detected:
            return self._env.episode_length - self._env.timestep
        return 1


class GoalImprovementReward(RewardTerm):
    """
    A class rewarding the agent for moving closer to the goal.
    """

    def measure(self) -> float:
        """
        Reward the agent for moving closer to the goal.
        @param data: the data required to compute the reward
        @return: the reward
        """
        return self._env.prev_distance_to_goal - self._env.distance_to_goal


class GoalReachedReward(RewardTerm):
    """
    A class rewarding the agent for reaching the goal location.
    """

    def __init__(self, goal_threshold: float = 0.5, *args, **kwargs) -> None:
        """
        Initialize the reward.
        @param goal_threshold: the threshold for reaching the goal location (in meters)
        """
        super().__init__(*args, **kwargs)
        self.goal_threshold = goal_threshold

    def measure(self) -> float:
        """
        Reward the agent when it reaches the goal location.
        @param data: the data required to compute the reward
        @return: the reward
        """
        return float(self._env.distance_to_goal < self.goal_threshold)


class CollisionPenalty(RewardTerm):
    """
    A class penalising the agent for colliding with other objects.
    """

    def measure(self) -> float:
        """
        Penalise the agent for colliding with other objects.
        """
        return float(self._env.collision_detected)


class IllegalLaneInvasions(RewardTerm):
    """
    A class penalising each illegal lane invasion.
    """

    def measure(self) -> float:
        """
        Penalise the agent for each illegal lane invasion.
        """
        reward = 0.0
        for lane_invasion in self._env.lane_invasions:
            for crossed_lane_marking in lane_invasion.crossed_lane_markings:
                if crossed_lane_marking.lane_change == carla.LaneChange.NONE:
                    reward += 1.0
        return reward


class RedLightViolation(RewardTerm):
    """
    A class penalising movement of the ego vehicle at a red light.
    """

    def measure(self) -> float:
        """
        Add a penalty for driving through a red light (+penalty per speed unit over 0 km/h).
        """
        # Add penalty if the agent is moving, while at a red traffic light.
        reward = 0.0
        if self._env.ego_vehicle.is_at_traffic_light():
            traffic_light = self._env.ego_vehicle.get_traffic_light()
            if traffic_light.state == carla.TrafficLightState.Red:
                reward += self._env.ego_speed
        return reward


class EgoIsTooFast(RewardTerm):
    """
    A class penalising the agent for driving too fast.
    """

    def __init__(self, max_speed: float = 8.333, *args, **kwargs) -> None:
        """
        Initialize the reward function.
        @param max_speed: the maximum speed at which the agent is allowed to drive
        """
        super().__init__(*args, **kwargs)
        self.max_speed = max_speed

    def measure(self) -> float:
        """
        Add a penalty for driving over the speed limit.
        """
        return max(self._env.ego_speed - self.max_speed, 0)


class TooSlowPenalty(RewardTerm):
    """
    A class penalising the agent for driving too slowly.
    """

    def __init__(self,
                 min_speed: float = 6.944,
                 goal_threshold: float = 0.5,
                 *args, **kwargs) -> None:
        """
        Initialize the reward function.
        @param too_slow_penalty: the penalty giving to the agent when it drives too slow
        @param min_speed: the minimum speed at which the agent must drive
        """
        super().__init__(*args, **kwargs)
        self.min_speed = min_speed
        # TODO(FU): Seen this as argument in multiple places
        # probably should be defined in env/task and retrieved, 
        # rather than stored
        self.goal_threshold = goal_threshold 

    def measure(self) -> float:
        """
        Add a penalty for driving under the speed limit without reason.
        """
        # Penalise the agent for driving too slowly (by default, below 25 km/h = 6.944 m/s).
        goal_reached = self._env.distance_to_goal < self.goal_threshold
        if not self._env.blocked_at_red_light and \
                not goal_reached and \
                (self._env.ego_speed < self.min_speed):
            return self.min_speed - self._env.ego_speed
        return 0.0


class DrivingOnSidewalks(RewardTerm):
    """
    A class penalising the agent for driving on the sidewalks.
    """

    def measure(self) -> float:
        """
        Add a penalty for driving on the sidewalks.
        """
        # Penalise the agent for driving on the sidewalks.
        reward = 0.0
        if self._env.current_waypoint.lane_type == carla.LaneType.Sidewalk:
            reward += 1.0
        return reward
