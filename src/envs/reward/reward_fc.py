import abc
from typing import Any, List, Optional, Dict

import carla


class RewardFunction(abc.ABC):
    """
    A class representing a reward function.
    """

    @abc.abstractmethod
    def compute(self, data: Dict[str, Any]) -> float:
        """
        Compute the reward function.
        @param data: the data required to compute the reward
        @return: the reward
        """
        ...

    @abc.abstractmethod
    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        ...


class CompositeReward(RewardFunction):
    """
    A class allowing the composition of reward functions, by weighting the reward sub-functions.
    """

    def __init__(self, sub_rewards: List[RewardFunction], weights: Optional[List[float]] = None) -> None:
        """
        Create a composite reward function.
        @param sub_rewards: a list of reward functions
        @param weights: a list of weights
        """
        self.sub_rewards = sub_rewards
        self.weights = [1.0] * len(sub_rewards) if weights is None else weights
        assert len(self.weights) == len(self.sub_rewards)
        self.requirements = []
        for sub_reward in sub_rewards:
            self.requirements += sub_reward.required_keys()
        self.requirements = list(set(self.requirements))

    def compute(self, data: Dict[str, Any]) -> float:
        """
        Compute the reward function.
        @param data: the data required to compute the reward
        @return: the reward
        """
        rewards = 0
        for sub_reward, weight in zip(self.sub_rewards, self.weights):
            rewards += weight * sub_reward.compute(data)
        return rewards

    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        return self.requirements


class EveryTimestepPenalty(RewardFunction):
    """
    A class applying a penalty at each time step.
    """

    def __init__(self, every_timestep_penalty: float = -1) -> None:
        """
        Initialize the reward function.

        NOTE: Add key "every_timestep_penalty" to the data.
        @param every_timestep_penalty: the penalty given to the agent at each timestep
        """
        self.every_timestep_penalty = every_timestep_penalty

    def compute(self, data: Dict[str, Any]) -> float:
        """
        Apply a penalty at each time step.
        @param data: the data required to compute the reward
        @return: the reward
        """
        data["every_timestep_penalty"] = self.every_timestep_penalty
        return self.every_timestep_penalty

    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        return ["every_timestep_penalty"]


class GoalImprovementReward(RewardFunction):
    """
    A class rewarding the agent for moving closer to the goal.
    """

    def compute(self, data: Dict[str, Any]) -> float:
        """
        Reward the agent for moving closer to the goal.
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Compute reward based on the progress made by the agent.
        if data["prev_distance"] is None:
            return 0.0
        return data["prev_distance"] - data["distance_to_goal"]

    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        return ["prev_distance", "distance_to_goal"]


class GoalReachedReward(RewardFunction):
    """
    A class rewarding the agent for reaching the goal location.

    NOTE: Add key "goal_reached" to the data.
    """

    def __init__(self, goal_reached_reward: float = 50.0, goal_threshold: float = 0.5) -> None:
        """
        Initialize the reward.
        @param goal_reached_reward: the reward given to the agent upon reaching the goal
        @param goal_threshold: the threshold for reaching the goal location (in meters)
        """
        self.goal_reached_reward = goal_reached_reward
        self.goal_threshold = goal_threshold

    def compute(self, data: Dict[str, Any]) -> float:
        """
        Reward the agent when it reaches the goal location.
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Compute the reward.
        data["goal_reached"] = data["distance_to_goal"] < self.goal_threshold
        return self.goal_reached_reward if data["goal_reached"] else 0.0

    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        return ["distance_to_goal"]


class CollisionPenalty(RewardFunction):
    """
    A class penalising the agent for colliding with other objects.
    """

    def __init__(self, collision_penalty: float = -25) -> None:
        """
        Initialize the reward function.

        NOTE: Must be called after "EveryTimestepPenalty".
        @param collision_penalty: the penalty given to the agent for each collision
        """
        self.collision_penalty = collision_penalty

    def compute(self, data: Dict[str, Any]) -> float:
        """
        Penalise the agent for colliding with other objects.
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Compute the collision reward.
        reward = 0.0
        if data["collision"]:
            reward += self.collision_penalty

            # Ensure that the car does not learn to crash to avoid small penalty at each time step,
            reward += data["every_timestep_penalty"] * (data["episode_length"] - data["timestep"])
        return reward

    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        return ["collision", "episode_length", "timestep", "every_timestep_penalty"]


class IllegalLaneInvasions(RewardFunction):
    """
    A class penalising each illegal lane invasion.
    """

    def __init__(self, lane_invasion_penalty: float = -25) -> None:
        """
        Initialize the reward function.
        @param lane_invasion_penalty: the penalty given to the agent for each illegal lane invasion
        """
        self.lane_invasion_penalty = lane_invasion_penalty

    def compute(self, data: Dict[str, Any]) -> float:
        """
        Penalise the agent for each illegal lane invasion.
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Compute a penalty for illegal lane invasions.
        reward = 0.0
        for lane_invasion in data["lane_invasions"]:
            for crossed_lane_marking in lane_invasion.crossed_lane_markings:
                if crossed_lane_marking.lane_change == carla.LaneChange.NONE:
                    reward += self.lane_invasion_penalty
        return reward

    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        return ["lane_invasions"]


class RedLightViolation(RewardFunction):
    """
    A class penalising movement of the ego vehicle at a red light.
    """

    def __init__(self, red_light_penalty: float = -1) -> None:
        """
        Initialize the reward function.
        @param red_light_penalty: the penalty given to the agent for driving through a red light
        """
        self.red_light_penalty = red_light_penalty

    def compute(self, data: Dict[str, Any]) -> float:
        """
        Add a penalty for driving through a red light (+penalty per speed unit over 0 km/h).
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Add penalty if the agent is moving, while at a red traffic light.
        reward = 0.0
        if data["ego_vehicle"].is_at_traffic_light():
            traffic_light = data["ego_vehicle"].get_traffic_light()
            if traffic_light.state == carla.TrafficLightState.Red:
                reward += self.red_light_penalty * data["ego_speed"]
        return reward

    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        return ["ego_vehicle", "ego_speed"]


class EgoIsTooFast(RewardFunction):
    """
    A class penalising the agent for driving too fast.
    """

    def __init__(self, too_fast_penalty: float = -1, max_speed: float = 8.333) -> None:
        """
        Initialize the reward function.
        @param too_fast_penalty: the penalty given to the agent when driving too fast
        @param max_speed: the maximum speed at which the agent is allowed to drive
        """
        self.too_fast_penalty = too_fast_penalty
        self.max_speed = max_speed

    def compute(self, data: Dict[str, Any]) -> float:
        """
        Add a penalty for driving over the speed limit.
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Penalise the agent for driving too fast (By default, over 30 km/h = 8.333 m/s).
        reward = 0.0
        if self.max_speed < data["ego_speed"]:
            reward += self.too_fast_penalty * (data["ego_speed"] - self.max_speed)
        return reward

    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        return ["ego_speed"]


class TooSlowPenalty(RewardFunction):
    """
    A class penalising the agent for driving too slowly.
    """

    def __init__(self, too_slow_penalty: float = -1, min_speed: float = 6.944) -> None:
        """
        Initialize the reward function.
        @param too_slow_penalty: the penalty giving to the agent when it drives too slow
        @param min_speed: the minimum speed at which the agent must drive
        """
        self.too_slow_penalty = too_slow_penalty
        self.min_speed = min_speed

    def compute(self, data: Dict[str, Any]) -> float:
        """
        Add a penalty for driving under the speed limit without reason.
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Penalise the agent for driving too slowly (by default, below 25 km/h = 6.944 m/s).
        reward = 0.0
        if not data["blocked_at_red_light"] and not data["goal_reached"] and data["ego_speed"] < self.min_speed:
            reward += self.too_slow_penalty * (self.min_speed - data["ego_speed"])
        return reward

    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        return ["blocked_at_red_light", "goal_reached", "ego_speed"]


class DrivingOnSidewalks(RewardFunction):
    """
    A class penalising the agent for driving on the sidewalks.
    """

    def __init__(self, on_sidewalks_penalty: float = -25) -> None:
        """
        Initialize the reward function.
        @param on_sidewalks_penalty: the penalty for driving on the sidewalks
        """
        self.on_sidewalks_penalty = on_sidewalks_penalty

    def compute(self, data: Dict[str, Any]) -> float:
        """
        Add a penalty for driving on the sidewalks.
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Penalise the agent for driving on the sidewalks.
        reward = 0.0
        if data["current_waypoint"].lane_type == carla.LaneType.Sidewalk:
            reward += self.on_sidewalks_penalty
        return reward

    def required_keys(self) -> List[str]:
        """
        Return a list of keys required by the reward function.
        @return: the list of keys
        """
        return ["current_waypoint"]


class ExperimentalRewardFunction(CompositeReward):
    """
    A wrapper around the composite class that provides a more complex reward function.
    """

    def __init__(self) -> None:
        """
        Initialize the reward function.
        """
        sub_rewards = [
            EveryTimestepPenalty(),
            GoalImprovementReward(),
            GoalReachedReward(),
            CollisionPenalty(),
            IllegalLaneInvasions(),
            RedLightViolation(),
            EgoIsTooFast(),
            TooSlowPenalty(),
            DrivingOnSidewalks()
        ]
        super().__init__(sub_rewards)
