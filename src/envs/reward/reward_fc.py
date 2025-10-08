import abc
from typing import Optional, Protocol, TypeVar, Generic, get_type_hints
from dataclasses import dataclass
import carla


@dataclass
class RewardData:
    """Base reward data container."""


# placeholder for the input type
T = TypeVar('T', bound=RewardData)

class Reward(abc.ABC, Generic[T]):
    """
    A class representing a reward function.
    """
    input_type: type[T]

    @abc.abstractmethod
    def compute(self, data: T) -> float:
        """
        Compute the reward function.
        @param data: the data required to compute the reward
        @return: the reward
        """

    @property
    def required_keys(self) -> list[str]:
        """Get input keys required by the reward function."""
        return list(get_type_hints(self.input_type).keys())


class CompositeReward(Reward[T]):
    """
    A class allowing the composition of reward functions, by weighting the reward sub-functions.
    """

    def __init__(self, sub_rewards: list[Reward], weights: Optional[list[float]] = None) -> None:
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
            self.requirements += sub_reward.required_keys
        self.requirements = list(set(self.requirements))

    def compute(self, data: T) -> float:
        """
        Compute the reward function.
        @param data: the data required to compute the reward
        @return: the reward
        """
        rewards = 0
        for sub_reward, weight in zip(self.sub_rewards, self.weights):
            rewards += weight * sub_reward.compute(data)
        return rewards


##########################################################################
# Reward elements
##########################################################################


class HasTime(Protocol):
    """Data needed for TimePenalty."""
    collision: bool
    episode_length: int
    timestep: int


class TimePenalty(Reward):
    """
    A class applying a penalty at each time step.
    """
    input_type = HasTime

    def __init__(self, time_penalty: float = -1.0) -> None:
        """
        Initialize the reward function.
        @param every_timestep_penalty: the penalty given to the agent at each timestep
        """
        self.time_penalty = time_penalty

    def compute(self, data: HasTime) -> float:
        """
        Apply a penalty at each time step. Add a penalty for all future steps if the episode
        ends prematurely due to a collision.
        @param data: the data required to compute the reward
        @return: the reward
        """
        reward = self.time_penalty
        if data.collision:
            # Ensure that the car does not learn to crash to avoid small penalty at each time step,
            reward += self.time_penalty * (data.episode_length - data.timestep -1)
        return reward


class HasGoalDistanceChange(Protocol):
    """Data needed for GoalImprovementReward."""
    distance_to_goal: float
    prev_distance_to_goal: float


class GoalImprovementReward(Reward):
    """
    A class rewarding the agent for moving closer to the goal.
    """
    input_type = HasGoalDistanceChange

    def compute(self, data: HasGoalDistanceChange) -> float:
        """
        Reward the agent for moving closer to the goal.
        @param data: the data required to compute the reward
        @return: the reward
        """
        return data.prev_distance_to_goal - data.distance_to_goal


class HasGoalDistance(Protocol):
    """Data needed for GoalReachedReward."""
    distance_to_goal: float


class GoalReachedReward(Reward):
    """
    A class rewarding the agent for reaching the goal location.
    """
    input_type = HasGoalDistance

    def __init__(self, goal_reached_reward: float = 50.0, goal_threshold: float = 0.5) -> None:
        """
        Initialize the reward.
        @param goal_reached_reward: the reward given to the agent upon reaching the goal
        @param goal_threshold: the threshold for reaching the goal location (in meters)
        """
        self.goal_reached_reward = goal_reached_reward
        self.goal_threshold = goal_threshold

    def compute(self, data: HasGoalDistance) -> float:
        """
        Reward the agent when it reaches the goal location.
        @param data: the data required to compute the reward
        @return: the reward
        """
        goal_reached = data.distance_to_goal < self.goal_threshold
        return self.goal_reached_reward if goal_reached else 0.0


class HasCollision(Protocol):
    """Data needed for CollisionPenalty."""
    collision: bool


class CollisionPenalty(Reward):
    """
    A class penalising the agent for colliding with other objects.
    """
    input_type = HasCollision

    def __init__(self, collision_penalty: float = -25) -> None:
        """
        Initialize the reward function.

        @param collision_penalty: the penalty given to the agent for each collision
        """
        self.collision_penalty = collision_penalty

    def compute(self, data: HasCollision) -> float:
        """
        Penalise the agent for colliding with other objects.
        @param data: the data required to compute the reward
        @return: the reward
        """
        # Compute the collision reward.
        reward = 0.0
        if data.collision:
            reward += self.collision_penalty
        return reward


class HasLaneInvasions(Protocol):
    """Data needed for IllegalLaneInvasions."""
    lane_invasions: list[carla.LaneInvasionEvent]


class IllegalLaneInvasions(Reward):
    """
    A class penalising each illegal lane invasion.
    """
    input_type = HasLaneInvasions

    def __init__(self, lane_invasion_penalty: float = -25) -> None:
        """
        Initialize the reward function.
        @param lane_invasion_penalty: the penalty given to the agent for each illegal lane invasion
        """
        self.lane_invasion_penalty = lane_invasion_penalty

    def compute(self, data: HasLaneInvasions) -> float:
        """
        Penalise the agent for each illegal lane invasion.
        @param data: the data required to compute the reward
        @return: the reward
        """
        reward = 0.0
        for lane_invasion in data.lane_invasions:
            for crossed_lane_marking in lane_invasion.crossed_lane_markings:
                if crossed_lane_marking.lane_change == carla.LaneChange.NONE:
                    reward += self.lane_invasion_penalty
        return reward


class HasVehicleAndSpeed(Protocol):
    """Data needed for RedLightViolation."""
    ego_vehicle: carla.Vehicle
    ego_speed: float


class RedLightViolation(Reward):
    """
    A class penalising movement of the ego vehicle at a red light.
    """
    input_type = HasVehicleAndSpeed

    def __init__(self, red_light_penalty: float = -1) -> None:
        """
        Initialize the reward function.
        @param red_light_penalty: the penalty given to the agent for driving through a red light
        """
        self.red_light_penalty = red_light_penalty

    def compute(self, data: HasVehicleAndSpeed) -> float:
        """
        Add a penalty for driving through a red light (+penalty per speed unit over 0 km/h).
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Add penalty if the agent is moving, while at a red traffic light.
        reward = 0.0
        if data.ego_vehicle.is_at_traffic_light():
            traffic_light = data.ego_vehicle.get_traffic_light()
            if traffic_light.state == carla.TrafficLightState.Red:
                reward += self.red_light_penalty * data.ego_speed
        return reward


class HasSpeed(Protocol):
    """Data needed for EgoIsTooFast."""
    ego_speed: float


class EgoIsTooFast(Reward):
    """
    A class penalising the agent for driving too fast.
    """
    input_type = HasSpeed

    def __init__(self, too_fast_penalty: float = -1, max_speed: float = 8.333) -> None:
        """
        Initialize the reward function.
        @param too_fast_penalty: the penalty given to the agent when driving too fast
        @param max_speed: the maximum speed at which the agent is allowed to drive
        """
        self.too_fast_penalty = too_fast_penalty
        self.max_speed = max_speed

    def compute(self, data: HasSpeed) -> float:
        """
        Add a penalty for driving over the speed limit.
        @param data: the data required to compute the reward
        @return: the reward
        """
        reward = 0.0
        if self.max_speed < data.ego_speed:
            reward += self.too_fast_penalty * (data.ego_speed - self.max_speed)
        return reward


class HasSpeedAndGoal(Protocol):
    """Data needed for TooSlowPenalty."""
    ego_speed: float
    blocked_at_red_light: bool
    distance_to_goal: float


class TooSlowPenalty(Reward):
    """
    A class penalising the agent for driving too slowly.
    """
    input_type = HasSpeedAndGoal

    def __init__(self, too_slow_penalty: float = -1,
                 min_speed: float = 6.944,
                 goal_threshold: float = 0.5) -> None:
        """
        Initialize the reward function.
        @param too_slow_penalty: the penalty giving to the agent when it drives too slow
        @param min_speed: the minimum speed at which the agent must drive
        """
        self.too_slow_penalty = too_slow_penalty
        self.min_speed = min_speed
        self.goal_threshold = goal_threshold

    def compute(self, data: HasSpeedAndGoal) -> float:
        """
        Add a penalty for driving under the speed limit without reason.
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Penalise the agent for driving too slowly (by default, below 25 km/h = 6.944 m/s).
        reward = 0.0
        goal_reached = data.distance_to_goal < self.goal_threshold
        if not data.blocked_at_red_light and not goal_reached and (data.ego_speed < self.min_speed):
            reward += self.too_slow_penalty * (self.min_speed - data.ego_speed)
        return reward


class HasWaypoint(Protocol):
    """Data needed for DrivingOnSidewalks."""
    current_waypoint: carla.Waypoint


class DrivingOnSidewalks(Reward):
    """
    A class penalising the agent for driving on the sidewalks.
    """
    input_type = HasWaypoint

    def __init__(self, on_sidewalks_penalty: float = -25) -> None:
        """
        Initialize the reward function.
        @param on_sidewalks_penalty: the penalty for driving on the sidewalks
        """
        self.on_sidewalks_penalty = on_sidewalks_penalty

    def compute(self, data: HasWaypoint) -> float:
        """
        Add a penalty for driving on the sidewalks.
        @param data: the data required to compute the reward
        @return: the reward
        """

        # Penalise the agent for driving on the sidewalks.
        reward = 0.0
        if data.current_waypoint.lane_type == carla.LaneType.Sidewalk:
            reward += self.on_sidewalks_penalty
        return reward
