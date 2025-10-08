from dataclasses import dataclass
import carla
from src.envs.reward.reward_fc import (
    RewardData,
    CompositeReward,
    TimePenalty,
    GoalImprovementReward,
    GoalReachedReward,
    CollisionPenalty,
    IllegalLaneInvasions,
    RedLightViolation,
    EgoIsTooFast,
    TooSlowPenalty,
    DrivingOnSidewalks
)


@dataclass
class ExperimentalRewardData(RewardData):
    """
    A dataclass containing all the data required to compute the reward function.
    """
    collision: bool
    episode_length: int
    timestep: int
    prev_distance_to_goal: float
    distance_to_goal: float
    lane_invasions: list[carla.LaneInvasionEvent]
    ego_vehicle: carla.Vehicle
    ego_speed: float
    blocked_at_red_light: bool
    current_waypoint: carla.Waypoint


class ExperimentalRewardFunction(CompositeReward[ExperimentalRewardData]):
    """
    A wrapper around the composite class that provides a more complex reward function.
    """
    input_type = ExperimentalRewardData

    def __init__(self) -> None:
        """
        Initialize the reward function.
        """
        sub_rewards = [
            TimePenalty(),
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

