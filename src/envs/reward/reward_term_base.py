from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional, Protocol, cast
import carla

if TYPE_CHECKING:
    import gymnasium as gym


class GeneralEnv(Protocol):
    """A placeholder protocol for """
    cumulative_step: int
    collision_detected: bool
    episode_length: int
    timestep: int
    prev_distance_to_goal: float
    distance_to_goal: float
    lane_invasions: list[carla.LaneInvasionEvent]
    ego_vehicle: carla.Vehicle
    ego_speed: float
    blocked_at_red_light: bool
    current_waypoint: carla.Waypoint


class RewardTerm(abc.ABC):
    """
    A class representing a reward function.
    """

    def __init__(self, env: gym.Env, weight: float):
        # TODO(FU): I would also add config as an argument,
        #   but we'd have to agree on how the config works.
        super().__init__()

        # TODO(FU): Remove this cast once we implement an env general class.
        self._env = cast(GeneralEnv, env)

        assert weight > 0.0, "Weight must be positive" # internally agreed on this.
        self._weight: float = weight
        self._cached_measurement: Optional[float] = None
        self._cached_at_step: int = -1

    def _cache_measurement(self, measurement: float):
        """
        Cache the measurement and update the validity.
        """
        self._cached_measurement = measurement
        self._cached_at_step = self._env.cumulative_step

    def _check_cached_is_valid(self):
        """
        Check if the cached measurement is still valid.
        @return: boolean indicating the validity of the cached measurement.
        """
        return self._cached_at_step == self._env.cumulative_step

    def compute(self) -> float:
        """
        Compute the weighted reward. Works as an external API for the manager.
        @return: the weighted reward
        """

        if not self._check_cached_is_valid():
            measurement = self.measure()
            self._cache_measurement(measurement)

        # TODO(FU): Should be multiplied by shaper also, 
        # once we define the best way to implement it.
        return self._weight * self._cached_measurement

    def reset(self):
        """
        Reset the reward term.
        """
        pass # Only define in case of stateful rewards.

    @abc.abstractmethod
    def measure(self) -> float:
        """
        Compute the raw reward.
        @return: the raw reward measurement.
        """
        raise NotImplementedError

    @property
    def raw_reward(self):
        return self._cache_measurement # just for renaming purposes
