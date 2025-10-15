from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import gymnasium as gym


class RewardTerm(abc.ABC):
    """
    A class representing a reward function.
    """

    def __init__(self, env: gym.Env, weight: float):
        # TODO(FU): I would also add config as an argument, 
        # but we'd have to agree on how the config works.
        super().__init__()
        self._env: gym.Env = env

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
        pass # Only define in case of statefull rewards.

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
