from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING
import src.envs.reward.reward_terms as RewardTermsLib
from collections import defaultdict

if TYPE_CHECKING:
    from src.envs.reward.reward_term_base import RewardTerm
    import gym


class RewardManager:
    """
    A class allowing the composition of reward functions, by weighting the reward sub-functions.
    """
    def __init__(self, env: gym.Env, reward_terms: Dict[str, Dict]) -> None:
        """
        Create a composite reward function.
        @param reward_terms: a dictionary mapping term names to their arguments as dicts
        """
        # Functional
        self._env = env
        self.reward_term_cfgs = reward_terms
        self.reward_terms: dict[str, RewardTerm] = self._init_terms()
        
        # Logging 
        self.stepwise_logs: defaultdict[str, float] = defaultdict(lambda: 0.0)
        self.cumulative_logs: defaultdict[str, float] = defaultdict(lambda: 0.0)
        self.episodic_logs: defaultdict[str, List[float]] = defaultdict(lambda: [])

    def reset(self):
        """
        Reset the manager and its terms.
        """
        for term_name, term in self.reward_terms.items():
            term.reset()
            self.episodic_logs[term_name].append(
                self.cumulative_logs[term_name]
            )
            self.cumulative_logs[term_name] = 0.0
            self.stepwise_logs[term_name] = 0.0

    def compute(self) -> float:
        """
        Compute the step reward, summing all the reward terms computations.
        @return: the full step reward.
        """
        full_reward = 0.0
        for term_name, term in self.reward_terms.items():
            term_reward = term.compute()
            full_reward += term_reward
            
            # Logging
            self.stepwise_logs[term_name] = term_reward
            self.cumulative_logs[term_name] += term_reward
        
        return full_reward

    def _init_terms(self) -> dict[str, RewardTerm]:
        """
        Initialize the reward terms.
        @return: A mapping between the name and the object.
        """
        reward_terms = {}
        for term_class_name, term_cfg in self.reward_term_cfgs.items():
            term_cls = getattr(RewardTermsLib, term_class_name)
            reward_terms[term_class_name] = term_cls(env=self._env, **term_cfg)
        
        return reward_terms