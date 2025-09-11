import threading
import elements
import embodied
import numpy as np
from src.envs.carla_env import CarlaGymEnv


class Carla(embodied.Env):
  """
  A class wrapping the CarlaGymEnv.
  """

  def __init__(self, name):
    self.env = CarlaGymEnv()

  @property
  def obs_space(self):
    return {
        key: elements.Space(box.dtype, box.shape, low=box.low, high=box.high)
        for key, box in self.env.observation_manager.observation_space.items()
    } | {
      'reward': elements.Space(np.float32),
      'is_last': elements.Space(bool),
      'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    box = self.env.action_manager.action_space
    return {
        'frenet_param': elements.Space(box.dtype, box.shape, box.low, box.high)
    }

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    return obs | {
        'reward': reward,
        'is_last': done,
        'is_terminal': done,
    }

  def _reset(self):
    self.env.reset()

  def _render(self, reset=False):
    self.env.render()
