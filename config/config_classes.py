""" Dataclasses defining the content and types of all config objects."""

from dataclasses import dataclass, field
from typing import Any
from hydra.core.config_store import ConfigStore
from config.env_config_classes import CarlaEnvConfig


@dataclass
class TrainConfig:
    """ Config for train.py """
    # Environment configuration
    env: Any = field(default_factory=CarlaEnvConfig) # must specify type. could replace with union of env configs
    
    # Training parameters
    save_path: str = "./saved_rl_models/"
    max_n_steps: int = 500000
    
    # Evaluation parameters
    eval_frequency: int = 1000
    n_eval_episode: int = 5
    
    # Tensorboard parameters
    tensorboard_dir: str = "./tensorboard/"
    logging_freq: int = 1000
    
    # Verbosity
    verbose: int = 1


@dataclass
class EvalConfig:
    """ Config for eval.py """
    # Environment configuration
    env: Any = field(default_factory=CarlaEnvConfig)
    save_path: str = "./saved_rl_models/"


# Register configs when module is imported
cs = ConfigStore.instance()

# Register the main config as "config" (can be any name)
cs.store(name="train_config", node=TrainConfig)
cs.store(name="eval_config", node=EvalConfig)
# Use different envs by running:
# python train.py +env=carla
cs.store(group="env", name="carla", node=CarlaEnvConfig)
