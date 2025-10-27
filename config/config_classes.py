""" Dataclasses defining the content and types of all config objects."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    env: dict  # used to instantiate a gym.Env environment
    save_path: str
    max_n_steps: int
    eval_frequency: int
    n_eval_episode: int
    tensorboard_dir: str
    logging_freq: int
    verbose: int

@dataclass
class EvalConfig:
    env: dict  # used to instantiate a gym.Env environment
    save_path:str
