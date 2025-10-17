"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import numpy as np
import os
import yaml
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from stable_baselines3.common.callbacks import EvalCallback
from src.envs.callbacks import LoggerCallback
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from src.envs.carla_env import CarlaGymEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# Load configurations from YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "src/envs/configs/config.yaml")
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

RENDER_CAMERA = config["RENDER_CAMERA"]
SAVE_PATH = config["SAVE_PATH"]

rng = np.random.default_rng(0)

env = DummyVecEnv([lambda: CarlaGymEnv(render_enabled=RENDER_CAMERA)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

evaluation_env = DummyVecEnv([lambda: CarlaGymEnv(render_enabled=RENDER_CAMERA)])
evaluation_env = VecNormalize(evaluation_env, norm_obs=True, norm_reward=False, training=False)

# make_vec_env(
#     "seals:seals/CartPole-v0",
#     rng=rng,
#     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
# )

# print(env.observation_space.contains())
# print(env.action_space.contains())
# print(env.action_space)
# print(env.observation_space)

checkpoint_callback = EvalCallback(
    eval_env=evaluation_env,
    n_eval_episodes=5,
    eval_freq=1000,
    best_model_save_path=SAVE_PATH,
    log_path=None, # specify path to save results
    verbose=1,
)
logging_callback = LoggerCallback(
    save_freq=1000,
    verbose=1,
)

def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training a expert.")
    # expert = PPO(
    #     policy="MultiInputPolicy",
    #     env=env,
    #     seed=0,
    #     # batch_size=64,
    #     ent_coef=0.0,
    #     learning_rate=0.0003,
    #     # n_epochs=10,
    #     n_steps=64,
    # )

    expert = A2C(
        "MultiInputPolicy",
        env,
        n_steps=20,
        learning_rate=0.0007,
        use_sde=True,
        ent_coef=0.4,
        use_rms_prop=True, 
        verbose=1, 
        tensorboard_log="./tensorboard/", 
        squash_output=True,
        )
    
    expert.learn(10, callback=[logging_callback, checkpoint_callback])  # Note: change this to 100_000 to train a decent expert.
    return expert


# def download_expert():
#     print("Downloading a pretrained expert.")
#     expert = load_policy(
#         "ppo-huggingface",
#         organization="HumanCompatibleAI",
#         env_name="seals-CartPole-v0",
#         venv=env,
#     )
#     return expert


def sample_expert_transitions():
    expert = train_expert()  # uncomment to train your own expert
    expert.save(SAVE_PATH + "/a2c_carla_expert")  # save the trained expert
    # expert = download_expert()

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
        unwrap=False,
    )
    return rollout.flatten_trajectories(rollouts)


transitions = sample_expert_transitions()
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

# evaluation_env = make_vec_env(
#     "seals:seals/CartPole-v0",
#     rng=rng,
#     env_make_kwargs={"render_mode": "human"},  # for rendering
# )



print("Evaluating the untrained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1)

print("Evaluating the trained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward after training: {reward}")

# Save BC policy
bc_trainer.policy.save(SAVE_PATH + "/bc_carla_policy")