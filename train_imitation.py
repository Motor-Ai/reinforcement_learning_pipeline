"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.policies.serialize import load_policy

from stable_baselines3.common.vec_env import DummyVecEnv


def train_expert(env):
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training a expert.")
    expert = PPO(
        policy="MultiInputPolicy",
        env=env,
        seed=0,
        # batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        # n_epochs=10,
        n_steps=64,
    )
    expert.learn(1_000)  # Note: change this to 100_000 to train a decent expert.
    return expert


def download_expert(env):
    print("Downloading a pretrained expert.")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-CartPole-v0",
        venv=env,
    )
    return expert


def sample_expert_transitions(env, rng, pretrained = False):

    expert = download_expert(env) if pretrained else train_expert(env)

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)


@hydra.main(version_base="1.3.2", config_path="config", config_name="train_imitation")
def train_imitation(config: DictConfig):

    rng = np.random.default_rng(0)

    env = DummyVecEnv([lambda: instantiate(config.env)])

    # make_vec_env(
    #     "seals:seals/CartPole-v0",
    #     rng=rng,
    #     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    # )

    # print(env.observation_space.contains())
    # print(env.action_space.contains())
    print(env.action_space)
    print(env.observation_space)

    transitions = sample_expert_transitions(env, rng)
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

    evaluation_env = DummyVecEnv([lambda: instantiate(config.env)])

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


if __name__ == '__main__':
    train_imitation()
