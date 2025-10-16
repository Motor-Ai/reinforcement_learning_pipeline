import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.envs.callbacks import LoggerCallback
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


@hydra.main(version_base="1.3.2", config_path="config", config_name="train")
def train(config: DictConfig) -> None:

    # Create vectorized environment
    env = DummyVecEnv([lambda: instantiate(config.env)])
    eval_env = DummyVecEnv([lambda: Monitor(instantiate(config.env))])

    # Apply VecNormalize (normalizes both observations and rewards)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # Don't normalize rewards during eval

    #NOTE: this will save the model, which contains the VecNormalize layer.
    # you need to use that layer to wrap a new env when testing the model.
    # use model.get_vec_normalize_env().
    # Don't use model.set_env() before, it will erase the VecNormalize layer.
    checkpoint_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=config.n_eval_episode,
        eval_freq=config.eval_frequency,
        best_model_save_path=config.save_path,
        log_path=None, # specify path to save results
        verbose=config.verbose,
    )
    logging_callback = LoggerCallback(
        save_freq=config.logging_freq,
        verbose=config.verbose,
    )

    # Train A2C agent
    model = A2C("MultiInputPolicy", env, verbose=config.verbose, tensorboard_log=config.tensorboard_dir)

    try:
        model.learn(total_timesteps=config.max_n_steps, callback=[logging_callback, checkpoint_callback])
    except KeyboardInterrupt:
        print("Training interrupted.")

    env.close()
    print("Environment closed.")


if __name__ == '__main__':
    train()
