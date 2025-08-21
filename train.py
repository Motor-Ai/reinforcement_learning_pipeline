import os
import yaml
from envs.carla_env import CarlaGymEnv
from envs.callbacks import SaveBestlCallback, LoggerCallback
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Load configurations from YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "envs/configs/config.yaml")
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

RENDER_CAMERA = config["RENDER_CAMERA"]
SAVE_PATH = config["SAVE_PATH"]

if __name__ == '__main__':
    try:
        # Create vectorized environment
        env = DummyVecEnv([lambda: CarlaGymEnv(render_enabled=RENDER_CAMERA)])
        eval_env = DummyVecEnv([lambda: CarlaGymEnv(render_enabled=False)])

        # Apply VecNormalize (normalizes both observations and rewards)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # Don't normalize rewards during eval

        # Create a directory for saving models/checkpoints.
        save_dir = SAVE_PATH
        os.makedirs(save_dir, exist_ok=True)

        # Create the custom callback: save a checkpoint every 1,000 timesteps.
        checkpoint_callback = SaveBestlCallback(
            eval_env=eval_env,
            save_freq=1000,
            save_path=save_dir,
            vec_env=env,
            n_eval_episodes=5,
            verbose=1
        )
        logging_callback = LoggerCallback(
            save_freq=1000,
            verbose=1
        )

        # Train A2C model
        model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
        model.learn(total_timesteps=50_000, callback=[checkpoint_callback, logging_callback])

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()
        print("Environment closed.")