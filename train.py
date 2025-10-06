import os
import yaml
from src.envs.carla_env_v2 import CarlaGymEnv
from src.envs.callbacks import LoggerCallback
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# Load configurations from YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "src/envs/configs/config.yaml")
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

RENDER_CAMERA = config["RENDER_CAMERA"]
SAVE_PATH = config["SAVE_PATH"]

if __name__ == '__main__':
    # Create vectorized environment
    env = DummyVecEnv([lambda: CarlaGymEnv(render_enabled=RENDER_CAMERA)])
    eval_env = DummyVecEnv([lambda: Monitor(CarlaGymEnv(render_enabled=False))])

    # Apply VecNormalize (normalizes both observations and rewards)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # Don't normalize rewards during eval

    # Create a directory for saving models/checkpoints.
    save_dir = SAVE_PATH

    #NOTE: this will save the model, which contains the VecNormalize layer. you need to use that layer to
    # wrap a new env when testing the model. use model.get_vec_normalize_env().
    # Don't use model.set_env() before, it will erase the VecNormalize layer.
    checkpoint_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=5,
        eval_freq=1000,
        best_model_save_path=save_dir,
        log_path=None, # specify path to save results
        verbose=1,
    )
    logging_callback = LoggerCallback(
        save_freq=1000,
        verbose=1,
    )

    # Train A2C model
    model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

    try:
        model.learn(total_timesteps=500_000, callback=[logging_callback, checkpoint_callback])
    except KeyboardInterrupt:
        print("Training interrupted.")

    env.close()
    print("Environment closed.")
