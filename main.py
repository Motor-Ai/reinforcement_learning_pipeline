import os

import hydra
from omegaconf import DictConfig

from src.envs.carla_env import CarlaGymEnv
from src.envs.callbacks import SaveBestlCallback, LoggerCallback
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def find_best_model(path):
    """Finds the best model saved in the given path (based on reward value in the filename)."""
    best_model = None
    best_reward = float("-inf")

    if not os.path.exists(path):
        return None

    for file in os.listdir(path):
        if file.startswith("best_model_") and file.endswith(".zip"):
            try:
                reward_value = float(file.split("_")[2])  # Extract reward value from filename
                if reward_value > best_reward:
                    best_reward = reward_value
                    best_model = os.path.join(path, file)
            except (IndexError, ValueError):
                continue  # Skip if filename format is incorrect

    return best_model


@hydra.main(version_base="1.3.2", config_path="config", config_name="main")
def main(config: DictConfig):
    try:
        # Create vectorized environment
        env = DummyVecEnv([lambda: CarlaGymEnv(config.env, render_enabled=False)])
        eval_env = DummyVecEnv([lambda: CarlaGymEnv(config.env, render_enabled=config.env.render_camera)])

        # Apply VecNormalize (normalizes both observations and rewards)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # Don't normalize rewards during eval

        if config.train:
            # Create a directory for saving models/checkpoints.
            os.makedirs(config.save_path, exist_ok=True)

            # Create the custom callback: save a checkpoint every 1,000 timesteps.
            checkpoint_callback = SaveBestlCallback(
                eval_env=eval_env,
                save_freq=1000,
                save_path=config.save_path,
                vec_env=env,
                n_eval_episodes=200,
                verbose=1
            )
            logging_callback = LoggerCallback(
                save_freq=1000,
                verbose=1
            )
            # Train A2C model
            model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
            model.learn(total_timesteps=50_000, callback=[checkpoint_callback, logging_callback])

        if config.test:
            # Load VecNormalize statistics if available
            vec_norm_path = os.path.join(config.save_path, "vec_normalize.pkl")
            if os.path.exists(vec_norm_path):
                print(f"Loading VecNormalize statistics from: {vec_norm_path}")
                env = DummyVecEnv([lambda: CarlaGymEnv(config.env, render_enabled=False)])
                env = VecNormalize.load(vec_norm_path, env)
                eval_env = DummyVecEnv([lambda: CarlaGymEnv(config.env, render_enabled=config.env.render_camera)])
                eval_env = VecNormalize.load(vec_norm_path, eval_env)
                eval_env.training = False  # Disable normalization updates during testing
                eval_env.norm_reward = False  # Don't normalize rewards during testing
            else:
                print(f"Warning: VecNormalize statistics not found at {vec_norm_path}. Running without normalization.")

            # Find the best model available
            best_model_path = find_best_model(config.save_path)

            # Load the trained model
            if best_model_path:
                print(f"Loading best model: {best_model_path}")
                model = A2C.load(best_model_path, env=eval_env)
            elif config.load_model:
                print(f"Loading last saved model: {config.load_model_path}")
                model = A2C.load(config.load_model_path, env=eval_env)
            else:
                print("No saved model found, initializing new model.")
                model = A2C("MultiInputPolicy", env)

            # Testing loop
            obs = eval_env.reset()
            done = False
            step_count = 0

            step = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                step_count += 1
                print(f"Step: {step_count}, Reward: {reward}, Sim Time: {info['sim_time']}")
                step += 1

    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == '__main__':
    main()
