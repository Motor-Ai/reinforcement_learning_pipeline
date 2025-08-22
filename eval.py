import os
import yaml
from src.envs.carla_env import CarlaGymEnv
from src.envs.carla_env_render import MatplotlibAnimationRenderer
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Load configurations from YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "src/envs/configs/config.yaml")
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

RENDER_CAMERA = config["RENDER_CAMERA"]
LOAD_MODEL = config["LOAD_MODEL"]
SAVE_PATH = config["SAVE_PATH"]
SAVED_MODEL_PATH = os.path.join(os.path.dirname(__file__), config["SAVED_MODEL_PATH"])

VECNORM_PATH = os.path.join(SAVE_PATH, "vec_normalize.pkl")

def find_best_model(path):
    best_model = None
    best_reward = float("-inf")
    if not os.path.exists(path):
        return None
    for file in os.listdir(path):
        if file.startswith("best_model_") and file.endswith(".zip"):
            try:
                reward_value = float(file.split("_")[2])
                if reward_value > best_reward:
                    best_reward = reward_value
                    best_model = os.path.join(path, file)
            except (IndexError, ValueError):
                continue
    return best_model

if __name__ == '__main__':
    try:
        env = DummyVecEnv([lambda: CarlaGymEnv(render_enabled=False)])
        eval_env = DummyVecEnv([lambda: CarlaGymEnv(render_enabled=RENDER_CAMERA)])

        # Load VecNormalize statistics if available
        if os.path.exists(VECNORM_PATH):
            print(f"Loading VecNormalize statistics from: {VECNORM_PATH}")
            env = DummyVecEnv([lambda: CarlaGymEnv(render_enabled=False)])
            env = VecNormalize.load(VECNORM_PATH, env)
            eval_env = DummyVecEnv([lambda: CarlaGymEnv(render_enabled=RENDER_CAMERA)])
            eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            print(f"Warning: VecNormalize statistics not found at {VECNORM_PATH}. Running without normalization.")

        best_model_path = find_best_model(SAVE_PATH)

        if best_model_path:
            print(f"Loading best model: {best_model_path}")
            model = A2C.load(best_model_path, env=eval_env)
        elif LOAD_MODEL:
            print(f"Loading last saved model: {SAVED_MODEL_PATH}")
            model = A2C.load(SAVED_MODEL_PATH, env=eval_env)
        else:
            print("No saved model found, initializing new model.")
            model = A2C("MultiInputPolicy", env)

        obs = eval_env.reset()
        done = False
        step_count = 0

        renderer = MatplotlibAnimationRenderer()
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