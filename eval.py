import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


@hydra.main(version_base="1.3.2", config_path="config", config_name="eval")
def eval(config: DictConfig):

    vec_norm_path = os.path.join(config.save_path, "1.1.3.1/vec_normalize.pkl")
    # TODO: right now every new model is saved into the same dir, erasing the previous run. Should
    #  make a new dir for every run and make eval load the latest dir by default.

    eval_env = DummyVecEnv([lambda: instantiate(config.env)])
    eval_env = VecNormalize.load(vec_norm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    best_model_path = os.path.join(config.save_path, "1.1.3.1/best_model.zip")
    print(f"Loading best model: {best_model_path}")
    model = A2C.load(best_model_path, env=eval_env)

    try:
        obs = eval_env.reset()
        done = False
        step_count = 0

        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            step_count += 1
            print(f"Step: {step_count}, Reward: {reward}, Sim Time: {info[0]['sim_time']}")
            step += 1

    except KeyboardInterrupt:
        print("Simulation interrupted.")

    eval_env.close()
    print("Environment closed.")


if __name__ == '__main__':
    eval()
