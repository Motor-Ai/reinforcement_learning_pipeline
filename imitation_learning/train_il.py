import pickle
import torch
import gymnasium
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from imitation.algorithms.bc import BC
from imitation.data import types
from imitation.algorithms.dagger import SimpleDAggerTrainer, LinearBetaSchedule, ExponentialBetaSchedule
import tempfile
import numpy as np
import h5py
from src.envs.carla_env import CarlaGymEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy


import matplotlib.pyplot as plt

def flatten_obs(obs_dict):
    # Concatenate all values into one vector
    return np.concatenate([v.flatten() for v in obs_dict.values()]).astype(np.float32)

with h5py.File("carla_data.h5", "r") as f:
    ego   = f["observations/ego"][:]     # first frame RGB
    neighbors = f["observations/neighbors"][:]   # first frame LiDAR
    maps = f["observations/map"][:]   # all speed values
    global_route = f["observations/global_route"][:]   # all speed values
    actions   = f["actions"][:]  # all speed values
    next_location = f["next_location"][:]  # all speed values



print("Data loaded from h5 file and converted to Transitions format.")

print(ego.shape, neighbors.shape, maps.shape, global_route.shape, actions.shape)

# # Plot the data
# for i in range(len(ego)):
#     plt.scatter(maps[i][:,:, 0], maps[i][:,:, 1], c='red', s=1)
#     plt.scatter(global_route[i][:, 0], global_route[i][:, 1], c='purple', s=1)


#     # plt.figure()

#     plt.scatter(ego[i][:, 1], ego[i][:, 2], c='blue', s=1)
#     plt.scatter(next_location[i][0], next_location[i][1], c='orange', s=10, marker='x')

#     # plt.scatter(neighbors[i][:, :, 1], neighbors[i][:, :, 2], c='green', s=1)


#     plt.xlim(-50, 50)
#     plt.ylim(-50, 50)
#     plt.savefig("plots/plot_{:03d}.png".format(i))
#     plt.cla()

# Dummy CARLA environment wrapper (replace with your custom gym.Env)
# venv = DummyVecEnv([lambda: CarlaGymEnv(render_enabled=False)])


class FlattenObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_keys = list(env.observation_space.spaces.keys())
        low = np.concatenate([env.observation_space[k].low.flatten() for k in self.obs_keys])
        high = np.concatenate([env.observation_space[k].high.flatten() for k in self.obs_keys])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return flatten_obs(observation)
    
env = CarlaGymEnv(render_enabled=False)
# env = FlattenObservationWrapper(env)
evaluation_env = DummyVecEnv([lambda: env])
evaluation_env = VecNormalize(evaluation_env, norm_obs=True, norm_reward=False, training=False)

rng = np.random.default_rng(0)

def iter_transitions(batch_size=5000):
    # Copied the following from Rollout.py in imitation library - flatten trajectory
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    N = actions.shape[0]  # total steps
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        parts= {key: [] for key in keys}

        obs = {}
        obs['global_route'] = np.expand_dims(global_route[start:end], axis = 1)
        obs['map'] = np.expand_dims(maps[start:end], axis = 1)
        obs['neighbors'] = np.expand_dims(neighbors[start:end], axis = 1)
        obs['ego'] = np.expand_dims(ego[start:end], axis = 1)
        # obs = flatten_obs(obs)
        obs = types.DictObs(obs)
        parts["obs"].append(obs)
        parts["next_obs"].append(obs)

        parts["infos"].append(obs)  # dummy infos
        parts["acts"].append(
          np.pad(actions[start:end], ((0, 0), (0, 6)), mode="constant", constant_values=0))

    parts["dones"] = np.expand_dims(np.zeros(parts['acts'][0].shape[0], dtype = np.bool), axis = 0).tolist() # dummy dones
    cat_parts = {
        key: types.concatenate_maybe_dictobs(part_list)
        for key, part_list in parts.items()
    }

    # Convert to imitation library format
    # transitions = types.Transitions(**cat_parts)

    yield types.Transitions(**cat_parts)

lr_scheduler = lambda x: 0.003
policy = MultiInputActorCriticPolicy(
    observation_space=evaluation_env.observation_space,
    action_space=evaluation_env.action_space,
    lr_schedule=lr_scheduler,
    log_std_init= -0.5,
)
# Initialize Behavior Cloning
bc_trainer = BC(
    observation_space=evaluation_env.observation_space,
    action_space=evaluation_env.action_space,
    demonstrations=None,  # will set later
    rng = np.random.default_rng(0),
    policy= policy,
)

print("Evaluating the untrained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward before training: {reward}")


# Behaviour cloning Train
for chunk in iter_transitions(batch_size=300):
    bc_trainer.set_demonstrations(chunk)
    bc_trainer.train(n_epochs=1000)
    
print("Evaluating the trained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward after training: {reward}")

bc_trainer.policy.save("bc_carla_policy_act_OG")

# Beta is set to 1 for all time steps (i.e., always use expert action)
# Essential to have beta=1 all time to keep querying the expert for actions as we use 20 steps as observation
# beta_schedule = LinearBetaSchedule(rampdown_rounds=np.inf)
exponential_beta_schedule = ExponentialBetaSchedule(decay_probability=1) # no decay, i.e. constant beta=1

# DAgger training
with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=evaluation_env,
        scratch_dir=tmpdir,
        expert_policy=evaluation_env.venv.envs[0],
        bc_trainer=bc_trainer,
        rng=rng,
        beta_schedule= exponential_beta_schedule,
    )
    dagger_trainer.train(8_000)

print("BC training finished and policy saved.")
