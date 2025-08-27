import os
import numpy as np
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from torch.utils.tensorboard.writer import SummaryWriter
from stable_baselines3.common.vec_env import VecNormalize

N_SAVED_CHECKPOINTS = 5

class SaveBestlCallback(BaseCallback):
    """
    Custom callback that:
      - Saves a checkpoint every `save_freq` timesteps.
      - Keeps only the most recent N_SAVED_CHECKPOINTS checkpoint files.
      - Saves the best N_SAVED_CHECKPOINTS models based on training rewards.
      - Saves VecNormalize stats alongside the model.
      - Adds reward value to the saved model filename.
    """

    def __init__(self, eval_env, save_freq: int, save_path: str, vec_env: VecNormalize, n_eval_episodes: int=5, verbose: int=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.vec_env = vec_env  # VecNormalize environment
        self.best_models = []  # Stores best models based on training reward
        self.saved_checkpoints = []  # Stores recent checkpoints (last N_SAVED_CHECKPOINTS)
        self.episode_rewards = []
        # Create an evaluation callback for saving the best model
        self.eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            n_eval_episodes=n_eval_episodes,
            verbose=verbose
        )

    def _save_model_and_env(self, model, reward=None):
        """Saves the model and VecNormalize environment."""
        # Save the model with reward in the filename if provided
        model_filename = f"model_{self.num_timesteps}.zip" if reward is None else f"best_model_{reward:.2f}_{self.num_timesteps}.zip"
        model_path = os.path.join(self.save_path, model_filename)
        model.save(model_path)

        # Save VecNormalize statistics
        vec_path = os.path.join(self.save_path, "vec_normalize.pkl")
        self.vec_env.save(vec_path)

        if self.verbose > 0:
            print(f"Saved model: {model_path}")
            print(f"Saved VecNormalize stats: {vec_path}")

        return model_path


    def _on_step(self) -> bool:
        """Called every step during training."""
        # Store episode rewards
        reward = self.locals["rewards"][0]  # Assuming single agent
        self.episode_rewards.append(reward)

        # Track episode length
        if "dones" in self.locals and self.locals["dones"][0]:  # Check if episode ended
            self.episode_rewards = []  # Reset for next episode

        if self.num_timesteps % self.save_freq == 0:
            # Reset episode tracking
            self.episode_rewards = []

            # Save checkpoint
            ckpt_path = self._save_model_and_env(self.model)
            self.saved_checkpoints.append(ckpt_path)

            if self.verbose > 0:
                print(f"Saved checkpoint: {ckpt_path}")

            # Remove oldest checkpoint if more than 5 saved
            if len(self.saved_checkpoints) > 5:
                old_ckpt = self.saved_checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                    if self.verbose > 0:
                        print(f"Deleted old checkpoint: {old_ckpt}")

        return True

    def _on_rollout_end(self):
        """Called at the end of each rollout to save the best models based on training rewards."""
        self.eval_callback.on_rollout_end()

        # Compute mean training reward and episode length (extra safety check)
        mean_training_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

        if mean_training_reward is not None:
            # Save best models if reward is among top N_SAVED_CHECKPOINTS
            if len(self.best_models) < N_SAVED_CHECKPOINTS or mean_training_reward > min(self.best_models, key=lambda x: x[0])[0]:
                best_path = self._save_model_and_env(self.model, reward=mean_training_reward)
                self.best_models.append((mean_training_reward, best_path))

                if self.verbose > 0:
                    print(f"Saved new best model: {best_path} (Reward: {mean_training_reward:.2f})")

                # Keep only top N_SAVED_CHECKPOINTS best models
                if len(self.best_models) > N_SAVED_CHECKPOINTS:
                    worst_model = min(self.best_models, key=lambda x: x[0])  # Find worst
                    self.best_models.remove(worst_model)
                    if os.path.exists(worst_model[1]):
                        os.remove(worst_model[1])  # Delete worst model file
                        if self.verbose > 0:
                            print(f"Deleted worst best model: {worst_model[1]} (Reward: {worst_model[0]:.2f})")

    def _on_training_end(self):
        """Called when training ends to properly close TensorBoard logging."""
        self.eval_callback.on_training_end()

        # Final save of VecNormalize stats
        vec_path = os.path.join(self.save_path, "vec_normalize.pkl")
        self.vec_env.save(vec_path)
        if self.verbose > 0:
            print(f"Final VecNormalize stats saved at: {vec_path}")



class LoggerCallback(BaseCallback):
    """
    Custom callback that:
      - Logs rollout metrics (`ep_rew_mean`, `ep_len_mean`) every `save_freq` steps.
      - Logs action frequency distributions to TensorBoard every `save_freq` steps.
    """

    def __init__(self, save_freq: int, verbose: int=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.action_buffer_0 = []
        self.action_buffer_1 = []
        self.episode_rewards = []
        self.episode_lengths = []  # Track episode lengths
        self.crashes = 0
        self.goals_reached = 0
        self.speeds = []
        self.distance_to_goal = []
        self.writer: Optional[SummaryWriter] = None
        self.vecnormalize = True #if isinstance(self.locals['env'], VecNormalize) else False

    def _on_training_start(self):
        """Called at the beginning of training, initializes TensorBoard writer."""
        self.writer = SummaryWriter(self.logger.dir)

    def _on_step(self) -> bool:
        """Called every step during training."""
        actions_0 = self.locals["actions"][0, 0]
        actions_1 = self.locals["actions"][0, 1]
        self.action_buffer_0.append(actions_0)
        self.action_buffer_1.append(actions_1)

        # Store episode rewards
        reward = self.locals["rewards"][0] # Assuming single agent
        if self.vecnormalize:
            # Unnormalize reward if using VecNormalize
            assert isinstance(self.locals['env'], VecNormalize)
            reward = self.locals['env'].get_original_reward()[0] # Assuming single agent
        self.episode_rewards.append(reward)

        # Track episode length
        if "dones" in self.locals and self.locals["dones"][0]:  # Check if episode ended
            self.episode_lengths.append(len(self.episode_rewards))
            self.episode_rewards = []  # Reset for next episode

        infos = self.locals["infos"]
        # Count crashes
        self.crashes += sum(1 for info in infos if info.get("crash", False))
        # Count goal reached
        self.goals_reached += sum(1 for info in infos if info.get("goal_reached", False))
        # Track speeds
        self.speeds.extend(info["target_speed"] for info in infos if "target_speed" in info)
        # Track distance to goal
        self.distance_to_goal.extend(info["distance_to_goal"] for info in infos if "distance_to_goal" in info)

        if self.num_timesteps % self.save_freq == 0:
            assert isinstance(self.writer, SummaryWriter)
            # Compute mean reward and episode length
            mean_training_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            ep_len_mean = np.mean(self.episode_lengths) if self.episode_lengths else 0

            self.writer.add_scalar("rollout/ep_rew_mean", mean_training_reward, self.num_timesteps)
            self.writer.add_scalar("rollout/ep_len_mean", ep_len_mean, self.num_timesteps)
            self.episode_rewards = []
            self.episode_lengths = []

            self.writer.add_scalar("rollout/crash_rate", self.crashes/self.save_freq, self.num_timesteps)
            self.crashes = 0  # Reset crash count

            self.writer.add_scalar("rollout/goals_reached", self.goals_reached, self.num_timesteps)
            self.goals_reached = 0  # Reset goals reached count

            if self.distance_to_goal:
                dist_np = np.array(self.distance_to_goal)
                self.writer.add_scalar("rollout/mean_distance_to_goal", np.mean(dist_np), self.num_timesteps)
                self.writer.add_scalar("rollout/min_distance_to_goal", np.min(dist_np), self.num_timesteps)
                self.distance_to_goal = []

            if self.speeds:
                speeds_np = np.array(self.speeds)
                self.writer.add_scalar("rollout/mean_speed", np.mean(speeds_np), self.num_timesteps)
                self.writer.add_scalar("rollout/max_speed", np.max(speeds_np), self.num_timesteps)
                self.speeds = []

            # Log action distributions
            actions_np_0 = np.array(self.action_buffer_0)
            actions_np_1 = np.array(self.action_buffer_1)
            self.writer.add_histogram("action_distribution/maneuver", actions_np_0, self.num_timesteps)
            self.writer.add_histogram("action_distribution/long_distance", actions_np_1, self.num_timesteps)
            self.action_buffer_0 = []
            self.action_buffer_1 = []

        return True

    def _on_rollout_end(self):
        # Reset episode rewards and lengths
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_end(self):
        """Properly close TensorBoard logging."""
        if self.writer:
            self.writer.close()

