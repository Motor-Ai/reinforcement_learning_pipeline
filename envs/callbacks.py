import os
import numpy as np
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from torch.utils.tensorboard.writer import SummaryWriter
from stable_baselines3.common.vec_env import VecNormalize

class SaveBestAndManageCallback(BaseCallback):
    """
    Custom callback that:
      - Logs rollout metrics (`ep_rew_mean`, `ep_len_mean`) every `save_freq` steps.
      - Saves a checkpoint every `save_freq` timesteps.
      - Keeps only the most recent 5 checkpoint files.
      - Saves the best 5 models based on training rewards.
      - Saves VecNormalize stats alongside the model.
      - Adds reward value to the saved model filename.
      - Logs action frequency distributions to TensorBoard every `save_freq` steps.
    """

    def __init__(self, eval_env, save_freq, save_path, vec_env: VecNormalize, n_eval_episodes: int=5, verbose: int=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.vec_env = vec_env  # VecNormalize environment
        self.best_models = []  # Stores best models based on training reward
        self.saved_checkpoints = []  # Stores recent checkpoints (last 5)
        self.action_buffer_0 = []
        self.action_buffer_1 = []
        self.episode_rewards = []
        self.episode_lengths = []  # Track episode lengths
        self.writer: Optional[SummaryWriter] = None

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
        reward = self.locals["rewards"][0]  # Assuming single agent
        self.episode_rewards.append(reward)

        # Track episode length
        if "dones" in self.locals and self.locals["dones"][0]:  # Check if episode ended
            self.episode_lengths.append(len(self.episode_rewards))
            self.episode_rewards = []  # Reset for next episode

        if self.num_timesteps % self.save_freq == 0:
            # Compute mean reward and episode length
            mean_training_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            ep_len_mean = np.mean(self.episode_lengths) if self.episode_lengths else 0

            # Log rollout metrics
            if self.writer:
                self.writer.add_scalar("rollout/ep_rew_mean", mean_training_reward, self.num_timesteps)
                self.writer.add_scalar("rollout/ep_len_mean", ep_len_mean, self.num_timesteps)

            # Reset episode tracking
            self.episode_rewards = []
            self.episode_lengths = []

            # Log action distributions
            actions_np_0 = np.array(self.action_buffer_0)
            actions_np_1 = np.array(self.action_buffer_1)
            self.writer.add_histogram("action_distribution/maneuver", actions_np_0, self.num_timesteps)
            self.writer.add_histogram("action_distribution/long_distance", actions_np_1, self.num_timesteps)
            self.action_buffer_0 = []
            self.action_buffer_1 = []

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
        ep_len_mean = np.mean(self.episode_lengths) if self.episode_lengths else 0

        # Reset episode rewards and lengths
        self.episode_rewards = []
        self.episode_lengths = []

        if mean_training_reward is not None:
            # Save best models if reward is among top 5
            if len(self.best_models) < 5 or mean_training_reward > min(self.best_models, key=lambda x: x[0])[0]:
                best_path = self._save_model_and_env(self.model, reward=mean_training_reward)
                self.best_models.append((mean_training_reward, best_path))

                if self.verbose > 0:
                    print(f"Saved new best model: {best_path} (Reward: {mean_training_reward:.2f})")

                # Keep only top 5 best models
                if len(self.best_models) > 5:
                    worst_model = min(self.best_models, key=lambda x: x[0])  # Find worst
                    self.best_models.remove(worst_model)
                    if os.path.exists(worst_model[1]):
                        os.remove(worst_model[1])  # Delete worst model file
                        if self.verbose > 0:
                            print(f"Deleted worst best model: {worst_model[1]} (Reward: {worst_model[0]:.2f})")

    def _on_training_end(self):
        """Called when training ends to properly close TensorBoard logging."""
        if self.writer:
            self.writer.close()
        self.eval_callback.on_training_end()

        # Final save of VecNormalize stats
        vec_path = os.path.join(self.save_path, "vec_normalize.pkl")
        self.vec_env.save(vec_path)
        if self.verbose > 0:
            print(f"Final VecNormalize stats saved at: {vec_path}")
