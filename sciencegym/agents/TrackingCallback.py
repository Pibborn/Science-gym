from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np


class TrackingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env, printfreq=100, num_eval_episodes=10, verbose=0, is_test=True):
        super(TrackingCallback, self).__init__(verbose)
        self.printfreq = printfreq
        self.num_eval_episodes = num_eval_episodes
        self.is_test = is_test
        self.env = env
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.num_timesteps % self.printfreq == 0:
            matches = 0
            actions = []
            angles = []
            for episode in range(1, self.num_eval_episodes + 1):
                obs = self.env.reset()
                action, _states = self.model.predict(obs, deterministic=True)
                actions.append(action)
                obs, reward, done, info = self.env.step(action)
                obs = self.env.get_attr('old_state')
                if 'Scale' in self.env.envs[0].name:
                    angle = obs[0][2]
                    angles.append(angle)
                else:
                    angles.append(np.nan)
                if reward >= 1:
                    matches += 1
            actions = np.squeeze(np.array(actions))
            angles = np.mean(angles)
            if self.is_test:
                wandb.log({
                    'test_success_rate': matches/self.num_eval_episodes,
                    'test_actions_mean': np.mean(actions, axis=0),
                    'test_action_var': np.var(actions, axis=0),
                    'test_actions': wandb.Histogram(actions),
                    'test_mean_angle': angles
                   })
            else:
                wandb.log({
                    'train_success_rate': matches/self.num_eval_episodes,
                    'train_actions_mean': np.mean(actions, axis=0),
                    'train_action_var': np.var(actions, axis=0),
                    'train_actions': wandb.Histogram(actions),
                    'train_mean_angle': angles
                })
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass