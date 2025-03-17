import sys
sys.path.append("/Users/lennartbaur/Documents/Arbeit/ScienceGym/Repo/science-gym")

from stable_baselines3 import SAC, HerReplayBuffer

from .StableBaselinesAgent import StableBaselinesAgent

class SACAgent(StableBaselinesAgent):
    def __init__(self, input_dim, output_dim, lr=1e-4, policy='MlpPolicy'):
        super().__init__(input_dim, output_dim, policy)
        self.agent = None
        self.name = 'sac'
        self.lr = lr

    def create_model(self, train_env, verbose=1, use_sde=False, lr=1e-4):
        """self.agent = stable_bascleelines3.sac.SAC("MlpPolicy", train_env, learning_rate=0.0003, buffer_size=1000000, learning_starts=100,
                                                     batch_size=256, tau=0.005, gamma=DISCOUNT_FACTOR, train_freq=1, gradient_steps=1,
                                                     action_noise=None, replay_buffer_class=self.replay_buffer, replay_buffer_kwargs=None,
                                                     optimize_memory_usage=False, ent_coef='auto', target_update_interval=1,
                                                     target_entropy='auto', use_sde=False, sde_sample_freq=- 1, use_sde_at_warmup=False,
                                                     tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0,
                                                     seed=None, device='auto', _init_setup_model=True)"""

        # Available strategies (cf paper): future, final, episode
        goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

        """# If True the HER transitions will get sampled online
        online_sampling = True
        # Time limit for the episodes
        max_episode_length = 120

        # Initialize the model
        self.agent = SAC(
            "MultiInputPolicy",
            train_env,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=online_sampling,
                max_episode_length=max_episode_length,
            ),
            verbose=1,
        )"""

        self.agent = SAC(policy=self.policy, env=train_env, verbose=verbose, use_sde=use_sde,
                         tensorboard_log='results/temp', learning_rate=self.lr,
                         # play with the following parameters if you use pictures as image
                         buffer_size=800 if self.policy == 'CnnPolicy' else 1000000,
                         #batch_size=128 if self.policy == 'CnnPolicy' else 256,
                         )
        return self.agent

    def save_agent(self, location):
        self.agent.save(location)
        return

    def load_agent(self, location):
        self.agent = SAC.load(location)
        return self.agent



if __name__ == '__main__':
    agent = SACAgent(10, 10)
