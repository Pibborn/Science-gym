from stable_baselines3 import HER, SAC, HerReplayBuffer

from agents.StableBaselinesAgents.StableBaselinesAgent import StableBaselinesAgent


class HERAgent(StableBaselinesAgent):
    def __init__(self, input_dim, output_dim, lr=1e-4, policy='MlpPolicy'):
        super().__init__(input_dim, output_dim, policy)
        self.agent = None
        self.name = 'her'
        self.lr = lr

    def create_model(self, train_env, verbose=1, use_sde=False, lr=1e-4):
        # Available strategies (cf paper): future, final, episode
        goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

        # If True the HER transitions will get sampled online
        online_sampling = True
        # Time limit for the episodes
        max_episode_length = 120

        # Initialize the model
        self.agent = HER(
            policy=self.policy,
            env=train_env,
            verbose=verbose,
            use_sde=use_sde,
            model_class=SAC,
            n_sampled_goal=4,
            goal_selection_strategy='future',
            learning_starts=0,
            # max_episode_length=max_episode_length,
            # online_sampling=online_sampling,
        )

        """self.agent = SAC(policy=self.policy, env=train_env, verbose=verbose, use_sde=use_sde,
                         tensorboard_log='results/temp', learning_rate=self.lr,
                         # play with the following parameters if you use pictures as image
                         # buffer_size=800 if self.policy == 'CnnPolicy' else 1000000,
                         # batch_size=128 if self.policy == 'CnnPolicy' else 256,
                         )"""
        return self.agent

    def save_agent(self, location):
        self.agent.save(location)
        return

    def load_agent(self, location):
        self.agent = SAC.load(location)
        return self.agent
