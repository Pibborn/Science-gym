from stable_baselines3 import PPO, HerReplayBuffer

from agents.StableBaselinesAgents.StableBaselinesAgent import StableBaselinesAgent

class PPOAgent(StableBaselinesAgent):
    def __init__(self, input_dim, output_dim, lr=1e-4, policy='CnnPolicy'):
        super().__init__(input_dim, output_dim, policy)
        self.agent = None
        self.name = 'ppo'
        self.lr = lr

    def create_model(self, train_env, verbose=1, use_sde=False, lr=1e-4):
        self.agent = PPO(policy=self.policy, env=train_env, verbose=verbose, use_sde=use_sde,
                         tensorboard_log='results/temp', learning_rate=self.lr,
                         # play with the following parameters if you use pictures as image
                         #buffer_size=800 if self.policy == 'CnnPolicy' else 1000000,
                         #batch_size=128 if self.policy == 'CnnPolicy' else 256,
                         )
        return self.agent

    def save_agent(self, location):
        self.agent.save(location)
        return

    def load_agent(self, location):
        self.agent = PPO.load(location)
        return self.agent
