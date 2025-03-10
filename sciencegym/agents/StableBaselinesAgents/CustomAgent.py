import gym

from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines3 import A2C, SAC, HerReplayBuffer
from agents.StableBaselinesAgents.StableBaselinesAgent import StableBaselinesAgent

# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])],
                                           feature_extraction="mlp")

class CustomAgent(StableBaselinesAgent):
    def __init__(self, input_dim, output_dim, lr=1e-4, policy='MlpPolicy'):
        super().__init__(input_dim, output_dim, policy)
        self.agent = None
        self.name = 'sac'
        self.lr = lr

    def create_model(self, train_env, verbose=1, use_sde=False, lr=1e-4):
        """self.agent = stable_baselines3.sac.SAC("MlpPolicy", train_env, learning_rate=0.0003, buffer_size=1000000, learning_starts=100,
                                                     batch_size=256, tau=0.005, gamma=DISCOUNT_FACTOR, train_freq=1, gradient_steps=1,
                                                     action_noise=None, replay_buffer_class=self.replay_buffer, replay_buffer_kwargs=None,
                                                     optimize_memory_usage=False, ent_coef='auto', target_update_interval=1,
                                                     target_entropy='auto', use_sde=False, sde_sample_freq=- 1, use_sde_at_warmup=False,
                                                     tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0,
                                                     seed=None, device='auto', _init_setup_model=True)"""
        self.agent = SAC(policy=CustomPolicy, env=train_env, verbose=verbose, use_sde=use_sde,
                         tensorboard_log='results/temp', learning_rate=self.lr,
                         )
        return self.agent

    def save_agent(self, location):
        self.agent.save(location)
        return

    def load_agent(self, location):
        self.agent = SAC.load(location)
        return self.agent