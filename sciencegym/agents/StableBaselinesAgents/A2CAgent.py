from stable_baselines3 import A2C

from agents.StableBaselinesAgents.StableBaselinesAgent import StableBaselinesAgent


class A2CAgent(StableBaselinesAgent):
    def __init__(self, input_dim, output_dim, lr=1e-4, policy='MlpPolicy'):
        super().__init__(input_dim, output_dim, policy=policy)
        self.agent = None
        self.name = 'a2c'
        self.lr = lr

    def create_model(self, train_env, verbose=1, use_sde=False, lr=1e-4):
        """self.agent = stable_baselines3.a2c.A2C(policy, env, learning_rate=0.0007, n_steps=5, gamma=0.99,
                gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, rms_prop_eps=1e-05, use_rms_prop=True,
                use_sde=False, sde_sample_freq=- 1, normalize_advantage=False, tensorboard_log=None, create_eval_env=False,
                policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True) """
        self.agent = A2C(policy=self.policy, env=train_env, verbose=verbose, use_sde=use_sde,
                         tensorboard_log='results/temp', learning_rate=self.lr)
        return self.agent

    def save_agent(self, location):
        self.agent.save(location)
        return

    def load_agent(self, location):
        self.agent = A2C.load(location)
        return self.agent

if __name__ == '__main__':
    agent = A2CAgent(10, 10)
