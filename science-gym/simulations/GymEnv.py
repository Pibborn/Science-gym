import gym


class GymEnv():
    def __init__(self, env_str):
        self.env_str = env_str

    def create(self):
        return gym.make(self.env_str)
