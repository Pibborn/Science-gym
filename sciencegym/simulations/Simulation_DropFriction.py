__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import torch.nn as nn
import torch
import joblib
from sklearn.metrics import mean_absolute_error
from definitions import ROOT_DIR
from collections import deque


class Sim_DropFriction(gym.Env):
    cumulated_reward = []
    zero_reward = []
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, args, context):
        self.args = args
        self.in_features = {
            'time': {'low': -2, 'high': 5},
            'tilt_angle': {'low': -2, 'high': 2}
        }
        self.out_features = {
            'drop_length': {'low': -2, 'high': 2.5},
            'adv': {'low': -26, 'high': 4},
            'rec': {'low': -3, 'high': 2},
            'avg_vel': {'low': -2, 'high': 2.5},
            'width': {'low': -7, 'high': 1},
            'y': {'low': 0, 'high': 1}
        }

        self.screen = None
        self.isopen = True
        self.actions_sofar = ActionsSofar()
        self.max_num_experiments = 10
        self.num_experiments = 0
        self.context = context
        self.sum_reward = 0
        self.zero_rewards = 0

        system = 'Teflon-Au-Ethylene Glycol'

        high_in = np.array(
            [feature['high'] for _, feature in self.in_features.items()],
            dtype=np.float32
        )
        low_in = np.array(
            [feature['low'] for _, feature in self.in_features.items()],
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=low_in, high=high_in, dtype=np.float32
        )
        high_out = np.array(
            [feature['high'] for _, feature in self.out_features.items()],
            dtype=np.float32
        )
        low_out = np.array(
            [feature['low'] for _, feature in self.out_features.items()],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low_out, high=high_out, dtype=np.float32)
        # /home/jbrugger/PycharmProjects/box-gym/environments/drop_friction_models/Teflon-Au-Ethylene Glycol
        self.loaded_scaler_X = joblib.load(ROOT_DIR / f'environments/drop_friction_models/{system}/scaler_X.pkl')
        self.loaded_scaler_Y = joblib.load(ROOT_DIR / f'environments/drop_friction_models/{system}/scaler_Y.pkl')
        weights = torch.load(ROOT_DIR / f'environments/drop_friction_models/{system}/model_weights.pth')
        self.model = define_model(len(low_in), len(low_out))
        self.model.load_state_dict(weights)
        self.model.eval()
        self.state = self._get_state(self.np_random.uniform(
            low=[self.out_features[feature]['low'] for feature in self.out_features],
            high=[self.out_features[feature]['high'] for feature in self.out_features]
        )
        )

    def step(self, u):
        #string = f"current state: {self.state}"
        observation = self.model(torch.tensor(u, dtype=torch.float32))
        observation = observation.detach().cpu().numpy()
        reward = self.get_reward(u)
        self.sum_reward += reward
        if reward == -1:
            self.zero_rewards += 1
        #string = f"{string}: action: {u} new observation: {observation}  reward: {reward}"
        # print(string)
        done = self.max_num_experiments <= self.num_experiments
        if done:
            Sim_DropFriction.cumulated_reward.append(self.sum_reward)
            Sim_DropFriction.zero_reward.append(self.zero_rewards)
        info = {}
        self.state = self._get_state(observation)
        self.num_experiments +=1
        return self.state, reward, done, info

    def _get_state(self, observation):
        return np.array(observation, dtype=np.float32)

    def get_reward(self, action):
        time = action[0]
        tilt_angle = action[1]
        if time < -1.7:
            reward =  -1
        elif time > 1.7:
            reward = -1
        elif time > np.exp(-tilt_angle):
            reward = -1
        else:
            distance = [mean_absolute_error(action, past_action) for
                        past_action in self.actions_sofar.memory]
            self.actions_sofar.memory.append(action)
            if len(distance) > 0:
                reward = max(0,min(min(distance) , 1))
            else:
                reward = 1
        if self.context == 'classic':
            return reward
        if self.context == 'noise':
            return reward + np.random.normal(self.args.noise_loc, self.args.noise_scale)
        if self.context == 'sparse':
            return reward >= self.args.sparse_thr
        else:
            raise NotImplementedError('Currently, context {} is not implemented\
                                       in DropFriction'.format(self.context))



    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None
    ):
        super().reset()
        self.actions_sofar = ActionsSofar()
        self.num_experiments = 0
        self.state = self._get_state(self.np_random.uniform(
            low=[self.out_features[feature]['low'] for feature in self.out_features],
            high=[self.out_features[feature]['high'] for feature in self.out_features]
        )
        )
        self.sum_reward = 0
        self.zero_rewards = 0
        return self.state

    def render(self, mode="human"):
        pass

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def get_current_state(self):
        return self.state
    def get_state_space(self):
        return self.observation_space
    def get_action_space(self):
        return self.action_space


def define_model(len_input, len_output):
    model = nn.Sequential(
        nn.Linear(len_input, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 12),
        nn.ReLU(),
        nn.Linear(12, len_output),
    )
    return model

class ActionsSofar:
    def __init__(self):
        self.memory_length= 50
        self.memory = deque(maxlen=self.memory_length)