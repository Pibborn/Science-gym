from abc import ABC, abstractmethod

import gym
import numpy as np
import torch
from gym import spaces
import wandb


class Agent(ABC, torch.nn.Module):
    """Class for an agent which tries to train and test a special method/learning algorithm and returns the results."""

    @abstractmethod
    def __init__(self, input_dim: int, output_dim: int):
        """Create an agent object.

                :type input_dim: int
                :param input_dim: dimension of the input (number of actions per episode)
                :type output_dim: int
                :param output_dim: dimension of the output (number of observations in the observation space)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @classmethod
    def convert_observation_space(self, obs_space: spaces.Dict, order: list[str] = []) -> spaces.Box:
        """Convert an existing scale environment so that the observation space is a Box instead of a Dict.

                :type obs_space: gym.spaces.Dict
                :param obs_space: the given observation space in the old format
                :type order: list[str]
                :param order: give a list of the descriptions as an order for the new observation space
                :returns: the new observation space as a Box
                :rtype: gym.spaces.Box
        """
        # not the best solution
        if not order:
            low = [obs_space[x].low[0] for x in obs_space]
            high = [obs_space[x].high[0] for x in obs_space]
            # shape = obs_space[list(obs_space)[0]].shape[2]
        else:
            low = [obs_space[entry].low[0] for entry in order]
            high = [obs_space[entry].high[0] for entry in order]
        observation_space = spaces.Box(low=np.array(low), high=np.array(high),
                                       # shape=(shape,), # todo: fix shape
                                       dtype=np.float32)
        return observation_space

    @abstractmethod
    def train_episode(self, env, verbose=1):
        """Train one single episode. Can be used in the train_loop function.

                :type env: gym.Env
                :param env: the train environment
                :type verbose: int
                :param verbose: detailed information about the training?
                :returns: loss, episode reward
                :rtype: tuple(float, float)
        """
        raise NotImplementedError("Train Function not implemented")

    @classmethod
    def calculate_returns(self, rewards, discount_factor, normalize=True):
        """Calculate the returns regarding the discount_factor within an episode.

                :type rewards: list[float]
                :param rewards: rewards within one episode
                :type discount_factor: float
                :param discount_factor: value between 0 and 1.0
                :type normalize: bool
                :param normalize: normalize the calculation
                :rtype: list[float]
                :returns: the calculated (and normalized) rewards
        """
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if normalize:
            if returns.size()[0] > 1:
                returns = (returns - returns.mean()) / returns.std()
            else:
                pass
        return returns

    @classmethod
    def update_policy(self, returns, log_prob_actions, optimizer):
        """Update the policy (?)

                :type returns: list[float]
                :param returns: calculated rewards
                :type log_prob_actions: list(float)
                :param log_prob_actions:
                :type optimizer: optim.Adam
                :param optimizer:
                :rtype: float
                :returns: loss
        """
        returns = returns.detach()
        if not log_prob_actions:  # todo: fix (for QAgent)
            return 0
        loss = - (returns * log_prob_actions).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @abstractmethod
    def evaluate(self, env):
        """Evaluate one test episode. Can be used while testing.

                        :type env: gym.Env:
                        :param env: test environment
                        :rtype: tuple[float, list]
                        :returns: reward from episode and the state
                """
        raise NotImplementedError("Evaluate Function not implemented")

    @abstractmethod
    def train_loop(self, train_env, test_env, config, verbose=1, sde=False, only_testing=False):
        """Loop for the training and testing part. First train the model, then test it.

                :type train_env: gym.Env
                :param train_env: train environment
                :type test_env: gym.Env
                :param test_env: test environment
                :type config: Namespace
                :param config: configurations, set in the arguments
                :type verbose: int
                :param verbose: detailed information about testing?
                :type sde: bool
                :param sde: use sde (standard dependent exploration)?
                :type only_testing: bool
                :param only_testing: only test it?
                :rtype: tuple[list[float], list[float]]
                :returns: train and test rewards for plotting"""
        raise NotImplementedError("Train Function not implemented")

    @abstractmethod
    def save_agent(self, location):
        """Saves an agent to a given location

        :type location: string
        :param location: path of the location where the agent should be saved"""
        raise NotImplementedError("Save Agent Function not implemented")

    @abstractmethod
    def load_agent(self, location):
        """Saves an agent to a given location

        :type location: string
        :param location: path of the location from where the agent should be loaded
        :rtype: Agent
        :returns: Agent object"""
        raise NotImplementedError("Save Agent Function not implemented")
