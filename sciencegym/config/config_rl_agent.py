from argparse import ArgumentParser
import argparse
from definitions import ROOT_DIR
from sciencegym.config.config_general import str2bool


class ConfigRLAgent():
    @staticmethod
    def arguments_parser(parser = None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for reinforcement agent")

        parser.add_argument("--rl_agent", type=str, default="SAC",
                            choices=["SAC", "A2C"],
                            help="Algorithm for the reinforcement learning")

        parser.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate")

        parser.add_argument("--policy", type=str, default="MlpPolicy",
                            help="Policy of RL agent")

        parser.add_argument("--verbose", type=str2bool, default=True,
                            help="Creation of model verbose?")

        parser.add_argument("--rl_train_steps", type=int, default=1000,
                            help="How long to train the model")

        parser.add_argument("--rl_test_episodes", type=int, default=100,
                            help="How many episodes are played to genreate data"
                                 " for the symbolic regression")

        return parser