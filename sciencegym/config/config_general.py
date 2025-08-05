from argparse import ArgumentParser
import argparse
from definitions import ROOT_DIR

class ConfigGeneral():
    @staticmethod
    def arguments_parser(parser = None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for General setting")
        parser.add_argument("--exp_name", type=str, default="test",
                            help="Identifier of the experiment")
        parser.add_argument('--simulation', type=str, default="basketball",
                            choices=['basketball', "sirv", "lagrange", "plane", "drop_friction"])
        parser.add_argument("--seed", type=int,
                            default=1,
                            help='Seed for experiment')
        parser.add_argument("--logging_level", type=int, default=30,
                            help="CRITICAL = 50, ERROR = 40, "
                                 "WARNING = 30, INFO = 20, "
                                 "DEBUG = 10, NOTSET = 0")
        parser.add_argument("--root_dir", type=str, default=ROOT_DIR,
                            help="Path to project")
        parser.add_argument("--result_dir", type=str, default='results',
                            help="Where to load the datasets")
        parser.add_argument('--regress_only',
                            help='Run regression only', type=str2bool, default=False)
        parser.add_argument('--path_to_regression_table',
                            help='Path from the repo root to the data used in the regression',
                                 default='', type=str)
        parser.add_argument('--equation_discoverer', type=str, default='pysr',
                            help="Which Equation discoverer to use",
                            choices=['pysr', 'gplearn'])

        return parser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
