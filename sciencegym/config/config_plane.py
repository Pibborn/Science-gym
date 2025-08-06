from argparse import ArgumentParser
import argparse
from definitions import ROOT_DIR
from sciencegym.config.config_general import str2bool


class ConfigPlane():
    @staticmethod
    def arguments_parser(parser = None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for tilted plane environment")
        parser.add_argument("--input_cols", type=str, nargs='+',
                            default=["mass", "gravity", "angle"],
                            help="Features for symbolic regression")
        parser.add_argument("--output_col", type=str, default='force',
                            help="Target feature for symbolic regression")
        parser.add_argument("--downsample", type=str2bool, default='False',
                            help="Downsample table for symbolic regression")
        parser.add_argument("--every_n", type=int, default=1,
                            help="Keep every n row in table")
        parser.add_argument("--context", type=str, default='classic',
                            choices=['classic', 'noise', 'sparse'],
                            help="Select if context for experiment. ")
        parser.add_argument("--success_thr", type=float, default= - 0.1,
                            help="Success threshold for experiment. ")
        parser.add_argument("--rendering", type=str2bool, default=False,
                            help="If environment should be rendered.")
        parser.add_argument('--t_end_test', type=int, default=200,
                            help='time steps after which  the test episode is terminated')
        parser.add_argument('--noise_scale', type=float, default=0.5, help='Scale of noise')
        parser.add_argument('--noise_loc', type=float, default=0, help='Mean of noise')
        parser.add_argument('--sparse_thr', type=float, default=0.1,
                            help='Which reward must be archived to get a positive reward')
        parser.add_argument('--use_analytical_simulation', type=str2bool, default=True,
                            help='Which reward must be archived to get a positive reward')




        return parser