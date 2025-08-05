from argparse import ArgumentParser
import argparse
from definitions import ROOT_DIR
from sciencegym.config.config_general import str2bool


class ConfigBasketball():
    @staticmethod
    def arguments_parser(parser = None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for General setting")
        parser.add_argument("--input_cols", type=str, nargs='+',
                            default=['velocity_sin_angle', 'time', "g"],
                            help="Features for symbolic regression")
        parser.add_argument("--output_col", type=str, default='ball_y',
                            help="Target feature for symbolic regression")
        parser.add_argument("--downsample", type=str2bool, default='True',
                            help="Downsample table for symbolic regression")
        parser.add_argument("--every_n", type=int, default=10,
                            help="Keep every n row in table")
        parser.add_argument("--context", type=str, default='classic',
                            choices=['classic', 'noise', 'sparse'],
                            help="Select if context for experiment. ")
        parser.add_argument("--success_thr", type=float, default=90,
                            help="Success threshold for experiment. ")
        parser.add_argument("--rendering", type=str2bool, default=False,
                            help="If environment should be rendered.")
        parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize the data')
        parser.add_argument('--raw_pixels', type=str2bool, default=False, help='Use raw pixels')
        parser.add_argument('--random_ball_size', type=str2bool, default=True, help='Randomize ball size')
        parser.add_argument('--random_density', type=str2bool, default=False, help='Randomize density')
        parser.add_argument('--random_basket', type=str2bool, default=False, help='Randomize basket')
        parser.add_argument('--random_ball_position', type=str2bool, default=True, help='Randomize ball position')
        parser.add_argument('--walls', type=int, default=0, help='Number of walls')
        parser.add_argument('--t_end_test', type=int, default=200, help='Number of walls')
        parser.add_argument('--noise_scale', type=float, default=10, help='Scale of noise')
        parser.add_argument('--noise_loc', type=float, default=1, help='Mean of noise')
        parser.add_argument('--sparse_thr', type=float, default=99,
                            help='Which reward must be archived to get a positive reward')




        return parser