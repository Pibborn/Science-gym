from sciencegym.config.config_basketball import ConfigBasketball
from sciencegym.config.config_general import ConfigGeneral
from sciencegym.config.config_gplearn import ConfigGPlearn
from sciencegym.config.config_pysr import ConfigPySR
from sciencegym.config.config_rl_agent import ConfigRLAgent
from sciencegym.config.config_sirv import ConfigSIRV
from sciencegym.config.config_syntax_tree import ConfigSyntaxTree


def parse_arguments():
    __pre_parser = ConfigGeneral.arguments_parser()
    pre_args, _ = __pre_parser.parse_known_args()
    parser = ConfigGeneral.arguments_parser()
    parser = ConfigRLAgent.arguments_parser(parser)
    parser = ConfigSyntaxTree.arguments_parser(parser)
    if pre_args.equation_discoverer == 'pysr':
        parser = ConfigPySR.arguments_parser(parser)
    elif pre_args.equation_discoverer == 'gplearn':
        parser = ConfigGPlearn.arguments_parser(parser)
    else:
        raise Exception(f'Unknown equation discoverer type: {pre_args.equation_discoverer}')
    if pre_args.simulation == 'basketball':
        parser = ConfigBasketball.arguments_parser(parser)
    elif pre_args.simulation == 'sirv':
        parser = ConfigSIRV.arguments_parser(parser)
    else:
        raise Exception(f'Unknown simulation type: {pre_args.simulation}')
    args = parser.parse_args()
    return args
