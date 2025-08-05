from argparse import ArgumentParser

from sciencegym.config.config_general import str2bool


class ConfigGPlearn():
    @staticmethod
    def arguments_parser(parser = None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for GPlearn Options")
        parser.add_argument(
            "--niterations",
            type=int,
            default=40,
            help="Number of iterations"
        )

        parser.add_argument(
            "--binary_operators",
            type=str,
            nargs='*',
            default=["*", "-", "+", "/"],
            help="List of binary operators"
        )

        parser.add_argument(
            "--unary_operators",
            type=str,
            nargs='*',
            default=['sqrt', 'sin', 'cos'],
            help="List of unary operators"
        )

        parser.add_argument(
            "--should_simplify",
            type=str2bool,
            default=True,
            help="Whether to simplify the model"
        )




        return parser
