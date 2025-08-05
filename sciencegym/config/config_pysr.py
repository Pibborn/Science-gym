from argparse import ArgumentParser

from sciencegym.config.config_general import str2bool


class ConfigPySR():
    @staticmethod
    def arguments_parser(parser=None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for PYSR Options")
        parser.add_argument(
            "--model_selection",
            type=str,
            default="best",
            help="Model selection criterion"
        )

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
            "--progress",
            type=str2bool,
            default=True,
            help="Whether to show progress"
        )

        parser.add_argument(
            "--should_simplify",
            type=str2bool,
            default=True,
            help="Whether to simplify the model"
        )

        parser.add_argument(
            "--deterministic",
            type=str2bool,
            default=True,
            help="Whether to use deterministic mode"
        )

        parser.add_argument(
            "--parallelism",
            type=str,
            default='serial',
            help="Parallelism mode"
        )

        parser.add_argument(
            "--maxsize",
            type=int,
            default=10,
            help="Maximum size"
        )

        parser.add_argument(
            "--parsimony",
            type=float,
            default=0.3,
            help="Multiplicative factor for how much to punish complexity."
        )

        parser.add_argument(
            "--complexity_of_constants",
            type=int,
            default=3,
            help="Complexity of constants"
        )

        parser.add_argument(
            "--weight_optimize",
            type=float,
            default=0.001,
            help="Weight for optimization"
        )
        return parser
