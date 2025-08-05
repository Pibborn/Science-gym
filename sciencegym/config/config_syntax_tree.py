from argparse import ArgumentParser
import numpy as np
class ConfigSyntaxTree:
    @staticmethod
    def arguments_parser(parser=None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for syntax tree parameter")
        parser.add_argument('--max_branching_factor', type=np.float32,
                            default=2,
                            help='How many children a node will maximal have')
        parser.add_argument('--max_constants_in_tree', type=int,
                            default=3,
                            help='Maximum number of constants allowed in  equation'
                                 'afterwards equation will be invalid')
        parser.add_argument('--max_depth_of_tree', type=int,
                                    default=10,
                                    help='Maximum depth of generated equations')
        parser.add_argument('--max_num_nodes_in_syntax_tree', type=int,
                            help='Maximum nodes of generated equations', default=25)
        return parser
