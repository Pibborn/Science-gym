from argparse import ArgumentParser

from sciencegym.config.config_general import str2bool


class ConfigGPlearn():
    @staticmethod
    def arguments_parser(parser = None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for GPlearn Options")
        return parser