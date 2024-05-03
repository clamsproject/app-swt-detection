import yaml
import pprint
import argparse
from pathlib import Path

from metadata import appmetadata
from clams.app import ClamsApp
from app import SwtDetection



default_config_fname = Path(__file__).parent / 'modeling/config/classifier.yml'

json_type_map = {
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool,
}

parameter_names = (
    'metadata', 'map', 'minFrameCount', 'minFrameScore', 'minTimeframeScore',
    'modelName', 'pretty', 'sampleRate', 'startAt', 'stopAt', 'useStitcher')


def get_app():
    app = SwtDetection(preconf_fname=default_config_fname, log_to_file=False)
    return app


def get_metadata():
    """Gets the metadata from the metadata.py  filem, with the universal parameters added."""
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    return metadata


def create_argparser(metadata):
    parser = argparse.ArgumentParser(
        description=f"Command-Line Interface for {metadata.identifier}")
    parser.add_argument(
        "--metadata",
        help="Return the apps metadata and exit",
        action="store_true")
    parser.add_argument("--input", help="The input MMIF file")
    parser.add_argument("--output", help="The output MMIF file")
    for parameter in metadata.parameters:
        nargs = '*' if parameter.type == 'map' else '?'
        parser.add_argument(
            f"--{parameter.name}",
            help=parameter.description,
            nargs=nargs,
            type=json_type_map.get(parameter.type),
            choices=parameter.choices,
            default=parameter.default,
            action="store")
    return parser


def print_parameters(metadata):
    for parameter in metadata.parameters:
        continue
        print(f'\n{parameter.name}')
        print(f'   type={parameter.type}')
        print(f'   default={parameter.default}')
        print(f'   choices={parameter.choices}')


if __name__ == '__main__':

    app = get_app()
    metadata = get_metadata()

    argparser = create_argparser(metadata)
    args = argparser.parse_args()
    
    print(args)
    print()
    for arg in vars(args):
        value = getattr(args, arg)
        print(f'{arg:18s}  {str(type(value)):15s}  {value}')

    if args.metadata:
        print(metadata.jsonify(pretty=args.pretty))

