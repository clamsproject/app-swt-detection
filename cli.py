"""cli.py

Command Line Interface for the SWT app.

This script can be called with the same arguments as the _annotate method on the app,
except that we add --input, --output and --metadata parameters.

Example invocation:

$ python cli.py \
    --modelName 20240409-093229.convnext_tiny
    --input example-mmif-local.json
    --output out.json
    --map B:bars S:slate
    --pretty true

"""


import sys
import yaml
import argparse

from mmif import Mmif
from clams.app import ClamsApp

from metadata import appmetadata
from app import SwtDetection


json_type_map = {
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool,
}

parameter_names = (
    'metadata', 'map', 'minFrameCount', 'minFrameScore', 'minTimeframeScore',
    'modelName', 'pretty', 'sampleRate', 'startAt', 'stopAt', 'useStitcher')


def get_metadata():
    """Gets the metadata from the metadata.py file, with the universal parameters added."""
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


def print_args(args):
    print(args)
    print()
    for arg in vars(args):
        value = getattr(args, arg)
        print(f'{arg:18s}  {str(type(value)):15s}  {value}')


def build_app_parameters(args):
    parameters = {}
    for arg in vars(args):
        if arg in ('input', 'output', 'metadata'):
            continue
        value = getattr(args, arg)
        parameters[arg] = value
    return parameters


def adjust_parameters(parameters, args):
    # Adding the empty directory makes the app code work, but it still won't be able
    # to print the parameters as given by the user on the command line. So we loop
    # over the arguments to populate the raw parameters dictionary.
    parameters[ClamsApp._RAW_PARAMS_KEY] = {}
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            argname = arg[2:]
            argval = vars(args)[argname]
            argval = argval if type(argval) is list else [str(argval)]
            parameters[ClamsApp._RAW_PARAMS_KEY][argname] = argval



if __name__ == '__main__':

    app = SwtDetection()
    metadata = get_metadata()

    argparser = create_argparser(metadata)
    args = argparser.parse_args()

    if args.metadata:
        print(metadata.jsonify(pretty=args.pretty))
    else:
        parameters = build_app_parameters(args)
        # Simply calling _annotate() breaks when we try to create the view and copy the
        # parameters into it because the CLAMS code expects there to be raw parameters.
        # So we first adjust the parameters to match what the CLAMS code expects.
        adjust_parameters(parameters, args)
        mmif = Mmif(open(args.input).read())
        app._annotate(mmif, **parameters)
        with open(args.output, 'w') as fh:
            fh.write(mmif.serialize(pretty=args.pretty))
