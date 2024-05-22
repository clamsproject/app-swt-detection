"""cli.py

Command Line Interface for the SWT app.

This script can be called with the same arguments as the annotate method on the app,
except that we add --input, --output and --metadata parameters.

Example invocations:

$ python cli.py --metadata

$ python cli.py \
    --input example-mmif.json --output out.json \
    --modelName 20240409-093229.convnext_tiny \
    --map B:bars S:slate --pretty True

Instead of using --input and --output you can also use pipes:

$ cat example-mmif.json | python cli.py \
    --modelName 20240409-093229.convnext_tiny \
    --map B:bars S:slate --pretty True > out.json

The core of the code is a mapping from app parameters to ArgumentParser parameters.

The most funky aspect of this is that boolean parameters are mapped to string-valued
parameters with 'True' and 'False' as their only possible values.

"""


import sys
import yaml
import argparse

from mmif import Mmif
from clams.app import ClamsApp

from metadata import appmetadata
from app import SwtDetection



# Mapping from app metadata types to ArgumentParser types
type_map = {
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool,
}

# Mapping from metadata types to ArgumentParser metavars, relevant only for its
# use when you call the script with the -h option.
metavar_map = {
    "integer": 'INT',
    "number": 'FLOAT',
    "string": 'STRING',
    "boolean": 'BOOL',
    "map": 'PRE_LABEL:POST_LABEL'
}

# Get the arguments handed in by the user, assumes all parameters start with
# a double dash.
user_arguments = [p[2:] for p in sys.argv[1:] if p.startswith('--') and len(p) > 2]


def get_metadata():
    """Gets the metadata from the metadata.py file, with the universal parameters
    added."""
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    return metadata


def create_argparser(parameters: dict):
    """Create an ArgumentParser from the parameters taken from the app metadata."""
    parser = argparse.ArgumentParser(
        description=f"Command-Line Interface for {metadata.identifier}")
    parser.add_argument(
        "--metadata",
        help="Return the app's metadata and exit",
        action="store_true")
    parser.add_argument("--input", help="The input MMIF file")
    parser.add_argument("--output", help="The output MMIF file")
    for parameter in metadata.parameters:
        p_name = f'--{parameter.name}'
        p_type = type_map.get(parameter.type)
        p_metavar = metavar_map.get(parameter.type)
        default = parameter.default
        choices = parameter.choices
        if parameter.type == 'map':
            # Nothing special, but it adds the nargs='*' argument
            parser.add_argument(
                p_name, help=parameter.description, metavar=p_metavar,
                type=p_type, nargs='*', default=default, choices=choices)
        elif parameter.type == 'boolean':
            # We turn this boolean into a string, somewhat ugly but it was
            # the cleanest way I could find to simplify the CLI for user and
            # developer alike (MV)
            parser.add_argument(
                p_name, help=parameter.description, metavar=p_metavar,
                type=str, default=default, choices=['True', 'False'])
        else:
            parser.add_argument(
                p_name, help=parameter.description, metavar=p_metavar,
                type=p_type, default=default, choices=choices)
    return parser


def build_app_parameters(args) -> dict:
    """Return the parameters to be handed to the server-less app. Only include
    the parameters handed in by the user, but exclude those that are irrelevant
    for the app.""" 
    parameters = {}
    for arg in vars(args):
        if arg in ('input', 'output', 'metadata'):
            continue
        if arg in user_arguments:
            value = getattr(args, arg)
            value = value if type(value) is list else [str(value)]
            parameters[arg] = value
    return parameters


def print_metadata_parameters(metadata):
    """Debugging utility method."""
    print('\n>>> Metadata parameters')
    for parameter in metadata.parameters:
        print(f'--- {parameter.name:20} <{parameter.type}>',
              f'default={parameter.default} choices={parameter.choices}')


def print_parsed_parameters(parameters):
    """Debugging utility method."""
    print('\n>>> Parsed parameters')
    for param in parameters:
        if param == ClamsApp._RAW_PARAMS_KEY:
            for prop in parameters[param]:
                print(f'--- {param}  {prop:13s} -->  {repr(parameters[param][prop])}')
        else:
            print(f'--- {param:20s} -->  {repr(parameters[param])}')


def print_args(args):
    """Debugging utility method."""
    print('\n>>> Argument Namespace')
    for arg in vars(args):
        print(f'--- {arg:20s} -->  {repr(getattr(args, arg))}')



if __name__ == '__main__':

    metadata = get_metadata()
    argparser = create_argparser(metadata)
    args = argparser.parse_args()

    if args.metadata:
        print(metadata.jsonify(pretty=args.pretty))
    else:
        parameters = build_app_parameters(args)
        if args.input is None:
            mmif = Mmif(sys.stdin.read())
        else:
            mmif = Mmif(open(args.input).read())
        out_mmif = SwtDetection().annotate(mmif, **parameters)
        if args.output is None:
            sys.stdout.write(out_mmif)
        else:
            with open(args.output, 'w') as fh:
                fh.write(out_mmif)
