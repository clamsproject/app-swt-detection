#!/usr/bin/env python3

# Imports
import argparse

from clams import ClamsApp
import logging
import os

from mmif import Mmif
from typing import Union

from app import SwtDetection
import metadata


# Dict to convert json object types to python types
json_type_map = {
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool,
}


# ============================================================|
def _process_file(
    fname: Union[str, os.PathLike, Mmif], app: ClamsApp, params: dict
) -> Mmif:
    mmif_obj = fname if isinstance(fname, Mmif) else Mmif(fname)
    return app._annotate(mmif_obj, **params)


def _process_directory(dirname: Union[str, os.PathLike], app: ClamsApp, params: dict):
    for fname in os.listdir(dirname):
        yield _process_file(fname, app)


def parse_metadata() -> dict[str, Union[str, float, int, bool]]:
    """Separate argparser for runtime parameters,

    uses parameters accessible via `metadata.py` to
    construct the argparse arguments, parses them,
    and converts them to a dictionary of values before returning
    """

    # helper function for argparse 'nargs' field
    def _get_nargs(param):
        if param.multivalued:
            return "?" if param.type == "boolean" else "+"
        return 1

    # Construct & Parse Runtime Arguments
    meta_parser = argparse.ArgumentParser(
        description="Command-Line Interface for CLAMS app"
    )
    app_meta = dict(metadata.appmetadata())
    for parameter in app_meta["parameters"]:
        meta_parser.add_argument(
            f"--{parameter.name}",
            help=parameter.description,
            nargs=_get_nargs(parameter),
            type=json_type_map[parameter.type],
            choices=parameter.choices,
            default=parameter.default,
            action="store",
        )
    return dict(vars(meta_parser.parse_args()))


def main(input_args: argparse.Namespace):
    runtime_parameters = parse_metadata()

    app = SwtDetection(input_args.preconf, log_to_file=input_args.log)
    if input_args.directory:
        _process_directory(input_args.input_path, app, runtime_parameters)
    else:
        _process_file(input_args.input_path, runtime_parameters)


if __name__ == "__main__":
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument(
        "-d",
        "--directory",
        action="store_true",
        help="flag for passing a directory of mmifs. If false, assumes input is a file",
    )
    input_parser.add_argument(
        "-i",
        "--input_path",
        help="path to input file or directory",
        required=True,
    )
    input_parser.add_argument("--preconf", help="pre-configuration file for SWT App")
    input_parser.add_argument(
        "--log", help="Whether or not to log to a file.", action="store_true"
    )
    main(input_parser.parse_args())
