# MIT License
#
# Copyright (c) 2019 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Command line interfaces to tszip.
"""
import argparse
import logging
import pathlib
import sys
import contextlib

import tskit

import tszip
from . import exceptions

logger = logging.getLogger(__name__)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def exit(message):
    """
    Exit with the specified error message, setting error status.
    """
    sys.exit("{}: {}".format(sys.argv[0], message))


def setup_logging(args):
    log_level = "WARN"
    if args.verbosity > 0:
        log_level = "INFO"
    if args.verbosity > 1:
        log_level = "DEBUG"
    logging.basicConfig(level=log_level, format=log_format)


def tszip_cli_parser():
    parser = argparse.ArgumentParser(
        description="Compress/decompress tskit trees files.")
    parser.add_argument(
        "-V", "--version", action='version',
        version='%(prog)s {}'.format(tszip.__version__))
    parser.add_argument(
        "-v", "--verbosity", action='count', default=0,
        help="Increase the verbosity")
    parser.add_argument(
        "files", nargs="+", help="The files to compress/decompress.")
    parser.add_argument(
        "--variants-only", action='store_true',
        help=(
            "Lossy compression; throws out information not needed to "
            "represent variants"))
    parser.add_argument(
        "-S", "--suffix", default=".tsz",
        help="Use suffix SUFFIX on compressed files")
    parser.add_argument(
        "-k", "--keep", action='store_true',
        help="Keep (don't delete) input files")
    parser.add_argument(
        "-f", "--force", action='store_true',
        help="Force overwrite of output file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-d", "--decompress", action='store_true',
        help="Decompress")
    group.add_argument(
        "-l", "--list", action='store_true',
        help="List contents of the file")
    return parser


def remove_input(infile, args):
    if not args.keep:
        logger.info("Removing {}".format(infile))
        infile.unlink()


def check_output(outfile, args):
    if outfile.exists():
        if not args.force:
            exit("'{}' already exists; use --force to overwrite".format(outfile))


def run_compress(args):
    setup_logging(args)
    for file_arg in args.files:
        logger.info("Compressing {}".format(file_arg))
        try:
            ts = tskit.load(file_arg)
        except tskit.FileFormatError as ffe:
            exit("Error loading '{}': {}".format(file_arg, ffe))
        logger.debug("Loaded tree sequence")
        infile = pathlib.Path(file_arg)
        outfile = pathlib.Path(file_arg + args.suffix)
        check_output(outfile, args)
        tszip.compress(ts, outfile, variants_only=args.variants_only)
        remove_input(infile, args)


@contextlib.contextmanager
def check_load_errors(file_arg):
    try:
        yield
    except OSError as ose:
        exit(str(ose))
    except exceptions.FileFormatError as ffe:
        exit("Error reading '{}': {}".format(file_arg, ffe))


def run_decompress(args):
    setup_logging(args)
    for file_arg in args.files:
        logger.info("Decompressing {}".format(file_arg))
        if not file_arg.endswith(args.suffix):
            exit("Compressed file must have '{}' suffix".format(args.suffix))
        infile = pathlib.Path(file_arg)
        outfile = pathlib.Path(file_arg[:-len(args.suffix)])
        check_output(outfile, args)
        with check_load_errors(file_arg):
            ts = tszip.decompress(file_arg)
        logger.info("Writing to {}".format(outfile))
        ts.dump(str(outfile))
        remove_input(infile, args)


def run_list(args):
    for file_arg in args.files:
        with check_load_errors(file_arg):
            tszip.print_summary(file_arg, args.verbosity)


def main(args):
    if args.decompress:
        run_decompress(args)
    elif args.list:
        run_list(args)
    else:
        run_compress(args)


def tszip_main(arg_list=None):
    parser = tszip_cli_parser()
    args = parser.parse_args(arg_list)
    main(args)


def tsunzip_main(arg_list=None):
    parser = tszip_cli_parser()
    args = parser.parse_args(arg_list)
    args.decompress = True
    main(args)
