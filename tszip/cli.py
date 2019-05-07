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

import daiquiri
import tskit

import tszip

logger = logging.getLogger(__name__)


def setup_logging(args):
    log_level = "WARN"
    if args.verbosity > 0:
        log_level = "INFO"
    if args.verbosity > 1:
        log_level = "DEBUG"
    daiquiri.setup(log_level)


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
        "file", help="The file to operate on")
    parser.add_argument(
        "--variants-only", action='store_true',
        help=(
            "Lossy compression; throws out information not needed to "
            "represent variants"))
    parser.add_argument(
        "-d", "--decompress", action='store_true',
        help="Decompress")
    parser.add_argument(
        "-l", "--list", action='store_true',
        help="List contents of the file")
    return parser


def run_compress(args):
    logger.info("Compressing {}".format(args.file))
    ts = tskit.load(args.file)
    outfile = args.file + ".zarr"
    tszip.compress(ts, outfile, variants_only=args.variants_only)
    # TODO various gzip-like semantics with file


def run_decompress(args):
    logger.info("Decompressing {}".format(args.file))
    if not args.file.endswith(".zarr"):
        raise ValueError("Compressed file must have .zarr suffix")
    ts = tszip.decompress(args.file)
    outfile = args.file[:-5]
    logger.info("Writing to {}".format(outfile))
    ts.dump(outfile)


def run_list(args):
    tszip.print_summary(args.file)


def tszip_main(arg_list=None):
    parser = tszip_cli_parser()
    args = parser.parse_args(arg_list)
    setup_logging(args)
    if args.decompress:
        run_decompress(args)
    elif args.list:
        run_list(args)
    else:
        run_compress(args)
