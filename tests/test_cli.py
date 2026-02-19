# MIT License
#
# Copyright (c) 2019-2026 Tskit Developers
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
Test cases for the command line interface for tszip.
"""

import pathlib
import sys
import tempfile
from unittest import mock

import msprime
import numpy as np
import pytest
import tskit

import zarr
import tszip
import tszip.cli as cli


def get_stdout_for_pytest():
    """
    Pytest automatically wraps stdout to intercept output, but the object
    that it uses isn't fully compatible with the production implementation.
    Specifically, it doesn't provide a "buffer" attribute, which we
    need when writing binary data to it. This is a workaround to make
    out tests work.
    """
    return sys.stdout


class TestException(Exception):
    """
    Custom exception we can throw for testing.
    """

    # We don't want pytest to use this as a class to test
    __test__ = False


def capture_output(func, *args, binary=False, **kwargs):
    """
    Runs the specified function and arguments, and returns the
    tuple (stdout, stderr) as strings.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        stdout_path = pathlib.Path(tmpdir) / "stdout"
        stderr_path = pathlib.Path(tmpdir) / "stderr"
        mode = "wb+" if binary else "w+"
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        with open(stdout_path, mode) as stdout, open(stderr_path, mode) as stderr:
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with mock.patch("signal.signal"):
                    func(*args, **kwargs)
                stdout.seek(0)
                stderr.seek(0)
                stdout_output = stdout.read()
                stderr_output = stderr.read()
            finally:
                sys.stdout = saved_stdout
                sys.stderr = saved_stderr
    return stdout_output, stderr_output


class TestTszipArgumentParser:
    """
    Tests for the tszip argument parser.
    """

    def test_default_values(self):
        parser = cli.tszip_cli_parser()
        infile = "tmp.trees"
        args = parser.parse_args([infile])
        assert args.files == [infile]
        assert not args.keep
        assert not args.force
        assert not args.decompress
        assert not args.list
        assert not args.stdout
        assert args.chunk_size == tszip.DEFAULT_CHUNK_SIZE
        assert not args.variants_only
        assert args.suffix == ".tsz"

    def test_many_files(self):
        parser = cli.tszip_cli_parser()
        infiles = [f"{j}.trees" for j in range(1025)]
        args = parser.parse_args(infiles)
        assert args.files == infiles

    def test_suffix(self):
        parser = cli.tszip_cli_parser()
        infile = "tmp.trees"
        args = parser.parse_args([infile, "-S", ".XYZ"])
        assert args.suffix == ".XYZ"
        args = parser.parse_args([infile, "--suffix=abx"])
        assert args.suffix == "abx"

    def test_decompress(self):
        parser = cli.tszip_cli_parser()
        infile = "tmp.trees.tsz"
        args = parser.parse_args([infile, "-d"])
        assert args.decompress
        args = parser.parse_args([infile, "--decompress"])
        assert args.decompress

    def test_chunk_size(self):
        parser = cli.tszip_cli_parser()
        infile = "tmp.trees.tsz"
        args = parser.parse_args([infile, "-C", "1234"])
        assert args.chunk_size == 1234
        args = parser.parse_args([infile, "--chunk-size=1234"])
        assert args.chunk_size


class TestCli:
    """
    Superclass of tests that run the CLI.
    """

    # Need to mock out setup_logging here or we spew logging to the console
    # in later tests.
    @mock.patch("tszip.cli.setup_logging")
    def run_tszip(self, command, mock_setup_logging):
        stdout, stderr = capture_output(cli.tszip_main, command)
        assert stderr == ""
        assert stdout == ""
        assert mock_setup_logging.called

    @mock.patch("tszip.cli.setup_logging")
    def run_tsunzip(self, command, mock_setup_logging):
        stdout, stderr = capture_output(cli.tsunzip_main, command)
        assert stderr == ""
        assert stdout == ""
        assert mock_setup_logging.called

    @mock.patch("tszip.cli.setup_logging")
    def run_tszip_stdout(self, command, mock_setup_logging):
        stdout, stderr = capture_output(cli.tszip_main, command, binary=True)
        assert stderr == b""
        assert mock_setup_logging.called
        return stdout, stderr

    @mock.patch("tszip.cli.setup_logging")
    def run_tsunzip_stdout(self, command, mock_setup_logging):
        stdout, stderr = capture_output(cli.tsunzip_main, command, binary=True)
        assert stderr == b""
        assert mock_setup_logging.called
        return stdout, stderr


class TestBadFiles(TestCli):
    """
    Tests that we deal with IO errors appropriately.
    """

    def test_sys_exit(self):
        # We test for cli.exit elsewhere as it's easier, but test that sys.exit
        # is called here, so we get coverage.
        with mock.patch("sys.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                self.run_tszip(["/no/such/file"])
            mocked_exit.assert_called_once()
            args = mocked_exit.call_args[0]
            assert len(args) == 1
            assert "Error loading" in args[0]

    def test_compress_missing(self):
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                self.run_tszip(["/no/such/file"])
            mocked_exit.assert_called_once()
            args = mocked_exit.call_args[0]
            assert len(args) == 1
            assert args[0].startswith("Error loading")

    def test_decompress_missing(self):
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                self.run_tszip(["-d", "/no/such/file.tsz"])
            mocked_exit.assert_called_once()
            args = mocked_exit.call_args[0]
            assert len(args) == 1
            assert args[0].startswith("[Errno 2] No such file or directory")

    def test_list_missing(self):
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                self.run_tszip(["-l", "/no/such/file.tsz"])
            mocked_exit.assert_called_once()
            args = mocked_exit.call_args[0]
            assert len(args) == 1
            assert args[0].startswith("[Errno 2] No such file or directory")


class TestCompressSemantics(TestCli):
    """
    Tests that the semantics of the CLI work as expected.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.trees_path = tmp_path / "msprime.trees"
        self.ts = msprime.simulate(10, mutation_rate=10, random_seed=1)
        self.ts.dump(str(self.trees_path))

    def test_simple(self):
        assert self.trees_path.exists()
        self.run_tszip([str(self.trees_path)])
        assert not self.trees_path.exists()
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        assert outpath.exists()
        ts = tszip.decompress(outpath)
        assert ts.tables == self.ts.tables

    def test_suffix(self):
        assert self.trees_path.exists()
        self.run_tszip([str(self.trees_path), "-S", ".XYZasdf"])
        assert not self.trees_path.exists()
        outpath = pathlib.Path(str(self.trees_path) + ".XYZasdf")
        assert outpath.exists()
        ts = tszip.decompress(outpath)
        assert ts.tables == self.ts.tables

    def test_variants_only(self):
        assert self.trees_path.exists()
        self.run_tszip([str(self.trees_path), "--variants-only"])
        assert not self.trees_path.exists()
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        assert outpath.exists()
        ts = tszip.decompress(outpath)
        assert ts.tables != self.ts.tables
        G1 = ts.genotype_matrix()
        G2 = self.ts.genotype_matrix()
        assert np.array_equal(G1, G2)

    def test_chunk_size(self):
        assert self.trees_path.exists()
        self.run_tszip([str(self.trees_path), "--chunk-size=20"])
        assert not self.trees_path.exists()
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        assert outpath.exists()
        ts = tszip.decompress(outpath)
        assert ts.tables == self.ts.tables
        store = zarr.storage.ZipStore(str(outpath), mode="r")
        root = zarr.open_group(store=store, zarr_format=2, mode="r")
        for _, g in root.groups():
            for _, a in g.arrays():
                assert a.chunks == (20,)

    def test_keep(self):
        assert self.trees_path.exists()
        self.run_tszip([str(self.trees_path), "--keep"])
        assert self.trees_path.exists()
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        assert outpath.exists()
        ts = tszip.decompress(outpath)
        assert ts.tables == self.ts.tables

    def test_overwrite(self):
        assert self.trees_path.exists()
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        outpath.touch()
        assert self.trees_path.exists()
        self.run_tszip([str(self.trees_path), "--force"])
        assert not self.trees_path.exists()
        assert outpath.exists()
        ts = tszip.decompress(outpath)
        assert ts.tables == self.ts.tables

    def test_no_overwrite(self):
        assert self.trees_path.exists()
        outpath = pathlib.Path(str(self.trees_path) + ".tsz")
        outpath.touch()
        assert self.trees_path.exists()
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                self.run_tszip([str(self.trees_path)])
            mocked_exit.assert_called_once_with(
                f"'{outpath}' already exists; use --force to overwrite"
            )

    def test_bad_file_format(self):
        assert self.trees_path.exists()
        with open(str(self.trees_path), "w") as f:
            f.write("xxx")
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                self.run_tszip([str(self.trees_path)])
            args = mocked_exit.call_args[0]
            assert len(args) == 1
            error_message = args[0]
            assert "Error loading" in error_message
            assert "File not in kastore format" in error_message

    def test_compress_stdout_keep(self):
        assert self.trees_path.exists()
        with mock.patch("tszip.cli.get_stdout", wraps=get_stdout_for_pytest):
            self.run_tszip_stdout([str(self.trees_path)] + ["-c"])
        assert self.trees_path.exists()

    def test_compress_stdout_correct(self):
        assert self.trees_path.exists()
        tmp_file = self.tmp_path / "stdout.trees"
        with mock.patch("tszip.cli.get_stdout", wraps=get_stdout_for_pytest):
            stdout, stderr = self.run_tszip_stdout(["-c", str(self.trees_path)])
        with open(tmp_file, "wb+") as tmp:
            tmp.write(stdout)
        assert tmp_file.exists()
        ts = tszip.decompress(str(tmp_file))
        assert ts.tables == self.ts.tables

    def test_compress_stdout_multiple(self):
        assert self.trees_path.exists()
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                self.run_tszip_stdout(["-c", str(self.trees_path), str(self.trees_path)])
            mocked_exit.assert_called_once_with(
                "Only one file can be compressed on with '-c'"
            )


class DecompressSemanticsMixin:
    """
    Tests that the decompress semantics of the CLI work as expected.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.trees_path = tmp_path / "msprime.trees"
        self.ts = msprime.simulate(10, mutation_rate=10, random_seed=1)
        self.compressed_path = tmp_path / "msprime.trees.tsz"
        tszip.compress(self.ts, self.compressed_path)
        self.trees_path1 = tmp_path / "msprime1.trees"
        self.trees_path2 = tmp_path / "msprime2.trees"
        self.ts1 = msprime.simulate(10, mutation_rate=10, random_seed=3)
        self.ts2 = msprime.simulate(10, mutation_rate=5, random_seed=4)
        self.compressed_path1 = tmp_path / "msprime1.trees.tsz"
        self.compressed_path2 = tmp_path / "msprime2.trees.tsz"
        tszip.compress(self.ts1, self.compressed_path1)
        tszip.compress(self.ts2, self.compressed_path2)

    def test_simple(self):
        assert self.compressed_path.exists()
        self.run_decompress([str(self.compressed_path)])
        assert not self.compressed_path.exists()
        outpath = self.trees_path
        assert outpath.exists()
        ts = tskit.load(str(outpath))
        assert ts.tables == self.ts.tables

    def test_suffix(self):
        suffix = ".XYGsdf"
        self.compressed_path = self.compressed_path.with_suffix(suffix)
        tszip.compress(self.ts, self.compressed_path)
        assert self.compressed_path.exists()
        self.run_decompress([str(self.compressed_path), "-S", suffix])
        assert not self.compressed_path.exists()
        outpath = self.trees_path
        assert outpath.exists()
        ts = tskit.load(str(outpath))
        assert ts.tables == self.ts.tables

    def test_keep(self):
        assert self.compressed_path.exists()
        self.run_decompress([str(self.compressed_path), "--keep"])
        assert self.compressed_path.exists()
        outpath = self.trees_path
        assert outpath.exists()
        ts = tskit.load(str(outpath))
        assert ts.tables == self.ts.tables

    def test_keep_stdout(self):
        assert self.compressed_path.exists()
        self.run_decompress_stdout([str(self.compressed_path), "--stdout"])
        assert self.compressed_path.exists()
        self.run_decompress_stdout([str(self.compressed_path), "-c"])
        assert self.compressed_path.exists()

    def test_valid_stdout(self):
        tmp_file = self.tmp_path / "stdout.trees"
        stdout, stderr = self.run_decompress_stdout(["-c", str(self.compressed_path)])
        with open(tmp_file, "wb+") as tmp:
            tmp.write(stdout)
        ts = tskit.load(str(tmp_file))
        assert ts.tables == self.ts.tables
        assert self.compressed_path.exists()

    def test_valid_stdout_multiple(self):
        tmp_file = self.tmp_path / "stdout.trees"
        with open(tmp_file, "wb+") as tmp:
            stdout, stderr = self.run_decompress_stdout(
                ["-c", str(self.compressed_path1), str(self.compressed_path2)]
            )
            tmp.write(stdout)
        with open(tmp_file) as out_tmp:
            ts1 = tskit.load(out_tmp)
            ts2 = tskit.load(out_tmp)
        assert ts1.tables == self.ts1.tables
        assert ts2.tables == self.ts2.tables
        assert self.compressed_path1.exists()
        assert self.compressed_path2.exists()

    def test_overwrite(self):
        assert self.compressed_path.exists()
        outpath = self.trees_path
        outpath.touch()
        self.run_decompress([str(self.compressed_path), "-f"])
        assert not self.compressed_path.exists()
        assert outpath.exists()
        ts = tskit.load(str(outpath))
        assert ts.tables == self.ts.tables

    def test_no_overwrite(self):
        assert self.compressed_path.exists()
        outpath = self.trees_path
        outpath.touch()
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                self.run_decompress([str(self.compressed_path)])
            mocked_exit.assert_called_once_with(
                f"'{outpath}' already exists; use --force to overwrite"
            )

    def test_decompress_bad_suffix(self):
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                self.run_decompress([str(self.compressed_path), "-S", "asdf"])
            mocked_exit.assert_called_once_with(
                "Compressed file must have 'asdf' suffix"
            )

    def test_bad_file_format(self):
        assert self.compressed_path.exists()
        with open(str(self.compressed_path), "w") as f:
            f.write("xxx")
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                self.run_decompress([str(self.compressed_path)])
            mocked_exit.assert_called_once_with(
                f"Error reading '{self.compressed_path}': File is not in tszip format"
            )


class TestDecompressSemanticsTszip(DecompressSemanticsMixin, TestCli):
    def run_decompress(self, args):
        self.run_tszip(["-d"] + args)

    def run_decompress_stdout(self, args):
        x = self.run_tszip_stdout(["-d"] + args)
        return x


class TestDecompressSemanticsTsunzip(DecompressSemanticsMixin, TestCli):
    def run_decompress(self, args):
        self.run_tsunzip(args)

    def run_decompress_stdout(self, args):
        x = self.run_tsunzip_stdout(args)
        return x


class TestList:
    """
    Tests that the --list option works as expected.

    We don't need to mock out setup_logging here because it's not called for list.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.trees_path = tmp_path / "msprime.trees"
        self.ts = msprime.simulate(10, mutation_rate=10, random_seed=1)
        self.compressed_path = tmp_path / "msprime.trees.tsz"
        tszip.compress(self.ts, self.compressed_path)

    def test_simple(self):
        stdout, stderr = capture_output(
            cli.tszip_main, ["--list", str(self.compressed_path)]
        )
        assert stderr == ""
        lines = stdout.splitlines()
        assert lines[0].startswith(f"File: {self.compressed_path}")
        for line in lines:
            assert len(line) > 0

    def test_verbose(self):
        stdout, stderr = capture_output(
            cli.tszip_main, ["--list", "-v", str(self.compressed_path)]
        )
        assert stderr == ""
        lines = stdout.splitlines()
        assert lines[0].startswith(f"File: {self.compressed_path}")
        for line in lines:
            assert len(line) > 0

    def test_bad_file_format(self):
        assert self.compressed_path.exists()
        with open(str(self.compressed_path), "w") as f:
            f.write("xxx")
        with mock.patch("tszip.cli.exit", side_effect=TestException) as mocked_exit:
            with pytest.raises(TestException):
                cli.tszip_main([str(self.compressed_path), "-l"])
            mocked_exit.assert_called_once_with(
                f"Error reading '{self.compressed_path}': File is not in tszip format"
            )


class TestSetupLogging:
    """
    Tests that setup logging has the desired effect.
    """

    def test_default(self):
        parser = cli.tszip_cli_parser()
        args = parser.parse_args(["afile"])
        with mock.patch("logging.basicConfig") as mocked_setup:
            cli.setup_logging(args)
            mocked_setup.assert_called_once_with(level="WARN", format=cli.log_format)

    def test_verbose(self):
        parser = cli.tszip_cli_parser()
        args = parser.parse_args(["afile", "-v"])
        with mock.patch("logging.basicConfig") as mocked_setup:
            cli.setup_logging(args)
            mocked_setup.assert_called_once_with(level="INFO", format=cli.log_format)

    def test_very_verbose(self):
        parser = cli.tszip_cli_parser()
        args = parser.parse_args(["afile", "-vv"])
        with mock.patch("logging.basicConfig") as mocked_setup:
            cli.setup_logging(args)
            mocked_setup.assert_called_once_with(level="DEBUG", format=cli.log_format)
