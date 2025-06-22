(sec_cli)=
# Command line interface

Tszip is intended to be used primarily as a command line interface. The interface for tszip is modelled directly on [gzip](http://linuxcommand.org/lc3_man_pages/gzip1.html), and so it should hopefully be immediately familiar and useful to many people. Tszip automatically installs the `tszip` and `tsunzip` programs, but depending on your setup, these may not be on your `PATH`. A slightly less convenient (but reliable) method of running tszip is the following:

```sh
python3 -m tszip
```

Online help is available using the `--help` option.

The `tsunzip` program is an alias for `tszip -d`.

## tszip

```{argparse}
:module: tszip.cli
:func: tszip_cli_parser
:prog: tszip
```
