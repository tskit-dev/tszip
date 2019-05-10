.. _sec_development:

===========
Development
===========

If you would like to add some features to ``tszip``, please read the
following. If you think there is anything missing,
please open an `issue <http://github.com/tszip-dev/tszip/issues>`_ or
`pull request <http://github.com/tszip-dev/tszip/pulls>`_ on GitHub!

**********
Quickstart
**********

- Make a fork of the tszip repo on `GitHub <http://github.com/tszip-dev/tszip>`_
- Clone your fork into a local directory::

  $ git clone git@github.com:YOUR_GITHUB/tszip.git

- Install the development requirements using
  ``python3 -m pip install -r requirements/development.txt``.
- Run the tests to ensure everything has worked: ``python3 -m nose -vs``. These should
  all pass.
- Make your changes in a local branch, and open a pull request on GitHub when you
  are ready. Please make sure that (a) the tests pass before you open the PR; and
  (b) your code passes PEP8 checks  before opening the PR.
