#!/bin/bash
DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DOCKER_DIR/shared.env"

set -e -x

ARCH=`uname -p`
echo "arch=$ARCH"


for V in "${PYTHON_VERSIONS[@]}"; do
    PYBIN=/opt/python/$V/bin
    rm -rf build/       # Avoid lib build by narrow Python is used by wide python
    $PYBIN/python -m pip install build twine
    $PYBIN/python -m build --wheel
done

# Validate all wheels with twine
for wheel in dist/*.whl; do
    /opt/python/cp310-cp310/bin/python -m twine check "$wheel"
done

rm dist/*.tar.gz