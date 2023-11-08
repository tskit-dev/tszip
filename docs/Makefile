# Need to set PYTHONPATH so that we pick up the local tszip
PYPATH=${PWD}/..
TSZIP_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import tszip; print(tszip.__version__.split("+")[0])')

dev:
	PYTHONPATH=${PYPATH} ./build.sh

dist:
	@echo Building distribution for tszip version ${TSZIP_VERSION}
	sed -i s/__TSZIP_VERSION__/${TSZIP_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR _build
