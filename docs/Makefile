# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first four.
SPHINX_APIDOC		?= sphinx-apidoc
SPHINX_APIDOC_OPTS	?= -f -T
SPHINX_BUILD		?= sphinx-build
SPHINX_BUILD_OPTS	?= -W --keep-going
MODULE_DIR		= ..
SOURCE_DIR		= .
BUILD_DIR		= _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINX_BUILD) -M help "$(SOURCE_DIR)" "$(BUILD_DIR)" $(SPHINX_BUILD_OPTS)

clean:
	@$(SPHINX_BUILD) -M clean "$(SOURCE_DIR)" "$(BUILD_DIR)" $(SPHINX_BUILD_OPTS)

.PHONY: clean help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new "make mode" option.
%: Makefile
	@$(SPHINX_APIDOC) -o "$(SOURCE_DIR)" "$(MODULE_DIR)" $(SPHINX_APIDOC_OPTS)
	@$(SPHINX_BUILD) -M $@ "$(SOURCE_DIR)" "$(BUILD_DIR)" $(SPHINX_BUILD_OPTS)
