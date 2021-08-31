# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

.PHONY: tests docs

tests:
	black --check .
	isort . --check --diff
	flake8 .
	pydocstyle .
	mypy .
	pytest --cov=. --cov-report=term-missing

docs:
	$(MAKE) -C docs html
