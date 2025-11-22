# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import tomllib

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version


def parse_requirements(reqs: list[str]) -> dict[str, Version]:
    deps = {}
    for requirement in reqs:
        try:
            req = Requirement(requirement)
        except InvalidRequirement:
            continue

        for spec in req.specifier:
            ver = str(spec).replace('==', '').replace('>=', '')
            try:
                deps[req.name] = Version(ver)
            except InvalidVersion:
                pass

    return deps


@pytest.fixture(scope='module')
def pyproject() -> dict[str, Version]:
    with open('pyproject.toml', 'rb') as f:
        data = tomllib.load(f)['project']

    deps: dict[str, Version] = {}
    for extra in data['optional-dependencies']:
        if extra in {'all', 'docs', 'style'}:
            continue

        deps |= parse_requirements(data['optional-dependencies'][extra])
    deps |= parse_requirements(data['dependencies'])

    return deps


@pytest.fixture(scope='module')
def requirements() -> dict[str, Version]:
    with open(os.path.join('requirements', 'min-reqs.old')) as f:
        data = f.readlines()

    return parse_requirements(data)


def test_min_requirements(
    pyproject: dict[str, Version], requirements: dict[str, Version]
) -> None:
    assert pyproject == requirements
