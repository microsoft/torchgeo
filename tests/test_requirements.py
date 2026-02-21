# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import tomllib

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version


def parse_requirements(reqs: list[str]) -> dict[str, Version]:
    deps: dict[str, Version] = {}
    for requirement in reqs:
        try:
            req = Requirement(requirement)
        except InvalidRequirement:
            continue

        for spec in req.specifier:
            if spec.operator != '>=':
                continue

            try:
                deps[req.name] = Version(spec.version)
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


def test_min_requirements(pyproject: dict[str, Version]) -> None:
    with open('pyproject.toml', 'rb') as f:
        data = tomllib.load(f)['project']

    expected: set[str] = set()
    expected.update(Requirement(req).name for req in data['dependencies'])
    for extra in data['optional-dependencies']:
        if extra in {'all', 'docs', 'style'}:
            continue
        expected.update(
            Requirement(req).name for req in data['optional-dependencies'][extra]
        )

    assert set(pyproject) == expected
