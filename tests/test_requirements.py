# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import tomllib

import pytest
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version


def parse_requirements(reqs: list[str], extra: str | None = None) -> dict[str, Version]:
    deps: dict[str, Version] = {}
    env = dict(default_environment())
    env['extra'] = extra or ''

    for requirement in reqs:
        try:
            req = Requirement(requirement)
        except InvalidRequirement:
            continue

        if req.marker is not None and not req.marker.evaluate(env):
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

        deps |= parse_requirements(data['optional-dependencies'][extra], extra=extra)
    deps |= parse_requirements(data['dependencies'])

    return deps


def test_min_requirements(pyproject: dict[str, Version]) -> None:
    with open('pyproject.toml', 'rb') as f:
        data = tomllib.load(f)['project']

    expected: set[str] = set()
    expected.update(parse_requirements(data['dependencies']))
    for extra in data['optional-dependencies']:
        if extra in {'all', 'docs', 'style'}:
            continue
        expected.update(
            parse_requirements(data['optional-dependencies'][extra], extra=extra)
        )

    assert set(pyproject) == expected
