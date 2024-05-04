# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest


def importandskip(modname: str, reason: str | None = None) -> None:
    """Exact opposite of :func:`pytest.importorskip`.

    Args:
        modname: The name of the module to import.
        reason: If given, this reason is shown as the message when the module can
            be imported.
    """
    try:
        __import__(modname)
        if reason is None:
            reason = f'could import {modname!r}'
        pytest.skip(reason)
    except ImportError:
        pass
