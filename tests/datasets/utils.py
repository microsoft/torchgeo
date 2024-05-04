# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# https://github.com/pytest-dev/pytest/issues/7469#issuecomment-2094101076
from _pytest.outcomes import Skipped


def importandskip(modname: str, reason: str | None = None) -> None:
    """Exact opposite of :func:`pytest.importorskip`.

    Args:
        modname: The name of the module to import.
        reason: If given, this reason is shown as the message when the module can
            be imported.

    Raises:
        Skipped: If *modname* can be imported.
    """
    try:
        __import__(modname)
        if reason is None:
            reason = f'could import {modname!r}'
        raise Skipped(reason, allow_module_level=True) from None
    except ImportError:
        pass
