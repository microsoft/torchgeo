#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Fix changelog images by replacing signed URLs with permanent ones.

GitHub converts user-attachments URLs to signed private-user-images URLs with
5-minute JWT expiration. This script reverses that transformation to use
permanent URLs that don't expire.
"""

import os
import re
from pathlib import Path


def fix_changelog_images() -> None:
    """Replace signed GitHub image URLs with permanent user-attachments URLs."""
    # Read the Docs builds to $READTHEDOCS_OUTPUT/html/
    changelog = Path('$READTHEDOCS_OUTPUT/html/user/changelog.html')

    # Expand environment variable
    changelog = Path(os.path.expandvars(str(changelog)))

    if not changelog.exists():
        print(f'Changelog not found at {changelog}')
        return

    content = changelog.read_text()

    # Pattern to match signed URLs and extract UUID
    # Example: https://private-user-images.githubusercontent.com/123/456-abcd1234-5678-90ab-cdef-ghijklmnopqr.png?jwt=...
    pattern = r'https://private-user-images\.githubusercontent\.com/\d+/\d+-([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\.([a-z]+)\?jwt=[^"\'>\s]+'

    def replace_url(match: re.Match[str]) -> str:
        uuid = match.group(1)
        return f'https://github.com/user-attachments/assets/{uuid}'

    fixed_content = re.sub(pattern, replace_url, content)

    # Count replacements
    original_count = len(re.findall(pattern, content))

    if original_count > 0:
        changelog.write_text(fixed_content)
        print(f'Fixed {original_count} changelog image URL(s)')
    else:
        print('No signed image URLs found to fix')


if __name__ == '__main__':
    fix_changelog_images()
