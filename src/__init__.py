"""BIPV facade analysis package.

Import heavy Colab pipeline pieces from ``src.pipeline`` when needed.
"""

from __future__ import annotations

import os


def patch_pillow_compatibility() -> None:
    """Patch Pillow utility attributes expected by some Colab dependencies."""

    try:
        import PIL._util
    except Exception:
        return

    if not hasattr(PIL._util, "is_directory"):
        PIL._util.is_directory = os.path.isdir
    if not hasattr(PIL._util, "is_path"):
        PIL._util.is_path = lambda f: isinstance(f, (str, bytes, os.PathLike))


patch_pillow_compatibility()
