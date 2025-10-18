"""
Compatibility shim that exposes the ``dp_textgrad`` Python package while the
source files live inside the ``dp-textgrad`` directory (requested rename).
"""

import importlib.util
import pathlib
import sys
from types import ModuleType

_PKG_NAME = "dp_textgrad"
_SOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "dp-textgrad"
_INIT_FILE = _SOURCE_DIR / "__init__.py"

if not _SOURCE_DIR.is_dir():
    raise ImportError(
        f"Expected source directory {_SOURCE_DIR!s} for {_PKG_NAME} package."
    )

if not _INIT_FILE.is_file():
    raise ImportError(f"Missing __init__.py in {_SOURCE_DIR!s}")

_module = ModuleType(_PKG_NAME)
_module.__file__ = str(_INIT_FILE)
_module.__package__ = _PKG_NAME
_module.__path__ = [str(_SOURCE_DIR)]  # type: ignore[attr-defined]
_module.__spec__ = importlib.util.spec_from_file_location(  # type: ignore[attr-defined]
    _PKG_NAME,
    _module.__file__,
    submodule_search_locations=[str(_SOURCE_DIR)],
)

sys.modules[_PKG_NAME] = _module

code = _INIT_FILE.read_text(encoding="utf-8")
exec(compile(code, _module.__file__, "exec"), _module.__dict__)
