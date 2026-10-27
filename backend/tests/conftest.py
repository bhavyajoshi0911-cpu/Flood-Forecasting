"""Keep the existing backend/app package importable beside the Flask app.py script."""

from __future__ import annotations

import sys
import types
from pathlib import Path


PACKAGE_DIRECTORY = Path(__file__).parents[1] / "app"

if "app" not in sys.modules or not hasattr(sys.modules["app"], "__path__"):
    sys.modules.pop("app", None)
    package_module = types.ModuleType("app")
    package_module.__path__ = [str(PACKAGE_DIRECTORY)]
    package_module.__package__ = "app"
    sys.modules["app"] = package_module
