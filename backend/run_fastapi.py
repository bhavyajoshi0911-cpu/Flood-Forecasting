"""Run the FastAPI application without colliding with the Flask app.py module."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import uvicorn


PACKAGE_NAME = "flood_forecasting_api"
PACKAGE_DIRECTORY = Path(__file__).resolve().parent / "app"


def load_application():
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_DIRECTORY)]
    package.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = package

    import importlib.util

    main_spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.main", PACKAGE_DIRECTORY / "main.py")
    if main_spec is None or main_spec.loader is None:
        raise RuntimeError("Could not load the FastAPI application")
    main_module = importlib.util.module_from_spec(main_spec)
    sys.modules[main_spec.name] = main_module
    main_spec.loader.exec_module(main_module)
    return main_module.app


if __name__ == "__main__":
    uvicorn.run(load_application(), host="127.0.0.1", port=8000)