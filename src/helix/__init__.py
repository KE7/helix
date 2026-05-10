"""HELIX — Hierarchical Evolution via LLM-Informed eXploration."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from helix.asi import log

try:
    __version__ = _pkg_version("helix-evo")
except PackageNotFoundError:  # not installed (e.g. source checkout without pip install -e .)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "log"]
