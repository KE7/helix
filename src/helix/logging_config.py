"""HELIX logging configuration: attach a FileHandler to persist all log output.

Call :func:`setup_file_logging` once at CLI startup to ensure every
``logger.warning`` / ``logger.error`` from any ``helix.*`` module is
captured in ``<helix_dir>/helix.log`` for post-run diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_file_logging(helix_dir: Path) -> None:
    """Attach a rotating FileHandler to the ``helix`` package logger.

    Creates *helix_dir* / ``helix.log`` (appending, not truncating).  The
    handler uses formatter::

        %(asctime)s [%(levelname)s] %(name)s: %(message)s

    Safe to call multiple times — duplicate handlers for the same log path
    are silently skipped.

    Parameters
    ----------
    helix_dir:
        The ``.helix/`` directory for the current project.  Will be created
        if it does not exist.
    """
    helix_dir.mkdir(parents=True, exist_ok=True)
    log_file = helix_dir / "helix.log"

    helix_logger = logging.getLogger("helix")

    # Avoid adding a duplicate handler when setup_file_logging is called
    # multiple times (e.g. resume after evolve in the same process).
    log_path_str = str(log_file.resolve())
    for existing in helix_logger.handlers:
        if (
            isinstance(existing, logging.FileHandler)
            and existing.baseFilename == log_path_str
        ):
            return

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    helix_logger.addHandler(file_handler)

    # Ensure the helix logger is at least at INFO level so the handler fires.
    if helix_logger.level == logging.NOTSET or helix_logger.level > logging.INFO:
        helix_logger.setLevel(logging.INFO)
