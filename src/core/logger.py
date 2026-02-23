import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from from_root import from_root


def _resolve_log_level() -> int:
    """Provide internal support for resolve log level.

    Used by this module to keep the main workflow functions focused and readable.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, level_name, logging.INFO)


def configure_logger() -> None:
    """Execute configure logger as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    """
    logger = logging.getLogger()
    if logger.handlers:
        return

    root_dir = Path(from_root())
    log_dir = root_dir / os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    log_level = _resolve_log_level()
    max_bytes = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    file_handler = RotatingFileHandler(
        filename=log_dir / "pipeline.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


configure_logger()
